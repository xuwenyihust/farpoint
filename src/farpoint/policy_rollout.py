"""Dependency-light contracts and safety metrics for closed-loop policy rollout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from farpoint.contracts import validate_contract
from farpoint.demonstration import state_snapshot_sha256
from farpoint.so101 import (
    LEROBOT_JOINT_NAMES,
    LEROBOT_MAX,
    LEROBOT_MIN,
    USD_MAX_DEGREES,
    USD_MIN_DEGREES,
    lerobot_to_radians,
    radians_to_lerobot,
)


STABLE_GRASP_MIN_POLICY_STEPS = 3


@dataclass
class RolloutStageTracker:
    """Accumulate diagnostic stage evidence without changing task success."""

    policy_hz: int
    stable_grasp_min_steps: int = STABLE_GRASP_MIN_POLICY_STEPS
    first_contact_step: int | None = None
    first_bilateral_contact_step: int | None = None
    first_stable_grasp_step: int | None = None
    first_lift_step: int | None = None
    first_target_entry_step: int | None = None
    first_release_after_lift_step: int | None = None
    first_stable_release_step: int | None = None
    bilateral_streak: int = 0
    maximum_bilateral_streak: int = 0
    minimum_pre_contact_gripper_origin_to_object_center_distance_m: float | None = None

    def __post_init__(self) -> None:
        if self.policy_hz < 1:
            raise ValueError("policy_hz must be positive")
        if self.stable_grasp_min_steps < 1:
            raise ValueError("stable grasp duration must be positive")

    @staticmethod
    def _first(current: int | None, step: int, condition: bool) -> int | None:
        return step if current is None and condition else current

    def observe(
        self,
        *,
        policy_step: int,
        contact: bool,
        bilateral_contact: bool,
        lifted: bool,
        entered_target: bool,
        released: bool,
        stable_release: bool,
        gripper_origin_to_object_center_distance_m: float,
    ) -> None:
        if policy_step < 0:
            raise ValueError("policy_step must be non-negative")
        distance = float(gripper_origin_to_object_center_distance_m)
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("gripper-object distance must be finite and non-negative")
        if self.first_contact_step is None:
            if (
                self.minimum_pre_contact_gripper_origin_to_object_center_distance_m is None
                or distance < self.minimum_pre_contact_gripper_origin_to_object_center_distance_m
            ):
                self.minimum_pre_contact_gripper_origin_to_object_center_distance_m = distance
        self.first_contact_step = self._first(self.first_contact_step, policy_step, contact)
        self.first_bilateral_contact_step = self._first(
            self.first_bilateral_contact_step, policy_step, bilateral_contact
        )
        self.bilateral_streak = self.bilateral_streak + 1 if bilateral_contact else 0
        self.maximum_bilateral_streak = max(self.maximum_bilateral_streak, self.bilateral_streak)
        self.first_stable_grasp_step = self._first(
            self.first_stable_grasp_step,
            policy_step,
            self.bilateral_streak >= self.stable_grasp_min_steps,
        )
        self.first_lift_step = self._first(self.first_lift_step, policy_step, lifted)
        self.first_target_entry_step = self._first(
            self.first_target_entry_step, policy_step, entered_target
        )
        self.first_release_after_lift_step = self._first(
            self.first_release_after_lift_step,
            policy_step,
            released and self.first_lift_step is not None,
        )
        self.first_stable_release_step = self._first(
            self.first_stable_release_step, policy_step, stable_release
        )

    def result(self) -> dict[str, Any]:
        contact_to_lift_steps = (
            None
            if self.first_contact_step is None or self.first_lift_step is None
            else self.first_lift_step - self.first_contact_step
        )
        return {
            "metric_definition": {
                "stable_grasp_min_policy_steps": self.stable_grasp_min_steps,
                "stable_grasp_min_duration_s": self.stable_grasp_min_steps / self.policy_hz,
                "distance": "minimum through first contact: gripper rigid-body origin to object center",
                "diagnostic_only": True,
            },
            "ever_cube_contact": self.first_contact_step is not None,
            "ever_bilateral_contact": self.first_bilateral_contact_step is not None,
            "ever_stable_grasp": self.first_stable_grasp_step is not None,
            "ever_lifted": self.first_lift_step is not None,
            "ever_entered_target": self.first_target_entry_step is not None,
            "ever_released_after_lift": self.first_release_after_lift_step is not None,
            "ever_stable_release": self.first_stable_release_step is not None,
            "first_contact_step": self.first_contact_step,
            "first_bilateral_contact_step": self.first_bilateral_contact_step,
            "first_stable_grasp_step": self.first_stable_grasp_step,
            "first_lift_step": self.first_lift_step,
            "first_target_entry_step": self.first_target_entry_step,
            "first_release_after_lift_step": self.first_release_after_lift_step,
            "first_stable_release_step": self.first_stable_release_step,
            "maximum_consecutive_bilateral_contact_steps": self.maximum_bilateral_streak,
            "contact_to_lift_steps": contact_to_lift_steps,
            "contact_to_lift_s": (
                None if contact_to_lift_steps is None else contact_to_lift_steps / self.policy_hz
            ),
            "minimum_pre_contact_gripper_origin_to_object_center_distance_m": (
                self.minimum_pre_contact_gripper_origin_to_object_center_distance_m
            ),
        }


def json_default(value: Any) -> Any:
    """Convert NumPy scalar values emitted by simulation into JSON scalars."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def load_rollout_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_contract(payload)
    if errors:
        raise ValueError("invalid policy rollout contract:\n" + "\n".join(errors))
    scene_ids = [scene["scene_id"] for scene in payload["scenes"]]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("policy rollout scene_id values must be unique")
    acceptance = payload["acceptance"]
    if acceptance["required_completed_episodes"] != len(scene_ids):
        raise ValueError("required_completed_episodes must equal the frozen scene count")
    if acceptance["minimum_task_successes"] > len(scene_ids):
        raise ValueError("minimum_task_successes exceeds the frozen scene count")
    if payload["control"]["physics_hz"] % payload["control"]["policy_hz"] != 0:
        raise ValueError("physics_hz must be divisible by policy_hz")
    resolve_action_safety_profile(payload["control"])
    if payload["task"]["evaluation_class"].startswith("independent_holdout"):
        source = payload.get("holdout_source")
        if source is None:
            raise ValueError("independent holdout rollout requires holdout_source")
        if source["evaluated_scene_count"] != len(scene_ids):
            raise ValueError("evaluated holdout scene count does not match scenes")
    if payload["task"]["evaluation_class"] == "recovery_expert_replay":
        source = payload.get("recovery_replay_source")
        if source is None or source["evaluated_episode_count"] != len(scene_ids):
            raise ValueError("recovery replay source does not match scenes")
        state_restore = source["state_restore"]
        for scene in payload["scenes"]:
            initial = scene.get("initial_state")
            if state_restore == "handoff_snapshot_v1":
                if initial is None:
                    raise ValueError("recovery replay scene requires initial_state")
                snapshot = {
                    key: value for key, value in initial.items() if key != "snapshot_sha256"
                }
                if state_snapshot_sha256(snapshot) != initial["snapshot_sha256"]:
                    raise ValueError("recovery replay state snapshot hash mismatch")
            elif initial is not None:
                raise ValueError("full-history recovery replay must start from reset")
    return payload


def resolve_replan_interval(
    requested_steps: int | None,
    *,
    checkpoint_steps: int,
    chunk_size: int,
) -> int:
    """Resolve a frozen rollout replan interval against the ACT checkpoint."""
    if checkpoint_steps < 1 or chunk_size < 1 or checkpoint_steps > chunk_size:
        raise ValueError("invalid checkpoint action execution configuration")
    resolved = checkpoint_steps if requested_steps is None else requested_steps
    if resolved < 1:
        raise ValueError("replan_interval_steps must be positive")
    if resolved > chunk_size:
        raise ValueError("replan_interval_steps exceeds checkpoint chunk_size")
    return resolved


def resolve_action_safety_profile(control: dict[str, Any]) -> dict[str, Any]:
    """Resolve legacy or v1 action safety configuration into a six-joint profile.

    Legacy rollout specs intentionally retain their original target-versus-state
    limiter semantics. New profiles limit command slew against the previously
    applied command, which is the quantity comparable to a physical speed cap.
    """
    profile = control.get("action_safety_profile")
    if profile is None:
        maximum = float(control["max_delta_calibrated"])
        return {
            "schema_version": "farpoint.action-safety-profile.legacy-v0",
            "profile_id": "legacy-current-state-delta",
            "joint_order": list(LEROBOT_JOINT_NAMES),
            "limiter_reference": "current_position",
            "max_command_slew_calibrated_per_step": [maximum] * 6,
        }

    if profile["joint_order"] != list(LEROBOT_JOINT_NAMES):
        raise ValueError("action safety profile joint_order does not match SO-101")
    if "max_command_slew_calibrated_per_step" in profile:
        maximum = np.asarray(profile["max_command_slew_calibrated_per_step"], dtype=np.float64)
    else:
        degrees_per_step = float(profile["arm_max_command_speed_deg_s"]) / float(
            control["policy_hz"]
        )
        calibrated_per_degree = (LEROBOT_MAX - LEROBOT_MIN) / (USD_MAX_DEGREES - USD_MIN_DEGREES)
        maximum = np.concatenate(
            (
                degrees_per_step * calibrated_per_degree[:5],
                [float(profile["gripper_max_command_slew_calibrated_per_step"])],
            )
        )
    if maximum.shape != (6,) or not np.all(np.isfinite(maximum)) or np.any(maximum <= 0):
        raise ValueError("action safety command slew must be six finite positive values")
    return {
        **profile,
        "limiter_reference": "previous_applied_action",
        "max_command_slew_calibrated_per_step": maximum.tolist(),
    }


def initial_command_slew_reference(
    measured_state: Any, initial_state: dict[str, Any] | None
) -> np.ndarray:
    """Resolve the command preceding the first rollout action.

    A normal reset has no prior policy command, so the measured state is the
    only safe reference. A state-restored replay must instead continue from the
    exact command captured at handoff; using the lagging measured joints would
    introduce an artificial first-step limiter intervention.
    """
    reference = (
        measured_state
        if initial_state is None
        else initial_state.get("applied_policy_action_calibrated")
    )
    value = np.asarray(reference, dtype=np.float64)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("initial action slew reference must be six finite values")
    return value.copy()


def interpolate_command_endpoints(
    previous_action: Any, applied_action: Any, physics_substeps: int
) -> np.ndarray:
    """Resolve actuator targets between two policy-rate command endpoints."""
    previous = np.asarray(previous_action, dtype=np.float64)
    applied = np.asarray(applied_action, dtype=np.float64)
    if previous.shape != (6,) or applied.shape != (6,):
        raise ValueError("command endpoints must have shape (6,)")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(applied)):
        raise ValueError("command endpoints must be finite")
    if not isinstance(physics_substeps, int) or physics_substeps < 1:
        raise ValueError("physics_substeps must be a positive integer")
    fractions = np.arange(1, physics_substeps + 1, dtype=np.float64) / physics_substeps
    return previous[None, :] + fractions[:, None] * (applied - previous)[None, :]


def resolve_physics_action_group(
    previous_action: Any,
    applied_action: Any,
    policy_execution: dict[str, Any],
    physics_substeps: int,
) -> tuple[np.ndarray, str]:
    """Resolve physics targets in radians, preferring an exact audited trace."""
    replay_targets = policy_execution.get("physics_actions_radians")
    if replay_targets is None:
        calibrated = interpolate_command_endpoints(
            previous_action, applied_action, physics_substeps
        )
        return np.asarray(
            [lerobot_to_radians(row, clip=True) for row in calibrated],
            dtype=np.float64,
        ), "linear_endpoint"
    targets = np.asarray(replay_targets, dtype=np.float64)
    applied = np.asarray(applied_action, dtype=np.float64)
    if (
        policy_execution.get("physics_action_source") != "exact_trace"
        or targets.ndim != 2
        or targets.shape[1:] != (6,)
        or not 1 <= len(targets) <= physics_substeps
        or not np.isfinite(targets).all()
    ):
        raise ValueError("invalid exact physics action replay group")
    first_calibrated = radians_to_lerobot(targets[0], clip=True)
    if applied.shape != (6,) or not np.allclose(first_calibrated, applied, rtol=0.0, atol=1e-4):
        raise ValueError("exact physics action group does not start at policy action")
    return targets, "exact_trace"


def constrain_policy_action(
    raw_action: Any,
    current_position: Any,
    *,
    max_delta: float | None = None,
    action_safety_profile: dict[str, Any] | None = None,
    previous_applied_action: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(raw_action, dtype=np.float64)
    current = np.asarray(current_position, dtype=np.float64)
    if raw.shape != (6,) or current.shape != (6,):
        raise ValueError("SO-101 rollout actions and positions must have shape (6,)")
    if not np.all(np.isfinite(raw)):
        raise ValueError("SO-101 rollout action contains non-finite values")
    if not np.all(np.isfinite(current)):
        raise ValueError("SO-101 current position contains non-finite values")
    if (max_delta is None) == (action_safety_profile is None):
        raise ValueError("provide exactly one of max_delta or action_safety_profile")
    hard_mask = (raw < LEROBOT_MIN) | (raw > LEROBOT_MAX)
    hard_clipped = np.clip(raw, LEROBOT_MIN, LEROBOT_MAX)
    if action_safety_profile is None:
        maximum = np.full(6, float(max_delta), dtype=np.float64)
        reference = current
        limiter_reference = "current_position"
    else:
        maximum = np.asarray(
            action_safety_profile["max_command_slew_calibrated_per_step"], dtype=np.float64
        )
        if maximum.shape != (6,) or not np.all(np.isfinite(maximum)) or np.any(maximum <= 0):
            raise ValueError("action safety command slew must be six finite positive values")
        if previous_applied_action is None:
            raise ValueError("command-slew profile requires previous_applied_action")
        reference = np.asarray(previous_applied_action, dtype=np.float64)
        if reference.shape != (6,) or not np.all(np.isfinite(reference)):
            raise ValueError("previous applied action must contain six finite values")
        limiter_reference = "previous_applied_action"
    requested_delta = hard_clipped - reference
    numerical_tolerance = 1e-4 if action_safety_profile is not None else 0.0
    limited_mask = np.abs(requested_delta) > maximum + numerical_tolerance
    delta = np.where(
        limited_mask,
        np.clip(requested_delta, -maximum, maximum),
        requested_delta,
    )
    applied = np.clip(reference + delta, LEROBOT_MIN, LEROBOT_MAX)
    diagnostics = {
        "limiter_reference": limiter_reference,
        "hard_range_violation_count": int(np.count_nonzero(hard_mask)),
        "maximum_hard_range_excess_calibrated": float(np.max(np.abs(raw - hard_clipped))),
        "delta_limited_count": int(np.count_nonzero(limited_mask)),
        "command_slew_limited_count": int(np.count_nonzero(limited_mask)),
        "command_slew_numerical_tolerance_calibrated": numerical_tolerance,
        "maximum_raw_abs": float(np.max(np.abs(raw))),
        "maximum_applied_delta": float(np.max(np.abs(delta))),
        "maximum_tracking_error_calibrated": float(np.max(np.abs(applied - current))),
    }
    return applied.astype(np.float32), diagnostics


def summarize_action_errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize raw/applied one-step errors and safety interventions."""
    if not rows:
        raise ValueError("action error rows must not be empty")
    predicted = np.asarray([row["predicted"] for row in rows], dtype=np.float64)
    applied = np.asarray([row["applied"] for row in rows], dtype=np.float64)
    expert = np.asarray([row["expert"] for row in rows], dtype=np.float64)
    if (
        predicted.shape[1:] != (6,)
        or applied.shape != predicted.shape
        or expert.shape != predicted.shape
    ):
        raise ValueError("action error rows must contain shape-(6,) actions")
    if (
        not np.isfinite(predicted).all()
        or not np.isfinite(applied).all()
        or not np.isfinite(expert).all()
    ):
        raise ValueError("action error rows contain non-finite values")

    def error_metrics(values: np.ndarray) -> dict[str, Any]:
        absolute = np.abs(values - expert)
        l2 = np.linalg.norm(values - expert, axis=1)
        return {
            "mae_per_joint": np.mean(absolute, axis=0).tolist(),
            "rmse_per_joint": np.sqrt(np.mean((values - expert) ** 2, axis=0)).tolist(),
            "mean_l2": float(np.mean(l2)),
            "p50_l2": float(np.percentile(l2, 50)),
            "p95_l2": float(np.percentile(l2, 95)),
            "maximum_l2": float(np.max(l2)),
        }

    return {
        "sample_count": len(rows),
        "raw_prediction_error": error_metrics(predicted),
        "applied_prediction_error": error_metrics(applied),
        "prediction_safety": {
            "hard_range_violation_count": sum(
                row["prediction_safety"]["hard_range_violation_count"] for row in rows
            ),
            "delta_limited_count": sum(
                row["prediction_safety"]["delta_limited_count"] for row in rows
            ),
            "maximum_hard_range_excess_calibrated": max(
                row["prediction_safety"]["maximum_hard_range_excess_calibrated"] for row in rows
            ),
        },
        "expert_safety": {
            "hard_range_violation_count": sum(
                row["expert_safety"]["hard_range_violation_count"] for row in rows
            ),
            "delta_limited_count": sum(row["expert_safety"]["delta_limited_count"] for row in rows),
            "maximum_hard_range_excess_calibrated": max(
                row["expert_safety"]["maximum_hard_range_excess_calibrated"] for row in rows
            ),
        },
    }


def evaluate_rollout_acceptance(
    spec: dict[str, Any], episode_results: list[dict[str, Any]]
) -> dict[str, Any]:
    acceptance = spec["acceptance"]
    expected_ids = [scene["scene_id"] for scene in spec["scenes"]]
    actual_ids = [result["scene_id"] for result in episode_results]
    errors = []
    if actual_ids != expected_ids:
        errors.append("episode results do not match the frozen scene order")
    completed = sum(result.get("execution_status") == "FINISHED" for result in episode_results)
    successes = sum(bool(result.get("task_success")) for result in episode_results)
    nonfinite = sum(int(result.get("nonfinite_action_count", 0)) for result in episode_results)
    range_violations = sum(
        int(result.get("hard_range_violation_count", 0)) for result in episode_results
    )
    maximum_range_excess = max(
        (
            float(result.get("maximum_hard_range_excess_calibrated", 0.0))
            for result in episode_results
        ),
        default=0.0,
    )
    delta_limited = sum(int(result.get("delta_limited_count", 0)) for result in episode_results)
    if completed != acceptance["required_completed_episodes"]:
        errors.append("completed episode count does not meet the smoke contract")
    if successes < acceptance["minimum_task_successes"]:
        errors.append("task success count is below the frozen minimum")
    if nonfinite > acceptance["maximum_nonfinite_actions"]:
        errors.append("non-finite action count exceeds the frozen maximum")
    maximum_allowed_excess = acceptance.get("maximum_hard_range_excess_calibrated")
    if maximum_allowed_excess is None:
        if range_violations > acceptance["maximum_hard_range_violations"]:
            errors.append("hard-range action violations exceed the frozen maximum")
    elif maximum_range_excess > maximum_allowed_excess:
        errors.append("hard-range action excess exceeds the frozen safety envelope")
    maximum_delta_limited = acceptance.get("maximum_delta_limited_actions")
    if maximum_delta_limited is not None and delta_limited > maximum_delta_limited:
        errors.append("delta-limited action count exceeds the frozen maximum")
    stage_names = (
        "ever_cube_contact",
        "ever_bilateral_contact",
        "ever_stable_grasp",
        "ever_lifted",
        "ever_entered_target",
        "ever_released_after_lift",
        "ever_stable_release",
    )
    stage_progress = {
        name: sum(
            bool((result.get("stage_evidence") or {}).get(name)) for result in episode_results
        )
        for name in stage_names
    }
    terminal_reasons: dict[str, int] = {}
    for result in episode_results:
        reason = str(result.get("terminal_reason", "unknown"))
        terminal_reasons[reason] = terminal_reasons.get(reason, 0) + 1
    return {
        "status": "PASS" if not errors else "FAIL",
        "acceptance_errors": errors,
        "completed_episodes": completed,
        "task_successes": successes,
        "task_success_rate": successes / len(expected_ids),
        "nonfinite_action_count": nonfinite,
        "hard_range_violation_count": range_violations,
        "maximum_hard_range_excess_calibrated": maximum_range_excess,
        "delta_limited_count": delta_limited,
        "stage_progress": stage_progress,
        "terminal_reason_counts": dict(sorted(terminal_reasons.items())),
    }


def compare_paired_rollout_reports(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compare two policies on the same frozen scenes without hiding pairing."""
    baseline_episodes = baseline.get("episodes") or []
    candidate_episodes = candidate.get("episodes") or []
    baseline_ids = [row.get("scene_id") for row in baseline_episodes]
    candidate_ids = [row.get("scene_id") for row in candidate_episodes]
    if not baseline_ids or baseline_ids != candidate_ids:
        raise ValueError("paired rollout reports must contain the same ordered scenes")
    if len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("paired rollout scene IDs must be unique")
    if any(row.get("execution_status") != "FINISHED" for row in baseline_episodes):
        raise ValueError("baseline rollout contains incomplete episodes")
    if any(row.get("execution_status") != "FINISHED" for row in candidate_episodes):
        raise ValueError("candidate rollout contains incomplete episodes")
    for baseline_row, candidate_row in zip(baseline_episodes, candidate_episodes):
        if baseline_row.get("scene_context") != candidate_row.get("scene_context"):
            raise ValueError("paired rollout scene context differs")
    possible_context_keys = (
        "object_variant_id",
        "target_profile_id",
        "camera_profile_id",
        "region_band",
        "yaw_stratum_id",
    )
    context_keys = tuple(
        key
        for key in possible_context_keys
        if all(key in (row.get("scene_context") or {}) for row in baseline_episodes)
    )

    stage_names = (
        "ever_cube_contact",
        "ever_bilateral_contact",
        "ever_stable_grasp",
        "ever_lifted",
        "ever_entered_target",
        "ever_released_after_lift",
        "ever_stable_release",
    )

    def wilson_interval(successes: int, total: int) -> dict[str, Any]:
        if total < 1:
            raise ValueError("confidence intervals require at least one episode")
        z = 1.959963984540054
        rate = successes / total
        denominator = 1 + z**2 / total
        center = (rate + z**2 / (2 * total)) / denominator
        radius = z * np.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2))
        radius /= denominator
        return {
            "method": "wilson_score",
            "confidence_level": 0.95,
            "lower": float(max(0.0, center - radius)),
            "upper": float(min(1.0, center + radius)),
        }

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        episode_count = len(rows)
        task_successes = sum(bool(row.get("task_success")) for row in rows)
        stage_counts = {
            name: sum(bool((row.get("stage_evidence") or {}).get(name)) for row in rows)
            for name in stage_names
        }
        distances = [
            float(
                row["stage_evidence"][
                    "minimum_pre_contact_gripper_origin_to_object_center_distance_m"
                ]
            )
            for row in rows
            if (row.get("stage_evidence") or {}).get(
                "minimum_pre_contact_gripper_origin_to_object_center_distance_m"
            )
            is not None
        ]
        contact_to_lift = [
            float(row["stage_evidence"]["contact_to_lift_s"])
            for row in rows
            if (row.get("stage_evidence") or {}).get("contact_to_lift_s") is not None
        ]
        hard_range_counts = [int(row.get("hard_range_violation_count", 0)) for row in rows]
        delta_limited_counts = [int(row.get("delta_limited_count", 0)) for row in rows]
        nonfinite_counts = [int(row.get("nonfinite_action_count", 0)) for row in rows]
        maximum_range_excesses = [
            float(row.get("maximum_hard_range_excess_calibrated", 0.0)) for row in rows
        ]
        return {
            "episodes": episode_count,
            "task_successes": task_successes,
            "task_success_rate": task_successes / episode_count,
            "task_success_rate_ci95": wilson_interval(task_successes, episode_count),
            "stage_counts": stage_counts,
            "stage_rates": {name: count / episode_count for name, count in stage_counts.items()},
            "stage_rate_ci95": {
                name: wilson_interval(count, episode_count) for name, count in stage_counts.items()
            },
            "minimum_pre_contact_gripper_object_distance_median": (
                None if not distances else float(np.median(distances))
            ),
            "contact_to_lift_s_median": (
                None if not contact_to_lift else float(np.median(contact_to_lift))
            ),
            "hard_range_violation_count": sum(hard_range_counts),
            "hard_range_violation_count_per_episode_mean": float(np.mean(hard_range_counts)),
            "delta_limited_count": sum(delta_limited_counts),
            "delta_limited_count_per_episode_mean": float(np.mean(delta_limited_counts)),
            "nonfinite_action_count": sum(nonfinite_counts),
            "nonfinite_action_count_per_episode_mean": float(np.mean(nonfinite_counts)),
            "maximum_hard_range_excess_calibrated": max(maximum_range_excesses),
        }

    baseline_summary = summarize(baseline_episodes)
    candidate_summary = summarize(candidate_episodes)
    paired_outcomes = {
        "improved": 0,
        "regressed": 0,
        "unchanged_success": 0,
        "unchanged_failure": 0,
    }
    for baseline_row, candidate_row in zip(baseline_episodes, candidate_episodes):
        before = bool(baseline_row.get("task_success"))
        after = bool(candidate_row.get("task_success"))
        key = (
            "improved"
            if not before and after
            else "regressed"
            if before and not after
            else "unchanged_success"
            if before
            else "unchanged_failure"
        )
        paired_outcomes[key] += 1

    strata: dict[str, Any] = {}
    for context_key in context_keys:
        values = sorted({str(row["scene_context"][context_key]) for row in baseline_episodes})
        strata[context_key] = {}
        for value in values:
            indexes = [
                index
                for index, row in enumerate(baseline_episodes)
                if str(row["scene_context"][context_key]) == value
            ]
            strata[context_key][value] = {
                "baseline": summarize([baseline_episodes[index] for index in indexes]),
                "candidate": summarize([candidate_episodes[index] for index in indexes]),
            }

    return {
        "schema_version": "farpoint.paired-policy-rollout-comparison.v1",
        "scene_count": len(baseline_ids),
        "scene_ids": baseline_ids,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta": {
            "task_successes": candidate_summary["task_successes"]
            - baseline_summary["task_successes"],
            "stage_counts": {
                name: candidate_summary["stage_counts"][name]
                - baseline_summary["stage_counts"][name]
                for name in stage_names
            },
            "hard_range_violation_count": candidate_summary["hard_range_violation_count"]
            - baseline_summary["hard_range_violation_count"],
            "delta_limited_count": candidate_summary["delta_limited_count"]
            - baseline_summary["delta_limited_count"],
            "nonfinite_action_count": candidate_summary["nonfinite_action_count"]
            - baseline_summary["nonfinite_action_count"],
        },
        "paired_task_outcomes": paired_outcomes,
        "strata": strata,
        "confidence_interval_policy": {
            "binary_rates": "two-sided 95% Wilson score interval",
        },
    }
