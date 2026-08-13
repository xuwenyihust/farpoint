"""Dependency-light contracts and safety metrics for closed-loop policy rollout."""

from __future__ import annotations

import json
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
        for scene in payload["scenes"]:
            initial = scene.get("initial_state")
            if initial is None:
                raise ValueError("recovery replay scene requires initial_state")
            snapshot = {key: value for key, value in initial.items() if key != "snapshot_sha256"}
            if state_snapshot_sha256(snapshot) != initial["snapshot_sha256"]:
                raise ValueError("recovery replay state snapshot hash mismatch")
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
        "ever_lifted",
        "ever_entered_target",
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
