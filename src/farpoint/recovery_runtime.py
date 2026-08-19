"""Dependency-light contracts and triggers for live ACT-to-Oracle recovery."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np

from farpoint.contracts import validate_contract
from farpoint.handoff_stage import HANDOFF_STAGE_SCHEMA_VERSION, derive_handoff_stage
from farpoint.policy_rollout import resolve_action_safety_profile
from farpoint.so101 import lerobot_to_radians, radians_to_lerobot


def canonical_recovery_failure_taxonomy(
    failure_class: str, evidence: dict[str, Any]
) -> dict[str, str]:
    """Derive canonical stage taxonomy without rewriting legacy failure labels."""
    if failure_class != "transport_drift":
        return {}
    if evidence.get("gripper_released") and not evidence.get("cube_in_target"):
        subclass = "premature_release_outside_target"
    elif (
        not evidence.get("has_contact")
        or not evidence.get("cube_lifted")
        or not evidence.get("secure_grasp")
    ):
        subclass = "transport_drop"
    else:
        subclass = "transport_stall"
    return {
        "failure_stage": "transport",
        "last_completed_stage": "lift",
        "failure_subclass": subclass,
    }


def load_recovery_runtime(path: Path) -> dict[str, Any]:
    """Load and semantically validate an immutable recovery runtime spec."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_contract(payload)
    if errors:
        raise ValueError("invalid recovery runtime contract:\n" + "\n".join(errors))
    triggers = (
        {"legacy": payload["trigger"]}
        if payload["schema_version"] == "farpoint.recovery-runtime.v1"
        else payload["trigger_profiles"]
    )
    for trigger_class, trigger in triggers.items():
        if trigger["minimum_policy_steps"] >= trigger["maximum_policy_steps_before_handoff"]:
            raise ValueError(
                f"minimum_policy_steps must precede the handoff deadline: {trigger_class}"
            )
        window_key = "stall_window_steps" if "stall_window_steps" in trigger else "window_steps"
        if trigger[window_key] > trigger["minimum_policy_steps"]:
            raise ValueError(f"trigger window must fit before admission: {trigger_class}")
    scene_ids = [scene["source_scene_id"] for scene in payload["scenes"]]
    variation_ids = [scene["variation_id"] for scene in payload["scenes"]]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("recovery source_scene_id values must be unique")
    if len(variation_ids) != len(set(variation_ids)):
        raise ValueError("recovery variation_id values must be unique")
    if payload["schema_version"] == "farpoint.recovery-runtime.v2":
        unknown = sorted(
            {
                scene["trigger_class"]
                for scene in payload["scenes"]
                if scene["trigger_class"] not in payload["trigger_profiles"]
            }
        )
        if unknown:
            raise ValueError(
                "recovery scenes reference unknown trigger classes: " + ", ".join(unknown)
            )
        for key, profile in payload["trigger_profiles"].items():
            if key != profile["failure_class"]:
                raise ValueError(f"trigger profile key/failure_class mismatch: {key}")
    return payload


def scene_binding(spec: dict[str, Any], variation_id: str) -> dict[str, Any]:
    matches = [scene for scene in spec["scenes"] if scene["variation_id"] == variation_id]
    if len(matches) != 1:
        raise ValueError(f"recovery runtime has no unique binding for {variation_id}")
    return deepcopy(matches[0])


def recovery_trigger_for_scene(spec: dict[str, Any], variation_id: str) -> dict[str, Any]:
    """Resolve one frozen trigger profile while preserving v1 behavior."""
    if spec["schema_version"] == "farpoint.recovery-runtime.v1":
        return deepcopy(spec["trigger"])
    binding = scene_binding(spec, variation_id)
    profile = deepcopy(spec["trigger_profiles"][binding["trigger_class"]])
    required_subclass = binding.get("required_failure_subclass")
    if required_subclass is not None:
        profile["required_failure_subclass"] = required_subclass
    return profile


def recovery_oracle_entry_phase(trigger: dict[str, Any]) -> str:
    """Choose the Oracle phase that preserves the measured recovery stage.

    Early-stage failures need a fresh grasp.  A policy that is already carrying
    the object must not be sent through PREGRASP again: cube contact is expected
    in that state and the approach collision gate would reject it immediately.
    Place/release recovery can continue even later when the cube is already on
    the target.
    """
    failure_class = trigger.get("failure_class")
    handoff_stage = trigger.get("handoff_stage")
    if handoff_stage == "approach":
        return "pregrasp"
    has_contact = bool(trigger.get("has_contact", False))
    ever_lifted = bool(trigger.get("ever_lifted", trigger.get("cube_lifted", False)))
    forces = np.asarray(trigger.get("contact_forces_n", (0.0, 0.0)), dtype=np.float64)
    bilateral_contact = bool(forces.shape == (2,) and np.min(forces) >= 0.1)
    lift_height_m = float(trigger.get("lift_height_m", 0.0))
    securely_carried = (
        has_contact
        and bilateral_contact
        and bool(trigger.get("cube_lifted", False))
        and not bool(trigger.get("gripper_released", False))
        and lift_height_m >= 0.010
    )
    if handoff_stage == "grasp":
        # A live fingertip contact is legitimate recovery state, but PREGRASP
        # deliberately rejects any contact during its collision-safe route.
        # Open and retreat through HOME before attempting a fresh grasp.
        return "home" if has_contact else "pregrasp"
    if handoff_stage == "lift":
        # The Oracle's LIFT phase assumes private capture state established by
        # its own close/verification sequence.  A policy handoff cannot safely
        # synthesize that state, so preserve the scene and re-enter via HOME.
        return "home"
    if handoff_stage in {"transport", "place"}:
        # ``ever_lifted`` alone also describes a cube that has already fallen
        # back to the table.  Continue transport only from a currently secure,
        # bilateral grasp with useful clearance; otherwise retreat and regrasp
        # without resetting the live scene.
        return "preplace" if securely_carried and ever_lifted else "home"
    if handoff_stage == "release":
        return "open"
    if handoff_stage == "settle":
        return "settle"
    # Frozen v1/v2 evidence did not carry a versioned handoff stage.
    if failure_class == "approach_miss":
        return "pregrasp"
    if failure_class == "contact_without_lift":
        return "home" if has_contact else "pregrasp"
    if failure_class == "transport_drift":
        return "preplace" if securely_carried and ever_lifted else "home"
    if failure_class == "place_release_failure":
        if bool(trigger.get("cube_in_target", False)):
            return "settle" if bool(trigger.get("gripper_released", False)) else "open"
        return "preplace" if securely_carried and ever_lifted else "home"
    return "pregrasp"


def recovery_descent_duration_seconds(spec: dict[str, Any] | None) -> float:
    """Resolve the versioned Oracle insertion duration for a live handoff."""
    if spec is None:
        return 2.3333333333
    return float(spec["oracle_handoff_profile"]["descent_duration_seconds"])


def recovery_oracle_command_continuity_enabled(spec: dict[str, Any] | None) -> bool:
    """Keep legacy recovery runtimes unchanged unless the profile opts in."""
    if spec is None:
        return False
    return (
        spec["oracle_handoff_profile"].get("command_continuity")
        == "action_safety_profile_control_rate_v1"
    )


def recovery_oracle_slew_limits(spec: dict[str, Any]) -> np.ndarray:
    """Resolve the 120 Hz Oracle target envelope from the 30 Hz actuator profile."""
    control = spec["control"]
    control_hz = int(control["physics_hz"])
    policy_hz = int(control["policy_hz"])
    if control_hz <= 0 or policy_hz <= 0 or control_hz % policy_hz:
        raise ValueError("recovery control frequencies must have an integer positive ratio")
    per_policy_step = np.asarray(
        resolve_action_safety_profile(control)["max_command_slew_calibrated_per_step"],
        dtype=np.float64,
    )
    if per_policy_step.shape != (6,) or not np.all(np.isfinite(per_policy_step)):
        raise ValueError("recovery action safety profile must resolve six finite limits")
    return per_policy_step / (control_hz // policy_hz)


def slew_recovery_oracle_target(
    previous_target_rad: Any,
    requested_target_rad: Any,
    maximum_delta_calibrated: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bound one Oracle target without resetting or rebasing the physical state."""
    previous = np.asarray(previous_target_rad, dtype=np.float64)
    requested = np.asarray(requested_target_rad, dtype=np.float64)
    maximum = np.asarray(maximum_delta_calibrated, dtype=np.float64)
    if previous.shape != (6,) or requested.shape != (6,) or maximum.shape != (6,):
        raise ValueError("recovery Oracle targets and slew limits must have shape (6,)")
    if not all(np.all(np.isfinite(value)) for value in (previous, requested, maximum)):
        raise ValueError("recovery Oracle targets and slew limits must be finite")
    if np.any(maximum <= 0):
        raise ValueError("recovery Oracle slew limits must be positive")
    previous_calibrated = radians_to_lerobot(previous, clip=True)
    requested_calibrated = radians_to_lerobot(requested, clip=True)
    requested_delta = requested_calibrated - previous_calibrated
    applied_delta = np.clip(requested_delta, -maximum, maximum)
    applied_calibrated = previous_calibrated + applied_delta
    limited = np.abs(requested_delta) > maximum
    return lerobot_to_radians(applied_calibrated, clip=True).astype(np.float32), {
        "limiter_reference": "previous_oracle_target",
        "limited_joint_count": int(np.count_nonzero(limited)),
        "requested_delta_calibrated": requested_delta.tolist(),
        "applied_delta_calibrated": applied_delta.tolist(),
        "maximum_applied_delta_calibrated": float(np.max(np.abs(applied_delta))),
    }


class RecoveryTriggerDetector:
    """Detect one frozen recovery class from measured closed-loop state.

    A v2 scene is intentionally assigned exactly one class.  This prevents an
    easy early trigger from starving later transport/place recovery coverage.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = deepcopy(config)
        window = int(config.get("stall_window_steps", config.get("window_steps", 2)))
        self._distance_history: deque[float] = deque(maxlen=window)
        self._consecutive_safety = 0
        self._initial_object_position: np.ndarray | None = None
        self._ever_contact = False
        self._ever_lifted = False
        self._ever_near_target = False

    def _finalize_profile_trigger(
        self, result: dict[str, Any], *, admission_stage: str | None
    ) -> dict[str, Any] | None:
        result.update(canonical_recovery_failure_taxonomy(str(result["failure_class"]), result))
        result["handoff_stage_schema_version"] = HANDOFF_STAGE_SCHEMA_VERSION
        result["handoff_stage"] = derive_handoff_stage(result)
        result["stage"] = result["handoff_stage"]
        result["reason"] = result["trigger_reason"]
        if admission_stage:
            result["source_stage_label"] = admission_stage
        required_stage = self.config.get("required_handoff_stage")
        if required_stage is not None and result["handoff_stage"] != required_stage:
            return None
        required_evidence = self.config.get("required_trigger_evidence") or {}
        if any(result.get(key) != expected for key, expected in required_evidence.items()):
            return None
        required_fields = {
            "failure_stage": self.config.get("required_failure_stage"),
            "last_completed_stage": self.config.get("required_last_completed_stage"),
        }
        if any(
            expected is not None and result.get(key) != expected
            for key, expected in required_fields.items()
        ):
            return None
        allowed_subclasses = self.config.get("allowed_failure_subclasses")
        if allowed_subclasses is not None and result.get("failure_subclass") not in set(
            allowed_subclasses
        ):
            return None
        required_subclass = self.config.get("required_failure_subclass")
        if required_subclass is not None and result.get("failure_subclass") != required_subclass:
            return None
        return result

    def observe(
        self,
        *,
        policy_step: int,
        gripper_position_m: Any,
        object_position_m: Any,
        cube_lifted: bool,
        hard_range_violation_count: int,
        command_slew_limited_count: int,
        contact_forces_n: Any | None = None,
        target_position_m: Any | None = None,
        cube_in_target: bool = False,
        gripper_released: bool = False,
        cube_stable: bool = False,
    ) -> dict[str, Any] | None:
        gripper = np.asarray(gripper_position_m, dtype=np.float64)
        obj = np.asarray(object_position_m, dtype=np.float64)
        if gripper.shape != (3,) or obj.shape != (3,):
            raise ValueError("recovery trigger positions must have shape (3,)")
        if not np.isfinite(gripper).all() or not np.isfinite(obj).all():
            raise ValueError("recovery trigger positions must be finite")
        if policy_step < 0:
            raise ValueError("policy_step must be non-negative")
        distance = float(np.linalg.norm(gripper - obj))
        if self._initial_object_position is None:
            self._initial_object_position = obj.copy()
        forces = np.zeros(2, dtype=np.float64)
        if contact_forces_n is not None:
            forces = np.asarray(contact_forces_n, dtype=np.float64)
            if forces.shape != (2,) or not np.isfinite(forces).all():
                raise ValueError(
                    "recovery trigger contact forces must have shape (2,) and be finite"
                )
        force_threshold = float(self.config.get("contact_force_threshold_n", 0.1))
        has_contact = bool(np.max(forces) >= force_threshold)
        self._ever_contact = self._ever_contact or has_contact
        self._ever_lifted = self._ever_lifted or bool(cube_lifted)
        target_distance = None
        if target_position_m is not None:
            target = np.asarray(target_position_m, dtype=np.float64)
            if target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError(
                    "recovery trigger target position must have shape (3,) and be finite"
                )
            target_distance = float(np.linalg.norm(obj[:2] - target[:2]))
            self._ever_near_target = (
                self._ever_near_target
                or bool(cube_in_target)
                or (target_distance <= float(self.config.get("target_distance_threshold_m", 0.04)))
            )
        failure_class = self.config.get("failure_class")
        tracked_distance = (
            target_distance
            if failure_class in {"transport_drift", "place_release_failure"}
            and target_distance is not None
            else distance
        )
        self._distance_history.append(float(tracked_distance))
        safety_event = hard_range_violation_count > 0 or command_slew_limited_count > 0
        self._consecutive_safety = self._consecutive_safety + 1 if safety_event else 0
        if cube_lifted and self.config.get("require_not_lifted", False):
            return None
        if policy_step + 1 < int(self.config["minimum_policy_steps"]):
            return None

        common = {
            "policy_step": int(policy_step),
            "gripper_object_distance_m": distance,
            "cube_lifted": bool(cube_lifted),
            "ever_lifted": self._ever_lifted,
            "ever_contact": self._ever_contact,
            "ever_near_target": self._ever_near_target,
            "has_contact": has_contact,
            "contact_forces_n": forces.tolist(),
            "contact_force_threshold_n": force_threshold,
            "secure_grasp": bool(np.min(forces) >= force_threshold and not gripper_released),
            "cube_in_target": bool(cube_in_target),
            "gripper_released": bool(gripper_released),
            "cube_stable": bool(cube_stable),
            "lift_height_m": float(obj[2] - self._initial_object_position[2]),
            "consecutive_safety_event_steps": self._consecutive_safety,
        }
        if target_distance is not None:
            common["object_target_distance_m"] = target_distance
        if failure_class is not None:
            common["object_displacement_m"] = float(
                np.linalg.norm(obj - self._initial_object_position)
            )
            admission_stage = self.config.get("admission_stage", self.config.get("stage"))
            stage_admitted = {
                "approach_miss": not self._ever_contact,
                "contact_without_lift": self._ever_contact and not self._ever_lifted,
                "transport_drift": self._ever_lifted,
                "place_release_failure": self._ever_near_target,
            }.get(failure_class)
            if stage_admitted is None:
                raise ValueError(f"unsupported recovery trigger class: {failure_class}")
            if stage_admitted and self._consecutive_safety >= int(
                self.config["consecutive_safety_event_steps"]
            ):
                result = {
                    **common,
                    "failure_class": failure_class,
                    "trigger_reason": "consecutive_action_safety_intervention",
                }
                finalized = self._finalize_profile_trigger(result, admission_stage=admission_stage)
                if finalized is not None:
                    return finalized
            deadline = policy_step + 1 >= int(self.config["maximum_policy_steps_before_handoff"])
            progress = None
            if len(self._distance_history) == self._distance_history.maxlen:
                progress = float(self._distance_history[0] - self._distance_history[-1])
            stalled = progress is not None and progress < float(self.config["minimum_progress_m"])
            if failure_class == "approach_miss":
                ready = not self._ever_contact and (stalled or deadline)
            elif failure_class == "contact_without_lift":
                ready = self._ever_contact and not self._ever_lifted and (stalled or deadline)
            elif failure_class == "transport_drift":
                ready = self._ever_lifted and ((not has_contact) or stalled or deadline)
            elif failure_class == "place_release_failure":
                ready = (
                    self._ever_near_target
                    and not (cube_in_target and gripper_released and cube_stable)
                    and (stalled or deadline)
                )
            if ready:
                result = {
                    **common,
                    "failure_class": failure_class,
                    "trigger_reason": (
                        "stage_progress_stall" if stalled else "stage_handoff_deadline"
                    ),
                }
                if progress is not None:
                    result["window_progress_m"] = progress
                return self._finalize_profile_trigger(result, admission_stage=admission_stage)
            return None
        if self._consecutive_safety >= int(self.config["consecutive_safety_event_steps"]):
            result = {
                **common,
                "failure_class": "action_saturation",
                "trigger_reason": "consecutive_action_safety_intervention",
            }
            result["handoff_stage_schema_version"] = HANDOFF_STAGE_SCHEMA_VERSION
            result["handoff_stage"] = derive_handoff_stage(result)
            result["stage"] = result["handoff_stage"]
            result["source_stage_label"] = "pre_lift"
            result["reason"] = result["trigger_reason"]
            return result
        if len(self._distance_history) == self._distance_history.maxlen:
            progress = float(self._distance_history[0] - self._distance_history[-1])
            if progress < float(self.config["minimum_progress_m"]):
                result = {
                    **common,
                    "failure_class": "progress_stall",
                    "trigger_reason": "insufficient_gripper_object_progress",
                    "window_progress_m": progress,
                }
                result["handoff_stage_schema_version"] = HANDOFF_STAGE_SCHEMA_VERSION
                result["handoff_stage"] = derive_handoff_stage(result)
                result["stage"] = result["handoff_stage"]
                result["source_stage_label"] = "pre_lift"
                result["reason"] = result["trigger_reason"]
                return result
        if policy_step + 1 >= int(self.config["maximum_policy_steps_before_handoff"]):
            result = {
                **common,
                "failure_class": "progress_stall",
                "trigger_reason": "bounded_pre_lift_handoff_deadline",
            }
            result["handoff_stage_schema_version"] = HANDOFF_STAGE_SCHEMA_VERSION
            result["handoff_stage"] = derive_handoff_stage(result)
            result["stage"] = result["handoff_stage"]
            result["source_stage_label"] = "pre_lift"
            result["reason"] = result["trigger_reason"]
            return result
        return None
