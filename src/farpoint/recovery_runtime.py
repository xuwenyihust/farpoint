"""Dependency-light contracts and triggers for live ACT-to-Oracle recovery."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np

from farpoint.contracts import validate_contract


def load_recovery_runtime(path: Path) -> dict[str, Any]:
    """Load and semantically validate an immutable recovery runtime spec."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_contract(payload)
    if errors:
        raise ValueError("invalid recovery runtime contract:\n" + "\n".join(errors))
    trigger = payload["trigger"]
    if trigger["minimum_policy_steps"] >= trigger["maximum_policy_steps_before_handoff"]:
        raise ValueError("minimum_policy_steps must precede the handoff deadline")
    if trigger["stall_window_steps"] > trigger["minimum_policy_steps"]:
        raise ValueError("stall_window_steps must fit before trigger admission")
    scene_ids = [scene["source_scene_id"] for scene in payload["scenes"]]
    variation_ids = [scene["variation_id"] for scene in payload["scenes"]]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("recovery source_scene_id values must be unique")
    if len(variation_ids) != len(set(variation_ids)):
        raise ValueError("recovery variation_id values must be unique")
    return payload


def scene_binding(spec: dict[str, Any], variation_id: str) -> dict[str, Any]:
    matches = [scene for scene in spec["scenes"] if scene["variation_id"] == variation_id]
    if len(matches) != 1:
        raise ValueError(f"recovery runtime has no unique binding for {variation_id}")
    return deepcopy(matches[0])


def recovery_descent_duration_seconds(spec: dict[str, Any] | None) -> float:
    """Resolve the versioned Oracle insertion duration for a live handoff."""
    if spec is None:
        return 2.3333333333
    return float(spec["oracle_handoff_profile"]["descent_duration_seconds"])


class RecoveryTriggerDetector:
    """Detect a bounded pre-lift deviation using measured closed-loop state."""

    def __init__(self, config: dict[str, Any]):
        self.config = deepcopy(config)
        self._distance_history: deque[float] = deque(maxlen=int(config["stall_window_steps"]))
        self._consecutive_safety = 0

    def observe(
        self,
        *,
        policy_step: int,
        gripper_position_m: Any,
        object_position_m: Any,
        cube_lifted: bool,
        hard_range_violation_count: int,
        command_slew_limited_count: int,
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
        self._distance_history.append(distance)
        safety_event = hard_range_violation_count > 0 or command_slew_limited_count > 0
        self._consecutive_safety = self._consecutive_safety + 1 if safety_event else 0
        if cube_lifted and self.config["require_not_lifted"]:
            return None
        if policy_step + 1 < int(self.config["minimum_policy_steps"]):
            return None

        common = {
            "policy_step": int(policy_step),
            "gripper_object_distance_m": distance,
            "cube_lifted": bool(cube_lifted),
            "consecutive_safety_event_steps": self._consecutive_safety,
        }
        if self._consecutive_safety >= int(self.config["consecutive_safety_event_steps"]):
            return {
                **common,
                "failure_class": "action_saturation",
                "stage": "pre_lift",
                "reason": "consecutive_action_safety_intervention",
            }
        if len(self._distance_history) == self._distance_history.maxlen:
            progress = float(self._distance_history[0] - self._distance_history[-1])
            if progress < float(self.config["minimum_progress_m"]):
                return {
                    **common,
                    "failure_class": "progress_stall",
                    "stage": "pre_lift",
                    "reason": "insufficient_gripper_object_progress",
                    "window_progress_m": progress,
                }
        if policy_step + 1 >= int(self.config["maximum_policy_steps_before_handoff"]):
            return {
                **common,
                "failure_class": "progress_stall",
                "stage": "pre_lift",
                "reason": "bounded_pre_lift_handoff_deadline",
            }
        return None
