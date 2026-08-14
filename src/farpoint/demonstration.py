"""Generic nominal and intervention demonstration metadata builders."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Any


RECOVERY_FAILURE_CLASSES = frozenset(
    {
        "grasp_alignment",
        "contact_without_lift",
        "dropped_object",
        "transport_drift",
        "place_alignment",
        "release_instability",
        "action_saturation",
        "progress_stall",
        "approach_miss",
        "place_release_failure",
    }
)


def _finite_json_value(value: Any, *, path: str = "snapshot") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _finite_json_value(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_json_value(item, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite value")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{path} contains unsupported type {type(value).__name__}")


def state_snapshot_sha256(snapshot: dict[str, Any]) -> str:
    """Hash the measured state at handoff using canonical finite JSON."""
    if not snapshot:
        raise ValueError("recovery handoff state snapshot must not be empty")
    _finite_json_value(snapshot)
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def intervention_command_trace(
    *,
    path: str,
    sha256: str,
    control_hz: int,
    sample_count: int,
    first_control_step: int,
    last_control_step: int,
    joint_order: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Describe the exact physics-rate targets sent after an intervention."""
    trace_path = PurePosixPath(path)
    if trace_path.is_absolute() or ".." in trace_path.parts or str(trace_path) in {"", "."}:
        raise ValueError("command trace path must be a relative artifact path")
    if len(sha256) != 64:
        raise ValueError("command trace sha256 must be a lowercase SHA256")
    try:
        int(sha256, 16)
    except ValueError as error:
        raise ValueError("command trace sha256 must be a lowercase SHA256") from error
    if sha256 != sha256.lower():
        raise ValueError("command trace sha256 must be a lowercase SHA256")
    if not isinstance(control_hz, int) or isinstance(control_hz, bool) or control_hz <= 0:
        raise ValueError("command trace control_hz must be a positive integer")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0:
        raise ValueError("command trace sample_count must be a positive integer")
    if first_control_step < 0 or last_control_step < first_control_step:
        raise ValueError("command trace control-step bounds are invalid")
    if last_control_step - first_control_step + 1 != sample_count:
        raise ValueError("command trace must contain one sample per physics control step")
    names = list(joint_order)
    if not names or len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError("command trace joint_order must contain unique non-empty names")
    return {
        "schema_version": "farpoint.command-trace.v1",
        "path": str(trace_path),
        "sha256": sha256,
        "control_hz": control_hz,
        "sampling_stride": 1,
        "sample_count": sample_count,
        "first_control_step": first_control_step,
        "last_control_step": last_control_step,
        "joint_order": names,
        "unit": "radian",
        "action_semantics": "actual_joint_position_target_sent_before_physics_step",
    }


def nominal_demonstration(*, oracle_profile_id: str) -> dict[str, Any]:
    """Describe a conventional Oracle demonstration from a normal reset."""
    if not oracle_profile_id:
        raise ValueError("nominal demonstration requires an Oracle profile")
    return {
        "schema_version": "farpoint.demonstration.v1",
        "type": "nominal",
        "controller": {"type": "oracle", "profile_id": oracle_profile_id},
    }


def recovery_demonstration(
    *,
    oracle_profile_id: str,
    source_policy: dict[str, Any],
    trigger_id: str,
    failure_class: str,
    control_step: int,
    stage: str,
    trigger_evidence: dict[str, Any],
    source_rollout_id: str,
    source_scene_id: str,
    state_snapshot: dict[str, Any],
    recovery_strategy_id: str,
) -> dict[str, Any]:
    """Bind an Oracle recovery to a live ACT-to-Oracle physical handoff."""
    required_policy = {
        "policy_type",
        "checkpoint_step",
        "model_sha256",
        "training_run_id",
        "rollout_git_commit",
    }
    missing_policy = sorted(required_policy - set(source_policy))
    if missing_policy:
        raise ValueError("source policy is missing: " + ", ".join(missing_policy))
    if failure_class not in RECOVERY_FAILURE_CLASSES:
        raise ValueError(f"unsupported recovery failure class: {failure_class}")
    if control_step < 0:
        raise ValueError("recovery control_step must be non-negative")
    if not trigger_evidence:
        raise ValueError("recovery trigger evidence must not be empty")
    values = {
        "oracle_profile_id": oracle_profile_id,
        "trigger_id": trigger_id,
        "stage": stage,
        "source_rollout_id": source_rollout_id,
        "source_scene_id": source_scene_id,
        "recovery_strategy_id": recovery_strategy_id,
    }
    missing = sorted(key for key, value in values.items() if not value)
    if missing:
        raise ValueError("recovery metadata is missing: " + ", ".join(missing))
    _finite_json_value(trigger_evidence, path="trigger_evidence")
    return {
        "schema_version": "farpoint.demonstration.v1",
        "type": "recovery",
        "controller": {"type": "oracle", "profile_id": oracle_profile_id},
        "source_policy": deepcopy(source_policy),
        "intervention": {
            "trigger": {
                "trigger_id": trigger_id,
                "failure_class": failure_class,
                "control_step": int(control_step),
                "stage": stage,
                "evidence": deepcopy(trigger_evidence),
            },
            "handoff": {
                "mode": "live_continuous_state",
                "source_rollout_id": source_rollout_id,
                "source_scene_id": source_scene_id,
                "source_control_step": int(control_step),
                "recovery_start_frame": 0,
                "physics_state_continuous": True,
                "reset_performed": False,
                "state_snapshot_sha256": state_snapshot_sha256(state_snapshot),
            },
            "recovery_strategy_id": recovery_strategy_id,
        },
    }
