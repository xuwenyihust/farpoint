import json

import pytest

from farpoint.recovery_runtime import (
    RecoveryTriggerDetector,
    load_recovery_runtime,
    scene_binding,
)


def runtime_spec():
    return {
        "schema_version": "farpoint.recovery-runtime.v1",
        "runtime_id": "recovery-test-v1",
        "source_policy": {
            "policy_type": "act",
            "checkpoint_step": 20_000,
            "model_sha256": "1" * 64,
            "training_run_id": "act-test",
            "rollout_git_commit": "2" * 40,
        },
        "control": {
            "physics_hz": 120,
            "policy_hz": 30,
            "replan_interval_steps": 10,
            "action_safety_profile": {"profile_id": "test"},
        },
        "trigger": {
            "trigger_id": "pre-lift-v1",
            "strategy_id": "regrasp_from_live_state_v1",
            "minimum_policy_steps": 4,
            "maximum_policy_steps_before_handoff": 8,
            "stall_window_steps": 3,
            "minimum_progress_m": 0.001,
            "consecutive_safety_event_steps": 2,
            "require_not_lifted": True,
        },
        "scenes": [
            {
                "variation_id": "variation-0",
                "source_rollout_id": "rollout-0",
                "source_scene_id": "scene-0",
                "source_partition": "train",
            }
        ],
    }


def test_runtime_contract_and_binding(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(runtime_spec()))
    loaded = load_recovery_runtime(path)
    assert scene_binding(loaded, "variation-0")["source_partition"] == "train"
    with pytest.raises(ValueError, match="no unique binding"):
        scene_binding(loaded, "missing")


def test_runtime_rejects_duplicate_and_invalid_timing(tmp_path):
    spec = runtime_spec()
    spec["scenes"].append({**spec["scenes"][0], "variation_id": "variation-1"})
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="source_scene_id values must be unique"):
        load_recovery_runtime(path)

    spec = runtime_spec()
    spec["trigger"]["minimum_policy_steps"] = 8
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="must precede"):
        load_recovery_runtime(path)


def test_trigger_prefers_safety_and_never_hands_off_after_lift():
    detector = RecoveryTriggerDetector(runtime_spec()["trigger"])
    assert (
        detector.observe(
            policy_step=0,
            gripper_position_m=[0.0, 0.0, 0.1],
            object_position_m=[0.1, 0.0, 0.0],
            cube_lifted=False,
            hard_range_violation_count=1,
            command_slew_limited_count=0,
        )
        is None
    )
    for step in (1, 2):
        detector.observe(
            policy_step=step,
            gripper_position_m=[0.0, 0.0, 0.1],
            object_position_m=[0.1, 0.0, 0.0],
            cube_lifted=False,
            hard_range_violation_count=1,
            command_slew_limited_count=0,
        )
    trigger = detector.observe(
        policy_step=3,
        gripper_position_m=[0.0, 0.0, 0.1],
        object_position_m=[0.1, 0.0, 0.0],
        cube_lifted=False,
        hard_range_violation_count=1,
        command_slew_limited_count=0,
    )
    assert trigger["failure_class"] == "action_saturation"

    lifted = RecoveryTriggerDetector(runtime_spec()["trigger"])
    for step in range(12):
        assert (
            lifted.observe(
                policy_step=step,
                gripper_position_m=[0.0, 0.0, 0.1],
                object_position_m=[0.1, 0.0, 0.0],
                cube_lifted=True,
                hard_range_violation_count=1,
                command_slew_limited_count=1,
            )
            is None
        )


def test_trigger_detects_stall_and_deadline():
    detector = RecoveryTriggerDetector(runtime_spec()["trigger"])
    trigger = None
    for step in range(8):
        trigger = detector.observe(
            policy_step=step,
            gripper_position_m=[0.0, 0.0, 0.1],
            object_position_m=[0.1, 0.0, 0.0],
            cube_lifted=False,
            hard_range_violation_count=0,
            command_slew_limited_count=0,
        )
        if trigger:
            break
    assert trigger["failure_class"] == "progress_stall"
    assert trigger["reason"] == "insufficient_gripper_object_progress"
