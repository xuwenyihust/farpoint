import json

import numpy as np
import pytest

from farpoint.recovery_runtime import (
    RecoveryTriggerDetector,
    load_recovery_runtime,
    recovery_descent_duration_seconds,
    recovery_oracle_command_continuity_enabled,
    recovery_oracle_slew_limits,
    scene_binding,
    slew_recovery_oracle_target,
)
from farpoint.so101 import lerobot_to_radians, radians_to_lerobot


def runtime_spec():
    return {
        "schema_version": "farpoint.recovery-runtime.v1",
        "runtime_id": "recovery-test-v1",
        "source_policy": {
            "policy_type": "act",
            "checkpoint_step": 20_000,
            "model_sha256": "1" * 64,
            "training_run_id": "act-test",
        },
        "control": {
            "physics_hz": 120,
            "policy_hz": 30,
            "replan_interval_steps": 10,
            "action_safety_profile": {
                "schema_version": "farpoint.action-safety-profile.v1",
                "profile_id": "test",
                "joint_order": [
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                    "gripper.pos",
                ],
                "arm_max_command_speed_deg_s": 50.0,
                "gripper_max_command_slew_calibrated_per_step": 5.5,
                "source": {
                    "kind": "open_source_hardware_default",
                    "reference": "https://example.invalid/so101",
                    "resolved_revision": "test",
                    "statistic": "configured speed",
                },
            },
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
        "oracle_handoff_profile": {
            "profile_id": "gentle-descent-test-v1",
            "descent_duration_seconds": 4.0,
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
    assert recovery_descent_duration_seconds(loaded) == pytest.approx(4.0)
    assert recovery_descent_duration_seconds(None) == pytest.approx(2.3333333333)
    assert not recovery_oracle_command_continuity_enabled(loaded)
    assert not recovery_oracle_command_continuity_enabled(None)
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


def test_recovery_oracle_target_is_slew_bounded_at_control_rate():
    spec = runtime_spec()
    spec["oracle_handoff_profile"]["command_continuity"] = "action_safety_profile_control_rate_v1"
    assert recovery_oracle_command_continuity_enabled(spec)
    limits = recovery_oracle_slew_limits(spec)
    assert limits.tolist() == pytest.approx(
        [
            1.5151515151515151 / 4,
            1.6666666666666667 / 4,
            1.7543859649122806 / 4,
            1.7543859649122806 / 4,
            1.0416666666666667 / 4,
            5.5 / 4,
        ]
    )
    previous = lerobot_to_radians([0.0] * 6)
    requested = lerobot_to_radians([50.0, -50.0, 20.0, -20.0, 10.0, 100.0])
    applied, audit = slew_recovery_oracle_target(previous, requested, limits)
    applied_calibrated = radians_to_lerobot(applied)
    assert applied_calibrated.tolist() == pytest.approx(
        [limits[0], -limits[1], limits[2], -limits[3], limits[4], limits[5]],
        abs=1e-5,
    )
    assert audit["limited_joint_count"] == 6
    assert audit["limiter_reference"] == "previous_oracle_target"

    current = previous
    for _ in range(4):
        current, _audit = slew_recovery_oracle_target(current, requested, limits)
    assert np.abs(radians_to_lerobot(current) - radians_to_lerobot(previous)).tolist() == (
        pytest.approx((limits * 4).tolist(), abs=1e-5)
    )


def test_recovery_oracle_slew_rejects_non_integral_control_ratio():
    spec = runtime_spec()
    spec["control"]["physics_hz"] = 100
    with pytest.raises(ValueError, match="integer positive ratio"):
        recovery_oracle_slew_limits(spec)
