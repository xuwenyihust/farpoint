import math

import pytest

from farpoint.demonstration import (
    nominal_demonstration,
    recovery_demonstration,
    state_snapshot_sha256,
)


SOURCE_POLICY = {
    "policy_type": "act",
    "checkpoint_step": 20_000,
    "model_sha256": "a" * 64,
    "training_run_id": "act-v010-baseline-20k",
    "rollout_git_commit": "b" * 40,
}
SNAPSHOT = {
    "robot_state_calibrated": [0.0] * 6,
    "object_pose_xyzw": [0.2, -0.05, 0.04, 0.0, 0.0, 0.0, 1.0],
    "object_velocity_mps": [0.0, 0.0, 0.0],
    "contact_forces_n": [1.2, 1.1],
    "simulation_time_s": 22.8,
}


def test_nominal_demonstration_is_minimal_and_typed():
    assert nominal_demonstration(oracle_profile_id="default-v1") == {
        "schema_version": "farpoint.demonstration.v1",
        "type": "nominal",
        "controller": {"type": "oracle", "profile_id": "default-v1"},
    }


def test_recovery_demonstration_binds_live_handoff_and_snapshot_deterministically():
    arguments = {
        "oracle_profile_id": "transport-recovery-v1",
        "source_policy": SOURCE_POLICY,
        "trigger_id": "target-progress-stall-v1",
        "failure_class": "transport_drift",
        "control_step": 684,
        "stage": "transport",
        "trigger_evidence": {"window_steps": 90, "progress_m": 0.0},
        "source_rollout_id": "rollout-1",
        "source_scene_id": "train-scene-1",
        "state_snapshot": SNAPSHOT,
        "recovery_strategy_id": "stabilize-lift-preplace-v1",
    }
    first = recovery_demonstration(**arguments)
    second = recovery_demonstration(
        **{
            **arguments,
            "trigger_evidence": {"progress_m": 0.0, "window_steps": 90},
            "state_snapshot": dict(reversed(SNAPSHOT.items())),
        }
    )
    assert first == second
    handoff = first["intervention"]["handoff"]
    assert handoff["mode"] == "live_continuous_state"
    assert handoff["physics_state_continuous"] is True
    assert handoff["reset_performed"] is False
    assert handoff["state_snapshot_sha256"] == state_snapshot_sha256(SNAPSHOT)


def test_recovery_metadata_rejects_missing_policy_and_nonfinite_snapshot():
    with pytest.raises(ValueError, match="source policy is missing"):
        recovery_demonstration(
            oracle_profile_id="recovery-v1",
            source_policy={},
            trigger_id="stall-v1",
            failure_class="progress_stall",
            control_step=10,
            stage="transport",
            trigger_evidence={"window": 5},
            source_rollout_id="rollout",
            source_scene_id="scene",
            state_snapshot=SNAPSHOT,
            recovery_strategy_id="recover-v1",
        )
    invalid = dict(SNAPSHOT)
    invalid["simulation_time_s"] = math.inf
    with pytest.raises(ValueError, match="non-finite"):
        state_snapshot_sha256(invalid)
