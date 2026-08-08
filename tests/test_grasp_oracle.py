import numpy as np
import pytest

from farpoint.grasp_oracle import (
    ContactAwareGraspStateMachine,
    ControlRecordingSchedule,
    GraspEvidence,
    GraspPhase,
    advance_proof_lift_command,
    cartesian_motion_command_base,
    grasp_phase_allows_unilateral_recenter,
    gripper_target_for_object_local_offset,
    point_in_local_frame,
    rotary_jaw_capture_hold_target,
)


def _evidence(**overrides):
    values = {
        "left_force_n": 2.0,
        "right_force_n": 2.0,
        "aperture_aligned": True,
        "relative_translation_error_m": 0.0,
        "relative_speed_mps": 0.0,
        "proof_lift_m": 0.01,
    }
    values.update(overrides)
    return GraspEvidence(**values)


def test_control_schedule_records_exactly_30_hz_from_120_hz():
    schedule = ControlRecordingSchedule(control_hz=120, recording_hz=30)
    recorded = [step for step in range(120) if schedule.should_record(step)]

    assert schedule.recording_stride == 4
    assert len(recorded) == 30
    assert recorded[-1] == 116
    assert schedule.frame_index(116) == 29
    assert schedule.timestamp_seconds(116) == pytest.approx(29 / 30)
    assert schedule.steps_for_seconds(0.5) == 60


def test_control_schedule_rejects_aliased_recording_rate():
    with pytest.raises(ValueError, match="integer multiple"):
        ControlRecordingSchedule(control_hz=100, recording_hz=30)


def test_gripper_local_offset_tracks_rotated_gripper_frame():
    half_turn_z = [0.0, 0.0, 1.0, 0.0]
    object_world = np.asarray([0.20, 0.10, 0.05])
    desired_local = np.asarray([0.03, -0.01, 0.02])
    gripper_world = gripper_target_for_object_local_offset(
        object_world, half_turn_z, desired_local
    )
    pose = np.concatenate((gripper_world, half_turn_z))

    np.testing.assert_allclose(
        point_in_local_frame(pose, object_world), desired_local, atol=1e-7
    )


def test_rotary_jaw_capture_hold_applies_bounded_closing_preload():
    assert rotary_jaw_capture_hold_target(
        1.2328, closed_position=-0.175, open_position=1.7453
    ) == pytest.approx(1.2308)
    assert rotary_jaw_capture_hold_target(
        -0.174, closed_position=-0.175, open_position=1.7453
    ) == pytest.approx(-0.175)
    with pytest.raises(ValueError, match="non-negative"):
        rotary_jaw_capture_hold_target(
            1.0,
            closed_position=-0.175,
            open_position=1.7453,
            preload_rad=-0.1,
        )


def test_unilateral_recenter_starts_before_bilateral_contact():
    assert grasp_phase_allows_unilateral_recenter(GraspPhase.CONTACT_ALIGNMENT)
    assert grasp_phase_allows_unilateral_recenter(GraspPhase.SLOW_CLOSE)
    assert grasp_phase_allows_unilateral_recenter(GraspPhase.BILATERAL_SETTLE)
    assert not grasp_phase_allows_unilateral_recenter(GraspPhase.APPROACH)
    assert not grasp_phase_allows_unilateral_recenter(GraspPhase.PROOF_LIFT)


def test_proof_lift_rebases_once_then_preserves_accumulated_command():
    measured = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 1.0], dtype=np.float32)
    ahead = measured + 0.02

    base, height = advance_proof_lift_command(
        ahead, measured, 0.0, just_armed=True
    )
    np.testing.assert_allclose(base, measured)
    assert height == pytest.approx(0.0000625)

    accumulated = base + 0.01
    base, height = advance_proof_lift_command(
        accumulated, measured, height, just_armed=False
    )
    np.testing.assert_allclose(base, accumulated)
    assert height == pytest.approx(0.000125)


def test_cartesian_motion_command_rebases_only_on_phase_entry():
    measured = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    accumulated = measured + 0.03

    np.testing.assert_allclose(
        cartesian_motion_command_base(
            accumulated, measured, entering_motion=True
        ),
        measured,
    )
    np.testing.assert_allclose(
        cartesian_motion_command_base(
            accumulated, measured, entering_motion=False
        ),
        accumulated,
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"current_height_m": -0.1},
        {"increment_m": 0.0},
        {"maximum_height_m": 0.0},
    ),
)
def test_proof_lift_command_rejects_invalid_ramp(kwargs):
    arguments = {
        "commanded_joints": [0.0] * 6,
        "measured_joints": [0.0] * 6,
        "current_height_m": 0.0,
        "just_armed": False,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        advance_proof_lift_command(**arguments)


def test_grasp_requires_each_named_quasi_static_stage():
    machine = ContactAwareGraspStateMachine(
        control_hz=10,
        bilateral_settle_s=0.2,
        static_hold_s=0.2,
        proof_lift_hold_s=0.2,
    )
    decision = machine.step(_evidence(right_force_n=0.0))
    assert decision.phase is GraspPhase.FIRST_CONTACT
    assert decision.rebase_joint_command
    assert machine.step(_evidence(right_force_n=0.0)).phase is GraspPhase.CONTACT_ALIGNMENT
    assert machine.step(_evidence(right_force_n=0.0)).phase is GraspPhase.SLOW_CLOSE
    decision = machine.step(_evidence())
    assert decision.phase is GraspPhase.BILATERAL_SETTLE
    assert decision.rebase_joint_command
    assert decision.rebase_relative_tracking
    for expected in (
        GraspPhase.BILATERAL_SETTLE,
        GraspPhase.STATIC_HOLD,
        GraspPhase.STATIC_HOLD,
        GraspPhase.PROOF_LIFT,
        GraspPhase.PROOF_LIFT,
        GraspPhase.VALIDATED,
    ):
        assert machine.step(_evidence()).phase is expected


def test_relative_tracking_rebases_only_at_bilateral_capture():
    machine = ContactAwareGraspStateMachine(control_hz=10)

    first = machine.step(_evidence(right_force_n=0.0))
    assert first.phase is GraspPhase.FIRST_CONTACT
    assert first.rebase_joint_command
    assert not first.rebase_relative_tracking

    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    capture = machine.step(_evidence())
    assert capture.rebase_relative_tracking
    assert not machine.step(_evidence()).rebase_relative_tracking


def test_transient_bilateral_force_cannot_validate_grasp():
    machine = ContactAwareGraspStateMachine(
        control_hz=10,
        bilateral_settle_s=0.3,
        maximum_contact_loss_s=0.1,
    )
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence())
    machine.step(_evidence(right_force_n=0.0))
    decision = machine.step(_evidence(right_force_n=0.0))

    assert decision.phase is GraspPhase.FAILED
    assert decision.failure_reason == "bilateral_contact_lost:bilateral_settle"


def test_low_force_capture_still_requires_physical_proof_lift():
    machine = ContactAwareGraspStateMachine(
        control_hz=10,
        minimum_contact_force_n=0.10,
        bilateral_settle_s=0.1,
        static_hold_s=0.1,
        proof_lift_hold_s=0.1,
    )
    low_force = {
        "left_force_n": 0.15,
        "right_force_n": 0.20,
        "proof_lift_m": 0.0,
    }

    for _ in range(6):
        decision = machine.step(_evidence(**low_force))

    assert decision.phase is GraspPhase.PROOF_LIFT
    assert machine.step(_evidence(**low_force)).phase is GraspPhase.PROOF_LIFT
    assert machine.step(
        _evidence(
            **{
                **low_force,
                "proof_lift_m": machine.minimum_proof_lift_m,
            }
        )
    ).phase is GraspPhase.VALIDATED


def test_capture_threshold_is_distinct_from_contact_persistence_threshold():
    machine = ContactAwareGraspStateMachine(
        control_hz=120,
        minimum_contact_force_n=0.10,
        capture_contact_force_n=2.0,
        capture_confirmation_s=0.025,
    )
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))

    for _ in range(5):
        decision = machine.step(_evidence(left_force_n=1.5, right_force_n=1.5))

    assert decision.phase is GraspPhase.SLOW_CLOSE
    assert machine.capture_steps == 0


def test_capture_confirmation_requires_consecutive_strong_bilateral_samples():
    machine = ContactAwareGraspStateMachine(
        control_hz=120,
        minimum_contact_force_n=0.10,
        capture_contact_force_n=2.0,
        capture_confirmation_s=0.025,
    )
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))

    assert machine.step(_evidence()).phase is GraspPhase.SLOW_CLOSE
    assert machine.step(_evidence()).phase is GraspPhase.SLOW_CLOSE
    assert machine.capture_steps == 2
    assert machine.step(_evidence(right_force_n=1.0)).phase is GraspPhase.SLOW_CLOSE
    assert machine.capture_steps == 0
    assert machine.step(_evidence()).phase is GraspPhase.SLOW_CLOSE
    assert machine.step(_evidence()).phase is GraspPhase.SLOW_CLOSE
    decision = machine.step(_evidence())

    assert decision.phase is GraspPhase.BILATERAL_SETTLE
    assert decision.rebase_relative_tracking


def test_sustain_threshold_applies_after_strong_capture():
    machine = ContactAwareGraspStateMachine(
        control_hz=10,
        minimum_contact_force_n=0.10,
        capture_contact_force_n=2.0,
        capture_confirmation_s=0.1,
        bilateral_settle_s=0.1,
    )
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    assert machine.step(_evidence()).phase is GraspPhase.BILATERAL_SETTLE

    decision = machine.step(_evidence(left_force_n=0.15, right_force_n=0.20))

    assert decision.phase is GraspPhase.STATIC_HOLD


@pytest.mark.parametrize(
    "overrides",
    (
        {"capture_contact_force_n": 0.05},
        {"capture_contact_force_n": 60.0},
        {"capture_confirmation_s": -0.01},
    ),
)
def test_capture_admission_configuration_is_validated(overrides):
    with pytest.raises(ValueError):
        ContactAwareGraspStateMachine(**overrides)


def test_motion_during_bilateral_contact_resets_settle_window():
    machine = ContactAwareGraspStateMachine(control_hz=10, bilateral_settle_s=0.2)
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence())
    machine.step(_evidence())
    machine.step(_evidence(relative_speed_mps=0.2))

    assert machine.phase is GraspPhase.BILATERAL_SETTLE
    assert machine.stable_steps == 0
