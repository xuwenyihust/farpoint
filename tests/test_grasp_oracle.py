import numpy as np
import pytest

from farpoint.grasp_oracle import (
    capture_retention_reopen_active,
    capture_admission_retention_fraction,
    ContactAwareGraspStateMachine,
    ControlRecordingSchedule,
    GraspEvidence,
    GraspPhase,
    advance_proof_lift_command,
    cartesian_motion_command_base,
    capture_hold_preload_for_force,
    capture_retention_recenter_fallback_active,
    capture_retention_force_floor,
    contact_constrained_joint_step_limit,
    contact_force_vectors_opposed,
    capture_preload_force_floor,
    captured_force_imbalance_requires_recenter,
    captured_force_imbalance_requires_squeeze_pause,
    capture_aperture_laterally_aligned,
    grasp_phase_allows_unilateral_recenter,
    gripper_target_for_object_local_offset,
    gripper_xy_target_for_object_local_offset,
    latch_pre_capture_recenter_object_reference,
    point_in_local_frame,
    proof_lift_recovery_holds_xy,
    rotary_jaw_capture_hold_target,
    so101_recenter_contact_memory,
    unilateral_contact_requires_recenter,
)


def test_capture_retention_reopen_requires_material_aperture_error():
    assert capture_retention_reopen_active(
        True,
        [0.0189, -0.0260, -0.0857],
        [0.0224, -0.0086, -0.0479],
    )
    assert not capture_retention_reopen_active(
        True,
        [0.0227, -0.0118, -0.0688],
        [0.0224, -0.0086, -0.0479],
    )
    assert not capture_retention_reopen_active(
        False,
        [0.0189, -0.0260, -0.0857],
        [0.0224, -0.0086, -0.0479],
    )


def test_capture_retention_recenter_fallback_requires_stalled_static_hold():
    assert capture_retention_recenter_fallback_active(
        GraspPhase.STATIC_HOLD,
        8,
        2.6,
        1.7,
        4.0,
    )
    assert not capture_retention_recenter_fallback_active(
        GraspPhase.STATIC_HOLD,
        7,
        2.6,
        1.7,
        4.0,
    )
    assert not capture_retention_recenter_fallback_active(
        GraspPhase.STATIC_HOLD,
        8,
        4.1,
        4.2,
        4.0,
    )
    assert not capture_retention_recenter_fallback_active(
        GraspPhase.BILATERAL_SETTLE,
        20,
        2.6,
        1.7,
        4.0,
    )


def test_capture_retention_recenter_fallback_requires_opted_in_stalled_slow_close():
    assert not capture_retention_recenter_fallback_active(
        GraspPhase.SLOW_CLOSE,
        300,
        0.5,
        3.0,
        2.0,
    )
    assert not capture_retention_recenter_fallback_active(
        GraspPhase.SLOW_CLOSE,
        299,
        0.5,
        3.0,
        2.0,
        minimum_slow_close_steps=300,
    )
    assert capture_retention_recenter_fallback_active(
        GraspPhase.SLOW_CLOSE,
        300,
        0.5,
        3.0,
        2.0,
        minimum_slow_close_steps=300,
    )
    assert not capture_retention_recenter_fallback_active(
        GraspPhase.SLOW_CLOSE,
        300,
        2.5,
        3.0,
        2.0,
        minimum_slow_close_steps=300,
    )


@pytest.mark.parametrize("minimum_slow_close_steps", [0, -1, 1.5, True])
def test_capture_retention_recenter_fallback_rejects_invalid_slow_close_window(
    minimum_slow_close_steps,
):
    with pytest.raises(ValueError, match="minimum_slow_close_steps"):
        capture_retention_recenter_fallback_active(
            GraspPhase.SLOW_CLOSE,
            300,
            0.5,
            3.0,
            2.0,
            minimum_slow_close_steps=minimum_slow_close_steps,
        )


def test_pre_capture_recenter_reference_latches_first_contact_position():
    first = latch_pre_capture_recenter_object_reference([0.20, -0.08, 0.052])
    retained = latch_pre_capture_recenter_object_reference(
        [0.23, -0.05, 0.052],
        first,
    )

    assert first.tolist() == pytest.approx([0.20, -0.08, 0.052])
    assert retained.tolist() == pytest.approx(first.tolist())


@pytest.mark.parametrize(
    ("current", "previous"),
    (
        ([0.0, 0.0], None),
        ([0.0, float("nan"), 0.0], None),
        ([0.0, 0.0, 0.0], [0.0, float("inf"), 0.0]),
    ),
)
def test_pre_capture_recenter_reference_rejects_invalid_positions(
    current,
    previous,
):
    with pytest.raises(ValueError, match="three finite values"):
        latch_pre_capture_recenter_object_reference(current, previous)


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


def test_captured_force_imbalance_pauses_squeeze_only_for_bilateral_load():
    assert captured_force_imbalance_requires_squeeze_pause(
        2.18, 3.20, minimum_force_n=0.10, proof_entry_force_n=4.0
    )
    assert not captured_force_imbalance_requires_squeeze_pause(
        3.82, 5.66, minimum_force_n=0.10, proof_entry_force_n=4.0
    )
    assert not captured_force_imbalance_requires_squeeze_pause(
        0.0, 3.20, minimum_force_n=0.10, proof_entry_force_n=4.0
    )


def test_captured_force_imbalance_recenters_only_after_strong_side_proof():
    assert captured_force_imbalance_requires_recenter(
        20.8, 1.86, minimum_force_n=0.10, proof_entry_force_n=4.0
    )
    assert not captured_force_imbalance_requires_recenter(
        3.20, 2.18, minimum_force_n=0.10, proof_entry_force_n=4.0
    )
    assert not captured_force_imbalance_requires_recenter(
        5.0, 4.0, minimum_force_n=0.10, proof_entry_force_n=4.0
    )
    assert not captured_force_imbalance_requires_recenter(
        5.0, 0.0, minimum_force_n=0.10, proof_entry_force_n=4.0
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"left_force_n": float("nan")},
        {"right_force_n": -0.1},
        {"minimum_force_n": 0.0},
        {"proof_entry_force_n": 0.05},
        {"minimum_balance_ratio": 0.0},
        {"minimum_balance_ratio": 1.1},
    ),
)
def test_captured_force_imbalance_rejects_invalid_contract(kwargs):
    arguments = {
        "left_force_n": 2.0,
        "right_force_n": 3.0,
        "minimum_force_n": 0.10,
        "proof_entry_force_n": 4.0,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        captured_force_imbalance_requires_squeeze_pause(**arguments)
    with pytest.raises(ValueError):
        captured_force_imbalance_requires_recenter(**arguments)


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


def test_contact_force_vectors_require_opposing_sidewall_loads():
    assert contact_force_vectors_opposed([2.0, -4.0, 0.5], [-3.0, 3.0, 0.0])
    assert not contact_force_vectors_opposed(
        [-0.9, -3.85, 0.98], [-4.18, 1.44, -0.02]
    )
    assert not contact_force_vectors_opposed([0.01, 0.0, 0.0], [-1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("left", "right", "kwargs"),
    (
        ([1.0, 2.0], [-1.0, -2.0, 0.0], {}),
        ([float("nan"), 0.0, 0.0], [-1.0, 0.0, 0.0], {}),
        ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], {"maximum_cosine": 0.0}),
        ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], {"minimum_force_n": 0.0}),
    ),
)
def test_contact_force_vectors_reject_invalid_inputs(left, right, kwargs):
    with pytest.raises(ValueError):
        contact_force_vectors_opposed(left, right, **kwargs)


def test_rotary_jaw_capture_hold_applies_bounded_closing_preload():
    assert rotary_jaw_capture_hold_target(
        1.2328,
        closed_position=-0.175,
        open_position=1.7453,
        relative_speed_mps=0.0011,
    ) == pytest.approx(1.2248)
    assert rotary_jaw_capture_hold_target(
        1.2328,
        closed_position=-0.175,
        open_position=1.7453,
        relative_speed_mps=0.0007,
    ) == pytest.approx(1.2248)
    assert rotary_jaw_capture_hold_target(
        1.2328,
        closed_position=-0.175,
        open_position=1.7453,
        relative_speed_mps=0.002,
    ) == pytest.approx(1.2248)
    assert rotary_jaw_capture_hold_target(
        1.2328,
        closed_position=-0.175,
        open_position=1.7453,
        relative_speed_mps=0.002,
        moving_capture_preload_rad=0.004,
    ) == pytest.approx(1.2288)
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
    with pytest.raises(ValueError, match="relative_speed_mps"):
        rotary_jaw_capture_hold_target(
            1.0,
            closed_position=-0.175,
            open_position=1.7453,
            relative_speed_mps=float("inf"),
        )
    with pytest.raises(ValueError, match="moving_capture_preload_rad"):
        rotary_jaw_capture_hold_target(
            1.0,
            closed_position=-0.175,
            open_position=1.7453,
            moving_capture_preload_rad=-0.001,
        )
    with pytest.raises(ValueError, match="moving_capture_threshold_mps"):
        rotary_jaw_capture_hold_target(
            1.0,
            closed_position=-0.175,
            open_position=1.7453,
            moving_capture_threshold_mps=float("inf"),
        )
    with pytest.raises(ValueError, match="moving_capture_ceiling_mps"):
        rotary_jaw_capture_hold_target(
            1.0,
            closed_position=-0.175,
            open_position=1.7453,
            moving_capture_threshold_mps=0.002,
            moving_capture_ceiling_mps=0.002,
        )


def test_capture_hold_preload_uses_retention_only_after_proof_force():
    assert capture_hold_preload_for_force(
        6.1,
        4.5,
        proof_entry_force_n=4.0,
    ) == pytest.approx(0.002)
    assert capture_hold_preload_for_force(
        4.4,
        4.0,
        proof_entry_force_n=4.0,
    ) == pytest.approx(0.008)
    assert capture_hold_preload_for_force(
        6.1,
        3.9,
        proof_entry_force_n=4.0,
    ) == pytest.approx(0.008)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"left_force_n": -0.1, "right_force_n": 4.0, "proof_entry_force_n": 4.0},
        {
            "left_force_n": 4.0,
            "right_force_n": float("nan"),
            "proof_entry_force_n": 4.0,
        },
        {"left_force_n": 4.0, "right_force_n": 4.0, "proof_entry_force_n": 0.0},
        {
            "left_force_n": 4.0,
            "right_force_n": 4.0,
            "proof_entry_force_n": 4.0,
            "overload_margin_n": -0.1,
        },
        {
            "left_force_n": 4.0,
            "right_force_n": 4.0,
            "proof_entry_force_n": 4.0,
            "retention_preload_rad": 0.009,
            "balanced_preload_rad": 0.0085,
            "buildup_preload_rad": 0.008,
        },
    ),
)
def test_capture_hold_preload_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError, match="non-negative|positive|capture preloads"):
        capture_hold_preload_for_force(**kwargs)


def test_capture_preload_force_floor_tracks_admission_threshold():
    assert capture_preload_force_floor(2.0) == pytest.approx(1.8)
    assert capture_preload_force_floor(4.0, retention_fraction=0.5) == pytest.approx(
        2.0
    )


def test_capture_admission_retention_fraction_is_size_aware():
    assert capture_admission_retention_fraction(None) == pytest.approx(0.25)
    assert capture_admission_retention_fraction(0.03) == pytest.approx(0.25)
    assert capture_admission_retention_fraction(0.035) == pytest.approx(0.575)
    assert capture_admission_retention_fraction(0.04) == pytest.approx(0.90)


def test_large_cube_capture_waits_for_settle_force_floor():
    machine = ContactAwareGraspStateMachine(
        control_hz=120,
        object_width_m=0.04,
        capture_contact_force_n=2.0,
        capture_confirmation_s=0.025,
    )
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))

    for _ in range(3):
        decision = machine.step(
            _evidence(left_force_n=1.79, right_force_n=2.64)
        )

    assert decision.phase is GraspPhase.SLOW_CLOSE
    assert machine.capture_steps == 0

    for _ in range(3):
        decision = machine.step(
            _evidence(left_force_n=1.80, right_force_n=2.64)
        )

    assert decision.phase is GraspPhase.BILATERAL_SETTLE


@pytest.mark.parametrize("width", (0.0, float("nan"), float("inf")))
def test_capture_admission_retention_fraction_rejects_invalid_width(width):
    with pytest.raises(ValueError):
        capture_admission_retention_fraction(width)


@pytest.mark.parametrize(
    ("capture_force", "retention_fraction"),
    [
        (0.0, 0.90),
        (float("nan"), 0.90),
        (2.0, 0.0),
        (2.0, 1.01),
        (2.0, float("nan")),
    ],
)
def test_capture_preload_force_floor_rejects_invalid_contract(
    capture_force, retention_fraction
):
    with pytest.raises(ValueError):
        capture_preload_force_floor(
            capture_force,
            retention_fraction=retention_fraction,
        )


def test_unilateral_recenter_starts_before_bilateral_contact():
    assert grasp_phase_allows_unilateral_recenter(GraspPhase.FIRST_CONTACT)
    assert grasp_phase_allows_unilateral_recenter(GraspPhase.CONTACT_ALIGNMENT)
    assert grasp_phase_allows_unilateral_recenter(GraspPhase.SLOW_CLOSE)
    assert grasp_phase_allows_unilateral_recenter(GraspPhase.BILATERAL_SETTLE)
    assert grasp_phase_allows_unilateral_recenter(GraspPhase.PROOF_LIFT)
    assert not grasp_phase_allows_unilateral_recenter(GraspPhase.APPROACH)


def test_first_contact_memory_bridges_handoff_force_dropout():
    first_contact = so101_recenter_contact_memory(0.0, 0.675)
    dropout = so101_recenter_contact_memory(0.0, 0.0, first_contact["side"])

    assert first_contact["side"] == "right"
    assert dropout == {
        "forces": (0.0, 0.10),
        "side": "right",
        "used_memory": True,
    }
    assert unilateral_contact_requires_recenter(
        *dropout["forces"], minimum_force_n=0.10
    )


def test_xy_aperture_target_corrects_recorded_r01_c00_direction():
    target = gripper_xy_target_for_object_local_offset(
        [0.14585812, -0.08612937, 0.04701491],
        [
            0.11045856,
            -0.12946184,
            0.08191921,
            -0.57678396,
            0.00717221,
            0.24073146,
            -0.78058785,
        ],
        [0.02149719, -0.00846439, -0.05466565],
    )

    assert target == pytest.approx([0.11417308, -0.12505639, 0.08191921])


def test_capture_aperture_alignment_uses_aperture_plane_not_finger_depth():
    reference = [0.0215, -0.0085, -0.0547]

    assert capture_aperture_laterally_aligned(
        [0.0223, -0.0105, -0.1107], reference
    )
    assert not capture_aperture_laterally_aligned(
        [0.0470, -0.0085, -0.0547], reference
    )


@pytest.mark.parametrize(
    ("actual", "reference", "maximum_error"),
    [
        ([0.0, 0.0], [0.0, 0.0, 0.0], 0.025),
        ([0.0, 0.0, float("nan")], [0.0, 0.0, 0.0], 0.025),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
    ],
)
def test_capture_aperture_alignment_rejects_invalid_contract(
    actual, reference, maximum_error
):
    with pytest.raises(ValueError):
        capture_aperture_laterally_aligned(
            actual,
            reference,
            maximum_lateral_error_m=maximum_error,
        )


def test_unilateral_recenter_uses_grasp_persistence_force_floor():
    assert unilateral_contact_requires_recenter(
        0.0, 0.12, minimum_force_n=0.10
    )
    assert not unilateral_contact_requires_recenter(
        0.0, 0.09, minimum_force_n=0.10
    )
    assert not unilateral_contact_requires_recenter(
        0.11, 0.12, minimum_force_n=0.10
    )
    with pytest.raises(ValueError, match="non-negative"):
        unilateral_contact_requires_recenter(0.0, 0.12, minimum_force_n=-0.1)


def test_recenter_contact_memory_starts_recovery_during_zero_force_gap():
    bilateral = so101_recenter_contact_memory(2.1, 3.0)
    dropout = so101_recenter_contact_memory(0.0, 0.0, bilateral["side"])

    assert bilateral == {
        "forces": (2.1, 3.0),
        "side": "right",
        "used_memory": False,
    }
    assert dropout == {
        "forces": (0.0, 0.10),
        "side": "right",
        "used_memory": True,
    }
    assert unilateral_contact_requires_recenter(
        *dropout["forces"], minimum_force_n=0.10
    )


def test_recenter_contact_memory_prefers_live_unilateral_side():
    result = so101_recenter_contact_memory(0.2, 0.0, "right")

    assert result["forces"] == (0.2, 0.0)
    assert result["side"] == "left"
    assert not result["used_memory"]


def test_recenter_contact_memory_ignores_bilateral_force_order_noise():
    result = so101_recenter_contact_memory(3.0, 2.0, "right")

    assert result["forces"] == (3.0, 2.0)
    assert result["side"] == "right"
    assert not result["used_memory"]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"left_force_n": -0.1, "right_force_n": 0.0},
        {"left_force_n": float("nan"), "right_force_n": 0.0},
        {"left_force_n": 0.0, "right_force_n": 0.0, "previous_side": "center"},
        {"left_force_n": 0.0, "right_force_n": 0.0, "minimum_force_n": 0.0},
    ),
)
def test_recenter_contact_memory_rejects_invalid_contract(kwargs):
    with pytest.raises(ValueError):
        so101_recenter_contact_memory(**kwargs)


def test_proof_lift_rebases_once_then_preserves_accumulated_command():
    measured = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 1.0], dtype=np.float32)
    ahead = measured + 0.02

    base, height = advance_proof_lift_command(
        ahead, measured, 0.0, just_armed=True
    )
    np.testing.assert_allclose(base, measured)
    assert height == pytest.approx(0.000015625)

    accumulated = base + 0.01
    base, height = advance_proof_lift_command(
        accumulated, measured, height, just_armed=False
    )
    np.testing.assert_allclose(base, accumulated)
    assert height == pytest.approx(0.00003125)


def test_proof_lift_freezes_height_during_bounded_contact_recovery():
    commanded = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    measured = commanded - 0.01

    held, height = advance_proof_lift_command(
        commanded,
        measured,
        0.00125,
        just_armed=False,
        contact_retained=False,
    )

    assert held == pytest.approx(commanded)
    assert height == pytest.approx(0.00125)

    resumed, height = advance_proof_lift_command(
        held,
        measured,
        height,
        just_armed=False,
        contact_retained=True,
    )

    assert resumed == pytest.approx(commanded)
    assert height == pytest.approx(0.001265625)


@pytest.mark.parametrize("value", (None, 0, 1, "yes"))
def test_proof_lift_rejects_non_boolean_contact_retained(value):
    with pytest.raises(ValueError, match="contact_retained"):
        advance_proof_lift_command(
            np.zeros(6),
            np.zeros(6),
            0.0,
            just_armed=False,
            contact_retained=value,
        )


@pytest.mark.parametrize(
    ("proof_lift_armed", "unilateral_contact", "expected"),
    (
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ),
)
def test_proof_lift_unilateral_recovery_holds_xy(
    proof_lift_armed, unilateral_contact, expected
):
    assert (
        proof_lift_recovery_holds_xy(
            proof_lift_armed=proof_lift_armed,
            unilateral_contact=unilateral_contact,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("proof_lift_armed", "unilateral_contact"),
    ((1, True), (True, 1), (None, False), (False, None)),
)
def test_proof_lift_xy_hold_rejects_non_boolean_flags(
    proof_lift_armed, unilateral_contact
):
    with pytest.raises(ValueError, match="flags"):
        proof_lift_recovery_holds_xy(
            proof_lift_armed=proof_lift_armed,
            unilateral_contact=unilateral_contact,
        )


def test_proof_lift_uses_tighter_contact_constrained_joint_step_limit():
    assert contact_constrained_joint_step_limit(
        0.005, proof_lift_armed=False
    ) == pytest.approx(0.005)
    assert contact_constrained_joint_step_limit(
        0.005, proof_lift_armed=True
    ) == pytest.approx(0.001)
    assert contact_constrained_joint_step_limit(
        0.0005, proof_lift_armed=True
    ) == pytest.approx(0.0005)


def test_capture_validation_raises_only_the_controller_retention_floor():
    assert capture_retention_force_floor(
        3.0, capture_validation_active=False
    ) == pytest.approx(3.0)
    assert capture_retention_force_floor(
        3.0, capture_validation_active=True
    ) == pytest.approx(4.0)
    assert capture_retention_force_floor(
        5.0, capture_validation_active=True
    ) == pytest.approx(5.0)


@pytest.mark.parametrize("value", (-0.1, float("nan"), float("inf")))
def test_capture_retention_force_floor_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        capture_retention_force_floor(value, capture_validation_active=False)
    with pytest.raises(ValueError, match="finite and non-negative"):
        capture_retention_force_floor(
            3.0,
            capture_validation_active=True,
            minimum_capture_force_n=value,
        )


@pytest.mark.parametrize("value", (0.0, -0.001, float("nan"), float("inf")))
def test_contact_constrained_joint_step_limit_rejects_invalid_limits(value):
    with pytest.raises(ValueError, match="finite and positive"):
        contact_constrained_joint_step_limit(value, proof_lift_armed=False)
    with pytest.raises(ValueError, match="finite and positive"):
        contact_constrained_joint_step_limit(
            0.005,
            proof_lift_armed=True,
            maximum_proof_lift_joint_step=value,
        )


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
        capture_confirmation_s=0.0,
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
    machine = ContactAwareGraspStateMachine(
        control_hz=10,
        capture_confirmation_s=0.0,
    )

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
        capture_confirmation_s=0.0,
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
        capture_confirmation_s=0.0,
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


def test_proof_lift_waits_for_full_bilateral_preload_window():
    machine = ContactAwareGraspStateMachine(
        control_hz=10,
        minimum_contact_force_n=0.10,
        capture_contact_force_n=2.0,
        minimum_proof_entry_force_n=4.0,
        capture_confirmation_s=0.0,
        bilateral_settle_s=0.1,
        static_hold_s=0.2,
    )
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence())
    assert machine.step(_evidence()).phase is GraspPhase.STATIC_HOLD

    for _ in range(4):
        decision = machine.step(
            _evidence(left_force_n=5.28, right_force_n=3.82)
        )

    assert decision.phase is GraspPhase.STATIC_HOLD
    assert machine.stable_steps == 0
    assert machine.step(
        _evidence(left_force_n=5.87, right_force_n=4.72)
    ).phase is GraspPhase.STATIC_HOLD
    assert machine.step(
        _evidence(left_force_n=5.87, right_force_n=4.72)
    ).phase is GraspPhase.PROOF_LIFT


def test_static_hold_rebases_only_after_stable_proof_ready_recovery():
    machine = ContactAwareGraspStateMachine(
        control_hz=10,
        capture_confirmation_s=0.0,
        bilateral_settle_s=0.2,
        static_hold_s=0.2,
        minimum_proof_entry_force_n=4.0,
        maximum_relative_translation_error_m=0.003,
        maximum_capture_relative_speed_mps=0.002,
    )
    machine.step(_evidence(left_force_n=5.0, right_force_n=0.0))
    machine.step(_evidence(left_force_n=5.0, right_force_n=0.0))
    machine.step(_evidence(left_force_n=5.0, right_force_n=0.0))
    machine.step(_evidence(left_force_n=5.0, right_force_n=5.0))
    machine.step(_evidence(left_force_n=5.0, right_force_n=5.0))
    assert machine.step(
        _evidence(left_force_n=5.0, right_force_n=5.0)
    ).phase is GraspPhase.STATIC_HOLD

    first = machine.step(
        _evidence(
            left_force_n=5.0,
            right_force_n=5.0,
            relative_translation_error_m=0.004,
            relative_speed_mps=0.001,
        )
    )
    second = machine.step(
        _evidence(
            left_force_n=5.0,
            right_force_n=5.0,
            relative_translation_error_m=0.004,
            relative_speed_mps=0.001,
        )
    )
    assert not first.rebase_relative_tracking
    assert second.rebase_recovered_capture_tracking
    assert second.rebase_relative_tracking
    assert second.phase is GraspPhase.STATIC_HOLD

    assert machine.step(
        _evidence(left_force_n=5.0, right_force_n=5.0)
    ).phase is GraspPhase.STATIC_HOLD
    assert machine.step(
        _evidence(left_force_n=5.0, right_force_n=5.0)
    ).phase is GraspPhase.PROOF_LIFT


def test_static_hold_recovery_rebase_resets_after_dynamic_sample():
    machine = ContactAwareGraspStateMachine(
        control_hz=10,
        capture_confirmation_s=0.0,
        bilateral_settle_s=0.2,
        minimum_proof_entry_force_n=4.0,
        maximum_relative_translation_error_m=0.003,
        maximum_capture_relative_speed_mps=0.002,
    )
    machine.step(_evidence(left_force_n=5.0, right_force_n=0.0))
    machine.step(_evidence(left_force_n=5.0, right_force_n=0.0))
    machine.step(_evidence(left_force_n=5.0, right_force_n=0.0))
    machine.step(_evidence(left_force_n=5.0, right_force_n=5.0))
    machine.step(_evidence(left_force_n=5.0, right_force_n=5.0))
    machine.step(_evidence(left_force_n=5.0, right_force_n=5.0))

    machine.step(
        _evidence(
            left_force_n=5.0,
            right_force_n=5.0,
            relative_translation_error_m=0.004,
            relative_speed_mps=0.001,
        )
    )
    dynamic = machine.step(
        _evidence(
            left_force_n=5.0,
            right_force_n=5.0,
            relative_translation_error_m=0.004,
            relative_speed_mps=0.003,
        )
    )
    assert not dynamic.rebase_relative_tracking
    assert machine.recovery_rebase_steps == 0


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
        decision = machine.step(_evidence(left_force_n=0.49, right_force_n=0.49))

    assert decision.phase is GraspPhase.SLOW_CLOSE
    assert machine.capture_steps == 0


def test_capture_threshold_accepts_bounded_solver_hysteresis():
    machine = ContactAwareGraspStateMachine(
        control_hz=120,
        minimum_contact_force_n=0.10,
        capture_contact_force_n=2.0,
        capture_confirmation_s=0.025,
    )
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))

    for _ in range(3):
        decision = machine.step(
            _evidence(left_force_n=0.5, right_force_n=0.5)
        )

    assert decision.phase is GraspPhase.BILATERAL_SETTLE


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
    assert machine.step(_evidence(right_force_n=0.4)).phase is GraspPhase.SLOW_CLOSE
    assert machine.capture_steps == 0
    assert machine.step(_evidence()).phase is GraspPhase.SLOW_CLOSE
    assert machine.step(_evidence()).phase is GraspPhase.SLOW_CLOSE
    decision = machine.step(_evidence())

    assert decision.phase is GraspPhase.BILATERAL_SETTLE


def test_capture_confirmation_rejects_nonopposing_contact_geometry():
    machine = ContactAwareGraspStateMachine(
        control_hz=120,
        capture_confirmation_s=0.025,
    )
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))

    for _ in range(machine.capture_confirmation_steps + 2):
        decision = machine.step(_evidence(contact_geometry_valid=False))
        assert decision.phase is GraspPhase.SLOW_CLOSE

    for _ in range(machine.capture_confirmation_steps):
        decision = machine.step(_evidence(contact_geometry_valid=True))
    assert decision.phase is GraspPhase.BILATERAL_SETTLE
    assert decision.rebase_relative_tracking


def test_capture_confirmation_steps_exposes_shared_window():
    machine = ContactAwareGraspStateMachine(
        control_hz=120,
        object_width_m=0.04,
        capture_confirmation_s=0.025,
    )

    assert machine.capture_confirmation_steps == 3
    assert machine.object_width_m == pytest.approx(0.04)


@pytest.mark.parametrize("width", (0.0, float("nan"), float("inf")))
def test_grasp_state_machine_rejects_invalid_object_width(width):
    with pytest.raises(ValueError):
        ContactAwareGraspStateMachine(object_width_m=width)


def test_default_capture_confirmation_uses_six_control_ticks():
    machine = ContactAwareGraspStateMachine(control_hz=120)

    assert machine.capture_confirmation_steps == 6


def test_capture_confirmation_rejects_dynamic_bilateral_contact():
    machine = ContactAwareGraspStateMachine(
        control_hz=120,
        capture_confirmation_s=0.025,
        maximum_capture_relative_speed_mps=0.002,
    )
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))

    for _ in range(4):
        decision = machine.step(_evidence(relative_speed_mps=0.004))

    assert decision.phase is GraspPhase.SLOW_CLOSE
    assert machine.capture_steps == 0

    decision = machine.step(_evidence(relative_speed_mps=0.001))

    assert decision.phase is GraspPhase.SLOW_CLOSE
    assert machine.capture_steps == 1
    machine.step(_evidence(relative_speed_mps=0.001))
    decision = machine.step(_evidence(relative_speed_mps=0.001))

    assert decision.phase is GraspPhase.BILATERAL_SETTLE


def test_capture_confirmation_resets_after_dynamic_tick():
    machine = ContactAwareGraspStateMachine(
        control_hz=120,
        capture_confirmation_s=0.025,
        maximum_capture_relative_speed_mps=0.002,
    )
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))

    machine.step(_evidence(relative_speed_mps=0.001))
    machine.step(_evidence(relative_speed_mps=0.001))
    decision = machine.step(_evidence(relative_speed_mps=0.004))

    assert decision.phase is GraspPhase.SLOW_CLOSE
    assert machine.capture_steps == 0
    machine.step(_evidence(relative_speed_mps=0.001))
    machine.step(_evidence(relative_speed_mps=0.001))
    decision = machine.step(_evidence(relative_speed_mps=0.001))

    assert decision.phase is GraspPhase.BILATERAL_SETTLE


def test_capture_confirmation_rejects_translating_bilateral_enclosure():
    machine = ContactAwareGraspStateMachine(
        control_hz=120,
        capture_confirmation_s=0.025,
        maximum_relative_translation_error_m=0.003,
    )
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))

    for _ in range(4):
        decision = machine.step(
            _evidence(
                relative_translation_error_m=0.00495,
                relative_speed_mps=0.001,
            )
        )

    assert decision.phase is GraspPhase.SLOW_CLOSE
    assert machine.capture_steps == 0

    for _ in range(machine.capture_confirmation_steps):
        decision = machine.step(
            _evidence(
                relative_translation_error_m=0.002,
                relative_speed_mps=0.001,
            )
        )

    assert decision.phase is GraspPhase.BILATERAL_SETTLE
    assert decision.rebase_relative_tracking


def test_capture_confirmation_resets_after_translation_drift_tick():
    machine = ContactAwareGraspStateMachine(
        control_hz=120,
        capture_confirmation_s=0.025,
        maximum_relative_translation_error_m=0.003,
    )
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))

    machine.step(_evidence(relative_translation_error_m=0.002))
    machine.step(_evidence(relative_translation_error_m=0.002))
    decision = machine.step(_evidence(relative_translation_error_m=0.00495))

    assert decision.phase is GraspPhase.SLOW_CLOSE
    assert machine.capture_steps == 0
    machine.step(_evidence(relative_translation_error_m=0.002))
    machine.step(_evidence(relative_translation_error_m=0.002))
    decision = machine.step(_evidence(relative_translation_error_m=0.002))

    assert decision.phase is GraspPhase.BILATERAL_SETTLE


def test_stable_recentered_capture_rebases_candidate_before_admission():
    machine = ContactAwareGraspStateMachine(
        control_hz=120,
        capture_confirmation_s=0.025,
        bilateral_settle_s=0.10,
        maximum_relative_translation_error_m=0.003,
    )
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))

    for _ in range(11):
        decision = machine.step(
            _evidence(
                relative_translation_error_m=0.01258,
                relative_speed_mps=0.00004,
            )
        )
        assert not decision.rebase_relative_tracking

    decision = machine.step(
        _evidence(
            relative_translation_error_m=0.01258,
            relative_speed_mps=0.00004,
        )
    )

    assert decision.phase is GraspPhase.SLOW_CLOSE
    assert decision.rebase_capture_candidate_tracking
    assert decision.rebase_relative_tracking
    assert machine.capture_steps == 0

    for _ in range(machine.capture_confirmation_steps):
        decision = machine.step(
            _evidence(
                relative_translation_error_m=0.00002,
                relative_speed_mps=0.00004,
            )
        )

    assert decision.phase is GraspPhase.BILATERAL_SETTLE


def test_transient_sliding_edge_cannot_rebase_capture_candidate():
    machine = ContactAwareGraspStateMachine(
        control_hz=120,
        capture_confirmation_s=0.025,
        bilateral_settle_s=0.10,
        maximum_relative_translation_error_m=0.003,
    )
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))

    for speed in (0.00255, 0.00356, 0.00321, 0.00137):
        decision = machine.step(
            _evidence(
                relative_translation_error_m=0.00495,
                relative_speed_mps=speed,
            )
        )

    assert decision.phase is GraspPhase.SLOW_CLOSE
    assert not decision.rebase_relative_tracking
    assert machine.capture_rebase_steps == 1
    decision = machine.step(_evidence(right_force_n=0.0))
    assert not decision.rebase_relative_tracking
    assert machine.capture_rebase_steps == 0


def test_capture_admission_blocks_force_only_corner_contact():
    machine = ContactAwareGraspStateMachine(
        control_hz=120,
        minimum_contact_force_n=0.10,
        capture_contact_force_n=2.0,
        capture_confirmation_s=0.025,
    )
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))
    machine.step(_evidence(right_force_n=0.0))

    for _ in range(4):
        decision = machine.step(_evidence(capture_admissible=False))

    assert decision.phase is GraspPhase.SLOW_CLOSE
    assert machine.capture_steps == 0
    assert machine.step(_evidence()).phase is GraspPhase.SLOW_CLOSE
    assert machine.step(_evidence()).phase is GraspPhase.SLOW_CLOSE
    assert machine.step(_evidence()).phase is GraspPhase.BILATERAL_SETTLE


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
        {"minimum_proof_entry_force_n": 0.05},
        {"minimum_proof_entry_force_n": 60.0},
        {"capture_confirmation_s": -0.01},
        {"maximum_capture_relative_speed_mps": float("inf")},
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
