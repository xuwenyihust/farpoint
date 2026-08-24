import unittest

import pytest

from farpoint.control import (
    advance_so101_slow_close_target,
    apply_place_hover_guard,
    bilateral_grasp_ready,
    bounded_position_target,
    calibrated_recovery_grasp_target,
    cartesian_tracking_servo_target,
    collision_safe_pregrasp_waypoints,
    contact_pair_force_summary,
    filtered_contact_force,
    force_controlled_gripper_target,
    force_controlled_rotary_jaw_target,
    grasp_proof_evidence,
    grasp_validation_decision,
    gripper_aperture_alignment,
    integral_visual_servo_grasp_target,
    merge_contact_group_samples,
    placement_converged,
    rate_limit_revolute_joint_targets,
    relative_object_grasp_servo_target,
    rmpflow_world_target,
    settle_release_separation_target,
    so101_release_object_target,
    simulation_stop_reason,
    so101_approach_jaw_target,
    so101_adaptive_pre_capture_recenter_limit,
    so101_capture_admission_ready,
    so101_capture_admission_retention_fraction,
    so101_bilateral_capture_ready,
    so101_capture_contact_loss_grace_s,
    so101_capture_jaw_backoff_force_n,
    so101_pre_capture_recenter_step_m,
    so101_slow_close_bilateral_brake_force_n,
    so101_slow_close_backoff_step_rad,
    so101_balanced_capture_close_step,
    so101_cube_contact_handoff,
    so101_imbalanced_capture_close_step,
    so101_minimum_safe_descent_fraction,
    so101_post_capture_recenter_step,
    so101_proof_entry_force_floor,
    so101_pre_capture_recenter_limit,
    so101_reset_support_is_stable,
    tactile_contact_hold_target,
    tactile_search_active,
    temporal_contact_confirmed,
    track_observed_pick_target,
    transport_grasp_support,
    update_contact_loss_streak,
    unilateral_contact_recenter_target,
    undirected_axis_angle_error_degrees,
    unsafe_so101_approach_contact,
    visual_servo_grasp_target,
)


def test_so101_capture_admission_requires_calibrated_enclosure():
    assert not so101_capture_admission_ready(1.40, 0.04)
    assert so101_capture_admission_ready(1.05, 0.04)
    assert not so101_capture_admission_ready(1.051, 0.04)
    assert not so101_capture_admission_ready(0.91, 0.03)
    assert so101_capture_admission_ready(0.90, 0.04)
    assert so101_capture_admission_ready(0.55, 0.03)


@pytest.mark.parametrize(
    ("jaw", "width"),
    ((float("nan"), 0.04), (0.5, 0.0), (0.5, float("inf"))),
)
def test_so101_capture_admission_rejects_invalid_geometry(jaw, width):
    with pytest.raises(ValueError):
        so101_capture_admission_ready(jaw, width)


def test_so101_bilateral_capture_ready_uses_bounded_force_hysteresis():
    assert so101_bilateral_capture_ready(0.5, 0.5, True)
    assert so101_bilateral_capture_ready(
        0.5,
        0.5,
        True,
        object_width_m=0.03,
    )
    assert not so101_bilateral_capture_ready(0.5, 0.499, True)
    assert so101_bilateral_capture_ready(
        1.8,
        1.8,
        True,
        object_width_m=0.04,
    )
    assert not so101_bilateral_capture_ready(
        1.8,
        1.799,
        True,
        object_width_m=0.04,
    )
    assert not so101_bilateral_capture_ready(3.0, 3.0, False)
    assert not so101_bilateral_capture_ready(
        2.0,
        1.999,
        True,
        minimum_force_n=2.0,
    )


def test_so101_capture_admission_floor_is_shared_across_cube_sizes():
    assert so101_capture_admission_retention_fraction(None) == pytest.approx(0.25)
    assert so101_capture_admission_retention_fraction(0.03) == pytest.approx(0.25)
    assert so101_capture_admission_retention_fraction(0.035) == pytest.approx(0.575)
    assert so101_capture_admission_retention_fraction(0.04) == pytest.approx(0.90)


def test_so101_proof_entry_force_floor_is_size_aware():
    assert so101_proof_entry_force_floor(0.03) == pytest.approx(3.0)
    assert so101_proof_entry_force_floor(0.035) == pytest.approx(3.5)
    assert so101_proof_entry_force_floor(0.04) == pytest.approx(4.0)
    assert so101_proof_entry_force_floor(0.02) == pytest.approx(3.0)
    assert so101_proof_entry_force_floor(0.05) == pytest.approx(4.0)


def test_so101_imbalanced_capture_close_step_tapers_to_large_cube_pause():
    assert so101_imbalanced_capture_close_step(0.03) == pytest.approx(0.00025)
    assert so101_imbalanced_capture_close_step(0.035) == pytest.approx(0.000125)
    assert so101_imbalanced_capture_close_step(0.04) == pytest.approx(0.0)
    assert so101_imbalanced_capture_close_step(0.02) == pytest.approx(0.00025)
    assert so101_imbalanced_capture_close_step(0.05) == pytest.approx(0.0)


def test_so101_balanced_capture_close_step_slows_large_cube_settle():
    assert so101_balanced_capture_close_step(0.03) == pytest.approx(0.0005)
    assert so101_balanced_capture_close_step(0.035) == pytest.approx(0.0003125)
    assert so101_balanced_capture_close_step(0.04) == pytest.approx(0.000125)
    assert so101_balanced_capture_close_step(0.02) == pytest.approx(0.0005)
    assert so101_balanced_capture_close_step(0.05) == pytest.approx(0.000125)


def test_so101_capture_jaw_backoff_force_is_size_aware():
    assert so101_capture_jaw_backoff_force_n(0.03) == pytest.approx(20.0)
    assert so101_capture_jaw_backoff_force_n(0.035) == pytest.approx(18.5)
    assert so101_capture_jaw_backoff_force_n(0.04) == pytest.approx(17.0)
    assert so101_capture_jaw_backoff_force_n(0.02) == pytest.approx(20.0)
    assert so101_capture_jaw_backoff_force_n(0.05) == pytest.approx(17.0)


def test_so101_pre_capture_recenter_step_accelerates_only_invalid_large_geometry():
    assert so101_pre_capture_recenter_step_m(0.03, geometry_valid=False) == pytest.approx(0.000125)
    assert so101_pre_capture_recenter_step_m(0.04, geometry_valid=True) == pytest.approx(0.000125)
    assert so101_pre_capture_recenter_step_m(0.035, geometry_valid=False) == pytest.approx(0.0003125)
    assert so101_pre_capture_recenter_step_m(0.04, geometry_valid=False) == pytest.approx(0.0005)


def test_so101_slow_close_bilateral_brake_is_size_aware():
    assert so101_slow_close_bilateral_brake_force_n(0.03) == pytest.approx(20.0)
    assert so101_slow_close_bilateral_brake_force_n(0.035) == pytest.approx(16.0)
    assert so101_slow_close_bilateral_brake_force_n(0.04) == pytest.approx(12.0)
    assert so101_slow_close_bilateral_brake_force_n(0.05) == pytest.approx(12.0)


def test_so101_slow_close_backoff_step_avoids_large_cube_release_impulse():
    assert so101_slow_close_backoff_step_rad(0.03) == pytest.approx(0.002)
    assert so101_slow_close_backoff_step_rad(0.035) == pytest.approx(0.001)
    assert so101_slow_close_backoff_step_rad(0.04) == pytest.approx(0.0)
    assert so101_slow_close_backoff_step_rad(0.05) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("object_width_m", "balanced_close_step"),
    (
        (0.0, 0.0005),
        (float("nan"), 0.0005),
        (0.03, -0.0005),
        (0.03, float("nan")),
    ),
)
def test_so101_imbalanced_capture_close_step_rejects_invalid_contract(
    object_width_m,
    balanced_close_step,
):
    with pytest.raises(ValueError):
        so101_imbalanced_capture_close_step(
            object_width_m,
            balanced_close_step=balanced_close_step,
        )


@pytest.mark.parametrize(
    ("object_width_m", "balanced_close_step"),
    (
        (0.0, 0.0005),
        (float("nan"), 0.0005),
        (0.03, -0.0005),
        (0.03, float("nan")),
    ),
)
def test_so101_balanced_capture_close_step_rejects_invalid_contract(
    object_width_m,
    balanced_close_step,
):
    with pytest.raises(ValueError):
        so101_balanced_capture_close_step(
            object_width_m,
            balanced_close_step=balanced_close_step,
        )


@pytest.mark.parametrize("width", (0.0, float("nan"), float("inf")))
def test_so101_capture_jaw_backoff_force_rejects_invalid_width(width):
    with pytest.raises(ValueError, match="finite and positive"):
        so101_capture_jaw_backoff_force_n(width)


@pytest.mark.parametrize("width", (0.0, float("nan"), float("inf")))
def test_so101_proof_entry_force_floor_rejects_invalid_width(width):
    with pytest.raises(ValueError, match="finite and positive"):
        so101_proof_entry_force_floor(width)


@pytest.mark.parametrize(
    ("left", "right", "threshold"),
    (
        (float("nan"), 2.0, 2.0),
        (2.0, -0.1, 2.0),
        (2.0, 2.0, 0.0),
    ),
)
def test_so101_bilateral_capture_ready_rejects_invalid_inputs(
    left, right, threshold
):
    with pytest.raises(ValueError):
        so101_bilateral_capture_ready(
            left,
            right,
            True,
            minimum_force_n=threshold,
        )


@pytest.mark.parametrize("width", (0.0, float("nan"), float("inf")))
def test_so101_bilateral_capture_ready_rejects_invalid_object_width(width):
    with pytest.raises(ValueError):
        so101_bilateral_capture_ready(
            2.0,
            2.0,
            True,
            object_width_m=width,
        )


def test_so101_slow_close_threads_transparent_capture_admission():
    default = advance_so101_slow_close_target(
        1.0,
        1.0,
        3.0,
        3.0,
        open_position=1.7,
        closed_position=-0.2,
    )
    explicit = advance_so101_slow_close_target(
        1.0,
        1.0,
        3.0,
        3.0,
        open_position=1.7,
        closed_position=-0.2,
        capture_admissible=True,
    )

    assert explicit == default


def test_collision_safe_pregrasp_waypoints_raise_before_translating():
    waypoints = collision_safe_pregrasp_waypoints(
        [0.01, -0.30, 0.12],
        [0.13, -0.13, 0.14],
        clearance_z=0.22,
    )

    assert waypoints[0] == pytest.approx([0.01, -0.30, 0.22])
    assert waypoints[1] == pytest.approx([0.13, -0.13, 0.22])


def test_collision_safe_pregrasp_waypoints_reject_unsafe_clearance():
    with pytest.raises(ValueError, match="clearance_z"):
        collision_safe_pregrasp_waypoints(
            [0.01, -0.30, 0.12],
            [0.13, -0.13, 0.14],
            clearance_z=0.14,
        )


def test_so101_approach_jaw_uses_wide_calibrated_geometry_for_large_cube():
    assert so101_approach_jaw_target(0.03) == pytest.approx(0.90)
    assert so101_approach_jaw_target(0.035) == pytest.approx(1.15)
    assert so101_approach_jaw_target(0.04) == pytest.approx(1.40)
    assert so101_approach_jaw_target(0.10) == pytest.approx(1.40)
    with pytest.raises(ValueError, match="finite and positive"):
        so101_approach_jaw_target(0.0)


def test_so101_descent_contact_window_opens_for_large_cube():
    assert so101_minimum_safe_descent_fraction(0.03) == pytest.approx(0.75)
    assert so101_minimum_safe_descent_fraction(0.035) == pytest.approx(0.675)
    assert so101_minimum_safe_descent_fraction(0.04) == pytest.approx(0.60)
    assert so101_minimum_safe_descent_fraction(0.10) == pytest.approx(0.60)
    with pytest.raises(ValueError, match="finite and positive"):
        so101_minimum_safe_descent_fraction(float("nan"))


def test_so101_cube_contact_handoff_uses_first_filtered_finger_contact():
    assert not so101_cube_contact_handoff(0.0, 0.099)
    assert so101_cube_contact_handoff(0.0, 0.10)
    assert so101_cube_contact_handoff(0.731, 0.0)


def test_so101_pre_capture_recenter_limit_preserves_validated_corridor():
    assert so101_pre_capture_recenter_limit(0.03) == pytest.approx(0.008)
    assert so101_pre_capture_recenter_limit(0.04) == pytest.approx(0.008)


def test_so101_adaptive_pre_capture_recenter_requires_unilateral_axis_saturation():
    assert so101_adaptive_pre_capture_recenter_limit(
        0.04, (0.008, 0.008), unilateral_contact=True
    ) == pytest.approx(0.016)
    assert so101_adaptive_pre_capture_recenter_limit(
        0.03, (0.008, 0.008), unilateral_contact=True
    ) == pytest.approx(0.009)
    assert so101_adaptive_pre_capture_recenter_limit(
        0.04, (0.004, 0.008), unilateral_contact=True
    ) == pytest.approx(0.016)
    assert so101_adaptive_pre_capture_recenter_limit(
        0.03, (0.0, 0.008), unilateral_contact=True
    ) == pytest.approx(0.009)
    assert so101_adaptive_pre_capture_recenter_limit(
        0.04, (0.008, 0.008), unilateral_contact=False
    ) == pytest.approx(0.008)
    assert so101_adaptive_pre_capture_recenter_limit(
        0.035, (0.008, 0.008), unilateral_contact=True
    ) == pytest.approx(0.01225)


def test_so101_capture_contact_loss_grace_is_size_aware():
    assert so101_capture_contact_loss_grace_s(0.03) == pytest.approx(0.30)
    assert so101_capture_contact_loss_grace_s(0.035) == pytest.approx(0.25)
    assert so101_capture_contact_loss_grace_s(0.04) == pytest.approx(0.20)
    assert so101_capture_contact_loss_grace_s(0.02) == pytest.approx(0.30)
    assert so101_capture_contact_loss_grace_s(0.05) == pytest.approx(0.20)


def test_so101_post_capture_recenter_reaches_bound_before_shortest_grace_expires():
    step = so101_post_capture_recenter_step()

    assert step == pytest.approx(0.000125)
    assert step * 16 == pytest.approx(0.002)
    assert 16 < 0.20 * 120


@pytest.mark.parametrize(
    "kwargs",
    (
        {"maximum_correction_m": 0.0},
        {"maximum_correction_m": float("nan")},
        {"active_recovery_steps": 0},
        {"active_recovery_steps": 1.5},
    ),
)
def test_so101_post_capture_recenter_rejects_invalid_contract(kwargs):
    with pytest.raises(ValueError):
        so101_post_capture_recenter_step(**kwargs)


@pytest.mark.parametrize("value", (0.0, -0.01, float("nan"), float("inf")))
def test_so101_capture_contact_loss_grace_rejects_invalid_width(value):
    with pytest.raises(ValueError):
        so101_capture_contact_loss_grace_s(value)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"object_width_m": 0.0},
        {"object_width_m": float("nan")},
        {"object_width_m": 0.03, "maximum_correction_m": 0.0},
    ),
)
def test_so101_pre_capture_recenter_limit_rejects_invalid_contract(kwargs):
    with pytest.raises(ValueError):
        so101_pre_capture_recenter_limit(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"object_width_m": 0.0, "current_xy_correction_m": (0.0, 0.0)},
        {"object_width_m": 0.04, "current_xy_correction_m": (0.0,)},
        {"object_width_m": 0.04, "current_xy_correction_m": (0.0, float("nan"))},
        {
            "object_width_m": 0.04,
            "current_xy_correction_m": (0.0, 0.0),
            "maximum_correction_m": 0.007,
        },
        {
            "object_width_m": 0.04,
            "current_xy_correction_m": (0.008, 0.008),
            "large_width_fraction": 0.29,
        },
    ),
)
def test_so101_adaptive_pre_capture_recenter_rejects_invalid_contract(kwargs):
    with pytest.raises(ValueError):
        so101_adaptive_pre_capture_recenter_limit(**kwargs)


@pytest.mark.parametrize(
    ("left_force_n", "right_force_n", "minimum_force_n"),
    [
        (-0.1, 0.0, 0.1),
        (float("nan"), 0.0, 0.1),
        (0.0, float("inf"), 0.1),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, float("nan")),
    ],
)
def test_so101_cube_contact_handoff_rejects_invalid_force_contract(
    left_force_n, right_force_n, minimum_force_n
):
    with pytest.raises(ValueError):
        so101_cube_contact_handoff(
            left_force_n,
            right_force_n,
            minimum_force_n=minimum_force_n,
        )


def test_large_cube_first_corner_contact_is_not_a_collision():
    threshold = so101_minimum_safe_descent_fraction(0.04)

    assert unsafe_so101_approach_contact(
        "descend", True, 0.59, minimum_safe_descent_fraction=threshold
    )
    assert not unsafe_so101_approach_contact(
        "descend", True, 0.68, minimum_safe_descent_fraction=threshold
    )


def test_settle_release_separation_ramps_and_caps_vertical_clearance():
    start = [0.20, 0.10, 0.08]
    assert settle_release_separation_target(
        start, 0, control_hz=120
    ) == pytest.approx([0.20, 0.10, 0.080125])
    assert settle_release_separation_target(
        start, 500, control_hz=120
    ) == pytest.approx([0.20, 0.10, 0.10])


def test_release_descent_preserves_validated_transport_xy():
    assert so101_release_object_target(
        [0.153, 0.073, 0.091], 0.080
    ) == pytest.approx([0.153, 0.073, 0.080])


def test_release_descent_rejects_nonfinite_targets():
    with pytest.raises(ValueError, match="finite"):
        so101_release_object_target([0.153, float("nan"), 0.091], 0.080)


def test_unsafe_so101_approach_contact_rejects_route_and_early_insertion():
    assert unsafe_so101_approach_contact("pregrasp", True)
    assert unsafe_so101_approach_contact("descend", True, 0.25)
    assert not unsafe_so101_approach_contact("descend", True, 0.80)
    assert not unsafe_so101_approach_contact("descend", False, 0.25)


def test_grasp_proof_uses_cumulative_contact_evidence():
    evidence = grasp_proof_evidence(
        object_lift=0.018,
        minimum_object_lift=0.012,
        bilateral_contact_frames=12,
        required_contact_frames=12,
        support_present=True,
        maximum_rigidity_error=0.0002,
        maximum_allowed_rigidity_error=0.015,
    )

    assert evidence == {
        "lift": True,
        "contact_history": True,
        "support": True,
        "rigidity": True,
        "passed": True,
    }


def test_grasp_proof_still_rejects_missing_contact_or_support():
    evidence = grasp_proof_evidence(
        object_lift=0.018,
        minimum_object_lift=0.012,
        bilateral_contact_frames=11,
        required_contact_frames=12,
        support_present=False,
        maximum_rigidity_error=0.0002,
        maximum_allowed_rigidity_error=0.015,
    )

    assert not evidence["contact_history"]
    assert not evidence["support"]
    assert not evidence["passed"]


def test_recovery_grasp_target_reuses_aperture_calibration_and_bias():
    target = calibrated_recovery_grasp_target(
        [0.969, 0.196, 0.4175],
        [0.0, -0.015, -0.135],
        aperture_tool_offset=[-0.00025, -0.00056, -0.11997],
        aperture_bias_xy=[-0.006, -0.015],
    )

    assert target == pytest.approx([0.96325, 0.18156, 0.53747])


def test_recovery_grasp_target_falls_back_to_configured_offset():
    target = calibrated_recovery_grasp_target(
        [0.98, 0.25, 0.4175],
        [0.0, -0.015, -0.135],
    )

    assert target == pytest.approx([0.98, 0.265, 0.5525])


def test_bounded_position_target_limits_drive_error_in_both_directions():
    assert bounded_position_target(0.30, 0.04, 0.03) == pytest.approx(0.07)
    assert bounded_position_target(0.0, 0.20, 0.03) == pytest.approx(0.17)
    assert bounded_position_target(0.05, 0.04, 0.03) == pytest.approx(0.05)


def test_transport_grasp_support_uses_direct_contact_first():
    result = transport_grasp_support(
        True,
        None,
        None,
        None,
        max_rigidity_error=0.015,
    )
    assert result == {
        "present": True,
        "source": "direct_contact",
        "rigidity_error": 0.0,
    }


def test_transport_grasp_support_accepts_only_rigid_relative_motion():
    supported = transport_grasp_support(
        False,
        [0.95, 0.27, 0.43],
        [0.95, 0.27, 0.58],
        [0.0, 0.0, -0.15],
        max_rigidity_error=0.015,
    )
    slipped = transport_grasp_support(
        False,
        [0.95, 0.27, 0.41],
        [0.95, 0.27, 0.58],
        [0.0, 0.0, -0.15],
        max_rigidity_error=0.015,
    )
    assert supported["present"]
    assert supported["source"] == "rigidity_inferred"
    assert supported["rigidity_error"] == pytest.approx(0.0)
    assert not slipped["present"]
    assert slipped["source"] is None
    assert slipped["rigidity_error"] == pytest.approx(0.02)


def test_simulation_stops_as_soon_as_release_has_settled():
    assert (
        simulation_stop_reason(
            1119,
            nominal_frame_count=5200,
            extension_frames=1200,
            release_complete_frame=1000,
            required_settle_frames=120,
            grasp_validated=True,
        )
        == "release_settled"
    )


def test_simulation_extension_is_available_only_after_validated_grasp():
    common = {
        "nominal_frame_count": 5200,
        "extension_frames": 1200,
        "release_complete_frame": None,
        "required_settle_frames": 120,
    }
    assert (
        simulation_stop_reason(5199, grasp_validated=False, **common)
        == "nominal_budget_exhausted"
    )
    assert simulation_stop_reason(5199, grasp_validated=True, **common) is None
    assert (
        simulation_stop_reason(6399, grasp_validated=True, **common)
        == "extension_budget_exhausted"
    )


def test_tactile_contact_hold_target_starts_from_measured_preload():
    assert tactile_contact_hold_target(0.36, 0.01, 0.45) == pytest.approx(
        0.37
    )
    assert tactile_contact_hold_target(0.448, 0.01, 0.45) == pytest.approx(
        0.45
    )


def test_undirected_axis_angle_error_treats_opposite_axes_as_equal():
    assert undirected_axis_angle_error_degrees(-90.0, 90.0) == pytest.approx(
        0.0
    )
    assert undirected_axis_angle_error_degrees(123.0, 90.0) == pytest.approx(
        33.0
    )


def test_filtered_contact_force_excludes_self_contacts():
    result = filtered_contact_force(
        [
            {
                "body0": "/World/Robot/left_finger",
                "body1": "/World/PickObject",
                "impulse": [0.03, 0.04, 0.0],
            },
            {
                "body0": "/World/Robot/left_finger",
                "body1": "/World/Robot/left_knuckle",
                "impulse": [10.0, 0.0, 0.0],
            },
        ],
        "/World/PickObject",
        physics_dt=0.01,
    )

    assert result["matching_contacts"] == 1
    assert result["force"] == pytest.approx(5.0)


def test_contact_pair_force_summary_aggregates_and_sorts_pairs():
    result = contact_pair_force_summary(
        [
            {
                "body0": "/World/Robot/finger",
                "body1": "/World/WorkTable",
                "impulse": [0.03, 0.04, 0.0],
            },
            {
                "body0": "/World/WorkTable",
                "body1": "/World/Robot/finger",
                "impulse": [0.0, 0.02, 0.0],
            },
            {
                "body0": "/World/Robot/finger",
                "body1": "/World/PickObject",
                "impulse": [0.01, 0.0, 0.0],
            },
        ],
        physics_dt=0.01,
    )

    assert result == [
        {
            "body0": "/World/Robot/finger",
            "body1": "/World/WorkTable",
            "force": pytest.approx(7.0),
            "contact_count": 2,
            "impulse_on_body0": pytest.approx([0.03, 0.02, 0.0]),
        },
        {
            "body0": "/World/PickObject",
            "body1": "/World/Robot/finger",
            "force": pytest.approx(1.0),
            "contact_count": 1,
            "impulse_on_body0": pytest.approx([-0.01, 0.0, 0.0]),
        },
    ]


def test_merge_contact_group_samples_keeps_substep_peaks():
    merged = merge_contact_group_samples(
        [
            (
                2.0,
                True,
                [2.0, 0.0],
                [
                    {
                        "body0": "/World/PickObject",
                        "body1": "/World/Robot/finger",
                        "force": 2.0,
                    }
                ],
            ),
            (
                0.5,
                True,
                [0.0, 0.5],
                [
                    {
                        "body0": "/World/PickObject",
                        "body1": "/World/Robot/finger",
                        "force": 0.5,
                    },
                    {
                        "body0": "/World/Robot/finger",
                        "body1": "/World/WorkTable",
                        "force": 0.25,
                    },
                ],
            ),
        ]
    )

    assert merged[0] == pytest.approx(2.0)
    assert merged[1]
    assert merged[2] == pytest.approx([2.0, 0.5])
    assert [pair["force"] for pair in merged[3]] == pytest.approx(
        [2.0, 0.25]
    )


def test_rate_limit_revolute_joint_targets_uses_nearest_angle_branch():
    result = rate_limit_revolute_joint_targets(
        [0.0, 1.90, 0.25],
        [0.2, -1.45, 0.45],
        [1],
        max_step=0.01,
    )

    assert result == pytest.approx([0.2, 1.91, 0.45])


def test_force_controlled_gripper_backs_off_from_measured_position():
    result = force_controlled_gripper_target(
        0.265,
        0.260,
        26.0,
        0.0,
        min_force=1.0,
        max_force=8.0,
        close_step=0.0001,
        backoff_step=0.001,
        max_position=0.45,
    )

    assert result["action"] == "backoff"
    assert result["position"] == pytest.approx(0.259)


def test_force_controlled_gripper_closes_only_below_force_limit():
    close = force_controlled_gripper_target(
        0.24,
        0.23,
        1.5,
        0.2,
        min_force=1.0,
        max_force=8.0,
        close_step=0.0001,
        backoff_step=0.001,
        max_position=0.45,
    )
    hold = force_controlled_gripper_target(
        0.24,
        0.23,
        1.5,
        1.2,
        min_force=1.0,
        max_force=8.0,
        close_step=0.0001,
        backoff_step=0.001,
        max_position=0.45,
    )

    assert close["action"] == "close"
    assert close["position"] == pytest.approx(0.2401)
    assert hold == {"position": 0.24, "action": "hold"}


def test_force_controlled_gripper_holds_preload_on_unilateral_contact():
    result = force_controlled_gripper_target(
        0.331,
        0.326,
        0.0,
        16.0,
        min_force=1.0,
        max_force=25.0,
        close_step=0.0001,
        backoff_step=0.0001,
        max_position=0.45,
        close_on_unilateral=False,
    )

    assert result["position"] == pytest.approx(0.331)
    assert result["action"] == "unilateral_hold"


def test_force_controlled_gripper_clamps_unilateral_hold_to_ceiling():
    result = force_controlled_gripper_target(
        0.331,
        0.326,
        0.0,
        16.0,
        min_force=1.0,
        max_force=25.0,
        close_step=0.0001,
        backoff_step=0.0001,
        max_position=0.315,
        close_on_unilateral=False,
    )

    assert result["position"] == pytest.approx(0.315)
    assert result["action"] == "unilateral_hold"


def test_force_controlled_gripper_limits_preload_windup_and_backoff():
    close = force_controlled_gripper_target(
        0.308,
        0.278,
        0.0,
        0.0,
        min_force=3.0,
        max_force=25.0,
        close_step=0.0001,
        backoff_step=0.0001,
        max_position=0.45,
        max_preload_error=0.010,
    )
    backoff = force_controlled_gripper_target(
        0.308,
        0.278,
        30.0,
        10.0,
        min_force=3.0,
        max_force=25.0,
        close_step=0.0001,
        backoff_step=0.0001,
        max_position=0.45,
        max_preload_error=0.010,
    )

    assert close["position"] == pytest.approx(0.288)
    assert backoff["position"] == pytest.approx(0.2879)


def test_force_controlled_gripper_uses_fixed_preload_reference_across_updates():
    first = force_controlled_gripper_target(
        0.308,
        0.300,
        0.0,
        0.0,
        min_force=3.0,
        max_force=25.0,
        close_step=0.001,
        backoff_step=0.0001,
        max_position=0.45,
        max_preload_error=0.010,
        preload_reference_position=0.278,
    )
    later = force_controlled_gripper_target(
        first["position"],
        0.305,
        0.0,
        0.0,
        min_force=3.0,
        max_force=25.0,
        close_step=0.001,
        backoff_step=0.0001,
        max_position=0.45,
        max_preload_error=0.010,
        preload_reference_position=0.278,
    )

    assert first["position"] == pytest.approx(0.288)
    assert later["position"] == pytest.approx(0.288)


def test_force_controlled_gripper_respects_validated_transport_ceiling():
    validated_position = 0.312
    transport_ceiling = validated_position + 0.003
    first = force_controlled_gripper_target(
        validated_position,
        validated_position,
        0.0,
        0.0,
        min_force=0.7,
        max_force=25.0,
        close_step=0.002,
        backoff_step=0.0001,
        max_position=transport_ceiling,
        max_preload_error=0.020,
        preload_reference_position=validated_position,
    )
    capped = force_controlled_gripper_target(
        first["position"],
        first["position"],
        0.0,
        0.0,
        min_force=0.7,
        max_force=25.0,
        close_step=0.002,
        backoff_step=0.0001,
        max_position=transport_ceiling,
        max_preload_error=0.020,
        preload_reference_position=validated_position,
    )

    assert first["position"] == pytest.approx(0.314)
    assert capped["position"] == pytest.approx(transport_ceiling)


def test_force_controlled_rotary_jaw_closes_toward_lower_position():
    update = force_controlled_rotary_jaw_target(
        0.79,
        0.791,
        2.2,
        0.6,
        open_position=1.745,
        closed_position=-0.175,
        min_force=2.0,
        max_force=20.0,
        close_step=0.002,
        backoff_step=0.001,
        max_preload_error=0.02,
        preload_reference_position=0.791,
    )

    assert update == {"position": pytest.approx(0.788), "action": "close"}


def test_force_controlled_rotary_jaw_restores_30mm_settling_force_floor():
    update = force_controlled_rotary_jaw_target(
        0.2514,
        0.2544,
        1.55,
        1.65,
        open_position=1.7453,
        closed_position=-0.1746,
        min_force=2.0,
        max_force=60.0,
        close_step=0.0005,
        backoff_step=0.001,
        max_preload_error=0.012,
        preload_reference_position=0.2514,
    )

    assert update == {"position": pytest.approx(0.2509), "action": "close"}


def test_so101_reset_support_requires_settled_table_pose():
    expected = [0.18, -0.07, 0.052]
    assert so101_reset_support_is_stable(
        expected,
        [0.1805, -0.0705, 0.0524],
        [0.001, 0.0, -0.002],
    )
    assert not so101_reset_support_is_stable(
        expected,
        [0.18, -0.07, -0.10],
        [0.0, 0.0, -1.0],
    )
    assert not so101_reset_support_is_stable(
        expected,
        [0.18, -0.07, 0.052],
        [0.0, 0.0, 0.06],
    )


def test_force_controlled_rotary_jaw_backs_off_high_force_and_limits_preload():
    backoff = force_controlled_rotary_jaw_target(
        0.78,
        0.779,
        23.0,
        0.0,
        open_position=1.745,
        closed_position=-0.175,
        min_force=2.0,
        max_force=20.0,
        close_step=0.002,
        backoff_step=0.001,
        max_preload_error=0.02,
        preload_reference_position=0.791,
    )
    capped = force_controlled_rotary_jaw_target(
        0.771,
        0.771,
        0.0,
        0.0,
        open_position=1.745,
        closed_position=-0.175,
        min_force=2.0,
        max_force=20.0,
        close_step=0.002,
        backoff_step=0.001,
        max_preload_error=0.02,
        preload_reference_position=0.791,
    )

    assert backoff == {"position": pytest.approx(0.781), "action": "backoff"}
    assert capped == {"position": pytest.approx(0.771), "action": "close"}


def test_so101_slow_close_accumulates_command_under_actuator_lag():
    target = 1.40
    measured = 1.40
    for _ in range(120):
        update = advance_so101_slow_close_target(
            target,
            measured,
            0.0,
            0.0,
            open_position=1.7453,
            closed_position=-0.1746,
        )
        target = update["position"]
        measured = max(target, measured - 0.0002)

    assert target == pytest.approx(1.28)
    assert measured == pytest.approx(1.376)
    assert target - measured == pytest.approx(-0.096)


@pytest.mark.parametrize("approach_target", [0.90, 1.20])
def test_so101_slow_close_reaches_mechanical_limit_inside_phase_budget(
    approach_target,
):
    closed = -0.1746
    target = approach_target
    steps = 0
    while target > closed and steps < 2400:
        update = advance_so101_slow_close_target(
            target,
            target,
            0.0,
            0.0,
            open_position=1.7453,
            closed_position=closed,
        )
        target = update["position"]
        assert closed <= target <= 1.7453
        steps += 1

    assert target == pytest.approx(closed)
    assert steps < 2400


def test_so101_slow_close_force_actions_preserve_limits():
    unilateral = advance_so101_slow_close_target(
        0.50,
        0.51,
        3.0,
        0.0,
        open_position=1.7453,
        closed_position=-0.1746,
    )
    bilateral = advance_so101_slow_close_target(
        unilateral["position"],
        0.50,
        2.0,
        2.0,
        open_position=1.7453,
        closed_position=-0.1746,
    )
    high_force = advance_so101_slow_close_target(
        -0.1746,
        -0.17,
        21.0,
        0.0,
        open_position=1.7453,
        closed_position=-0.1746,
    )
    assert unilateral == {"position": pytest.approx(0.499), "action": "close"}
    assert bilateral == {"position": pytest.approx(0.499), "action": "hold"}
    assert high_force == {"position": pytest.approx(-0.168), "action": "backoff"}
    assert -0.1746 <= high_force["position"] <= 1.7453


def test_so101_slow_close_overforce_discards_stale_close_backlog():
    update = advance_so101_slow_close_target(
        0.411947,
        0.466314,
        17.803,
        4.579,
        open_position=1.7453,
        closed_position=-0.1746,
        max_force=17.0,
        backoff_step=0.002,
        capture_admissible=False,
    )

    assert update == {"position": pytest.approx(0.468314), "action": "backoff"}


def test_so101_slow_close_keeps_unilateral_search_above_bilateral_brake():
    update = advance_so101_slow_close_target(
        0.421947,
        0.476176,
        10.557,
        0.932,
        open_position=1.7453,
        closed_position=-0.1746,
        max_force=12.0,
        unilateral_backoff_force=17.0,
        backoff_step=0.0,
        capture_admissible=False,
    )

    assert update == {"position": pytest.approx(0.420947), "action": "close"}


def test_so101_slow_close_crosses_observed_unilateral_limit_cycles_then_backs_off():
    crossing = advance_so101_slow_close_target(
        0.40,
        0.405,
        16.93,
        0.0,
        open_position=1.7453,
        closed_position=-0.1746,
        min_force=2.0,
        max_force=20.0,
    )
    backoff = advance_so101_slow_close_target(
        crossing["position"],
        0.404,
        17.0,
        0.0,
        open_position=1.7453,
        closed_position=-0.1746,
        min_force=2.0,
        max_force=20.0,
    )

    assert crossing == {"position": pytest.approx(0.399), "action": "close"}
    assert backoff == {"position": pytest.approx(0.406), "action": "backoff"}


@pytest.mark.parametrize("fraction", (0.0, 1.0, float("nan")))
def test_so101_slow_close_rejects_invalid_unilateral_backoff_fraction(fraction):
    with pytest.raises(ValueError, match="unilateral_backoff_fraction"):
        advance_so101_slow_close_target(
            0.40,
            0.405,
            0.0,
            0.0,
            open_position=1.7453,
            closed_position=-0.1746,
            unilateral_backoff_fraction=fraction,
        )


def test_gripper_aperture_alignment_uses_finger_bounds_midpoint():
    result = gripper_aperture_alignment(
        {"center": [1.02, 0.24, 0.41]},
        {"center": [0.98, 0.36, 0.41]},
        [1.005, 0.295, 0.4075],
    )

    assert result["aperture_center"] == pytest.approx([1.0, 0.3, 0.41])
    assert result["finger_center_delta"] == pytest.approx([-0.04, 0.12, 0.0])
    assert result["finger_z_skew"] == pytest.approx(0.0)
    assert result["position_error"] == pytest.approx([0.005, -0.005, -0.0025])
    assert result["xy_error"] == pytest.approx(0.0070710678)
    assert result["z_error"] == pytest.approx(0.0025)


def test_relative_object_grasp_servo_tracks_offset_with_norm_limit():
    result = relative_object_grasp_servo_target(
        [0.15, -0.11, 0.06],
        [0.03, 0.01, -0.04],
        [0.13, -0.13, 0.09],
        [0.13, -0.13, 0.09],
        max_step=0.005,
        max_correction=[0.02, 0.02, 0.02],
    )

    assert result["desired_position"] == pytest.approx([0.12, -0.12, 0.10])
    delta = [
        result["position"][axis] - [0.13, -0.13, 0.09][axis]
        for axis in range(3)
    ]
    assert sum(value * value for value in delta) ** 0.5 == pytest.approx(0.005)


def test_relative_object_grasp_servo_recovers_yaw30_static_hold_drift():
    captured_object = [0.2514796, -0.0771627, 0.0469985]
    captured_grasp = [0.2116900, -0.1243032, 0.0793410]
    capture_offset = [
        captured_object[axis] - captured_grasp[axis] for axis in range(3)
    ]
    measured_grasp = [0.2096628, -0.1253432, 0.0771301]
    result = relative_object_grasp_servo_target(
        [0.2515510, -0.0766251, 0.0470775],
        capture_offset,
        measured_grasp,
        captured_grasp,
        max_step=0.000125,
        max_correction=[0.006, 0.006, 0.006],
    )

    correction = [
        result["position"][axis] - measured_grasp[axis] for axis in range(3)
    ]
    assert all(value > 0.0 for value in correction)
    assert sum(value * value for value in correction) ** 0.5 == pytest.approx(
        0.000125
    )


def test_relative_object_grasp_servo_recovers_c22_physical_capture_offset():
    captured_object = [0.1854020655, -0.0461667143, 0.0521944426]
    captured_grasp = [0.1527982503, -0.1056722030, 0.0845712945]
    capture_offset = [
        captured_object[axis] - captured_grasp[axis] for axis in range(3)
    ]
    measured_object = [0.1868820041, -0.0458244756, 0.0525271036]
    planar_object = measured_object.copy()
    planar_object[2] = capture_offset[2] + captured_grasp[2]

    result = relative_object_grasp_servo_target(
        planar_object,
        capture_offset,
        captured_grasp,
        captured_grasp,
        max_step=0.000125,
        max_correction=[0.002, 0.002, 0.0],
    )

    correction = [
        result["position"][axis] - captured_grasp[axis] for axis in range(3)
    ]
    assert correction[0] > 0.0
    assert abs(correction[0]) > abs(correction[1])
    assert correction[2] == pytest.approx(0.0)
    assert sum(value * value for value in correction) ** 0.5 == pytest.approx(
        0.000125
    )


def test_relative_object_grasp_servo_limits_total_axis_correction():
    result = relative_object_grasp_servo_target(
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
        max_step=2.0,
        max_correction=[0.01, 0.02, 0.03],
    )

    assert result["position"] == pytest.approx([0.11, 0.22, 0.33])


def test_relative_object_grasp_servo_rejects_invalid_limits():
    with pytest.raises(ValueError, match="max_step"):
        relative_object_grasp_servo_target(
            [0.0] * 3,
            [0.0] * 3,
            [0.0] * 3,
            [0.0] * 3,
            max_step=0.0,
        )


def test_unilateral_contact_recenter_moves_toward_contact_side():
    bounds_left = {"center": [1.0, 0.2, 0.42]}
    bounds_right = {"center": [1.0, 0.3, 0.41]}

    left = unilateral_contact_recenter_target(
        [1.0, 0.25, 0.54],
        [1.0, 0.25, 0.54],
        bounds_left,
        bounds_right,
        1.0,
        0.0,
        step=0.001,
    )
    right = unilateral_contact_recenter_target(
        [1.0, 0.25, 0.54],
        [1.0, 0.25, 0.54],
        bounds_left,
        bounds_right,
        0.0,
        1.0,
        step=0.001,
    )

    assert left["position"] == pytest.approx([1.0, 0.249, 0.54])
    assert left["contact_side"] == "left"
    assert right["position"] == pytest.approx([1.0, 0.251, 0.54])
    assert right["contact_side"] == "right"


def test_unilateral_contact_recenter_limits_total_xy_correction():
    result = unilateral_contact_recenter_target(
        [1.0, 0.2699, 0.54],
        [1.0, 0.25, 0.54],
        {"center": [1.0, 0.2, 0.42]},
        {"center": [1.0, 0.3, 0.41]},
        0.0,
        1.0,
        step=0.001,
        max_correction=0.02,
    )

    assert result["position"] == pytest.approx([1.0, 0.27, 0.54])


def test_transport_recenter_moves_away_from_high_force_contact_side():
    result = unilateral_contact_recenter_target(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        {"center": [1.0, 0.2, 0.42]},
        {"center": [1.0, 0.3, 0.41]},
        0.0,
        8.0,
        step=0.001,
        move_toward_contact=False,
    )

    assert result["position"] == pytest.approx([0.0, -0.001, 0.0])
    assert result["contact_side"] == "right"


def test_observed_pick_tracking_preserves_tactile_correction():
    result = track_observed_pick_target(
        [0.92, 0.28, 0.4075],
        [-0.02, -0.035, -0.135],
        [0.973, 0.297, 0.5425],
        [0.974, 0.315, 0.5425],
        max_step=[0.01, 0.01, 0.01],
    )

    assert result["nominal_position"] == pytest.approx(
        [0.964, 0.315, 0.5425]
    )
    assert result["position"] == pytest.approx(
        [0.963, 0.297, 0.5425]
    )
    assert result["tactile_correction"] == pytest.approx(
        [-0.001, -0.018, 0.0]
    )


def test_grasp_validation_waits_for_terminal_raw_contact():
    assert grasp_validation_decision(
        60,
        60,
        min_frames=60,
        max_frames=180,
        min_support_frames=30,
        confirmed_raw_contact=False,
    ) == "pending"
    assert grasp_validation_decision(
        75,
        62,
        min_frames=60,
        max_frames=180,
        min_support_frames=30,
        confirmed_raw_contact=True,
    ) == "validated"


def test_grasp_validation_requires_consecutive_terminal_stability():
    assert grasp_validation_decision(
        75,
        62,
        min_frames=60,
        max_frames=180,
        min_support_frames=30,
        confirmed_raw_contact=True,
        terminal_stable_frames=19,
        required_terminal_stable_frames=20,
    ) == "pending"
    assert grasp_validation_decision(
        76,
        63,
        min_frames=60,
        max_frames=180,
        min_support_frames=30,
        confirmed_raw_contact=True,
        terminal_stable_frames=20,
        required_terminal_stable_frames=20,
    ) == "validated"


def test_grasp_validation_fails_at_maximum_window():
    assert grasp_validation_decision(
        180,
        90,
        min_frames=60,
        max_frames=180,
        min_support_frames=30,
        confirmed_raw_contact=False,
    ) == "failed"


def test_temporal_contact_confirmation_allows_solver_oscillation():
    assert temporal_contact_confirmed(
        [100, 103, 108],
        110,
        required_events=3,
        window_frames=12,
    )
    assert not temporal_contact_confirmed(
        [90, 100, 108],
        110,
        required_events=3,
        window_frames=12,
    )


class ControlTests(unittest.TestCase):
    def test_rmpflow_target_remains_in_world_frame(self):
        self.assertEqual(
            rmpflow_world_target([0.96, -0.04, 0.72]),
            [0.96, -0.04, 0.72],
        )

    def test_visual_servo_uses_observed_object_error(self):
        result = visual_servo_grasp_target(
            current_grasp_position=[0.82, 0.01, 0.75],
            object_position_estimate=[0.855, -0.054, 0.618],
            target_object_position=[0.850, 0.049, 0.418],
            commanded_grasp_position=[0.870, 0.084, 0.543],
            nominal_grasp_position=[0.870, 0.084, 0.543],
            gain=1.0,
            max_step=0.03,
            max_correction=0.18,
        )
        self.assertAlmostEqual(result["grasp_position"][0], 0.84)
        self.assertAlmostEqual(result["grasp_position"][1], 0.113)
        self.assertAlmostEqual(result["grasp_position"][2], 0.55)
        self.assertAlmostEqual(result["xy_error"], 0.103121, places=5)
        self.assertAlmostEqual(result["z_error"], 0.2)

    def test_visual_servo_limits_total_correction(self):
        result = visual_servo_grasp_target(
            current_grasp_position=[0.0, 0.0, 0.0],
            object_position_estimate=[1.0, 1.0, 1.0],
            target_object_position=[0.0, 0.0, 0.0],
            commanded_grasp_position=[0.0, 0.0, 0.0],
            nominal_grasp_position=[0.0, 0.0, 0.0],
            gain=1.0,
            max_step=1.0,
            max_correction=0.2,
        )
        self.assertEqual(result["grasp_position"], [-0.2, -0.2, -0.2])

    def test_integral_visual_servo_accumulates_error_with_anti_windup(self):
        result = integral_visual_servo_grasp_target(
            object_position_estimate=[0.90, -0.10, 0.53],
            target_object_position=[0.96, -0.04, 0.42],
            commanded_grasp_position=[0.93, 0.00, 0.64],
            nominal_grasp_position=[0.98, 0.00, 0.54],
            gain=0.35,
            max_step=0.03,
            max_correction=0.18,
        )
        for actual, expected in zip(
            result["grasp_position"],
            [0.951, 0.021, 0.61],
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result["saturated_axes"], [False, False, False])

        saturated = integral_visual_servo_grasp_target(
            object_position_estimate=[0.0, 0.0, 0.0],
            target_object_position=[1.0, 1.0, 1.0],
            commanded_grasp_position=[0.19, 0.19, 0.19],
            nominal_grasp_position=[0.0, 0.0, 0.0],
            gain=1.0,
            max_step=0.5,
            max_correction=0.2,
        )
        self.assertEqual(saturated["grasp_position"], [0.2, 0.2, 0.2])
        self.assertEqual(saturated["saturated_axes"], [True, True, True])

    def test_integral_visual_servo_supports_per_axis_gains(self):
        result = integral_visual_servo_grasp_target(
            object_position_estimate=[0.90, -0.10, 0.60],
            target_object_position=[1.00, 0.00, 0.40],
            commanded_grasp_position=[1.00, 0.00, 0.70],
            nominal_grasp_position=[1.00, 0.00, 0.70],
            gain=[0.10, 0.10, 0.50],
            max_step=[0.02, 0.02, 0.03],
            max_correction=0.40,
        )
        self.assertAlmostEqual(result["grasp_position"][0], 1.01)
        self.assertAlmostEqual(result["grasp_position"][1], 0.01)
        self.assertAlmostEqual(result["grasp_position"][2], 0.67)

    def test_cartesian_tracking_servo_compensates_steady_state_error(self):
        result = cartesian_tracking_servo_target(
            current_position=[0.95, 0.20, 0.59],
            desired_position=[0.98, 0.22, 0.54],
            commanded_position=[0.98, 0.22, 0.54],
            nominal_position=[0.98, 0.22, 0.54],
            gain=[0.25, 0.25, 0.40],
            max_step=[0.01, 0.01, 0.015],
            max_correction=0.12,
        )
        self.assertEqual(result["position"], [0.9875, 0.225, 0.525])
        self.assertAlmostEqual(result["xy_error"], 0.0360555, places=6)
        self.assertAlmostEqual(result["z_error"], 0.05)

    def test_cartesian_tracking_servo_limits_integral_correction(self):
        result = cartesian_tracking_servo_target(
            current_position=[0.0, 0.0, 0.0],
            desired_position=[1.0, 1.0, 1.0],
            commanded_position=[0.11, 0.11, 0.11],
            nominal_position=[0.0, 0.0, 0.0],
            gain=1.0,
            max_step=0.5,
            max_correction=0.12,
        )
        self.assertEqual(result["position"], [0.12, 0.12, 0.12])
        self.assertEqual(result["saturated_axes"], [True, True, True])

    def test_cartesian_tracking_servo_does_not_wind_up_during_lag(self):
        first = cartesian_tracking_servo_target(
            current_position=[0.95, 0.20, 0.59],
            desired_position=[0.98, 0.22, 0.54],
            commanded_position=[0.98, 0.22, 0.54],
            nominal_position=[0.98, 0.22, 0.54],
            gain=1.0,
            max_step=0.1,
            max_correction=0.12,
        )
        second = cartesian_tracking_servo_target(
            current_position=[0.95, 0.20, 0.59],
            desired_position=[0.98, 0.22, 0.54],
            commanded_position=first["position"],
            nominal_position=[0.98, 0.22, 0.54],
            gain=1.0,
            max_step=0.1,
            max_correction=0.12,
        )
        self.assertEqual(second["position"], first["position"])

    def test_placement_convergence_checks_object_and_grasp(self):
        self.assertTrue(
            placement_converged(
                [0.01, -0.01, 0.02],
                0.03,
                max_xy_error=0.03,
                max_z_error=0.04,
                max_grasp_tracking_error=0.04,
            )
        )
        self.assertFalse(
            placement_converged(
                [0.01, -0.01, 0.02],
                0.08,
                max_xy_error=0.03,
                max_z_error=0.04,
                max_grasp_tracking_error=0.04,
            )
        )

    def test_tactile_search_stays_active_after_contact_deflection(self):
        self.assertTrue(
            tactile_search_active(
                close_started=True,
                measured_finger_position=0.188,
                activation_position=0.20,
                hold_active=True,
                recovery_state="initial",
            )
        )
        self.assertFalse(
            tactile_search_active(
                close_started=True,
                measured_finger_position=0.188,
                activation_position=0.20,
                hold_active=False,
                recovery_state="initial",
            )
        )

    def test_bilateral_grasp_requires_force_and_closure(self):
        self.assertFalse(
            bilateral_grasp_ready(
                0.75,
                0.045,
                0.25,
                min_force=0.2,
                min_finger_position=0.23,
            )
        )
        self.assertTrue(
            bilateral_grasp_ready(
                2.7,
                3.8,
                0.238,
                min_force=0.2,
                min_finger_position=0.23,
            )
        )

    def test_contact_loss_streak_requires_continuous_monitored_loss(self):
        streak = 0
        for _ in range(4):
            streak = update_contact_loss_streak(
                streak,
                monitoring=True,
                contact_present=False,
            )
        self.assertEqual(streak, 4)
        self.assertEqual(
            update_contact_loss_streak(
                streak,
                monitoring=True,
                contact_present=True,
            ),
            0,
        )
        self.assertEqual(
            update_contact_loss_streak(
                streak,
                monitoring=False,
                contact_present=False,
            ),
            0,
        )

    def test_hover_guard_blocks_descent_until_xy_converges(self):
        guarded = apply_place_hover_guard(
            [0.92, 0.01, 0.55],
            [0.98, 0.00, 0.54],
            object_xy_error=0.08,
            grasp_xy_tracking_error=0.02,
            max_object_xy_error=0.03,
            max_grasp_xy_tracking_error=0.04,
            hover_clearance=0.10,
        )
        self.assertTrue(guarded["active"])
        self.assertAlmostEqual(guarded["grasp_position"][2], 0.64)

        converged = apply_place_hover_guard(
            [0.92, 0.01, 0.55],
            [0.98, 0.00, 0.54],
            object_xy_error=0.02,
            grasp_xy_tracking_error=0.02,
            max_object_xy_error=0.03,
            max_grasp_xy_tracking_error=0.04,
            hover_clearance=0.10,
        )
        self.assertFalse(converged["active"])
        self.assertAlmostEqual(converged["grasp_position"][2], 0.55)


if __name__ == "__main__":
    unittest.main()
