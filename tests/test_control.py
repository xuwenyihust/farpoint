import unittest

import pytest

from farpoint.control import (
    apply_place_hover_guard,
    bilateral_grasp_ready,
    bounded_position_target,
    cartesian_tracking_servo_target,
    contact_pair_force_summary,
    filtered_contact_force,
    force_controlled_gripper_target,
    grasp_validation_decision,
    gripper_aperture_alignment,
    integral_visual_servo_grasp_target,
    merge_contact_group_samples,
    placement_converged,
    rate_limit_revolute_joint_targets,
    rmpflow_world_target,
    simulation_stop_reason,
    tactile_contact_hold_target,
    tactile_search_active,
    temporal_contact_confirmed,
    track_observed_pick_target,
    transport_grasp_support,
    update_contact_loss_streak,
    unilateral_contact_recenter_target,
    undirected_axis_angle_error_degrees,
    visual_servo_grasp_target,
)


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
