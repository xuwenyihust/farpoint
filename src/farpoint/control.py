import math


def so101_cube_requires_yaw_alignment(object_width_m, orientation_xyzw):
    """Select the orientation task only for a large axis-aligned cube."""
    width = float(object_width_m)
    quaternion = tuple(float(value) for value in orientation_xyzw)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("object_width_m must be finite and positive")
    if len(quaternion) != 4 or not all(math.isfinite(value) for value in quaternion):
        raise ValueError("orientation_xyzw must contain four finite values")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12:
        raise ValueError("orientation_xyzw must be non-zero")
    x, y, z, w = (value / norm for value in quaternion)
    yaw_degrees = math.degrees(
        math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    )
    canonical_yaw = ((yaw_degrees + 45.0) % 90.0) - 45.0
    return width >= 0.035 and abs(canonical_yaw) <= 5.0


def so101_approach_jaw_target(object_width_m):
    """Return the validated open-jaw target for the supported cube sizes."""
    width = float(object_width_m)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("object_width_m must be finite and positive")
    interpolation = _clamp((width - 0.03) / 0.01, 0.0, 1.0)
    # The balanced50 source demonstrations used the 1.2-rad exact-mesh
    # calibration for 40 mm cubes.  A later 1.7-rad recovery profile changed
    # the aperture center and produced position-dependent unilateral wedging
    # in the mirrored 0.03 kg gate.  Preserve the validated 0.9-rad 30 mm path
    # while returning 40 mm cubes to the source collection geometry.
    return 0.90 + 0.30 * interpolation


def so101_minimum_safe_descent_fraction(object_width_m):
    """Return the size-aware insertion fraction where cube contact is expected."""
    width = float(object_width_m)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("object_width_m must be finite and positive")
    interpolation = _clamp((width - 0.03) / 0.01, 0.0, 1.0)
    # The 40 mm gate at merged commit 5277dd4 produced repeatable first-corner
    # contact at 68--70% insertion with an unmoved cube and a fully open
    # 1.7-rad jaw.  Keeping the 30 mm threshold at its proven 75% while
    # lowering the large-cube endpoint to 60% distinguishes that intended
    # alignment contact from an actual pregrasp sweep.
    return 0.75 - 0.15 * interpolation


def so101_cube_contact_handoff(
    left_force_n,
    right_force_n,
    *,
    minimum_force_n=0.10,
):
    """Stop DESCEND on the first real, cube-filtered fingertip contact."""
    forces = (float(left_force_n), float(right_force_n))
    threshold = float(minimum_force_n)
    if any(not math.isfinite(force) or force < 0.0 for force in forces):
        raise ValueError("cube contact forces must be finite and non-negative")
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("minimum_force_n must be finite and positive")
    return max(forces) >= threshold


def so101_pre_capture_recenter_limit(
    object_width_m,
    *,
    maximum_correction_m=0.008,
    width_fraction=0.30,
):
    """Bound capture search while leaving room for the opposite fingertip.

    The former 4 mm bound saturated on every uncovered mass variation while
    the cube moved 6--10 mm under unilateral contact.  Permit at most 30% of
    the object width, capped at 8 mm, only before bilateral capture.
    """
    width = float(object_width_m)
    maximum = float(maximum_correction_m)
    fraction = float(width_fraction)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("object_width_m must be finite and positive")
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum_correction_m must be finite and positive")
    if not math.isfinite(fraction) or not 0.0 < fraction < 0.5:
        raise ValueError("width_fraction must be finite and between zero and 0.5")
    return min(maximum, fraction * width)


def so101_capture_contact_loss_grace_s(object_width_m):
    """Return a size-aware grace period for bounded capture recovery.

    A 30 mm cube can briefly lose both contacts while the rotary jaw closes
    through its final few milliradians. Give that smaller geometry three
    additional 30 Hz control ticks to recover, while preserving the validated
    0.20 s behavior for 40 mm cubes.
    """
    width = float(object_width_m)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("object_width_m must be finite and positive")
    interpolation = _clamp((width - 0.03) / 0.01, 0.0, 1.0)
    return 0.30 - 0.10 * interpolation


def so101_reset_support_is_stable(
    expected_position_m,
    measured_position_m,
    linear_velocity_mps,
    *,
    maximum_xy_error_m=0.002,
    maximum_z_error_m=0.001,
    maximum_speed_mps=0.05,
):
    """Check that a reset object settled on its support before oracle motion."""
    if not all(
        len(values) == 3
        for values in (
            expected_position_m,
            measured_position_m,
            linear_velocity_mps,
        )
    ):
        raise ValueError("reset support vectors must have three coordinates")
    expected = [float(value) for value in expected_position_m]
    measured = [float(value) for value in measured_position_m]
    velocity = [float(value) for value in linear_velocity_mps]
    limits = (
        float(maximum_xy_error_m),
        float(maximum_z_error_m),
        float(maximum_speed_mps),
    )
    if any(not math.isfinite(value) for value in (*expected, *measured, *velocity)):
        return False
    if any(not math.isfinite(value) or value <= 0.0 for value in limits):
        raise ValueError("reset support limits must be finite and positive")
    xy_error = math.hypot(measured[0] - expected[0], measured[1] - expected[1])
    z_error = abs(measured[2] - expected[2])
    speed = math.sqrt(sum(value * value for value in velocity))
    return xy_error <= limits[0] and z_error <= limits[1] and speed <= limits[2]


def settle_release_separation_target(
    release_hold_position,
    phase_steps,
    *,
    control_hz,
    separation_speed_mps=0.015,
    maximum_separation_m=0.020,
):
    """Ramp the open gripper upward so a released object cannot hang on one finger."""
    if len(release_hold_position) != 3:
        raise ValueError("release_hold_position must have three coordinates")
    steps = int(phase_steps)
    rate = float(control_hz)
    speed = float(separation_speed_mps)
    maximum = float(maximum_separation_m)
    if steps < 0:
        raise ValueError("phase_steps must be non-negative")
    if rate <= 0.0 or speed <= 0.0 or maximum <= 0.0:
        raise ValueError("release separation rates and limits must be positive")
    distance = min(maximum, speed * (steps + 1) / rate)
    return [
        float(release_hold_position[0]),
        float(release_hold_position[1]),
        float(release_hold_position[2]) + distance,
    ]


def unsafe_so101_approach_contact(
    phase,
    has_contact,
    descent_fraction=None,
    *,
    minimum_safe_descent_fraction=0.75,
):
    """Reject contact during routing or before the calibrated insertion window."""
    if not bool(has_contact):
        return False
    phase_name = str(getattr(phase, "value", phase))
    if phase_name == "pregrasp":
        return True
    if phase_name != "descend" or descent_fraction is None:
        return False
    fraction = float(descent_fraction)
    threshold = float(minimum_safe_descent_fraction)
    if not math.isfinite(fraction) or not math.isfinite(threshold):
        raise ValueError("descent fractions must be finite")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum_safe_descent_fraction must be in [0, 1]")
    return fraction < threshold


def collision_safe_pregrasp_waypoints(
    home_position,
    final_position,
    *,
    clearance_z,
):
    """Route above the workspace before moving over the grasp target."""
    if len(home_position) != 3 or len(final_position) != 3:
        raise ValueError("pregrasp positions must have three coordinates")
    clearance = float(clearance_z)
    minimum_clearance = max(float(home_position[2]), float(final_position[2]))
    if clearance <= minimum_clearance:
        raise ValueError("clearance_z must be above home and final positions")
    return [
        [float(home_position[0]), float(home_position[1]), clearance],
        [float(final_position[0]), float(final_position[1]), clearance],
    ]


def _clamp(value, lower, upper):
    return max(float(lower), min(float(upper), float(value)))


def _axis_values(value, name):
    if isinstance(value, (int, float)):
        return [float(value)] * 3
    if len(value) != 3:
        raise ValueError(f"{name} must be a scalar or three values")
    return [float(item) for item in value]


def bounded_position_target(desired_position, measured_position, max_error):
    error_limit = max(0.0, float(max_error))
    measured = float(measured_position)
    return _clamp(
        desired_position,
        measured - error_limit,
        measured + error_limit,
    )


def tactile_contact_hold_target(
    measured_position,
    preload,
    max_position,
):
    return min(
        float(max_position),
        float(measured_position) + max(0.0, float(preload)),
    )


def gripper_aperture_alignment(left_bounds, right_bounds, object_position):
    if not all(
        isinstance(bounds, dict)
        and isinstance(bounds.get("center"), (list, tuple))
        and len(bounds["center"]) == 3
        for bounds in (left_bounds, right_bounds)
    ):
        raise ValueError("both finger bounds must provide three-value centers")
    if len(object_position) != 3:
        raise ValueError("object position must have three coordinates")

    aperture_center = [
        (
            float(left_bounds["center"][axis])
            + float(right_bounds["center"][axis])
        )
        * 0.5
        for axis in range(3)
    ]
    finger_center_delta = [
        float(right_bounds["center"][axis])
        - float(left_bounds["center"][axis])
        for axis in range(3)
    ]
    position_error = [
        float(object_position[axis]) - aperture_center[axis]
        for axis in range(3)
    ]
    finger_axis_yaw_degrees = math.degrees(
        math.atan2(finger_center_delta[1], finger_center_delta[0])
    )
    return {
        "aperture_center": aperture_center,
        "finger_center_delta": finger_center_delta,
        "finger_axis_yaw_degrees": finger_axis_yaw_degrees,
        "finger_z_skew": abs(finger_center_delta[2]),
        "position_error": position_error,
        "xy_error": math.hypot(position_error[0], position_error[1]),
        "z_error": abs(position_error[2]),
    }


def undirected_axis_angle_error_degrees(actual, target):
    difference = (float(actual) - float(target) + 90.0) % 180.0 - 90.0
    return abs(difference)


def relative_object_grasp_servo_target(
    object_position,
    desired_object_minus_grasp,
    commanded_grasp_position,
    nominal_grasp_position,
    *,
    max_step=0.0005,
    max_correction=(0.015, 0.015, 0.020),
):
    """Track a measured object while preserving a calibrated grasp offset."""
    values = (
        object_position,
        desired_object_minus_grasp,
        commanded_grasp_position,
        nominal_grasp_position,
        max_correction,
    )
    if not all(len(value) == 3 for value in values):
        raise ValueError("relative grasp servo values must have three coordinates")
    step_limit = float(max_step)
    if step_limit <= 0.0:
        raise ValueError("max_step must be positive")
    corrections = [float(value) for value in max_correction]
    if any(value < 0.0 for value in corrections):
        raise ValueError("max_correction values must be non-negative")

    desired = [
        float(object_position[axis])
        - float(desired_object_minus_grasp[axis])
        for axis in range(3)
    ]
    error = [
        desired[axis] - float(commanded_grasp_position[axis])
        for axis in range(3)
    ]
    error_norm = math.sqrt(sum(value * value for value in error))
    scale = min(1.0, step_limit / error_norm) if error_norm > 0.0 else 0.0
    target = []
    for axis in range(3):
        stepped = float(commanded_grasp_position[axis]) + scale * error[axis]
        nominal = float(nominal_grasp_position[axis])
        target.append(
            _clamp(
                stepped,
                nominal - corrections[axis],
                nominal + corrections[axis],
            )
        )
    return {"position": target, "error": error, "desired_position": desired}


def unilateral_contact_recenter_target(
    commanded_position,
    nominal_position,
    left_bounds,
    right_bounds,
    left_force,
    right_force,
    *,
    min_force=0.2,
    step=0.00025,
    max_correction=0.02,
    move_toward_contact=True,
):
    if len(commanded_position) != 3 or len(nominal_position) != 3:
        raise ValueError("Cartesian positions must have three coordinates")
    alignment = gripper_aperture_alignment(
        left_bounds,
        right_bounds,
        [0.0, 0.0, 0.0],
    )
    left_active = float(left_force) >= float(min_force)
    right_active = float(right_force) >= float(min_force)
    if left_active == right_active:
        return {
            "position": [float(value) for value in commanded_position],
            "active": False,
            "contact_side": None,
            "axis_xy": [0.0, 0.0],
        }

    finger_delta = alignment["finger_center_delta"]
    axis_norm = math.hypot(finger_delta[0], finger_delta[1])
    if axis_norm <= 1e-9:
        return {
            "position": [float(value) for value in commanded_position],
            "active": False,
            "contact_side": None,
            "axis_xy": [0.0, 0.0],
        }
    axis_xy = [
        finger_delta[0] / axis_norm,
        finger_delta[1] / axis_norm,
    ]
    direction = -1.0 if left_active else 1.0
    if not bool(move_toward_contact):
        direction *= -1.0
    proposed_xy = [
        float(commanded_position[axis])
        + direction * float(step) * axis_xy[axis]
        for axis in range(2)
    ]
    correction_xy = [
        proposed_xy[axis] - float(nominal_position[axis])
        for axis in range(2)
    ]
    correction_norm = math.hypot(*correction_xy)
    if correction_norm > float(max_correction):
        scale = float(max_correction) / correction_norm
        proposed_xy = [
            float(nominal_position[axis]) + correction_xy[axis] * scale
            for axis in range(2)
        ]
    return {
        "position": proposed_xy + [float(commanded_position[2])],
        "active": True,
        "contact_side": "left" if left_active else "right",
        "axis_xy": axis_xy,
    }


def track_observed_pick_target(
    object_position_estimate,
    object_grasp_offset,
    commanded_position,
    nominal_position,
    *,
    max_step=0.01,
):
    if not all(
        len(position) == 3
        for position in (
            object_position_estimate,
            object_grasp_offset,
            commanded_position,
            nominal_position,
        )
    ):
        raise ValueError("all pick tracking positions must have three coordinates")

    max_steps = _axis_values(max_step, "max_step")
    observed_nominal = [
        float(object_position_estimate[axis])
        - float(object_grasp_offset[axis])
        for axis in range(3)
    ]
    tactile_correction = [
        float(commanded_position[axis]) - float(nominal_position[axis])
        for axis in range(3)
    ]
    updated_nominal = [
        float(nominal_position[axis])
        + _clamp(
            observed_nominal[axis] - float(nominal_position[axis]),
            -max_steps[axis],
            max_steps[axis],
        )
        for axis in range(3)
    ]
    return {
        "position": [
            updated_nominal[axis] + tactile_correction[axis]
            for axis in range(3)
        ],
        "nominal_position": updated_nominal,
        "observed_nominal_position": observed_nominal,
        "tactile_correction": tactile_correction,
    }


def calibrated_recovery_grasp_target(
    object_position_estimate,
    object_grasp_offset,
    *,
    aperture_tool_offset=None,
    aperture_bias_xy=(0.0, 0.0),
):
    if len(object_position_estimate) != 3 or len(object_grasp_offset) != 3:
        raise ValueError("recovery positions must have three coordinates")
    if len(aperture_bias_xy) != 2:
        raise ValueError("aperture_bias_xy must contain two values")
    if aperture_tool_offset is not None and len(aperture_tool_offset) != 3:
        raise ValueError("aperture_tool_offset must have three coordinates")

    object_target = [float(value) for value in object_position_estimate]
    object_target[0] += float(aperture_bias_xy[0])
    object_target[1] += float(aperture_bias_xy[1])
    tool_offset = (
        aperture_tool_offset
        if aperture_tool_offset is not None
        else object_grasp_offset
    )
    return [
        object_target[axis] - float(tool_offset[axis])
        for axis in range(3)
    ]


def rmpflow_world_target(cartesian_world_target):
    if len(cartesian_world_target) != 3:
        raise ValueError("RMPflow target must have three world coordinates")
    return [float(value) for value in cartesian_world_target]


def visual_servo_grasp_target(
    current_grasp_position,
    object_position_estimate,
    target_object_position,
    commanded_grasp_position,
    nominal_grasp_position,
    *,
    gain=1.0,
    max_step=0.03,
    max_correction=0.18,
):
    if not all(
        len(position) == 3
        for position in (
            current_grasp_position,
            object_position_estimate,
            target_object_position,
            commanded_grasp_position,
            nominal_grasp_position,
        )
    ):
        raise ValueError("all visual-servo positions must have three coordinates")

    object_error = [
        float(target_object_position[axis])
        - float(object_position_estimate[axis])
        for axis in range(3)
    ]
    desired_grasp_position = [
        float(current_grasp_position[axis]) + float(gain) * object_error[axis]
        for axis in range(3)
    ]
    updated_grasp_position = []
    for axis in range(3):
        stepped = float(commanded_grasp_position[axis]) + _clamp(
            desired_grasp_position[axis] - float(commanded_grasp_position[axis]),
            -float(max_step),
            float(max_step),
        )
        updated_grasp_position.append(
            _clamp(
                stepped,
                float(nominal_grasp_position[axis]) - float(max_correction),
                float(nominal_grasp_position[axis]) + float(max_correction),
            )
        )
    return {
        "grasp_position": updated_grasp_position,
        "object_error": object_error,
        "xy_error": math.hypot(object_error[0], object_error[1]),
        "z_error": abs(object_error[2]),
    }


def integral_visual_servo_grasp_target(
    object_position_estimate,
    target_object_position,
    commanded_grasp_position,
    nominal_grasp_position,
    *,
    gain=0.35,
    max_step=0.03,
    max_correction=0.18,
):
    if not all(
        len(position) == 3
        for position in (
            object_position_estimate,
            target_object_position,
            commanded_grasp_position,
            nominal_grasp_position,
        )
    ):
        raise ValueError("all visual-servo positions must have three coordinates")

    object_error = [
        float(target_object_position[axis])
        - float(object_position_estimate[axis])
        for axis in range(3)
    ]
    gains = _axis_values(gain, "gain")
    max_steps = _axis_values(max_step, "max_step")
    updated_grasp_position = []
    saturated_axes = []
    for axis in range(3):
        delta = _clamp(
            gains[axis] * object_error[axis],
            -max_steps[axis],
            max_steps[axis],
        )
        lower = (
            float(nominal_grasp_position[axis])
            - float(max_correction)
        )
        upper = (
            float(nominal_grasp_position[axis])
            + float(max_correction)
        )
        requested = float(commanded_grasp_position[axis]) + delta
        updated = _clamp(requested, lower, upper)
        updated_grasp_position.append(updated)
        saturated_axes.append(not math.isclose(requested, updated, abs_tol=1e-9))

    return {
        "grasp_position": updated_grasp_position,
        "object_error": object_error,
        "xy_error": math.hypot(object_error[0], object_error[1]),
        "z_error": abs(object_error[2]),
        "saturated_axes": saturated_axes,
    }


def cartesian_tracking_servo_target(
    current_position,
    desired_position,
    commanded_position,
    nominal_position,
    *,
    gain=0.25,
    max_step=0.01,
    max_correction=0.12,
):
    if not all(
        len(position) == 3
        for position in (
            current_position,
            desired_position,
            commanded_position,
            nominal_position,
        )
    ):
        raise ValueError("all Cartesian tracking positions must have three coordinates")

    position_error = [
        float(desired_position[axis]) - float(current_position[axis])
        for axis in range(3)
    ]
    gains = _axis_values(gain, "gain")
    max_steps = _axis_values(max_step, "max_step")
    updated_position = []
    saturated_axes = []
    for axis in range(3):
        desired_command = (
            float(nominal_position[axis])
            + gains[axis] * position_error[axis]
        )
        stepped = float(commanded_position[axis]) + _clamp(
            desired_command - float(commanded_position[axis]),
            -max_steps[axis],
            max_steps[axis],
        )
        lower = float(nominal_position[axis]) - float(max_correction)
        upper = float(nominal_position[axis]) + float(max_correction)
        updated = _clamp(stepped, lower, upper)
        updated_position.append(updated)
        saturated_axes.append(not math.isclose(stepped, updated, abs_tol=1e-9))

    return {
        "position": updated_position,
        "position_error": position_error,
        "xy_error": math.hypot(position_error[0], position_error[1]),
        "z_error": abs(position_error[2]),
        "saturated_axes": saturated_axes,
    }


def placement_converged(
    object_error,
    grasp_tracking_error,
    *,
    max_xy_error,
    max_z_error,
    max_grasp_tracking_error,
):
    if len(object_error) != 3:
        raise ValueError("object_error must have three coordinates")
    return (
        math.hypot(float(object_error[0]), float(object_error[1]))
        <= float(max_xy_error)
        and abs(float(object_error[2])) <= float(max_z_error)
        and float(grasp_tracking_error) <= float(max_grasp_tracking_error)
    )


def tactile_search_active(
    *,
    close_started,
    measured_finger_position,
    activation_position,
    hold_active,
    recovery_state,
):
    return (
        bool(close_started)
        and (
            bool(hold_active)
            or float(measured_finger_position)
            >= float(activation_position)
        )
        and recovery_state in {"initial", "close"}
    )


def bilateral_grasp_ready(
    left_force,
    right_force,
    measured_finger_position,
    *,
    min_force,
    min_finger_position,
):
    return (
        float(left_force) >= float(min_force)
        and float(right_force) >= float(min_force)
        and float(measured_finger_position) >= float(min_finger_position)
    )


def rate_limit_revolute_joint_targets(
    current_positions,
    desired_positions,
    joint_indices,
    *,
    max_step,
):
    current = [float(value) for value in current_positions]
    desired = [float(value) for value in desired_positions]
    if len(current) != len(desired):
        raise ValueError("current_positions and desired_positions must match")
    limited = list(desired)
    step_limit = max(0.0, float(max_step))
    for raw_index in joint_indices:
        index = int(raw_index)
        if index < 0 or index >= len(current):
            raise ValueError("joint index is out of range")
        delta = math.atan2(
            math.sin(desired[index] - current[index]),
            math.cos(desired[index] - current[index]),
        )
        limited[index] = current[index] + max(
            -step_limit,
            min(step_limit, delta),
        )
    return limited


def force_controlled_gripper_target(
    target_position,
    measured_position,
    left_force,
    right_force,
    *,
    min_force,
    max_force,
    close_step,
    backoff_step,
    max_position,
    close_on_unilateral=True,
    max_preload_error=None,
    preload_reference_position=None,
):
    target = float(target_position)
    measured = float(measured_position)
    lower_force = min(float(left_force), float(right_force))
    upper_force = max(float(left_force), float(right_force))
    preload_reference = (
        measured
        if preload_reference_position is None
        else float(preload_reference_position)
    )
    close_ceiling = (
        float(max_position)
        if max_preload_error is None
        else min(
            float(max_position),
            preload_reference
            + max(0.0, float(max_preload_error)),
        )
    )
    backoff_ceiling = (
        measured if max_preload_error is None else close_ceiling
    )
    if upper_force > float(max_force):
        return {
            "position": max(
                0.0,
                min(target, backoff_ceiling) - float(backoff_step),
            ),
            "action": "backoff",
        }
    if (
        not bool(close_on_unilateral)
        and lower_force < float(min_force) <= upper_force
    ):
        return {
            "position": min(target, close_ceiling),
            "action": "unilateral_hold",
        }
    if lower_force < float(min_force):
        return {
            "position": min(
                close_ceiling,
                target + float(close_step),
            ),
            "action": "close",
        }
    return {"position": target, "action": "hold"}


def force_controlled_rotary_jaw_target(
    target_position,
    measured_position,
    left_force,
    right_force,
    *,
    open_position,
    closed_position,
    min_force,
    max_force,
    close_step,
    backoff_step,
    max_preload_error=None,
    preload_reference_position=None,
):
    """Adapt force control to a rotary jaw whose position decreases on close."""
    open_value = float(open_position)
    closed_value = float(closed_position)
    if closed_value >= open_value:
        raise ValueError("closed_position must be below open_position")

    reference = (
        None
        if preload_reference_position is None
        else open_value - float(preload_reference_position)
    )
    update = force_controlled_gripper_target(
        open_value - float(target_position),
        open_value - float(measured_position),
        left_force,
        right_force,
        min_force=min_force,
        max_force=max_force,
        close_step=close_step,
        backoff_step=backoff_step,
        max_position=open_value - closed_value,
        max_preload_error=max_preload_error,
        preload_reference_position=reference,
    )
    return {
        "position": open_value - float(update["position"]),
        "action": update["action"],
    }


def advance_so101_slow_close_target(
    previous_command_target,
    measured_position,
    left_force,
    right_force,
    *,
    open_position,
    closed_position,
    min_force=2.0,
    max_force=20.0,
    close_step=0.001,
    backoff_step=0.002,
):
    """Advance the persistent SO-101 jaw command during slow close.

    ``previous_command_target`` is deliberately separate from the measured
    joint position.  Rebasing the target to the measurement every tick leaves
    only a 1 mrad servo error, which was too small to overcome the simulated
    jaw load and made every workspace-recovery trial time out before bilateral
    contact.
    """
    open_value = float(open_position)
    closed_value = float(closed_position)
    previous_target = _clamp(previous_command_target, closed_value, open_value)
    update = force_controlled_rotary_jaw_target(
        previous_target,
        measured_position,
        left_force,
        right_force,
        open_position=open_value,
        closed_position=closed_value,
        min_force=min_force,
        max_force=max_force,
        close_step=close_step,
        backoff_step=backoff_step,
    )
    return {
        "position": _clamp(update["position"], closed_value, open_value),
        "action": update["action"],
    }


def filtered_contact_force(
    contacts,
    required_body_path,
    *,
    physics_dt,
):
    dt = float(physics_dt)
    if dt <= 0.0:
        raise ValueError("physics_dt must be positive")
    force = 0.0
    matching_contacts = 0
    for contact in contacts or []:
        if required_body_path not in {
            str(contact.get("body0", "")),
            str(contact.get("body1", "")),
        }:
            continue
        impulse = contact.get("impulse", [0.0, 0.0, 0.0])
        if len(impulse) != 3:
            raise ValueError("contact impulse must have three coordinates")
        force += math.sqrt(
            sum(float(component) ** 2 for component in impulse)
        ) / dt
        matching_contacts += 1
    return {
        "force": force,
        "matching_contacts": matching_contacts,
    }


def contact_pair_force_summary(contacts, *, physics_dt):
    dt = float(physics_dt)
    if dt <= 0.0:
        raise ValueError("physics_dt must be positive")
    pairs = {}
    for contact in contacts or []:
        body0 = str(contact.get("body0", ""))
        body1 = str(contact.get("body1", ""))
        impulse = contact.get("impulse", [0.0, 0.0, 0.0])
        if len(impulse) != 3:
            raise ValueError("contact impulse must have three coordinates")
        key = tuple(sorted((body0, body1)))
        direction = 1.0 if body0 == key[0] else -1.0
        directed_impulse = [
            direction * float(component) for component in impulse
        ]
        force = math.sqrt(
            sum(float(component) ** 2 for component in impulse)
        ) / dt
        record = pairs.setdefault(
            key,
            {
                "body0": key[0],
                "body1": key[1],
                "force": 0.0,
                "contact_count": 0,
            },
        )
        record["force"] += force
        record["contact_count"] += 1
        if "position" in contact and len(contact["position"]) == 3:
            position = [float(value) for value in contact["position"]]
            position_sum = record.setdefault(
                "_position_weighted_sum",
                [0.0, 0.0, 0.0],
            )
            position_weight = max(force, 1.0e-12)
            for axis in range(3):
                position_sum[axis] += position[axis] * position_weight
            record["_position_weight"] = (
                record.get("_position_weight", 0.0) + position_weight
            )
        if "normal" in contact and len(contact["normal"]) == 3:
            normal_sum = record.setdefault(
                "_normal_weighted_sum",
                [0.0, 0.0, 0.0],
            )
            normal_weight = max(force, 1.0e-12)
            for axis in range(3):
                normal_sum[axis] += (
                    direction
                    * float(contact["normal"][axis])
                    * normal_weight
                )
            record["_normal_weight"] = (
                record.get("_normal_weight", 0.0) + normal_weight
            )
        impulse_sum = record.setdefault(
            "impulse_on_body0",
            [0.0, 0.0, 0.0],
        )
        for axis in range(3):
            impulse_sum[axis] += directed_impulse[axis]
    for record in pairs.values():
        position_weight = record.pop("_position_weight", 0.0)
        position_sum = record.pop("_position_weighted_sum", None)
        if position_sum is not None and position_weight > 0.0:
            record["position"] = [
                value / position_weight for value in position_sum
            ]
        normal_weight = record.pop("_normal_weight", 0.0)
        normal_sum = record.pop("_normal_weighted_sum", None)
        if normal_sum is not None and normal_weight > 0.0:
            normal = [value / normal_weight for value in normal_sum]
            magnitude = math.sqrt(sum(value**2 for value in normal))
            if magnitude > 0.0:
                normal = [value / magnitude for value in normal]
            record["normal_on_body0"] = normal
    return sorted(
        pairs.values(),
        key=lambda record: record["force"],
        reverse=True,
    )


def merge_contact_group_samples(samples):
    samples = list(samples or [])
    if not samples:
        return 0.0, False, [], []
    link_count = max((len(sample[2]) for sample in samples), default=0)
    link_forces = [0.0] * link_count
    pair_forces = {}
    for _, _, sample_link_forces, sample_pairs in samples:
        for index, link_force in enumerate(sample_link_forces):
            link_forces[index] = max(
                link_forces[index],
                float(link_force),
            )
        for pair in sample_pairs:
            key = (pair["body0"], pair["body1"])
            existing = pair_forces.get(key)
            if existing is None or float(pair["force"]) > float(
                existing["force"]
            ):
                pair_forces[key] = dict(pair)
    return (
        max(float(sample[0]) for sample in samples),
        any(bool(sample[1]) for sample in samples),
        link_forces,
        sorted(
            pair_forces.values(),
            key=lambda record: record["force"],
            reverse=True,
        ),
    )


def temporal_contact_confirmed(
    contact_event_frames,
    current_frame,
    *,
    required_events,
    window_frames,
):
    current = int(current_frame)
    window = max(1, int(window_frames))
    required = max(1, int(required_events))
    cutoff = current - window + 1
    recent_events = sum(
        cutoff <= int(event_frame) <= current
        for event_frame in contact_event_frames
    )
    return recent_events >= required


def grasp_validation_decision(
    elapsed_frames,
    support_frames,
    *,
    min_frames,
    max_frames,
    min_support_frames,
    confirmed_raw_contact,
    terminal_stable_frames=1,
    required_terminal_stable_frames=1,
):
    elapsed = max(0, int(elapsed_frames))
    support = max(0, int(support_frames))
    minimum = max(0, int(min_frames))
    maximum = max(minimum, int(max_frames))
    required_support = max(1, int(min_support_frames))
    terminal_stability = max(0, int(terminal_stable_frames))
    required_terminal_stability = max(
        1,
        int(required_terminal_stable_frames),
    )
    if (
        elapsed >= minimum
        and support >= required_support
        and bool(confirmed_raw_contact)
        and terminal_stability >= required_terminal_stability
    ):
        return "validated"
    if elapsed >= maximum:
        return "failed"
    return "pending"


def update_contact_loss_streak(
    current_streak,
    *,
    monitoring,
    contact_present,
):
    if not monitoring or contact_present:
        return 0
    return max(0, int(current_streak)) + 1


def grasp_proof_evidence(
    *,
    object_lift,
    minimum_object_lift,
    bilateral_contact_frames,
    required_contact_frames,
    support_present,
    maximum_rigidity_error,
    maximum_allowed_rigidity_error,
):
    checks = {
        "lift": float(object_lift) >= float(minimum_object_lift),
        "contact_history": int(bilateral_contact_frames)
        >= int(required_contact_frames),
        "support": bool(support_present),
        "rigidity": float(maximum_rigidity_error)
        <= float(maximum_allowed_rigidity_error),
    }
    return {**checks, "passed": all(checks.values())}


def transport_grasp_support(
    direct_contact,
    object_position,
    grasp_position,
    reference_offset,
    *,
    max_rigidity_error,
):
    if direct_contact:
        return {
            "present": True,
            "source": "direct_contact",
            "rigidity_error": 0.0,
        }
    vectors = (object_position, grasp_position, reference_offset)
    if not all(
        isinstance(vector, (list, tuple)) and len(vector) == 3
        for vector in vectors
    ):
        return {
            "present": False,
            "source": None,
            "rigidity_error": None,
        }
    current_offset = [
        float(object_position[axis]) - float(grasp_position[axis])
        for axis in range(3)
    ]
    rigidity_error = math.sqrt(
        sum(
            (
                current_offset[axis] - float(reference_offset[axis])
            )
            ** 2
            for axis in range(3)
        )
    )
    inferred = rigidity_error <= float(max_rigidity_error)
    return {
        "present": inferred,
        "source": "rigidity_inferred" if inferred else None,
        "rigidity_error": rigidity_error,
    }


def simulation_stop_reason(
    frame,
    *,
    nominal_frame_count,
    extension_frames,
    release_complete_frame,
    required_settle_frames,
    grasp_validated,
):
    """Return why a bounded episode should stop, or ``None`` to continue."""
    simulated_frames = int(frame) + 1
    nominal_frames = max(1, int(nominal_frame_count))
    maximum_frames = nominal_frames + max(0, int(extension_frames))
    if release_complete_frame is not None:
        settled_frames = int(frame) - int(release_complete_frame) + 1
        if settled_frames >= max(1, int(required_settle_frames)):
            return "release_settled"
    if simulated_frames < nominal_frames:
        return None
    if not grasp_validated:
        return "nominal_budget_exhausted"
    if simulated_frames >= maximum_frames:
        return "extension_budget_exhausted"
    return None


def apply_place_hover_guard(
    grasp_position,
    nominal_grasp_position,
    *,
    object_xy_error,
    grasp_xy_tracking_error,
    max_object_xy_error,
    max_grasp_xy_tracking_error,
    hover_clearance,
):
    guarded = [float(value) for value in grasp_position]
    active = (
        float(object_xy_error) > float(max_object_xy_error)
        or float(grasp_xy_tracking_error)
        > float(max_grasp_xy_tracking_error)
    )
    if active:
        guarded[2] = max(
            guarded[2],
            float(nominal_grasp_position[2])
            + float(hover_clearance),
        )
    return {
        "grasp_position": guarded,
        "active": active,
    }
