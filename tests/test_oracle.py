import numpy as np
import pytest

from farpoint.oracle import (
    OracleObservation,
    OraclePhase,
    OracleStateMachine,
    damped_least_squares,
    oriented_box_footprint_inside_target,
    oriented_box_xy_half_extents,
    quaternion_direction_error,
    quaternion_rotation_vector_error,
)


def test_oriented_box_footprint_uses_rotation_not_only_center():
    half_extents = oriented_box_xy_half_extents(
        [0.04, 0.04, 0.04],
        [0.0, 0.0, np.sin(np.deg2rad(45.0) / 2.0), np.cos(np.deg2rad(45.0) / 2.0)],
    )
    np.testing.assert_allclose(half_extents, [np.sqrt(2) * 0.02] * 2)

    assert oriented_box_footprint_inside_target(
        [0.20, 0.02, 0.062],
        [0.04, 0.04, 0.04],
        [0.0, 0.0, 0.0, 1.0],
        [0.20, 0.02, 0.037],
        [0.16, 0.14, 0.01],
        margin_m=0.005,
    )
    assert not oriented_box_footprint_inside_target(
        [0.1498970091, -0.0234679021, 0.062],
        [0.04, 0.04, 0.04],
        [-0.6008957028, -0.3727255464, 0.3727254272, 0.6008957624],
        [0.20, 0.02, 0.037],
        [0.16, 0.14, 0.01],
        margin_m=0.005,
    )


def test_damped_least_squares_solves_identity_task():
    jacobian = np.eye(3, 5)
    delta = damped_least_squares(jacobian, np.asarray([1.0, -2.0, 0.5]), damping=1e-3)
    np.testing.assert_allclose(delta[:3], [1.0, -2.0, 0.5], atol=1e-4)
    np.testing.assert_allclose(delta[3:], 0.0, atol=1e-6)


def test_damped_least_squares_validates_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        damped_least_squares(np.eye(3), np.ones(2))


def test_quaternion_rotation_vector_error_uses_shortest_xyzw_rotation():
    identity = [0.0, 0.0, 0.0, 1.0]
    quarter_turn_z = [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]

    np.testing.assert_allclose(
        quaternion_rotation_vector_error(identity, identity),
        [0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        quaternion_rotation_vector_error(quarter_turn_z, identity),
        [0.0, 0.0, np.pi / 2],
    )
    np.testing.assert_allclose(
        quaternion_rotation_vector_error(np.negative(quarter_turn_z), identity),
        [0.0, 0.0, np.pi / 2],
    )


def test_quaternion_direction_error_constrains_axis_but_leaves_roll_free():
    identity = [0.0, 0.0, 0.0, 1.0]
    quarter_turn_x = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    quarter_turn_z = [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]

    np.testing.assert_allclose(
        quaternion_direction_error(quarter_turn_x, identity),
        [1.0, 0.0, 0.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        quaternion_direction_error(quarter_turn_z, identity),
        [0.0, 0.0, 0.0],
        atol=1e-6,
    )


def test_oracle_requires_contact_lift_release_and_stability():
    machine = OracleStateMachine(required_stable_steps=2, required_contact_steps=2)
    for _phase in (OraclePhase.HOME, OraclePhase.PREGRASP, OraclePhase.DESCEND):
        machine.step(OracleObservation(reached_target=True))
    assert machine.phase is OraclePhase.CLOSE
    machine.step(OracleObservation(has_contact=True))
    assert machine.phase is OraclePhase.CLOSE
    machine.step(OracleObservation(has_contact=True))
    assert machine.phase is OraclePhase.VERIFY_CONTACT
    machine.step(OracleObservation(has_contact=True, cube_lifted=True))
    assert machine.phase is OraclePhase.LIFT
    for _phase in (OraclePhase.LIFT, OraclePhase.PREPLACE, OraclePhase.PLACE_DESCEND):
        machine.step(OracleObservation(reached_target=True))
    machine.step(OracleObservation(gripper_released=True))
    assert machine.phase is OraclePhase.SETTLE
    valid = OracleObservation(cube_in_target=True, gripper_released=True, cube_stable=True)
    machine.step(valid)
    machine.step(valid)
    assert machine.phase is OraclePhase.RETREAT
    machine.step(OracleObservation(reached_target=True))
    assert machine.phase is OraclePhase.SUCCEEDED


def test_oracle_reports_timeout_without_silent_progress():
    machine = OracleStateMachine(phase_timeout_steps=1)
    machine.step(OracleObservation())
    machine.step(OracleObservation())
    assert machine.phase is OraclePhase.FAILED
    assert machine.failure_reason == "phase_timeout:home"
