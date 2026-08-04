import numpy as np
import pytest

from farpoint.oracle import (
    OracleObservation,
    OraclePhase,
    OracleStateMachine,
    damped_least_squares,
)


def test_damped_least_squares_solves_identity_task():
    jacobian = np.eye(3, 5)
    delta = damped_least_squares(jacobian, np.asarray([1.0, -2.0, 0.5]), damping=1e-3)
    np.testing.assert_allclose(delta[:3], [1.0, -2.0, 0.5], atol=1e-4)
    np.testing.assert_allclose(delta[3:], 0.0, atol=1e-6)


def test_damped_least_squares_validates_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        damped_least_squares(np.eye(3), np.ones(2))


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
