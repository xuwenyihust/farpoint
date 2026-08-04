"""Dependency-light SO-101 pick-and-place oracle primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class OraclePhase(str, Enum):
    HOME = "home"
    PREGRASP = "pregrasp"
    DESCEND = "descend"
    CLOSE = "close"
    VERIFY_CONTACT = "verify_contact"
    LIFT = "lift"
    PREPLACE = "preplace"
    PLACE_DESCEND = "place_descend"
    OPEN = "open"
    SETTLE = "settle"
    RETREAT = "retreat"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


PHASE_ORDER = (
    OraclePhase.HOME,
    OraclePhase.PREGRASP,
    OraclePhase.DESCEND,
    OraclePhase.CLOSE,
    OraclePhase.VERIFY_CONTACT,
    OraclePhase.LIFT,
    OraclePhase.PREPLACE,
    OraclePhase.PLACE_DESCEND,
    OraclePhase.OPEN,
    OraclePhase.SETTLE,
    OraclePhase.RETREAT,
    OraclePhase.SUCCEEDED,
)


@dataclass(frozen=True)
class OracleObservation:
    reached_target: bool = False
    has_contact: bool = False
    cube_lifted: bool = False
    cube_in_target: bool = False
    gripper_released: bool = False
    cube_stable: bool = False
    collision: bool = False
    cube_dropped: bool = False


@dataclass
class OracleStateMachine:
    phase: OraclePhase = OraclePhase.HOME
    phase_steps: int = 0
    stable_steps: int = 0
    failure_reason: str | None = None
    phase_timeout_steps: int = 180
    required_stable_steps: int = 15

    def _advance(self) -> None:
        index = PHASE_ORDER.index(self.phase)
        self.phase = PHASE_ORDER[index + 1]
        self.phase_steps = 0

    def step(self, observation: OracleObservation) -> OraclePhase:
        if self.phase in {OraclePhase.SUCCEEDED, OraclePhase.FAILED}:
            return self.phase
        self.phase_steps += 1
        if observation.collision:
            return self.fail("collision")
        if observation.cube_dropped and self.phase not in {
            OraclePhase.HOME,
            OraclePhase.PREGRASP,
            OraclePhase.DESCEND,
        }:
            return self.fail("cube_dropped")
        if self.phase_steps > self.phase_timeout_steps:
            return self.fail(f"phase_timeout:{self.phase.value}")

        if self.phase in {
            OraclePhase.HOME,
            OraclePhase.PREGRASP,
            OraclePhase.DESCEND,
            OraclePhase.LIFT,
            OraclePhase.PREPLACE,
            OraclePhase.PLACE_DESCEND,
            OraclePhase.RETREAT,
        } and observation.reached_target:
            self._advance()
        elif self.phase is OraclePhase.CLOSE and observation.has_contact:
            self._advance()
        elif self.phase is OraclePhase.VERIFY_CONTACT and observation.has_contact and observation.cube_lifted:
            self._advance()
        elif self.phase is OraclePhase.OPEN and observation.gripper_released:
            self._advance()
        elif self.phase is OraclePhase.SETTLE:
            valid = observation.cube_in_target and observation.gripper_released and observation.cube_stable
            self.stable_steps = self.stable_steps + 1 if valid else 0
            if self.stable_steps >= self.required_stable_steps:
                self._advance()
        return self.phase

    def fail(self, reason: str) -> OraclePhase:
        self.phase = OraclePhase.FAILED
        self.failure_reason = reason
        return self.phase


def damped_least_squares(
    jacobian: np.ndarray,
    error: np.ndarray,
    *,
    damping: float = 0.05,
    nullspace_error: np.ndarray | None = None,
    nullspace_gain: float = 0.1,
) -> np.ndarray:
    """Solve a damped Jacobian step with optional posture regularization."""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    if jacobian.ndim != 2 or error.shape != (jacobian.shape[0],):
        raise ValueError("jacobian and task error dimensions do not match")
    if damping <= 0:
        raise ValueError("damping must be positive")
    task_inverse = jacobian.T @ np.linalg.inv(
        jacobian @ jacobian.T + (damping**2) * np.eye(jacobian.shape[0])
    )
    delta = task_inverse @ error
    if nullspace_error is not None:
        posture = np.asarray(nullspace_error, dtype=np.float64)
        if posture.shape != (jacobian.shape[1],):
            raise ValueError("nullspace error dimension does not match joint count")
        projector = np.eye(jacobian.shape[1]) - task_inverse @ jacobian
        delta += nullspace_gain * projector @ posture
    return delta.astype(np.float32)
