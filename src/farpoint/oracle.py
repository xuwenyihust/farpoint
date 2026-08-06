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


def oriented_box_xy_half_extents(dimensions_m, orientation_xyzw) -> np.ndarray:
    """Project an oriented 3D box onto the world XY axes."""
    dimensions = np.asarray(dimensions_m, dtype=np.float64)
    quaternion = np.asarray(orientation_xyzw, dtype=np.float64)
    if dimensions.shape != (3,) or np.any(dimensions <= 0.0):
        raise ValueError("box dimensions must be three positive values")
    if quaternion.shape != (4,):
        raise ValueError("orientation quaternion must have shape (4,)")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("orientation quaternion must be non-zero")
    x, y, z, w = quaternion / norm
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return (np.abs(rotation[:2, :]) @ (0.5 * dimensions)).astype(np.float32)


def oriented_box_footprint_inside_target(
    object_position_m,
    object_dimensions_m,
    object_orientation_xyzw,
    target_position_m,
    target_dimensions_m,
    *,
    margin_m: float = 0.0,
) -> bool:
    """Return whether the complete projected box footprint fits on a target."""
    object_position = np.asarray(object_position_m, dtype=np.float64)
    target_position = np.asarray(target_position_m, dtype=np.float64)
    target_dimensions = np.asarray(target_dimensions_m, dtype=np.float64)
    if object_position.shape != (3,) or target_position.shape != (3,):
        raise ValueError("object and target positions must have shape (3,)")
    if target_dimensions.shape != (3,) or np.any(target_dimensions <= 0.0):
        raise ValueError("target dimensions must be three positive values")
    if margin_m < 0.0:
        raise ValueError("margin_m must be non-negative")
    object_half_extent = oriented_box_xy_half_extents(
        object_dimensions_m, object_orientation_xyzw
    )
    available_half_extent = 0.5 * target_dimensions[:2] - float(margin_m)
    return bool(
        np.all(
            np.abs(object_position[:2] - target_position[:2])
            + object_half_extent
            <= available_half_extent
        )
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
    contact_steps: int = 0
    stable_steps: int = 0
    failure_reason: str | None = None
    phase_timeout_steps: int = 180
    required_stable_steps: int = 15
    required_contact_steps: int = 10

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
        elif self.phase is OraclePhase.CLOSE:
            self.contact_steps = self.contact_steps + 1 if observation.has_contact else 0
            if self.contact_steps >= self.required_contact_steps:
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


def quaternion_rotation_vector_error(target_xyzw, current_xyzw) -> np.ndarray:
    """Return the shortest rotation vector from current to target (xyzw)."""
    target = np.asarray(target_xyzw, dtype=np.float64)
    current = np.asarray(current_xyzw, dtype=np.float64)
    if target.shape != (4,) or current.shape != (4,):
        raise ValueError("quaternions must have shape (4,)")
    target_norm = float(np.linalg.norm(target))
    current_norm = float(np.linalg.norm(current))
    if target_norm <= 1e-12 or current_norm <= 1e-12:
        raise ValueError("quaternions must be non-zero")
    target /= target_norm
    current /= current_norm
    if float(np.dot(target, current)) < 0.0:
        target = -target
    tx, ty, tz, tw = target
    cx, cy, cz, cw = current
    error = np.asarray(
        [
            tw * cw + tx * cx + ty * cy + tz * cz,
            -tw * cx + tx * cw - ty * cz + tz * cy,
            -tw * cy + tx * cz + ty * cw - tz * cx,
            -tw * cz - tx * cy + ty * cx + tz * cw,
        ],
        dtype=np.float64,
    )
    if error[0] < 0.0:
        error = -error
    vector_norm = float(np.linalg.norm(error[1:]))
    if vector_norm <= 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(vector_norm, float(error[0]))
    return (error[1:] * (angle / vector_norm)).astype(np.float32)


def quaternion_direction_error(
    target_xyzw,
    current_xyzw,
    local_axis=(0.0, 0.0, 1.0),
) -> np.ndarray:
    """Return angular error aligning one body-fixed axis, leaving roll free."""

    def rotate(quaternion, vector):
        quaternion = np.asarray(quaternion, dtype=np.float64)
        if quaternion.shape != (4,):
            raise ValueError("quaternions must have shape (4,)")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12:
            raise ValueError("quaternions must be non-zero")
        x, y, z, w = quaternion / norm
        rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        return rotation @ vector

    axis = np.asarray(local_axis, dtype=np.float64)
    if axis.shape != (3,):
        raise ValueError("local_axis must have shape (3,)")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        raise ValueError("local_axis must be non-zero")
    axis /= axis_norm
    current_direction = rotate(current_xyzw, axis)
    target_direction = rotate(target_xyzw, axis)
    return np.cross(current_direction, target_direction).astype(np.float32)
