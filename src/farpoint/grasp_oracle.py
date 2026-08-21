"""Contact-aware, rate-independent grasp primitives for the SO-101 oracle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from farpoint.control import so101_capture_admission_retention_fraction


class GraspPhase(str, Enum):
    APPROACH = "approach"
    FIRST_CONTACT = "first_contact"
    CONTACT_ALIGNMENT = "contact_alignment"
    SLOW_CLOSE = "slow_close"
    BILATERAL_SETTLE = "bilateral_settle"
    STATIC_HOLD = "static_hold"
    PROOF_LIFT = "proof_lift"
    VALIDATED = "validated"
    FAILED = "failed"


def grasp_phase_allows_unilateral_recenter(phase: GraspPhase) -> bool:
    """Return whether a one-finger load should still steer the aperture."""
    return phase in {
        GraspPhase.FIRST_CONTACT,
        GraspPhase.CONTACT_ALIGNMENT,
        GraspPhase.SLOW_CLOSE,
        GraspPhase.BILATERAL_SETTLE,
        GraspPhase.STATIC_HOLD,
    }


def capture_aperture_laterally_aligned(
    object_in_gripper_m,
    aperture_reference_m,
    *,
    maximum_lateral_error_m: float = 0.025,
) -> bool:
    """Check enclosure alignment without treating finger depth as lateral error."""
    object_local = np.asarray(object_in_gripper_m, dtype=np.float64)
    reference = np.asarray(aperture_reference_m, dtype=np.float64)
    maximum_error = float(maximum_lateral_error_m)
    if object_local.shape != (3,) or reference.shape != (3,):
        raise ValueError("aperture alignment vectors must have shape (3,)")
    if not np.all(np.isfinite(object_local)) or not np.all(np.isfinite(reference)):
        raise ValueError("aperture alignment vectors must be finite")
    if not np.isfinite(maximum_error) or maximum_error <= 0.0:
        raise ValueError("maximum_lateral_error_m must be finite and positive")
    return float(np.linalg.norm(object_local[:2] - reference[:2])) <= maximum_error


def advance_proof_lift_command(
    commanded_joints,
    measured_joints,
    current_height_m: float,
    *,
    just_armed: bool,
    increment_m: float = 0.0000625,
    maximum_height_m: float = 0.010,
) -> tuple[np.ndarray, float]:
    """Advance proof lift without erasing accumulated gravity compensation."""
    command_base = cartesian_motion_command_base(
        commanded_joints, measured_joints, entering_motion=just_armed
    )
    if not np.isfinite(current_height_m) or current_height_m < 0.0:
        raise ValueError("current proof-lift height must be finite and non-negative")
    if increment_m <= 0.0 or maximum_height_m <= 0.0:
        raise ValueError("proof-lift increment and maximum must be positive")
    if current_height_m > maximum_height_m:
        raise ValueError("current proof-lift height exceeds the maximum")
    next_height = min(maximum_height_m, current_height_m + increment_m)
    return command_base, float(next_height)


def cartesian_motion_command_base(
    commanded_joints, measured_joints, *, entering_motion: bool
) -> np.ndarray:
    """Rebase once on entry, then retain the integrated Cartesian command."""
    commanded = np.asarray(commanded_joints, dtype=np.float32)
    measured = np.asarray(measured_joints, dtype=np.float32)
    if commanded.shape != measured.shape or commanded.ndim != 1:
        raise ValueError("commanded and measured joints must be matching vectors")
    if not np.all(np.isfinite(commanded)) or not np.all(np.isfinite(measured)):
        raise ValueError("joint vectors must contain only finite values")
    return (measured if entering_motion else commanded).copy()


@dataclass(frozen=True)
class ControlRecordingSchedule:
    """Separate high-rate control ticks from policy recording frames."""

    control_hz: int = 120
    recording_hz: int = 30

    def __post_init__(self) -> None:
        if self.control_hz <= 0 or self.recording_hz <= 0:
            raise ValueError("control and recording rates must be positive")
        if self.control_hz % self.recording_hz:
            raise ValueError("control_hz must be an integer multiple of recording_hz")

    @property
    def recording_stride(self) -> int:
        return self.control_hz // self.recording_hz

    def should_record(self, control_step: int) -> bool:
        if control_step < 0:
            raise ValueError("control_step must be non-negative")
        return control_step % self.recording_stride == 0

    def frame_index(self, control_step: int) -> int:
        if not self.should_record(control_step):
            raise ValueError("control step does not coincide with a recording frame")
        return control_step // self.recording_stride

    def timestamp_seconds(self, control_step: int) -> float:
        return self.frame_index(control_step) / self.recording_hz

    def steps_for_seconds(self, seconds: float) -> int:
        if seconds <= 0:
            raise ValueError("duration must be positive")
        return max(1, int(round(float(seconds) * self.control_hz)))


def quaternion_rotation_matrix_xyzw(quaternion) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError("quaternion must have shape (4,)")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion must be non-zero")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def point_in_local_frame(frame_pose_xyzw, point_world) -> np.ndarray:
    """Express a world-space point in a pose's local coordinate frame."""
    pose = np.asarray(frame_pose_xyzw, dtype=np.float64)
    point = np.asarray(point_world, dtype=np.float64)
    if pose.shape != (7,) or point.shape != (3,):
        raise ValueError("frame pose and point must have shapes (7,) and (3,)")
    rotation = quaternion_rotation_matrix_xyzw(pose[3:])
    return (rotation.T @ (point - pose[:3])).astype(np.float32)


def gripper_target_for_object_local_offset(
    object_position_world,
    gripper_orientation_xyzw,
    desired_object_in_gripper,
) -> np.ndarray:
    """Place the gripper so an object occupies a desired gripper-local point."""
    object_position = np.asarray(object_position_world, dtype=np.float64)
    local_offset = np.asarray(desired_object_in_gripper, dtype=np.float64)
    if object_position.shape != (3,) or local_offset.shape != (3,):
        raise ValueError("object position and local offset must have shape (3,)")
    rotation = quaternion_rotation_matrix_xyzw(gripper_orientation_xyzw)
    return (object_position - rotation @ local_offset).astype(np.float32)


def gripper_xy_target_for_object_local_offset(
    object_position_world,
    gripper_pose_xyzw,
    desired_object_in_gripper,
) -> np.ndarray:
    """Align the aperture in XY without changing the captured descent height."""
    pose = np.asarray(gripper_pose_xyzw, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError("gripper pose must be a finite XYZW pose")
    target = gripper_target_for_object_local_offset(
        object_position_world,
        pose[3:7],
        desired_object_in_gripper,
    )
    target[2] = pose[2]
    return target


def rotary_jaw_capture_hold_target(
    measured_position: float,
    *,
    closed_position: float,
    open_position: float,
    preload_rad: float = 0.008,
    relative_speed_mps: float | None = None,
    moving_capture_preload_rad: float = 0.008,
    moving_capture_threshold_mps: float = 0.001,
    moving_capture_ceiling_mps: float = 0.002,
) -> float:
    """Hold a captured rotary jaw with a continuously tapered preload.

    Exact-mesh captures need the validated 8 mrad preload to keep bilateral
    contact.  The independent capture-speed gate already rejects a moving
    enclosure above 2 mm/s, so the default hold must not weaken again inside
    that admitted window. Immutable outer-workspace evidence showed the former
    4 mrad moving endpoint entering BILATERAL_SETTLE correctly, then shedding a
    finger before force control could recover.  The taper remains configurable
    for explicit experiments; force limits and grasp-success evidence remain
    unchanged.
    """
    if closed_position > open_position:
        raise ValueError("closed_position must not exceed open_position")
    if preload_rad < 0.0:
        raise ValueError("preload_rad must be non-negative")
    if moving_capture_preload_rad < 0.0:
        raise ValueError("moving_capture_preload_rad must be non-negative")
    if (
        not np.isfinite(moving_capture_threshold_mps)
        or moving_capture_threshold_mps < 0.0
    ):
        raise ValueError("moving_capture_threshold_mps must be non-negative")
    if (
        not np.isfinite(moving_capture_ceiling_mps)
        or moving_capture_ceiling_mps <= moving_capture_threshold_mps
    ):
        raise ValueError(
            "moving_capture_ceiling_mps must exceed moving_capture_threshold_mps"
        )
    if relative_speed_mps is not None and (
        not np.isfinite(relative_speed_mps) or relative_speed_mps < 0.0
    ):
        raise ValueError("relative_speed_mps must be finite and non-negative")
    effective_preload = float(preload_rad)
    if (
        relative_speed_mps is not None
        and relative_speed_mps > moving_capture_threshold_mps
    ):
        moving_fraction = float(
            np.clip(
                (relative_speed_mps - moving_capture_threshold_mps)
                / (moving_capture_ceiling_mps - moving_capture_threshold_mps),
                0.0,
                1.0,
            )
        )
        effective_preload = effective_preload + moving_fraction * (
            min(effective_preload, float(moving_capture_preload_rad))
            - effective_preload
        )
    return float(
        np.clip(
            float(measured_position) - effective_preload,
            float(closed_position),
            float(open_position),
        )
    )


def capture_preload_force_floor(
    capture_contact_force_n: float,
    *,
    retention_fraction: float = 0.90,
) -> float:
    """Keep a confirmed capture above a bounded fraction of admission force.

    The admission threshold rejects weak bilateral impacts.  Immediately after
    admission, retaining only the much lower persistence threshold can let both
    contacts decay before the rotary jaw reacts.  This floor keeps force control
    proactive without changing either the validator's contact truth or its
    independent maximum-force safety limit.
    """
    capture_force = float(capture_contact_force_n)
    fraction = float(retention_fraction)
    if not np.isfinite(capture_force) or capture_force <= 0.0:
        raise ValueError("capture_contact_force_n must be finite and positive")
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("retention_fraction must be finite and in (0, 1]")
    return capture_force * fraction


def capture_admission_retention_fraction(object_width_m: float | None) -> float:
    """Return the evidence-bounded capture floor fraction for object width.

    Large cubes require the same 90% force floor used by bilateral settle.
    Admitting a 40 mm capture below that floor can transition away from the
    active slow-close/recenter controller while one fingertip is already
    losing contact.  The 30 mm hysteresis remains unchanged.
    """
    return so101_capture_admission_retention_fraction(object_width_m)


def unilateral_contact_requires_recenter(
    left_force_n: float,
    right_force_n: float,
    *,
    minimum_force_n: float,
) -> bool:
    """Return whether exactly one finger retains meaningful object contact."""
    if minimum_force_n < 0.0:
        raise ValueError("minimum force must be non-negative")
    return (
        min(float(left_force_n), float(right_force_n)) < minimum_force_n
        <= max(float(left_force_n), float(right_force_n))
    )


def so101_recenter_contact_memory(
    left_force_n,
    right_force_n,
    previous_side=None,
    *,
    minimum_force_n=0.10,
):
    """Preserve a bounded recenter direction across a brief zero-force gap.

    A captured cube can leave both sensors between 120 Hz samples, so waiting
    for a later unilateral sample starts recovery after the object is already
    moving out of the aperture. Remember the stronger side while bilateral
    contact is still present, prefer a live unilateral side, and synthesize
    only the minimum persistence force when both sides briefly read zero.
    """
    forces = (float(left_force_n), float(right_force_n))
    threshold = float(minimum_force_n)
    if previous_side not in {None, "left", "right"}:
        raise ValueError("previous_side must be left, right, or None")
    if any(not np.isfinite(force) or force < 0.0 for force in forces):
        raise ValueError("contact forces must be finite and non-negative")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("minimum_force_n must be finite and positive")

    left_active = forces[0] >= threshold
    right_active = forces[1] >= threshold
    if left_active and right_active:
        side = previous_side
        if side is None:
            if forces[0] > forces[1]:
                side = "left"
            elif forces[1] > forces[0]:
                side = "right"
        return {"forces": forces, "side": side, "used_memory": False}
    if left_active or right_active:
        side = "left" if left_active else "right"
        return {"forces": forces, "side": side, "used_memory": False}
    if previous_side is None:
        return {"forces": forces, "side": None, "used_memory": False}
    remembered_forces = (
        (threshold, 0.0) if previous_side == "left" else (0.0, threshold)
    )
    return {
        "forces": remembered_forces,
        "side": previous_side,
        "used_memory": True,
    }


@dataclass(frozen=True)
class GraspEvidence:
    left_force_n: float
    right_force_n: float
    aperture_aligned: bool = False
    capture_admissible: bool = True
    relative_translation_error_m: float = float("inf")
    relative_speed_mps: float = float("inf")
    proof_lift_m: float = 0.0
    collision: bool = False


@dataclass(frozen=True)
class GraspDecision:
    phase: GraspPhase
    entered_phase: bool
    rebase_joint_command: bool
    hold_cartesian_pose: bool
    failure_reason: str | None

    @property
    def rebase_relative_tracking(self) -> bool:
        """Return whether object-in-gripper tracking must start from this capture."""
        return self.entered_phase and self.phase is GraspPhase.BILATERAL_SETTLE


@dataclass
class ContactAwareGraspStateMachine:
    """Validate a quasi-static grasp; transient bilateral impacts cannot pass."""

    control_hz: int = 120
    object_width_m: float | None = None
    minimum_contact_force_n: float = 0.10
    capture_contact_force_n: float | None = None
    capture_confirmation_s: float = 0.05
    maximum_capture_relative_speed_mps: float = 0.002
    maximum_force_n: float = 60.0
    maximum_relative_translation_error_m: float = 0.003
    maximum_relative_speed_mps: float = 0.015
    minimum_proof_lift_m: float = 0.005
    bilateral_settle_s: float = 0.10
    static_hold_s: float = 0.20
    proof_lift_hold_s: float = 0.05
    phase_timeout_s: float = 5.0
    maximum_contact_loss_s: float = 0.05
    phase: GraspPhase = GraspPhase.APPROACH
    phase_steps: int = 0
    stable_steps: int = 0
    contact_loss_steps: int = 0
    capture_steps: int = 0
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if self.object_width_m is not None and (
            not np.isfinite(self.object_width_m) or self.object_width_m <= 0.0
        ):
            raise ValueError("object_width_m must be finite and positive")
        if self.minimum_contact_force_n < 0 or self.maximum_force_n <= 0:
            raise ValueError("contact force thresholds must be valid")
        if self.minimum_contact_force_n >= self.maximum_force_n:
            raise ValueError("minimum contact force must be below maximum force")
        if self.capture_contact_force_n is None:
            self.capture_contact_force_n = self.minimum_contact_force_n
        if not (
            self.minimum_contact_force_n
            <= self.capture_contact_force_n
            < self.maximum_force_n
        ):
            raise ValueError(
                "capture contact force must be at least the minimum contact "
                "force and below the maximum force"
            )
        if self.capture_confirmation_s < 0:
            raise ValueError("capture confirmation duration must be non-negative")
        if (
            not np.isfinite(self.maximum_capture_relative_speed_mps)
            or self.maximum_capture_relative_speed_mps < 0
        ):
            raise ValueError("capture relative speed limit must be non-negative")

    def _steps(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.control_hz)))

    @property
    def capture_confirmation_steps(self) -> int:
        """Return the shared bilateral-capture confirmation window."""
        return self._steps(self.capture_confirmation_s)

    def _enter(self, phase: GraspPhase) -> None:
        self.phase = phase
        self.phase_steps = 0
        self.stable_steps = 0
        self.contact_loss_steps = 0
        self.capture_steps = 0

    def _fail(self, reason: str) -> None:
        self.phase = GraspPhase.FAILED
        self.failure_reason = reason

    def step(self, evidence: GraspEvidence) -> GraspDecision:
        previous = self.phase
        if self.phase in {GraspPhase.VALIDATED, GraspPhase.FAILED}:
            return GraspDecision(self.phase, False, False, True, self.failure_reason)
        self.phase_steps += 1
        if evidence.collision:
            self._fail("collision")
        elif max(evidence.left_force_n, evidence.right_force_n) > self.maximum_force_n:
            self._fail("contact_force_limit")
        elif self.phase_steps > self._steps(self.phase_timeout_s):
            self._fail(f"grasp_phase_timeout:{self.phase.value}")

        left = evidence.left_force_n >= self.minimum_contact_force_n
        right = evidence.right_force_n >= self.minimum_contact_force_n
        any_contact = left or right
        bilateral = left and right
        capture_bilateral = (
            evidence.left_force_n
            >= capture_preload_force_floor(
                self.capture_contact_force_n,
                retention_fraction=capture_admission_retention_fraction(
                    self.object_width_m
                ),
            )
            and evidence.right_force_n
            >= capture_preload_force_floor(
                self.capture_contact_force_n,
                retention_fraction=capture_admission_retention_fraction(
                    self.object_width_m
                ),
            )
            and evidence.capture_admissible
        )
        rigid = (
            evidence.relative_translation_error_m
            <= self.maximum_relative_translation_error_m
            and evidence.relative_speed_mps <= self.maximum_relative_speed_mps
        )

        if self.phase is GraspPhase.APPROACH and any_contact:
            self._enter(GraspPhase.FIRST_CONTACT)
        elif self.phase is GraspPhase.FIRST_CONTACT:
            self._enter(GraspPhase.CONTACT_ALIGNMENT)
        elif self.phase is GraspPhase.CONTACT_ALIGNMENT and evidence.aperture_aligned:
            self._enter(GraspPhase.SLOW_CLOSE)
        elif self.phase is GraspPhase.SLOW_CLOSE:
            # Confirmation is a joint force-and-motion window. Counting
            # dynamic bilateral samples and checking speed only on the final
            # tick lets a moving edge impact accumulate the whole force
            # window, pause momentarily, and enter capture before the cube is
            # settled in the aperture. Any dynamic sample therefore resets
            # the same consecutive confirmation counter.
            capture_stable = (
                capture_bilateral
                and evidence.relative_speed_mps
                <= self.maximum_capture_relative_speed_mps
            )
            self.capture_steps = self.capture_steps + 1 if capture_stable else 0
            if self.capture_steps >= self.capture_confirmation_steps:
                self._enter(GraspPhase.BILATERAL_SETTLE)
        elif self.phase is GraspPhase.BILATERAL_SETTLE:
            self.stable_steps = self.stable_steps + 1 if bilateral and rigid else 0
            if self.stable_steps >= self._steps(self.bilateral_settle_s):
                self._enter(GraspPhase.STATIC_HOLD)
        elif self.phase is GraspPhase.STATIC_HOLD:
            self.stable_steps = self.stable_steps + 1 if bilateral and rigid else 0
            if self.stable_steps >= self._steps(self.static_hold_s):
                self._enter(GraspPhase.PROOF_LIFT)
        elif self.phase is GraspPhase.PROOF_LIFT:
            valid = (
                bilateral
                and rigid
                and evidence.proof_lift_m >= self.minimum_proof_lift_m
            )
            self.stable_steps = self.stable_steps + 1 if valid else 0
            if self.stable_steps >= self._steps(self.proof_lift_hold_s):
                self._enter(GraspPhase.VALIDATED)

        if self.phase in {
            GraspPhase.BILATERAL_SETTLE,
            GraspPhase.STATIC_HOLD,
            GraspPhase.PROOF_LIFT,
        }:
            self.contact_loss_steps = self.contact_loss_steps + 1 if not bilateral else 0
            if self.contact_loss_steps > self._steps(self.maximum_contact_loss_s):
                self._fail(f"bilateral_contact_lost:{self.phase.value}")

        entered = self.phase is not previous
        return GraspDecision(
            phase=self.phase,
            entered_phase=entered,
            rebase_joint_command=entered
            and self.phase
            in {GraspPhase.FIRST_CONTACT, GraspPhase.BILATERAL_SETTLE},
            hold_cartesian_pose=self.phase
            in {
                GraspPhase.FIRST_CONTACT,
                GraspPhase.CONTACT_ALIGNMENT,
                GraspPhase.SLOW_CLOSE,
                GraspPhase.BILATERAL_SETTLE,
                GraspPhase.STATIC_HOLD,
            },
            failure_reason=self.failure_reason,
        )
