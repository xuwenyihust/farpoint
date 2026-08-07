"""Version-pinned SO-101 rotary-jaw geometry diagnostics.

The SO-101 ``gripper`` and ``jaw`` rigid-body origins are servo/joint frames,
not fingertip contact centers.  These helpers keep the pinned workshop USD
collision bounds explicit and evaluate a candidate posture without using
either body origin as an aperture proxy.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from farpoint.grasp_oracle import quaternion_rotation_matrix_xyzw


SO101_WORKSHOP_COMMIT = "ce807d99724cb65671abec01f908a2fcb4a6eab7"
SO101_WORKSHOP_ASSET_SHA256 = (
    "11f5f0bb5f2fae3eefebbcd07dfafc6b14602f6c4e5dae8f21a4a46892991006"
)
SO101_RUNTIME_QUATERNION_ORDER = "xyzw"

# Collision AABBs expressed in their respective rigid-link frames.  They are
# extracted from the pinned official USD.  Transforming all eight corners is
# conservative for table-clearance screening: the transformed AABB encloses
# the actual collision mesh.
SO101_GRIPPER_COLLISION_BOUNDS_M = (
    (
        (-0.0150000006, -0.0193000313, -0.0358000512),
        (0.0304000005, 0.0203000053, -0.0109999475),
    ),
    (
        (-0.0351999998, -0.0279999853, -0.1044252933),
        (0.0299999993, 0.0240000749, 0.0010000305),
    ),
)
SO101_JAW_COLLISION_BOUNDS_M = (
    (-0.0122999996, -0.0820000023, -0.0051000006),
    (0.0099979769, 0.0099946642, 0.0428999998),
)

# Terminal 20% centers of the fixed and moving finger collision meshes.  The
# vector between these points is diagnostic only; it is deliberately not used
# as a control axis or as proof of bilateral contact.
SO101_FIXED_FINGER_DISTAL_CENTER_M = np.asarray(
    (-0.0139828045, -0.0002179978, -0.0944252863), dtype=np.float32
)
SO101_MOVING_FINGER_DISTAL_CENTER_M = np.asarray(
    (-0.0053084364, -0.0728107095, 0.0188927737), dtype=np.float32
)

# Exact-mesh medial-axis candidate for a 40 mm cube at jaw=1.0 rad,
# wrist-pitch=0.5 rad, and wrist-roll=0.5 rad.  It replaces the run171/run173
# corner-wedge point ([+55, -16, +3] mm), but remains a diagnostic reference
# until a PhysX slow-close/proof-lift run validates the posture.
SO101_APERTURE_REFERENCE_IN_GRIPPER_M = np.asarray(
    (0.0190589216, -0.0084643856, -0.0578673638), dtype=np.float32
)

# Production aperture centers from the exact collision meshes at the frozen
# wrist posture (pitch=0.5 rad, roll=0.5 rad) and 45-degree cube yaw.  The
# 1.2-rad point is the reference used by the validated 30 mm episodes.  The
# larger openings were recomputed from the pinned workshop USD after the
# 40 mm slow-close regression showed that reusing the 1.2-rad point at a
# 1.7-rad approach leaves the cube 12.7 mm too deep in the finger throat.
SO101_CAPTURE_APERTURE_CALIBRATION = (
    (
        1.2,
        np.asarray((0.0214971882, -0.0084643886, -0.0546656502), dtype=np.float32),
    ),
    (
        1.4,
        np.asarray(
            (0.0224196905, -0.0086327176, -0.0479056704), dtype=np.float32
        ),
    ),
    (
        1.7,
        np.asarray(
            (0.0236281415, -0.0086327433, -0.0420040267), dtype=np.float32
        ),
    ),
)


def so101_capture_aperture_reference(jaw_position_rad: float) -> np.ndarray:
    """Return the calibrated local aperture center for an approach opening.

    Openings below 1.2 rad deliberately retain the validated 30 mm reference.
    Between measured exact-mesh anchors the center is linearly interpolated;
    the small interpolation error is bounded by the adjacent calibration
    points instead of extrapolating rotary-jaw geometry.  Values above the
    largest production anchor retain the 1.7-rad reference up to the pinned
    USD open limit.
    """
    jaw_position = float(jaw_position_rad)
    if not np.isfinite(jaw_position):
        raise ValueError("jaw_position_rad must be finite")
    if not -0.1746 <= jaw_position <= 1.7453:
        raise ValueError("jaw_position_rad must be within pinned USD limits")

    positions = np.asarray(
        [position for position, _reference in SO101_CAPTURE_APERTURE_CALIBRATION],
        dtype=np.float64,
    )
    references = np.asarray(
        [reference for _position, reference in SO101_CAPTURE_APERTURE_CALIBRATION],
        dtype=np.float64,
    )
    calibrated_position = float(np.clip(jaw_position, positions[0], positions[-1]))
    return np.asarray(
        [
            np.interp(calibrated_position, positions, references[:, axis])
            for axis in range(3)
        ],
        dtype=np.float32,
    )


def so101_capture_feed_axis_local(jaw_position_rad: float) -> np.ndarray:
    """Return the gripper-local collision-free insertion direction.

    The validated 30 mm side insertion follows local +Z.  At jaw=1.7 rad the
    exact-mesh fixed/moving contact pair lies in the local X-Z closing plane
    with nearly identical local-Y coordinates, so its open channel is local
    +Y.  Blend and normalize between the production calibration endpoints;
    this preserves the 30 mm path while avoiding the rotating jaw sweeping a
    40 mm cube during side insertion.
    """
    jaw_position = float(jaw_position_rad)
    if not np.isfinite(jaw_position):
        raise ValueError("jaw_position_rad must be finite")
    if not -0.1746 <= jaw_position <= 1.7453:
        raise ValueError("jaw_position_rad must be within pinned USD limits")
    blend = float(np.clip((jaw_position - 1.2) / (1.7 - 1.2), 0.0, 1.0))
    direction = np.asarray((0.0, blend, 1.0 - blend), dtype=np.float64)
    return (direction / np.linalg.norm(direction)).astype(np.float32)


def _vector(value, *, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def aabb_corners(bounds) -> np.ndarray:
    """Return the eight corners of a three-dimensional AABB."""
    if len(bounds) != 2:
        raise ValueError("bounds must contain minimum and maximum vectors")
    lower = _vector(bounds[0], length=3, name="bounds minimum")
    upper = _vector(bounds[1], length=3, name="bounds maximum")
    if np.any(lower > upper):
        raise ValueError("bounds minimum must not exceed maximum")
    return np.asarray(list(product(*zip(lower, upper))), dtype=np.float64)


def transform_points_xyzw(pose_xyzw, points) -> np.ndarray:
    """Transform link-local points by an Isaac Lab 3.0 ``[pos, xyzw]`` pose."""
    pose = _vector(pose_xyzw, length=7, name="pose_xyzw")
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("points must have shape (N, 3) and contain finite values")
    rotation = quaternion_rotation_matrix_xyzw(pose[3:]).astype(np.float64)
    return values @ rotation.T + pose[:3]


def posture_geometry_diagnostics(
    gripper_pose_xyzw,
    jaw_pose_xyzw,
    object_center_world_m,
    *,
    object_half_height_m: float,
    aperture_center_local_m=SO101_APERTURE_REFERENCE_IN_GRIPPER_M,
) -> dict:
    """Screen an open-jaw posture for aperture alignment and table clearance.

    The returned clearance is conservative because it uses transformed link
    AABB corners.  A positive value means the AABBs remain above the inferred
    table plane after translating the aperture reference onto the object.
    """
    gripper_pose = _vector(gripper_pose_xyzw, length=7, name="gripper pose")
    jaw_pose = _vector(jaw_pose_xyzw, length=7, name="jaw pose")
    object_center = _vector(object_center_world_m, length=3, name="object center")
    aperture_local = _vector(
        aperture_center_local_m, length=3, name="aperture center"
    )
    half_height = float(object_half_height_m)
    if not np.isfinite(half_height) or half_height <= 0.0:
        raise ValueError("object_half_height_m must be positive and finite")

    gripper_rotation = quaternion_rotation_matrix_xyzw(gripper_pose[3:]).astype(
        np.float64
    )
    aperture_world = gripper_pose[:3] + gripper_rotation @ aperture_local
    alignment_translation = object_center - aperture_world

    fixed_corners = np.concatenate(
        [aabb_corners(bounds) for bounds in SO101_GRIPPER_COLLISION_BOUNDS_M]
    )
    jaw_corners = aabb_corners(SO101_JAW_COLLISION_BOUNDS_M)
    collision_world = np.concatenate(
        [
            transform_points_xyzw(gripper_pose, fixed_corners),
            transform_points_xyzw(jaw_pose, jaw_corners),
        ]
    )
    aligned_minimum_z = float(
        np.min(collision_world[:, 2]) + alignment_translation[2]
    )
    table_height = float(object_center[2] - half_height)

    fixed_distal_world = transform_points_xyzw(
        gripper_pose, SO101_FIXED_FINGER_DISTAL_CENTER_M[None, :]
    )[0]
    moving_distal_world = transform_points_xyzw(
        jaw_pose, SO101_MOVING_FINGER_DISTAL_CENTER_M[None, :]
    )[0]
    tip_connection = moving_distal_world - fixed_distal_world
    tip_connection_norm = float(np.linalg.norm(tip_connection))
    if tip_connection_norm <= 1e-12:
        tip_connection_unit = np.zeros(3, dtype=np.float64)
    else:
        tip_connection_unit = tip_connection / tip_connection_norm

    aperture_distance = float(np.linalg.norm(aperture_local))
    if aperture_distance <= 1e-12:
        raise ValueError("aperture center must not coincide with the gripper origin")
    feed_local = aperture_local / aperture_distance
    feed_world = gripper_rotation @ feed_local
    aligned_gripper_target = gripper_pose[:3] + alignment_translation

    return {
        "quaternion_order": SO101_RUNTIME_QUATERNION_ORDER,
        "aperture_center_local_m": aperture_local.astype(float).tolist(),
        "aperture_center_world_m": aperture_world.astype(float).tolist(),
        "alignment_translation_m": alignment_translation.astype(float).tolist(),
        "aligned_gripper_target_m": aligned_gripper_target.astype(float).tolist(),
        "finger_feed_axis_world": feed_world.astype(float).tolist(),
        "finger_feed_axis_vertical_component": float(feed_world[2]),
        "fixed_finger_distal_world_m": fixed_distal_world.astype(float).tolist(),
        "moving_finger_distal_world_m": moving_distal_world.astype(float).tolist(),
        "tip_connection_axis_world": tip_connection_unit.astype(float).tolist(),
        "tip_connection_vertical_component": float(tip_connection_unit[2]),
        "conservative_aligned_collision_minimum_z_m": aligned_minimum_z,
        "inferred_table_height_m": table_height,
        "conservative_table_clearance_m": aligned_minimum_z - table_height,
    }
