import math

import numpy as np


class PerceptionError(RuntimeError):
    pass


def look_at_calibration(position, target, resolution, focal_length_mm=24.0, aperture_mm=20.955):
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - position
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    width, height = (int(resolution[0]), int(resolution[1]))
    focal_pixels = width * float(focal_length_mm) / float(aperture_mm)
    intrinsics = np.asarray(
        [
            [focal_pixels, 0.0, width * 0.5],
            [0.0, focal_pixels, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, 0] = right
    camera_to_world[:3, 1] = down
    camera_to_world[:3, 2] = forward
    camera_to_world[:3, 3] = position
    return intrinsics, camera_to_world


def color_mask(rgb, channel, min_channel=80, min_dominance=30):
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("rgb must have shape (height, width, 3+)")
    channel_indices = {"red": 0, "green": 1, "blue": 2}
    if channel not in channel_indices:
        raise ValueError(f"unsupported dominant channel: {channel}")
    index = channel_indices[channel]
    selected = image[:, :, index].astype(np.int16)
    competitors = np.max(np.delete(image[:, :, :3], index, axis=2).astype(np.int16), axis=2)
    return (selected >= int(min_channel)) & (
        selected - competitors >= int(min_dominance)
    )


def backproject_pixels(points_uv, depth, intrinsics, camera_to_world):
    points_uv = np.asarray(points_uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    if points_uv.ndim != 2 or points_uv.shape[1] != 2:
        raise ValueError("points_uv must have shape (n, 2)")
    if len(points_uv) != len(depth):
        raise ValueError("points_uv and depth must have the same length")
    homogeneous = np.column_stack([points_uv, np.ones(len(points_uv))])
    camera_points = (
        np.linalg.inv(np.asarray(intrinsics, dtype=np.float64))
        @ (homogeneous * depth[:, None]).T
    ).T
    camera_homogeneous = np.column_stack([camera_points, np.ones(len(camera_points))])
    world = (
        np.asarray(camera_to_world, dtype=np.float64) @ camera_homogeneous.T
    ).T
    return world[:, :3]


def estimate_dominant_color_pose(
    rgb,
    depth,
    intrinsics,
    camera_to_world,
    channel,
    min_pixels=20,
    min_channel=80,
    min_dominance=30,
    surface_to_center_m=0.0,
    xy_center_method="median",
):
    image = np.asarray(rgb)
    depth_image = np.asarray(depth, dtype=np.float64)
    if depth_image.shape != image.shape[:2]:
        raise ValueError("depth shape must match the RGB image")
    mask = color_mask(
        image,
        channel,
        min_channel=min_channel,
        min_dominance=min_dominance,
    )
    valid = mask & np.isfinite(depth_image) & (depth_image > 0.0)
    rows, columns = np.nonzero(valid)
    if len(rows) < int(min_pixels):
        raise PerceptionError(
            f"{channel} segmentation found {len(rows)} valid pixels; "
            f"at least {int(min_pixels)} are required"
        )

    values = depth_image[rows, columns]
    median_depth = float(np.median(values))
    mad = float(np.median(np.abs(values - median_depth)))
    tolerance = max(0.01, 3.5 * mad)
    inliers = np.abs(values - median_depth) <= tolerance
    rows = rows[inliers]
    columns = columns[inliers]
    values = values[inliers]
    if len(rows) < int(min_pixels):
        raise PerceptionError(
            f"{channel} depth filtering retained {len(rows)} pixels; "
            f"at least {int(min_pixels)} are required"
        )

    points_uv = np.column_stack([columns.astype(np.float64), rows.astype(np.float64)])
    world_points = backproject_pixels(
        points_uv,
        values,
        intrinsics,
        camera_to_world,
    )
    position = np.median(world_points, axis=0)
    if xy_center_method == "bounds":
        position[:2] = 0.5 * (
            np.min(world_points[:, :2], axis=0)
            + np.max(world_points[:, :2], axis=0)
        )
    elif xy_center_method != "median":
        raise ValueError(
            "xy_center_method must be either 'median' or 'bounds'"
        )
    position[2] -= float(surface_to_center_m)
    spread = np.linalg.norm(world_points[:, :2] - position[:2], axis=1)
    return {
        "position": [round(float(value), 6) for value in position],
        "pixel_centroid": [
            round(float(np.median(columns)), 3),
            round(float(np.median(rows)), 3),
        ],
        "valid_pixels": int(len(rows)),
        "median_depth": round(float(np.median(values)), 6),
        "xy_center_method": xy_center_method,
        "xy_spread_m": round(float(np.median(spread)), 6),
        "confidence": round(
            min(1.0, len(rows) / max(float(min_pixels) * 8.0, 1.0)),
            4,
        ),
    }


def _undirected_yaw_difference_degrees(first, second):
    """Angular distance when a cube has four equivalent upright rotations."""
    return abs((float(first) - float(second) + 45.0) % 90.0 - 45.0)


def estimate_dominant_color_yaw(
    rgb,
    depth,
    intrinsics,
    camera_to_world,
    channel,
    *,
    min_pixels=20,
    min_channel=80,
    min_dominance=30,
    min_confidence=0.15,
):
    """Estimate an upright cube's yaw from RGB-D support points.

    The result is deliberately modulo 90 degrees: a uniformly coloured cube
    has no observable distinction between its four upright face rotations.
    This estimator never receives scene state and is therefore safe for use by
    the controller; simulator yaw may only be compared by an offline audit.
    """
    image = np.asarray(rgb)
    depth_image = np.asarray(depth, dtype=np.float64)
    if depth_image.shape != image.shape[:2]:
        raise ValueError("depth shape must match the RGB image")
    mask = color_mask(image, channel, min_channel=min_channel, min_dominance=min_dominance)
    valid = mask & np.isfinite(depth_image) & (depth_image > 0.0)
    rows, columns = np.nonzero(valid)
    if len(rows) < int(min_pixels):
        raise PerceptionError(f"{channel} segmentation found {len(rows)} valid pixels; at least {int(min_pixels)} are required")
    points = backproject_pixels(
        np.column_stack([columns.astype(np.float64), rows.astype(np.float64)]),
        depth_image[rows, columns], intrinsics, camera_to_world,
    )[:, :2]
    center = np.median(points, axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    # PCA cannot orient a square: its second moments are isotropic at every
    # upright 90-degree rotation.  Instead, fit the orientation of the
    # projected support's minimum-area bounding box.  This uses RGB-D pixels
    # only and is intrinsically modulo 90 degrees, matching cube symmetry.
    candidate_angles = np.arange(0.0, 90.0, 0.5)
    areas = []
    for angle in candidate_angles:
        radians = math.radians(float(angle))
        rotation = np.array([[math.cos(radians), math.sin(radians)], [-math.sin(radians), math.cos(radians)]])
        rotated = centered @ rotation.T
        spans = np.ptp(rotated, axis=0)
        areas.append(float(spans[0] * spans[1]))
    best_index = int(np.argmin(areas))
    yaw = float(candidate_angles[best_index])
    best_area = areas[best_index]
    offset = max(1, int(10.0 / 0.5))
    nearby = [areas[(best_index - offset) % len(areas)], areas[(best_index + offset) % len(areas)]]
    orientation_separation = max(0.0, min(nearby) / max(best_area, 1e-12) - 1.0)
    pixel_support = min(1.0, len(points) / max(float(min_pixels) * 8.0, 1.0))
    confidence = round(pixel_support * min(1.0, orientation_separation), 4)
    if confidence < float(min_confidence):
        raise PerceptionError(f"cube yaw confidence {confidence:.4f} is below {float(min_confidence):.4f}")
    return {
        "yaw_degrees": round(float(yaw), 6),
        "symmetry_period_degrees": 90.0,
        "valid_pixels": int(len(points)),
        "confidence": confidence,
        "xy_covariance": covariance.tolist(),
        "orientation_separation": round(float(orientation_separation), 6),
    }


def cube_yaw_error_degrees(estimated_yaw_degrees, ground_truth_yaw_degrees):
    """Return modulo-90 yaw error for audit-only simulator ground truth."""
    return round(_undirected_yaw_difference_degrees(estimated_yaw_degrees, ground_truth_yaw_degrees), 6)


def xy_error(estimate, ground_truth):
    return math.hypot(
        float(estimate[0]) - float(ground_truth[0]),
        float(estimate[1]) - float(ground_truth[1]),
    )
