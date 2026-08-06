"""Rank SO-101 open-jaw aperture candidates from runtime poses and USD meshes.

This diagnostic intentionally uses the collision meshes rather than link origins
or transformed AABBs.  The minimum world Z of a triangle mesh occurs at one of
its vertices, so the reported table clearance is exact for the pinned meshes.
Candidate medial points are approximate (vertices rounded to a configurable
resolution) and must still pass a PhysX slow-close and proof-lift test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom


def _matrix(prim: Usd.Prim) -> Gf.Matrix4d:
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )


def _rotation_xyzw(quaternion) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm((x, y, z, w))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("quaternion must be finite and non-zero")
    x, y, z, w = (x, y, z, w) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform(points: np.ndarray, pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    return points @ _rotation_xyzw(pose[3:]).T + pose[:3]


def _inverse_transform(points: np.ndarray, pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    return (points - pose[:3]) @ _rotation_xyzw(pose[3:])


def _collision_vertices(stage: Usd.Stage, link_name: str) -> dict[str, np.ndarray]:
    links = [
        prim
        for prim in stage.Traverse()
        if prim.GetName() == link_name
        and prim.GetTypeName() == "Xform"
        and str(prim.GetPath()).startswith("/so101_new_calib/")
    ]
    if len(links) != 1:
        raise RuntimeError(f"expected one {link_name} link, got {len(links)}")
    link = links[0]
    link_inverse = _matrix(link).GetInverse()
    meshes = {}
    for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if "/collisions/" not in path or not prim.IsA(UsdGeom.Mesh):
            continue
        # v0 disables the workshop wrist-camera mount collider at startup.
        if "/camera_mount/" in path:
            continue
        values = UsdGeom.Mesh(prim).GetPointsAttr().Get(Usd.TimeCode.Default())
        if not values:
            continue
        mesh_to_world = _matrix(prim)
        meshes[path] = np.asarray(
            [
                link_inverse.Transform(mesh_to_world.Transform(Gf.Vec3d(point)))
                for point in values
            ],
            dtype=np.float64,
        )
    return meshes


def _rounded_unique(points: np.ndarray, resolution_m: float) -> np.ndarray:
    keys = np.round(points / resolution_m).astype(np.int64)
    _, indexes = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indexes)]


def _nearest_pairs(
    fixed: np.ndarray, moving: np.ndarray, *, chunk_size: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixed_indexes = []
    distances = []
    for start in range(0, len(moving), chunk_size):
        chunk = moving[start : start + chunk_size]
        squared = np.sum((chunk[:, None, :] - fixed[None, :, :]) ** 2, axis=2)
        indexes = np.argmin(squared, axis=1)
        fixed_indexes.append(indexes)
        distances.append(np.sqrt(squared[np.arange(len(chunk)), indexes]))
    indexes = np.concatenate(fixed_indexes)
    return fixed[indexes], moving, np.concatenate(distances)


def _candidate_report(
    candidate: dict,
    fixed_vertices: np.ndarray,
    jaw_vertices: np.ndarray,
    object_center: np.ndarray,
    object_edge_m: float,
    aperture_reference: np.ndarray,
    resolution_m: float,
    top_k: int,
) -> dict:
    gripper_pose = candidate["gripper_pose_xyzw"]
    jaw_pose = candidate["jaw_pose_xyzw"]
    fixed_world = _transform(fixed_vertices, gripper_pose)
    jaw_world = _transform(jaw_vertices, jaw_pose)
    jaw_in_gripper = _inverse_transform(jaw_world, gripper_pose)

    collision_world = np.concatenate((fixed_world, jaw_world))
    rotation = _rotation_xyzw(gripper_pose[3:])
    table_height = float(object_center[2] - object_edge_m / 2.0)

    def aligned_clearance(aperture_local: np.ndarray) -> float:
        aperture_world = np.asarray(gripper_pose[:3]) + rotation @ aperture_local
        translation = object_center - aperture_world
        return float(np.min(collision_world[:, 2] + translation[2]) - table_height)

    # Restrict the search to the elongated finger sections, excluding servo
    # housings.  The thresholds are expressed in the pinned link-local frames.
    fixed_finger = fixed_vertices[fixed_vertices[:, 2] < -0.055]
    moving_finger = jaw_in_gripper[
        _inverse_transform(jaw_world, jaw_pose)[:, 1] < -0.035
    ]
    fixed_finger = _rounded_unique(fixed_finger, resolution_m)
    moving_finger = _rounded_unique(moving_finger, resolution_m)
    fixed_nearest, moving_points, gaps = _nearest_pairs(fixed_finger, moving_finger)

    pair_vectors = moving_points - fixed_nearest
    axes = pair_vectors / np.maximum(gaps[:, None], 1e-12)
    midpoints = (moving_points + fixed_nearest) * 0.5
    projected_cube_widths = object_edge_m * np.sum(
        np.abs(axes @ rotation.T), axis=1
    )
    clearances = np.asarray([aligned_clearance(point) for point in midpoints])

    # The open gap needs at least 2 mm margin over the axis-aligned cube.  Rank
    # usable medial points by table clearance, then by smaller excess gap.
    usable = np.flatnonzero(gaps >= projected_cube_widths + 0.002)
    order = sorted(
        usable,
        key=lambda index: (
            -clearances[index],
            gaps[index] - projected_cube_widths[index],
        ),
    )[:top_k]
    ranked = []
    for index in order:
        ranked.append(
            {
                "aperture_center_local_m": midpoints[index].tolist(),
                "fixed_contact_local_m": fixed_nearest[index].tolist(),
                "moving_contact_local_m": moving_points[index].tolist(),
                "closing_axis_local": axes[index].tolist(),
                "open_gap_m": float(gaps[index]),
                "projected_cube_width_m": float(projected_cube_widths[index]),
                "exact_table_clearance_m": float(clearances[index]),
            }
        )

    return {
        "candidate_index": candidate["candidate_index"],
        "pitch_target_rad": candidate["pitch_target_rad"],
        "roll_target_rad": candidate["roll_target_rad"],
        "jaw_target_rad": candidate["jaw_target_rad"],
        "rounded_fixed_finger_vertex_count": len(fixed_finger),
        "rounded_moving_finger_vertex_count": len(moving_finger),
        "reference_exact_table_clearance_m": aligned_clearance(aperture_reference),
        "ranked_apertures": ranked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd", type=Path)
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--resolution-m", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.resolution_m <= 0.0 or args.top_k <= 0:
        parser.error("resolution and top-k must be positive")

    source = json.loads(args.results_json.read_text())
    stage = Usd.Stage.Open(str(args.usd))
    if stage is None:
        raise RuntimeError(f"could not open USD: {args.usd}")
    gripper_meshes = _collision_vertices(stage, "gripper")
    jaw_meshes = _collision_vertices(stage, "jaw")
    fixed_vertices = np.concatenate(tuple(gripper_meshes.values()))
    jaw_vertices = np.concatenate(tuple(jaw_meshes.values()))
    object_center = np.asarray(
        source["object"]["center_world_m"], dtype=np.float64
    )
    object_edge_m = float(source["object"]["edge_m"])
    aperture_reference = np.asarray(
        source["aperture_reference_in_gripper_m"], dtype=np.float64
    )
    reports = [
        _candidate_report(
            candidate,
            fixed_vertices,
            jaw_vertices,
            object_center,
            object_edge_m,
            aperture_reference,
            args.resolution_m,
            args.top_k,
        )
        for candidate in source["candidates"]
    ]
    result = {
        "schema_version": 1,
        "source_results_json": str(args.results_json),
        "camera_mount_collision_included": False,
        "fixed_collision_vertex_count": len(fixed_vertices),
        "jaw_collision_vertex_count": len(jaw_vertices),
        "resolution_m": args.resolution_m,
        "candidates": reports,
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
