"""Print SO-101 finger collision bounds in each rigid-link frame."""

from __future__ import annotations

import argparse
import math

from pxr import Gf, Usd, UsdGeom


def _matrix(prim: Usd.Prim) -> Gf.Matrix4d:
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _transform_xyzw(point, pose):
    px, py, pz, x, y, z, w = pose
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    vx, vy, vz = point
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty) + px,
        vy + w * ty + (z * tx - x * tz) + py,
        vz + w * tz + (x * ty - y * tx) + pz,
    )


def _transform_wxyz(point, pose):
    px, py, pz, w, x, y, z = pose
    return _transform_xyzw(point, (px, py, pz, x, y, z, w))


def _sub(a, b):
    return tuple(a[index] - b[index] for index in range(3))


def _add(a, b):
    return tuple(a[index] + b[index] for index in range(3))


def _scale(vector, scalar):
    return tuple(value * scalar for value in vector)


def _dot(a, b):
    return sum(a[index] * b[index] for index in range(3))


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(vector):
    length = math.sqrt(_dot(vector, vector))
    return tuple(value / length for value in vector) if length else (0.0, 0.0, 0.0)


def _distal_summary(points, fraction=0.2):
    """Summarize the far end of a finger-like mesh in its rigid-link frame."""
    low = [min(point[index] for point in points) for index in range(3)]
    high = [max(point[index] for point in points) for index in range(3)]
    axis = max(range(3), key=lambda index: high[index] - low[index])
    use_low_end = abs(low[axis]) >= abs(high[axis])
    span = high[axis] - low[axis]
    threshold = low[axis] + fraction * span if use_low_end else high[axis] - fraction * span
    distal = [
        point
        for point in points
        if (point[axis] <= threshold if use_low_end else point[axis] >= threshold)
    ]
    distal_low = [min(point[index] for point in distal) for index in range(3)]
    distal_high = [max(point[index] for point in distal) for index in range(3)]
    center = tuple((distal_low[index] + distal_high[index]) * 0.5 for index in range(3))
    return {
        "axis": axis,
        "end": "low" if use_low_end else "high",
        "count": len(distal),
        "min": distal_low,
        "max": distal_high,
        "center": center,
    }


def _closest_point_on_triangle(point, a, b, c):
    """Return the closest point using the region tests from RTCD section 5.1.5."""
    ab = _sub(b, a)
    ac = _sub(c, a)
    ap = _sub(point, a)
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = _sub(point, b)
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return _add(a, _scale(ab, d1 / (d1 - d3)))

    cp = _sub(point, c)
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return _add(a, _scale(ac, d2 / (d2 - d6)))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        edge = _sub(c, b)
        weight = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return _add(b, _scale(edge, weight))

    denominator = 1.0 / (va + vb + vc)
    return _add(a, _add(_scale(ab, vb * denominator), _scale(ac, vc * denominator)))


def _triangles(mesh, points):
    counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default()) or []
    indexes = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default()) or []
    cursor = 0
    for count in counts:
        face = indexes[cursor : cursor + count]
        cursor += count
        for index in range(1, count - 1):
            yield points[face[0]], points[face[index]], points[face[index + 1]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd")
    parser.add_argument("--gripper-pose", nargs=7, type=float)
    parser.add_argument("--jaw-pose", nargs=7, type=float)
    parser.add_argument(
        "--pose-order",
        choices=("wxyz", "xyzw"),
        default="xyzw",
        help=(
            "Quaternion order of runtime poses; pinned Isaac Lab 3.0 body poses "
            "use xyzw (Isaac Lab 2.x used wxyz)."
        ),
    )
    parser.add_argument("--probe", nargs=3, type=float, action="append")
    parser.add_argument(
        "--include-visuals",
        action="store_true",
        help="Also print visual meshes so missing collider proxies can be diagnosed.",
    )
    parser.add_argument(
        "--distal-summary",
        action="store_true",
        help="Print the terminal 20%% of each mesh along its longest local axis.",
    )
    args = parser.parse_args()
    transform = _transform_wxyz if args.pose_order == "wxyz" else _transform_xyzw
    stage = Usd.Stage.Open(args.usd)
    if stage is None:
        raise RuntimeError(f"could not open USD: {args.usd}")
    for link_name in ("gripper", "jaw"):
        links = [
            prim
            for prim in stage.Traverse()
            if prim.GetName() == link_name
            and prim.GetTypeName() == "Xform"
            and str(prim.GetPath()).startswith("/so101_new_calib/")
        ]
        if len(links) != 1:
            raise RuntimeError(f"expected one {link_name} link, got {[str(p.GetPath()) for p in links]}")
        link = links[0]
        link_inverse = _matrix(link).GetInverse()
        for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            is_collision = "/collisions/" in path
            if (not is_collision and not args.include_visuals) or not prim.IsA(
                UsdGeom.Mesh
            ):
                continue
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get(Usd.TimeCode.Default())
            if not points:
                continue
            mesh_to_world = _matrix(prim)
            local = [
                link_inverse.Transform(mesh_to_world.Transform(Gf.Vec3d(point)))
                for point in points
            ]
            low = [min(point[index] for point in local) for index in range(3)]
            high = [max(point[index] for point in local) for index in range(3)]
            center = [(low[index] + high[index]) * 0.5 for index in range(3)]
            print(
                f"{link_name}\t{'collision' if is_collision else 'visual'}\t{path}\t"
                f"min={low}\tmax={high}\tcenter={center}",
                flush=True,
            )
            pose = args.gripper_pose if link_name == "gripper" else args.jaw_pose
            if args.distal_summary:
                distal = _distal_summary(local)
                print(
                    f"distal\t{link_name}\t{path}\taxis={distal['axis']}\t"
                    f"end={distal['end']}\tcount={distal['count']}\t"
                    f"min={distal['min']}\tmax={distal['max']}\t"
                    f"center={distal['center']}",
                    flush=True,
                )
                if pose:
                    print(
                        f"distal_world\t{link_name}\t{path}\t"
                        f"center={transform(distal['center'], pose)}",
                        flush=True,
                    )
            if pose:
                world = [transform(point, pose) for point in local]
                world_low = [min(point[index] for point in world) for index in range(3)]
                world_high = [max(point[index] for point in world) for index in range(3)]
                print(
                    f"world\t{link_name}\t{path}\tmin={world_low}\tmax={world_high}",
                    flush=True,
                )
            if pose:
                for probe in args.probe or []:
                    candidates = []
                    for triangle in _triangles(UsdGeom.Mesh(prim), world):
                        closest = _closest_point_on_triangle(probe, *triangle)
                        delta = _sub(closest, probe)
                        distance_squared = _dot(delta, delta)
                        normal = _normalize(
                            _cross(
                                _sub(triangle[1], triangle[0]),
                                _sub(triangle[2], triangle[0]),
                            )
                        )
                        candidates.append((distance_squared, closest, normal))
                    distance_squared, closest, normal = min(
                        candidates, key=lambda value: value[0]
                    )
                    distance = math.sqrt(distance_squared)
                    print(
                        f"probe\t{link_name}\t{path}\tpoint={probe}\tclosest={closest}\t"
                        f"distance={distance}\tnormal={normal}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
