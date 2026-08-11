"""Derive conservative continuous feasible regions from simulator probes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable

from farpoint.variation_engine import FeasibleRegion, _segment_distance


WORKSPACE_PROBE_SCHEMA = "farpoint.workspace-probe.v1"
REQUIRED_CONSTRAINTS = (
    "full_path_ik",
    "joint_limits",
    "table_collision_free",
    "self_collision_free",
    "front_camera_visible",
    "wrist_camera_visible",
)


@dataclass(frozen=True)
class WorkspaceProbe:
    x_m: float
    y_m: float
    checks: tuple[tuple[str, bool], ...]
    probe_id: str

    def __post_init__(self) -> None:
        if not self.probe_id:
            raise ValueError("workspace probe_id must be non-empty")
        if not math.isfinite(self.x_m) or not math.isfinite(self.y_m):
            raise ValueError("workspace probe position must be finite")
        check_map = dict(self.checks)
        if set(check_map) != set(REQUIRED_CONSTRAINTS):
            raise ValueError("workspace probe checks do not match the required constraints")
        if any(type(value) is not bool for value in check_map.values()):
            raise ValueError("workspace probe checks must be booleans")

    @property
    def passed(self) -> bool:
        return all(dict(self.checks).values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORKSPACE_PROBE_SCHEMA,
            "probe_id": self.probe_id,
            "position_xy_m": [self.x_m, self.y_m],
            "checks": dict(self.checks),
            "passed": self.passed,
        }


def _cross(origin, first, second) -> float:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (
        first[1] - origin[1]
    ) * (second[0] - origin[0])


def convex_hull(points: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    ordered = sorted(set(points))
    if len(ordered) < 3:
        raise ValueError("at least three distinct passing probes are required")
    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = tuple(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        raise ValueError("passing probes are collinear")
    return hull


def _maximum_clearance(polygon: tuple[tuple[float, float], ...]) -> float:
    """Deterministically approximate the in-polygon maximum edge clearance."""
    region = FeasibleRegion(
        region_id="temporary",
        version="1",
        frame_id="temporary",
        polygon_xy_m=polygon,
        max_clearance_m=1.0,
        object_anchor="center",
        footprint_xy_m=(1.0, 1.0),
        generator_sha256="0" * 64,
        constraints_sha256="0" * 64,
    )
    x_min, x_max, y_min, y_max = region.bounds
    best_point = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    best_clearance = 0.0
    half_x = (x_max - x_min) / 2
    half_y = (y_max - y_min) / 2
    for _ in range(5):
        for x_index in range(21):
            for y_index in range(21):
                point = (
                    best_point[0] - half_x + 2 * half_x * x_index / 20,
                    best_point[1] - half_y + 2 * half_y * y_index / 20,
                )
                if not region.contains(point):
                    continue
                clearance = min(
                    _segment_distance(point, start, end)
                    for start, end in zip(polygon, polygon[1:] + polygon[:1])
                )
                if clearance > best_clearance:
                    best_point, best_clearance = point, clearance
        half_x /= 10
        half_y /= 10
    if best_clearance <= 0:
        raise ValueError("derived feasible region has no positive interior clearance")
    return best_clearance


def derive_feasible_region(
    probes: Iterable[WorkspaceProbe],
    *,
    region_id: str,
    version: str,
    frame_id: str,
    object_anchor: str,
    footprint_xy_m: tuple[float, float],
    generator_identity: dict[str, Any],
) -> FeasibleRegion:
    """Create a hull only when no known failed probe would be hidden inside it."""
    probes = tuple(probes)
    passing = tuple(probe for probe in probes if probe.passed)
    polygon = convex_hull((probe.x_m, probe.y_m) for probe in passing)
    temporary = FeasibleRegion(
        region_id=region_id,
        version=version,
        frame_id=frame_id,
        polygon_xy_m=polygon,
        max_clearance_m=_maximum_clearance(polygon),
        object_anchor=object_anchor,
        footprint_xy_m=footprint_xy_m,
        generator_sha256=hashlib.sha256(
            json.dumps(generator_identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        constraints_sha256=hashlib.sha256(
            json.dumps(
                [probe.as_dict() for probe in probes],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    )
    hidden_failures = [
        probe.probe_id
        for probe in probes
        if not probe.passed and temporary.contains((probe.x_m, probe.y_m))
    ]
    if hidden_failures:
        raise ValueError(
            "failed probes lie inside the derived convex region: "
            + ", ".join(sorted(hidden_failures))
        )
    return temporary
