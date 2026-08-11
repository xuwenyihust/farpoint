"""Composable axes, samplers, and split policies for variation plans."""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
import hashlib
import math
from typing import Any, Iterable, Protocol


@dataclass(frozen=True)
class VariationAxis:
    """One ordered categorical axis in a deterministic variation product."""

    name: str
    path: str
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.path:
            raise ValueError("variation axis name and path must be non-empty")
        if not self.values:
            raise ValueError(f"variation axis {self.name!r} must define values")


def axis_product(axes: Iterable[VariationAxis]) -> Iterable[dict[str, Any]]:
    """Yield ordered path/value assignments for an arbitrary axis product."""
    axes = tuple(axes)
    for values in product(*(axis.values for axis in axes)):
        yield {axis.path: value for axis, value in zip(axes, values)}


class Sampler(Protocol):
    def sample(self, row: int, column: int, seed: int) -> dict[str, Any]: ...


@dataclass(frozen=True)
class StratifiedGridSampler:
    x_bounds_m: tuple[float, float]
    y_bounds_m: tuple[float, float]
    rows: int
    columns: int
    interior_fraction: tuple[float, float] = (0.2, 0.8)

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("stratified grid dimensions must be positive")
        if self.x_bounds_m[0] >= self.x_bounds_m[1] or self.y_bounds_m[0] >= self.y_bounds_m[1]:
            raise ValueError("stratified grid bounds must be increasing")
        low, high = self.interior_fraction
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("interior_fraction must be increasing within [0, 1]")

    def cells(self) -> Iterable[tuple[int, int]]:
        return product(range(self.rows), range(self.columns))

    def sample(self, row: int, column: int, seed: int) -> dict[str, Any]:
        if not 0 <= row < self.rows or not 0 <= column < self.columns:
            raise ValueError("grid cell is outside the sampler")
        x_width = (self.x_bounds_m[1] - self.x_bounds_m[0]) / self.columns
        y_width = (self.y_bounds_m[1] - self.y_bounds_m[0]) / self.rows
        low, high = self.interior_fraction
        rng = random.Random(seed)
        return {
            "x": round(self.x_bounds_m[0] + column * x_width + x_width * rng.uniform(low, high), 9),
            "y": round(self.y_bounds_m[0] + row * y_width + y_width * rng.uniform(low, high), 9),
        }


class SplitPolicy(Protocol):
    def split_for(self, index: int, total: int) -> str: ...


@dataclass(frozen=True)
class ProportionalSplitPolicy:
    """Assign ordered items using deterministic largest-remainder quotas."""

    weights: tuple[tuple[str, float], ...] = (
        ("train", 0.8),
        ("validation", 0.1),
        ("test", 0.1),
    )

    def counts(self, total: int) -> dict[str, int]:
        if total <= 0:
            raise ValueError("split population must be positive")
        if not self.weights or any(not name or weight < 0 for name, weight in self.weights):
            raise ValueError("split weights must contain non-negative named entries")
        weight_sum = sum(weight for _, weight in self.weights)
        if weight_sum <= 0:
            raise ValueError("split weights must have positive total weight")
        exact = [(name, total * weight / weight_sum) for name, weight in self.weights]
        counts = {name: int(value) for name, value in exact}
        remainder = total - sum(counts.values())
        order = sorted(exact, key=lambda item: (-(item[1] - int(item[1])), item[0]))
        for name, _ in order[:remainder]:
            counts[name] += 1
        return counts

    def split_for(self, index: int, total: int) -> str:
        if not 0 <= index < total:
            raise ValueError("split index is outside the population")
        stop = 0
        for name, _ in self.weights:
            stop += self.counts(total)[name]
            if index < stop:
                return name
        raise AssertionError("split policy did not assign an item")


def _segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    projection = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator
    projection = min(1.0, max(0.0, projection))
    return math.hypot(
        point[0] - (start[0] + projection * dx),
        point[1] - (start[1] + projection * dy),
    )


@dataclass(frozen=True)
class FeasibleRegion:
    """Versioned continuous placement region derived outside the sampler.

    The polygon represents the already-intersected IK, collision, joint-limit,
    table-clearance, and camera-visibility constraints. ``max_clearance_m`` is
    frozen by the region generator so classification never depends on a grid.
    """

    region_id: str
    version: str
    frame_id: str
    polygon_xy_m: tuple[tuple[float, float], ...]
    max_clearance_m: float
    object_anchor: str
    footprint_xy_m: tuple[float, float]
    generator_sha256: str
    constraints_sha256: str

    def __post_init__(self) -> None:
        if not self.region_id or not self.version or not self.frame_id:
            raise ValueError("feasible region identity fields must be non-empty")
        if len(self.polygon_xy_m) < 3:
            raise ValueError("feasible region polygon must have at least three vertices")
        if any(len(point) != 2 or not all(math.isfinite(value) for value in point) for point in self.polygon_xy_m):
            raise ValueError("feasible region polygon vertices must be finite XY pairs")
        if self.max_clearance_m <= 0.0 or not math.isfinite(self.max_clearance_m):
            raise ValueError("max_clearance_m must be positive and finite")
        if len(self.footprint_xy_m) != 2 or any(value <= 0.0 for value in self.footprint_xy_m):
            raise ValueError("footprint_xy_m must contain two positive values")
        for name, digest in (
            ("generator_sha256", self.generator_sha256),
            ("constraints_sha256", self.constraints_sha256),
        ):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"{name} must be a lowercase SHA256")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = [point[0] for point in self.polygon_xy_m]
        ys = [point[1] for point in self.polygon_xy_m]
        return min(xs), max(xs), min(ys), max(ys)

    def contains(self, point: tuple[float, float]) -> bool:
        """Return whether a point is in or on the polygon (even-odd rule)."""
        x, y = point
        inside = False
        vertices = self.polygon_xy_m
        for start, end in zip(vertices, vertices[1:] + vertices[:1]):
            if _segment_distance(point, start, end) <= 1e-12:
                return True
            if (start[1] > y) != (end[1] > y):
                crossing_x = (end[0] - start[0]) * (y - start[1]) / (end[1] - start[1]) + start[0]
                if x < crossing_x:
                    inside = not inside
        return inside

    def clearance_m(self, point: tuple[float, float]) -> float:
        if not self.contains(point):
            raise ValueError("point is outside the feasible region")
        vertices = self.polygon_xy_m
        return min(
            _segment_distance(point, start, end)
            for start, end in zip(vertices, vertices[1:] + vertices[:1])
        )

    def normalized_clearance(self, point: tuple[float, float]) -> float:
        return min(1.0, self.clearance_m(point) / self.max_clearance_m)

    def band(self, point: tuple[float, float]) -> str:
        clearance = self.normalized_clearance(point)
        if clearance < 1.0 / 3.0:
            return "outer"
        if clearance < 2.0 / 3.0:
            return "middle"
        return "core"


def _sobol_uint32(index: int, dimension: int) -> int:
    if index < 0:
        raise ValueError("Sobol index must be non-negative")
    if dimension not in (0, 1):
        raise ValueError("the built-in Sobol sampler supports two dimensions")
    directions = []
    value = 1 << 31
    for _ in range(32):
        directions.append(value)
        value = value >> 1 if dimension == 0 else value ^ (value >> 1)
    gray = index ^ (index >> 1)
    result = 0
    bit = 0
    while gray:
        if gray & 1:
            result ^= directions[bit]
        gray >>= 1
        bit += 1
    return result


@dataclass(frozen=True)
class ScrambledSobolSampler:
    """Deterministic 2-D Sobol sampler with a seed-derived digital shift."""

    max_candidates_per_sample: int = 4096

    def __post_init__(self) -> None:
        if self.max_candidates_per_sample <= 0:
            raise ValueError("max_candidates_per_sample must be positive")

    @staticmethod
    def _scramble(seed: int, dimension: int) -> int:
        material = f"farpoint-sobol-v1:{seed}:{dimension}".encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")

    def unit(self, index: int, seed: int) -> tuple[float, float]:
        denominator = float(1 << 32)
        return tuple(
            (_sobol_uint32(index, dimension) ^ self._scramble(seed, dimension)) / denominator
            for dimension in (0, 1)
        )  # type: ignore[return-value]

    def sample(
        self,
        region: FeasibleRegion,
        *,
        ordinal: int,
        seed: int,
        band: str | None = None,
    ) -> dict[str, Any]:
        if ordinal < 0:
            raise ValueError("sample ordinal must be non-negative")
        if band not in {None, "outer", "middle", "core"}:
            raise ValueError("band must be outer, middle, core, or None")
        x_min, x_max, y_min, y_max = region.bounds
        first_index = ordinal * self.max_candidates_per_sample + 1
        for offset in range(self.max_candidates_per_sample):
            sobol_index = first_index + offset
            unit_x, unit_y = self.unit(sobol_index, seed)
            point = (
                x_min + unit_x * (x_max - x_min),
                y_min + unit_y * (y_max - y_min),
            )
            if region.contains(point) and (band is None or region.band(point) == band):
                return {
                    "position_xy_m": [round(point[0], 9), round(point[1], 9)],
                    "region_band": region.band(point),
                    "normalized_clearance": round(region.normalized_clearance(point), 9),
                    "sobol_index": sobol_index,
                    "scramble_seed": seed,
                    "sampler_version": "farpoint.scrambled-sobol.v1",
                }
        raise ValueError(f"could not sample band {band!r} inside region {region.region_id!r}")


@dataclass(frozen=True)
class ExactQuotaSplitPolicy:
    """Assign exact train/validation quotas within arbitrary named strata."""

    quotas: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    seed: int

    def assignments(self, stratum: str, item_ids: Iterable[str]) -> dict[str, str]:
        quota_map = dict(dict(self.quotas).get(stratum, ()))
        ids = list(item_ids)
        if sum(quota_map.values()) != len(ids):
            raise ValueError(f"split quotas for {stratum!r} do not match its population")
        if any(not name or count < 0 for name, count in quota_map.items()):
            raise ValueError("split quotas must be non-negative named entries")
        ordered = sorted(
            ids,
            key=lambda item_id: hashlib.sha256(
                f"farpoint-split-v1:{self.seed}:{stratum}:{item_id}".encode()
            ).digest(),
        )
        result = {}
        cursor = 0
        for split, count in quota_map.items():
            for item_id in ordered[cursor : cursor + count]:
                result[item_id] = split
            cursor += count
        return result
