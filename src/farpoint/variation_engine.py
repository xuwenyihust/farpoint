"""Composable axes, samplers, and split policies for variation plans."""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
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
