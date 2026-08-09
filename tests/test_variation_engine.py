import pytest

from farpoint.variation_engine import (
    ProportionalSplitPolicy,
    StratifiedGridSampler,
    VariationAxis,
    axis_product,
)


def test_arbitrary_axis_product_preserves_declared_order():
    axes = [
        VariationAxis("shape", "object.shape", ("cube", "cylinder")),
        VariationAxis("mass", "object.mass_kg", (0.03, 0.04, 0.05)),
    ]

    assignments = list(axis_product(axes))

    assert len(assignments) == 6
    assert assignments[0] == {"object.shape": "cube", "object.mass_kg": 0.03}
    assert assignments[-1] == {
        "object.shape": "cylinder",
        "object.mass_kg": 0.05,
    }


def test_grid_sampler_supports_non_5x5_workspaces_deterministically():
    sampler = StratifiedGridSampler(
        x_bounds_m=(0.0, 0.3),
        y_bounds_m=(-0.2, 0.2),
        rows=2,
        columns=3,
    )

    assert list(sampler.cells()) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert sampler.sample(1, 2, 7) == sampler.sample(1, 2, 7)
    with pytest.raises(ValueError, match="outside"):
        sampler.sample(2, 0, 7)


def test_split_policy_uses_largest_remainder_for_any_population():
    policy = ProportionalSplitPolicy()

    assert policy.counts(7) == {"train": 5, "validation": 1, "test": 1}
    assert [policy.split_for(index, 7) for index in range(7)] == [
        "train",
        "train",
        "train",
        "train",
        "train",
        "validation",
        "test",
    ]
