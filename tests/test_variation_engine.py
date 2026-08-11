import pytest

from farpoint.variation_engine import (
    ExactQuotaSplitPolicy,
    FeasibleRegion,
    ProportionalSplitPolicy,
    ScrambledSobolSampler,
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


def _region():
    digest = "a" * 64
    return FeasibleRegion(
        region_id="cube-bottom-center-v1",
        version="1",
        frame_id="isaac_world",
        polygon_xy_m=((0.0, 0.0), (0.3, 0.0), (0.3, 0.3), (0.0, 0.3)),
        max_clearance_m=0.15,
        object_anchor="bottom_center",
        footprint_xy_m=(0.04, 0.04),
        generator_sha256=digest,
        constraints_sha256="b" * 64,
    )


def test_feasible_region_bands_use_normalized_boundary_clearance():
    region = _region()
    assert region.band((0.01, 0.15)) == "outer"
    assert region.band((0.075, 0.15)) == "middle"
    assert region.band((0.15, 0.15)) == "core"
    with pytest.raises(ValueError, match="outside"):
        region.clearance_m((0.4, 0.1))


def test_scrambled_sobol_is_deterministic_and_can_target_each_region_band():
    sampler = ScrambledSobolSampler()
    samples = {
        band: sampler.sample(_region(), ordinal=2, seed=41, band=band)
        for band in ("outer", "middle", "core")
    }
    assert samples == {
        band: sampler.sample(_region(), ordinal=2, seed=41, band=band)
        for band in ("outer", "middle", "core")
    }
    assert {row["region_band"] for row in samples.values()} == {"outer", "middle", "core"}
    assert all(row["sampler_version"] == "farpoint.scrambled-sobol.v1" for row in samples.values())


def test_exact_quota_split_is_deterministic_and_has_no_test_split():
    policy = ExactQuotaSplitPolicy(
        quotas=(("red/yaw0/core", (("train", 3), ("validation", 2))),),
        seed=20260811,
    )
    item_ids = [f"variation-{index}" for index in range(5)]
    first = policy.assignments("red/yaw0/core", item_ids)
    assert first == policy.assignments("red/yaw0/core", reversed(item_ids))
    assert sorted(first.values()).count("validation") == 2
    assert "test" not in first.values()
    with pytest.raises(ValueError, match="population"):
        policy.assignments("red/yaw0/core", item_ids[:-1])
