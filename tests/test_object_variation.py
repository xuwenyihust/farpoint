import copy
from collections import Counter
from pathlib import Path

import pytest

from farpoint.contracts import validate_contract
from farpoint.object_variation import (
    ObjectSpec,
    generate_variation_plan,
    load_variation_config,
    validate_variation_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/variations/so101_cube_pick_place_v1.json"


def test_so101_plan_is_deterministic_stratified_and_valid():
    config = load_variation_config(CONFIG)
    first = generate_variation_plan(config)
    second = generate_variation_plan(config)

    assert (
        first["plan_sha256"] == "c93cb6378efac2e4febe6b6027bf5fd13581f54caae7831095d84535806d75fc"
    )
    assert first == second
    assert not validate_contract(first)
    assert len(first["trials"]) == 100
    assert Counter(row["split"] for row in first["trials"]) == {
        "train": 80,
        "validation": 10,
        "test": 10,
    }
    assert set(Counter(row["cell_id"] for row in first["trials"]).values()) == {4}
    assert {tuple(row["resolved"]["dimensions_m"]) for row in first["trials"]} == {
        (0.03, 0.03, 0.03),
        (0.04, 0.04, 0.04),
    }
    trial = first["trials"][0]
    assert trial["resolved"]["entities"]["pick_object"]["entity_type"] == "cube"
    assert trial["resolved"]["entities"]["placement_target"]["pose"]["position_m"] == [
        0.20,
        0.10,
        0.037,
    ]
    assert first["varied_axes"] == [
        "entities.pick_object.pose.position_m.x",
        "entities.pick_object.pose.position_m.y",
        "entities.pick_object.geometry.dimensions_m",
        "entities.pick_object.appearance.rgba",
    ]


def test_object_spec_rejects_invalid_physics():
    with pytest.raises(ValueError, match="dynamic_friction"):
        ObjectSpec(
            shape="cube",
            asset_id="cube",
            dimensions_m=(0.03, 0.03, 0.03),
            position_m=(0.2, 0.0, 0.05),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            rgba=(1.0, 0.0, 0.0, 1.0),
            mass_kg=0.04,
            static_friction=0.5,
            dynamic_friction=0.8,
        ).validate()


def test_variation_config_rejects_invalid_mass():
    config = load_variation_config(CONFIG)
    invalid = copy.deepcopy(config)
    invalid["object"]["mass_kg"] = 0
    with pytest.raises(ValueError, match="mass_kg"):
        validate_variation_config(invalid)


def test_generic_dimension_profiles_support_non_cube_assets():
    config = load_variation_config(CONFIG)
    config["object"].pop("edge_sizes_m")
    config["object"].update(
        {
            "shape": "cylinder",
            "asset_id": "procedural_cylinder_v1",
            "dimension_profiles_m": [[0.03, 0.03, 0.06], [0.04, 0.04, 0.08]],
        }
    )

    plan = generate_variation_plan(config)

    assert plan["trials"][0]["resolved"]["shape"] == "cylinder"
    assert plan["trials"][0]["trial_id"].startswith("cylinder_")
    assert plan["trials"][0]["resolved"]["dimensions_m"] == [0.03, 0.03, 0.06]
    assert plan["trials"][0]["resolved"]["entities"]["pick_object"]["entity_type"] == "cylinder"
