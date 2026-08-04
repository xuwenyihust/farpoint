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
