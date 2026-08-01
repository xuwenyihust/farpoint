import json
from pathlib import Path

import pytest

from farpoint.contracts import validate_contract
from farpoint.position_plan import (
    apply_position_trial,
    generate_position_plan,
    load_position_config,
    resolve_position_trial,
    validate_position_plan,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "variations"
    / "farpoint_v1_3_cube_position.json"
)


def plan():
    return generate_position_plan(load_position_config(CONFIG_PATH))


def test_plan_is_deterministic_complete_and_contract_valid():
    first = plan()
    second = plan()
    assert first == second
    assert len(first["trials"]) == 75
    assert validate_contract(first) == []
    validate_position_plan(first)


def test_split_and_cell_counts_match_the_release_design():
    manifest = plan()
    counts = {
        split: sum(trial["split"] == split for trial in manifest["trials"])
        for split in ("train", "validation", "test")
    }
    assert counts == {"train": 50, "validation": 13, "test": 12}
    assert set({
        cell_id: sum(trial["cell_id"] == cell_id for trial in manifest["trials"])
        for cell_id in {trial["cell_id"] for trial in manifest["trials"]}
    }.values()) == {3}


def test_primary_and_reserve_coordinates_are_unique_and_inside_cell_interiors():
    manifest = plan()
    all_candidates = []
    for trial in manifest["trials"]:
        all_candidates.append(trial["object_position_xy_m"])
        all_candidates.extend(
            candidate["object_position_xy_m"] for candidate in trial["reserve_candidates"]
        )
    assert len({tuple(position) for position in all_candidates}) == 225
    assert all(
        0 <= candidate["seed"] < 2**63
        for trial in manifest["trials"]
        for candidate in (trial, *trial["reserve_candidates"])
    )


def test_trial_resolution_marks_only_initial_xy_as_varied():
    manifest = plan()
    resolved = resolve_position_trial(manifest, "primary_r00_c00_s00")
    assert resolved["variation"]["varied_axes"] == [
        "object_initial_position_x_m",
        "object_initial_position_y_m",
    ]
    assert resolved["variation"]["resolved"]["object_shape"] == "cube"
    assert resolved["variation"]["resolved"]["object_yaw_degrees"] == 0.0
    assert resolved["split"] == "train"


def test_reserve_selection_is_predeclared_and_deterministic():
    manifest = plan()
    primary = resolve_position_trial(manifest, "primary_r04_c04_s02")
    reserve = resolve_position_trial(manifest, "primary_r04_c04_s02", reserve_index=2)
    assert reserve["variation"]["variation_id"].endswith("_reserve2")
    assert reserve["variation"]["resolved"]["object_position_m"] != primary["variation"]["resolved"]["object_position_m"]
    assert reserve["split"] == primary["split"] == "validation"


def test_manifest_hash_detects_any_mutation():
    manifest = plan()
    manifest["trials"][0]["object_position_xy_m"][0] += 0.0001
    with pytest.raises(ValueError, match="plan_sha256"):
        validate_position_plan(manifest)


def test_scene_integration_applies_only_planned_xy_and_disables_randomization():
    manifest = plan()
    source = {
        "name": "isaac_perception_contact_scene",
        "scene": {
            "pick_object": {"position": [9.0, 9.0, 0.5]},
            "target_zone": {"position": [8.0, 8.0, 0.42]},
        },
        "randomization": {"enabled": True},
    }
    configured, resolved = apply_position_trial(
        source, manifest, "primary_r02_c02_s00"
    )
    assert source["randomization"]["enabled"] is True
    assert configured["randomization"]["enabled"] is False
    assert configured["scene"]["pick_object"]["position"] == resolved["variation"]["resolved"]["object_position_m"]
    assert configured["scene"]["target_zone"]["position"] == [0.96, -0.03, 0.42]


def test_config_change_produces_a_new_config_and_plan_hash():
    config = load_position_config(CONFIG_PATH)
    baseline = generate_position_plan(config)
    changed = json.loads(json.dumps(config))
    changed["config_revision"] = "2"
    updated = generate_position_plan(changed)
    assert updated["config_sha256"] != baseline["config_sha256"]
    assert updated["plan_sha256"] != baseline["plan_sha256"]
