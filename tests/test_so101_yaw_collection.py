import copy
from pathlib import Path

import pytest

from farpoint.object_variation import load_variation_config
from farpoint.so101_collection import create_manifest, next_attempt, record_attempt
from farpoint.so101_yaw_collection import (
    build_yaw_collection_plan,
    load_yaw_collection_config,
    validate_yaw_collection_balance,
    validate_yaw_collection_config,
    yaw_collection_balance,
)


ROOT = Path(__file__).resolve().parents[1]


def build_plan():
    return build_yaw_collection_plan(
        load_variation_config(ROOT / "configs/variations/so101_cube_pick_place_v1.json"),
        load_yaw_collection_config(ROOT / "configs/collections/so101_cube_yaw0_30mm_v0_0_2.json"),
    )


def build_yaw30_plan():
    return build_yaw_collection_plan(
        load_variation_config(
            ROOT / "configs/variations/so101_cube_pick_place_v1.json"
        ),
        load_yaw_collection_config(
            ROOT / "configs/collections/so101_cube_yaw30_30mm_v0_0_3.json"
        ),
    )


def test_yaw_formal_plan_is_deterministic_and_balanced():
    plan = build_plan()
    assert plan == build_plan()
    assert len(plan["trials"]) == 50
    assert plan["collection"]["required_successes"] == 50
    assert plan["collection"]["maximum_attempts"] == 150
    balance = yaw_collection_balance(plan)
    assert validate_yaw_collection_balance(balance) == []
    assert balance["splits"] == {"test": 5, "train": 40, "validation": 5}
    assert balance["sizes"] == {"size_0": 50}
    assert balance["colors"] == {"color_0": 25, "color_1": 25}
    assert balance["masses_kg"] == {"0.03": 25, "0.04": 25}
    assert sorted(balance["mass_color"].values()) == [12, 12, 13, 13]
    assert set(balance["workspace_cells"].values()) == {2}
    assert balance["yaw_degrees"] == {"0.0": 50}
    override = next(trial for trial in plan["trials"] if trial["source_trial_id"] == "cube_r04_c02_s0_k1")
    assert override["split"] == "test"
    assert override["split_source"] == "yaw_formal_40_5_5_override_v1"


def test_published_yaw0_plan_hash_remains_compatible():
    assert (
        build_plan()["plan_sha256"]
        == "c0b66f4bef95fceaeaf3e1cbab7b23a7f9f5a1f2014f9100df42b42296693a61"
    )


def test_yaw30_formal_plan_freezes_balanced30_with_sixty_attempt_budget():
    plan = build_yaw30_plan()
    assert plan == build_yaw30_plan()
    assert len(plan["trials"]) == 30
    assert plan["collection"]["required_successes"] == 30
    assert plan["collection"]["maximum_attempts"] == 60
    balance = yaw_collection_balance(plan)
    assert validate_yaw_collection_balance(
        balance, plan["collection"]["balance_contract"]
    ) == []
    assert balance["splits"] == {"test": 3, "train": 24, "validation": 3}
    assert set(balance["workspace_cells"]) == {
        f"r{row:02d}_c{column:02d}" for row in range(5) for column in range(5)
    }
    assert set(balance["workspace_rows"].values()) == {6}
    assert set(balance["workspace_columns"].values()) == {6}
    assert balance["colors"] == {"color_0": 15, "color_1": 15}
    assert balance["masses_kg"] == {"0.03": 15, "0.04": 15}
    assert sorted(balance["mass_color"].values()) == [7, 7, 8, 8]
    assert balance["yaw_degrees"] == {"30.0": 30}
    by_source = {trial["source_trial_id"]: trial for trial in plan["trials"]}
    assert by_source["cube_r04_c00_s0_k1"]["resolved"]["mass_kg"] == 0.03
    assert by_source["cube_r04_c01_s0_k0"]["resolved"]["mass_kg"] == 0.04


def test_yaw30_manifest_freezes_sixty_attempt_budget():
    plan = build_yaw30_plan()
    manifest = create_manifest(
        plan,
        collection_id="yaw30_formal_test",
        git_commit="a" * 40,
        maximum_attempts=60,
    )
    assert manifest["maximum_attempts"] == 60
    with pytest.raises(ValueError, match="frozen collection profile"):
        create_manifest(
            plan,
            collection_id="bad",
            git_commit="a" * 40,
            maximum_attempts=59,
        )


def test_every_cell_has_complementary_mass_and_color():
    plan = build_plan()
    for cell in {trial["cell_id"] for trial in plan["trials"]}:
        trials = [trial for trial in plan["trials"] if trial["cell_id"] == cell]
        assert {trial["seed_material"]["color_index"] for trial in trials} == {0, 1}
        assert {trial["resolved"]["mass_kg"] for trial in trials} == {0.03, 0.04}
        assert all(trial["resolved"]["dimensions_m"] == [0.03, 0.03, 0.03] for trial in trials)
        assert all(trial["resolved"]["orientation_xyzw"] == [0.0, 0.0, 0.0, 1.0] for trial in trials)


def test_manifest_freezes_budget_and_retries_uncovered_variations_evenly():
    plan = build_plan()
    manifest = create_manifest(plan, collection_id="yaw_formal_test", git_commit="a" * 40, maximum_attempts=150)
    assert manifest["release_status"] == "CANDIDATE"
    first = next_attempt(manifest, plan)
    record_attempt(manifest, plan, first, episode_id="episode_first", success=False, dataset_valid=True, failure_reason="missed_grasp")
    for _ in range(49):
        attempt = next_attempt(manifest, plan)
        record_attempt(manifest, plan, attempt, episode_id=f"episode_{attempt['attempt_id']}", success=True, dataset_valid=True)
    retry = next_attempt(manifest, plan)
    assert retry["variation_id"] == first["variation_id"]
    assert retry["attempt_index"] == 1
    with pytest.raises(ValueError, match="frozen collection profile"):
        create_manifest(plan, collection_id="bad", git_commit="a" * 40, maximum_attempts=149)


def test_yaw_collection_config_rejects_contract_drift():
    config = load_yaw_collection_config(ROOT / "configs/collections/so101_cube_yaw0_30mm_v0_0_2.json")
    bad = copy.deepcopy(config)
    bad["maximum_attempts"] = 151
    with pytest.raises(ValueError, match="150-attempt"):
        validate_yaw_collection_config(bad)


def test_explicit_yaw_collection_config_rejects_duplicate_source_trials():
    config = load_yaw_collection_config(
        ROOT / "configs/collections/so101_cube_yaw30_30mm_v0_0_3.json"
    )
    bad = copy.deepcopy(config)
    bad["trial_profiles"][1]["source_trial_id"] = bad["trial_profiles"][0][
        "source_trial_id"
    ]
    with pytest.raises(ValueError, match="must be unique"):
        validate_yaw_collection_config(bad)


def test_explicit_yaw_collection_config_requires_every_balance_axis():
    config = load_yaw_collection_config(
        ROOT / "configs/collections/so101_cube_yaw30_30mm_v0_0_3.json"
    )
    bad = copy.deepcopy(config)
    bad["expected_balance"].pop("mass_color")
    with pytest.raises(ValueError, match="axes are incomplete"):
        validate_yaw_collection_config(bad)
