import copy
from pathlib import Path

import pytest

from farpoint.object_variation import load_variation_config
from farpoint.so101_collection import create_manifest, next_attempt, record_attempt
from farpoint.so101_mass_collection import (
    build_mirrored_mass_collection_plan,
    load_mass_collection_config,
    mirrored_balance,
    validate_mass_collection_config,
    validate_mirrored_balance,
)


ROOT = Path(__file__).resolve().parents[1]


def build_plan():
    return build_mirrored_mass_collection_plan(
        load_variation_config(
            ROOT / "configs/variations/so101_cube_pick_place_v1.json"
        ),
        load_mass_collection_config(
            ROOT / "configs/collections/so101_cube_mass_003_v0_0_1.json"
        ),
    )


def test_plan_exactly_mirrors_balanced50_and_changes_only_mass_identity():
    plan = build_plan()

    assert len(plan["trials"]) == 50
    assert plan["collection"]["required_successes"] == 50
    assert plan["collection"]["maximum_attempts"] == 150
    assert plan["collection"]["target_mass_kg"] == 0.03
    assert plan["collection"]["source_plan_sha256"] == (
        "c93cb6378efac2e4febe6b6027bf5fd13581f54caae7831095d84535806d75fc"
    )
    assert [trial["source_trial_id"] for trial in plan["trials"]][:3] == [
        "cube_r00_c00_s0_k0",
        "cube_r00_c00_s0_k1",
        "cube_r00_c00_s1_k0",
    ]
    assert all(trial["trial_id"].endswith("_m030g") for trial in plan["trials"])
    for trial in plan["trials"]:
        for key in ("requested", "resolved"):
            assert trial[key]["mass_kg"] == 0.03
            assert (
                trial[key]["entities"]["pick_object"]["physics"]["mass_kg"]
                == 0.03
            )


def test_plan_preserves_frozen_balance_and_is_deterministic():
    first = build_plan()
    second = build_plan()

    assert first == second
    balance = mirrored_balance(first)
    assert validate_mirrored_balance(balance) == []
    assert balance["splits"] == {"test": 5, "train": 40, "validation": 5}
    assert balance["sizes"] == {"size_0": 25, "size_1": 25}
    assert balance["colors"] == {"color_0": 25, "color_1": 25}
    assert len(balance["workspace_cells"]) == 25


def test_formal_manifest_freezes_50_successes_and_150_attempts():
    plan = build_plan()
    manifest = create_manifest(
        plan,
        collection_id="so101_cube_mass_003_formal_test",
        git_commit="a" * 40,
        maximum_attempts=150,
    )

    assert manifest["required_successes"] == 50
    assert manifest["maximum_attempts"] == 150
    assert manifest["release_status"] == "CANDIDATE"
    assert manifest["collection_profile"] == plan["collection"]
    first = next_attempt(manifest, plan)
    record_attempt(
        manifest,
        plan,
        first,
        episode_id="episode_first",
        success=False,
        dataset_valid=True,
        failure_category="task",
        failure_reason="missed_grasp",
    )
    for _ in range(49):
        attempt = next_attempt(manifest, plan)
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=f"episode_{attempt['attempt_id']}",
            success=True,
            dataset_valid=True,
        )
    retry = next_attempt(manifest, plan)
    assert retry["source_trial_id"] == first["source_trial_id"]
    assert retry["attempt_index"] == 1


def test_formal_manifest_rejects_operator_budget_override():
    plan = build_plan()
    with pytest.raises(ValueError, match="frozen collection profile"):
        create_manifest(
            plan,
            collection_id="bad_budget",
            git_commit="a" * 40,
            maximum_attempts=75,
        )


def test_mass_collection_config_rejects_invalid_contract():
    config = load_mass_collection_config(
        ROOT / "configs/collections/so101_cube_mass_003_v0_0_1.json"
    )
    duplicate = copy.deepcopy(config)
    duplicate["source_trial_ids"][1] = duplicate["source_trial_ids"][0]
    with pytest.raises(ValueError, match="unique"):
        validate_mass_collection_config(duplicate)
    bad_budget = copy.deepcopy(config)
    bad_budget["maximum_attempts"] = 49
    with pytest.raises(ValueError, match="smaller"):
        validate_mass_collection_config(bad_budget)
