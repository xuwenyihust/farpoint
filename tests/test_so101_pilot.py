import pytest

from farpoint.object_variation import load_variation_config
from farpoint.so101_collection import create_pilot_manifest, next_attempt, record_attempt
from farpoint.so101_pilot import (
    DEFAULT_FALLBACK_TRIAL_IDS,
    DEFAULT_PRIMARY_TRIAL_IDS,
    build_so101_pilot_plan,
    build_targeted_mass_diagnostic_pilot_plan,
)


def _config():
    return load_variation_config("configs/variations/so101_cube_pick_place_v1.json")


def test_pilot_plan_freezes_stratified_primary_and_fallback_order():
    plan = build_so101_pilot_plan(_config(), pilot_id="so101_pilot_v1")

    assert len(plan["trials"]) == 100
    assert plan["pilot"] == {
        "kind": "stratified_success_pilot",
        "required_successes": 10,
        "maximum_attempts": 15,
        "primary_trial_ids": list(DEFAULT_PRIMARY_TRIAL_IDS),
        "fallback_trial_ids": list(DEFAULT_FALLBACK_TRIAL_IDS),
    }
    assert [trial["trial_id"] for trial in plan["trials"][:15]] == list(
        DEFAULT_PRIMARY_TRIAL_IDS + DEFAULT_FALLBACK_TRIAL_IDS
    )
    primary = plan["trials"][:10]
    assert {trial["resolved"]["dimensions_m"][0] for trial in primary} == {0.03, 0.04}
    assert {"red" if trial["resolved"]["rgba"][0] > 0.5 else "blue" for trial in primary} == {
        "red",
        "blue",
    }
    assert [trial["split"] for trial in primary].count("train") == 8
    assert [trial["split"] for trial in primary].count("validation") == 1
    assert [trial["split"] for trial in primary].count("test") == 1


def test_pilot_manifest_stops_at_ten_distinct_successes():
    plan = build_so101_pilot_plan(_config(), pilot_id="so101_pilot_v1")
    manifest = create_pilot_manifest(
        plan, collection_id=plan["plan_id"], git_commit="a" * 40
    )

    for index in range(10):
        attempt = next_attempt(manifest, plan)
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=f"episode_{index}",
            success=True,
            dataset_valid=True,
        )

    assert next_attempt(manifest, plan) is None
    assert manifest["execution_status"] == "FINISHED"
    assert manifest["quality_status"] == "PASS"
    assert len(manifest["selected_variations"]) == 10


def test_pilot_failure_consumes_fallback_without_retrying_or_changing_split():
    plan = build_so101_pilot_plan(_config(), pilot_id="so101_pilot_v1")
    manifest = create_pilot_manifest(
        plan, collection_id=plan["plan_id"], git_commit="a" * 40
    )
    failed = next_attempt(manifest, plan)
    record_attempt(
        manifest,
        plan,
        failed,
        episode_id="episode_failed",
        success=False,
        dataset_valid=True,
        failure_category="oracle",
        failure_reason="missed_grasp",
    )

    following = next_attempt(manifest, plan)

    assert following["trial_id"] == DEFAULT_PRIMARY_TRIAL_IDS[1]
    assert following["attempt_index"] == 0
    assert following["split"] == plan["trials"][1]["split"]
    for index in range(10):
        attempt = following if index == 0 else next_attempt(manifest, plan)
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=f"episode_success_{index}",
            success=True,
            dataset_valid=True,
        )

    assert manifest["quality_status"] == "PASS"
    assert len(manifest["attempts"]) == 11
    assert len(manifest["selected_variations"]) == 10
    assert failed["variation_id"] not in manifest["selected_variations"]
    assert DEFAULT_FALLBACK_TRIAL_IDS[0] in manifest["selected_variations"]


def test_pilot_plan_rejects_unknown_or_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        build_so101_pilot_plan(
            _config(),
            pilot_id="bad",
            primary_trial_ids=(DEFAULT_PRIMARY_TRIAL_IDS[0],) * 10,
        )
    with pytest.raises(ValueError, match="unknown"):
        build_so101_pilot_plan(
            _config(),
            pilot_id="bad",
            primary_trial_ids=("missing",) + DEFAULT_PRIMARY_TRIAL_IDS[1:],
        )


def test_targeted_mass_pilot_freezes_named_order_mass_and_budget():
    source_ids = (
        "cube_r03_c00_s1_k1",
        "cube_r04_c01_s0_k0",
        "cube_r02_c00_s1_k0",
        "cube_r03_c03_s1_k0",
        "cube_r01_c01_s0_k1",
    )
    plan = build_targeted_mass_diagnostic_pilot_plan(
        _config(),
        pilot_id="capture_fix",
        source_trial_ids=source_ids,
        target_mass_kg=0.03,
        required_successes=3,
    )

    assert len(plan["trials"]) == 100
    assert plan["pilot"] == {
        "kind": "targeted_mass_diagnostic_pilot",
        "required_successes": 3,
        "maximum_attempts": 5,
        "trial_ids": [f"{trial_id}_m030g" for trial_id in source_ids],
        "source_trial_ids": list(source_ids),
        "target_mass_kg": 0.03,
        "actual_mass_tolerance_kg": 1e-6,
    }
    assert [trial["source_trial_id"] for trial in plan["trials"][:5]] == list(
        source_ids
    )
    assert {
        trial["requested"]["entities"]["pick_object"]["physics"]["mass_kg"]
        for trial in plan["trials"]
    } == {0.03}
    assert {trial["resolved"]["mass_kg"] for trial in plan["trials"]} == {0.03}

    manifest = create_pilot_manifest(
        plan, collection_id=plan["plan_id"], git_commit="a" * 40
    )
    assert manifest["required_successes"] == 3
    assert manifest["maximum_attempts"] == 5
    assert next_attempt(manifest, plan)["trial_id"] == f"{source_ids[0]}_m030g"


def test_targeted_mass_pilot_rejects_unknown_and_invalid_threshold():
    with pytest.raises(ValueError, match="unknown"):
        build_targeted_mass_diagnostic_pilot_plan(
            _config(),
            pilot_id="bad",
            source_trial_ids=("missing",),
            target_mass_kg=0.03,
            required_successes=1,
        )
    with pytest.raises(ValueError, match="fit"):
        build_targeted_mass_diagnostic_pilot_plan(
            _config(),
            pilot_id="bad",
            source_trial_ids=("cube_r03_c00_s1_k1",),
            target_mass_kg=0.03,
            required_successes=2,
        )
