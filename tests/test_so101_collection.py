from pathlib import Path

import pytest

from farpoint.object_variation import generate_variation_plan, load_variation_config
from farpoint.so101_collection import (
    build_attempt_run_state,
    build_export_selection,
    create_manifest,
    episode_id_for_attempt,
    load_manifest,
    next_attempt,
    record_attempt,
    write_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_episode_id_is_namespaced_by_collection():
    first = episode_id_for_attempt("pilot_v1", "cube_r00_c00__attempt00")
    second = episode_id_for_attempt("pilot_v2", "cube_r00_c00__attempt00")

    assert first == "episode_pilot_v1__cube_r00_c00__attempt00"
    assert second == "episode_pilot_v2__cube_r00_c00__attempt00"
    assert first != second


@pytest.mark.parametrize(
    ("collection_id", "attempt_id"),
    [("../pilot", "attempt00"), ("pilot", "attempt/00"), ("", "attempt00")],
)
def test_episode_id_rejects_unsafe_identifiers(collection_id, attempt_id):
    with pytest.raises(ValueError, match="must contain only"):
        episode_id_for_attempt(collection_id, attempt_id)


def test_attempt_run_state_preserves_live_collection_and_variation_context():
    variation_plan = plan()
    attempt = next_attempt(
        create_manifest(
            variation_plan, collection_id="pilot_live", git_commit="a" * 40
        ),
        variation_plan,
    )

    state = build_attempt_run_state(
        attempt, collection_id="pilot_live", git_commit="b" * 40
    )

    assert state["schema_version"] == "farpoint.episode-run.v1"
    assert state["execution_status"] == "RUNNING"
    assert state["identity"]["episode_id"].startswith("episode_pilot_live__")
    assert state["identity"]["split"] == attempt["split"]
    assert state["provenance"]["collection_id"] == "pilot_live"
    assert state["variation"]["requested"] == attempt["requested"]
    assert state["variation"]["resolved"] == attempt["resolved"]
    assert state["recording"]["cameras"] == ["observation.images.front"]


def plan():
    return generate_variation_plan(
        load_variation_config(ROOT / "configs/variations/so101_cube_pick_place_v1.json")
    )


def test_collection_retries_failure_without_changing_split(tmp_path):
    variation_plan = plan()
    manifest = create_manifest(
        variation_plan, collection_id="pilot_1", git_commit="a" * 40
    )
    first = next_attempt(manifest, variation_plan)
    assert first["varied_axes"] == variation_plan["varied_axes"]
    assert first["frozen_axes"] == variation_plan["frozen_axes"]
    record_attempt(
        manifest,
        variation_plan,
        first,
        episode_id="episode_failed",
        success=False,
        dataset_valid=True,
        failure_category="task",
        failure_reason="missed_grasp",
    )
    second = next_attempt(manifest, variation_plan)
    assert second["variation_id"] != first["variation_id"]
    assert second["attempt_index"] == 0

    for trial in variation_plan["trials"][1:]:
        attempt = next_attempt(manifest, variation_plan)
        record_attempt(
            manifest,
            variation_plan,
            attempt,
            episode_id=f"episode_{attempt['attempt_id']}",
            success=True,
            dataset_valid=True,
        )
    retry = next_attempt(manifest, variation_plan)
    assert retry["variation_id"] == first["variation_id"]
    assert retry["attempt_index"] == 1
    assert retry["split"] == first["split"]

    record_attempt(
        manifest,
        variation_plan,
        retry,
        episode_id="episode_retry",
        success=True,
        dataset_valid=True,
    )
    assert manifest["quality_status"] == "PASS"
    selection = build_export_selection(manifest)
    assert len(selection["episodes"]) == 100
    assert all("episode_failed" not in row["episode_dir"] for row in selection["episodes"])

    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    assert load_manifest(path, variation_plan) == manifest


def test_collection_rejects_attempt_budget_below_target():
    with pytest.raises(ValueError, match="maximum_attempts"):
        create_manifest(plan(), collection_id="bad", git_commit="a" * 40, maximum_attempts=99)


def test_collection_preserves_frozen_plan_order_within_retry_round():
    variation_plan = plan()
    variation_plan["trials"][0], variation_plan["trials"][1] = (
        variation_plan["trials"][1],
        variation_plan["trials"][0],
    )
    manifest = create_manifest(
        variation_plan,
        collection_id="ordered_pilot",
        git_commit="a" * 40,
    )

    attempt = next_attempt(manifest, variation_plan)

    assert attempt["variation_id"] == variation_plan["trials"][0]["variation_id"]


def test_collection_attempt_seed_is_deterministic_uint32_for_isaac_lab():
    variation_plan = plan()
    first = next_attempt(
        create_manifest(variation_plan, collection_id="seed_a", git_commit="a" * 40),
        variation_plan,
    )
    second = next_attempt(
        create_manifest(variation_plan, collection_id="seed_b", git_commit="b" * 40),
        variation_plan,
    )

    assert first["attempt_seed"] == second["attempt_seed"]
    assert 0 <= first["attempt_seed"] <= 2**32 - 1
