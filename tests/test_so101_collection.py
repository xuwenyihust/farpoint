from pathlib import Path

import pytest

from farpoint.object_variation import generate_variation_plan, load_variation_config
from farpoint.so101_collection import (
    build_export_selection,
    create_manifest,
    load_manifest,
    next_attempt,
    record_attempt,
    write_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


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
