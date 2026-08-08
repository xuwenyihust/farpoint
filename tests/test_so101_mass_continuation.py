import json
from pathlib import Path

import pytest

from farpoint.object_variation import load_variation_config
from farpoint.so101_collection import (
    abort_collection_manifest,
    create_manifest,
    next_attempt,
    record_attempt,
)
from farpoint.so101_mass_collection import (
    build_mirrored_mass_collection_plan,
    load_mass_collection_config,
)
from farpoint.so101_mass_continuation import (
    build_mass_completion_report,
    build_mass_completion_selection,
    build_mass_continuation_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def full_plan():
    return build_mirrored_mass_collection_plan(
        load_variation_config(
            ROOT / "configs/variations/so101_cube_pick_place_v1.json"
        ),
        load_mass_collection_config(
            ROOT / "configs/collections/so101_cube_mass_003_v0_0_1.json"
        ),
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_success_episode(root, episode_id):
    audit = {
        "requested_mass_kg": 0.03,
        "resolved_mass_kg": 0.03,
        "physx_actual_mass_kg": 0.0300000001,
        "tolerance_kg": 1e-6,
        "verified": True,
    }
    episode = root / episode_id
    write_json(
        episode / "metadata.json",
        {"scene": {"object": {"mass_audit": audit}}},
    )
    write_json(
        episode / "metrics.json",
        {
            "success": True,
            "dataset_valid": True,
            "physics_audit": {"mass": audit},
        },
    )
    write_json(
        episode / "run-state.json", {"execution_status": "FINISHED"}
    )
    (episode / "observations.jsonl").write_text("{}\n", encoding="utf-8")


def aborted_parent(tmp_path, success_count=35):
    plan = full_plan()
    manifest = create_manifest(
        plan,
        collection_id="mass_parent",
        git_commit="a" * 40,
        maximum_attempts=150,
    )
    episodes = tmp_path / "parent"
    for _ in range(success_count):
        attempt = next_attempt(manifest, plan)
        episode_id = f"episode_parent__{attempt['attempt_id']}"
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=episode_id,
            success=True,
            dataset_valid=True,
        )
        write_success_episode(episodes, episode_id)
    abort_collection_manifest(
        manifest,
        "watchdog:stop:recent_structural_failure:bilateral_contact_lost:8/10",
    )
    return plan, manifest, episodes


def complete_continuation(tmp_path, parent_plan, parent_manifest):
    plan = build_mass_continuation_plan(
        parent_plan,
        parent_manifest,
        continuation_id="mass_continuation",
    )
    manifest = create_manifest(
        plan,
        collection_id="mass_continuation",
        git_commit="b" * 40,
        maximum_attempts=plan["collection"]["maximum_attempts"],
    )
    episodes = tmp_path / "continuation"
    while manifest["execution_status"] == "RUNNING":
        attempt = next_attempt(manifest, plan)
        episode_id = f"episode_continuation__{attempt['attempt_id']}"
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=episode_id,
            success=True,
            dataset_valid=True,
        )
        write_success_episode(episodes, episode_id)
    return plan, manifest, episodes


def test_continuation_plan_freezes_only_missing_variations_and_budget(tmp_path):
    parent_plan, parent_manifest, _episodes = aborted_parent(tmp_path)

    continuation = build_mass_continuation_plan(
        parent_plan,
        parent_manifest,
        continuation_id="mass_continuation",
    )

    assert len(continuation["trials"]) == 15
    assert continuation["collection"]["required_successes"] == 15
    assert continuation["collection"]["maximum_attempts"] == 115
    assert continuation["collection"]["parent_collection"] == {
        **continuation["collection"]["parent_collection"],
        "collection_id": "mass_parent",
        "attempted_count": 35,
        "selected_successes": 35,
    }
    assert not (
        set(parent_manifest["selected_variations"])
        & {trial["variation_id"] for trial in continuation["trials"]}
    )


def test_continuation_plan_rejects_non_aborted_parent(tmp_path):
    parent_plan, parent_manifest, _episodes = aborted_parent(tmp_path)
    parent_manifest["execution_status"] = "RUNNING"

    with pytest.raises(ValueError, match="must be ABORTED"):
        build_mass_continuation_plan(
            parent_plan,
            parent_manifest,
            continuation_id="bad",
        )


def test_completion_selection_requires_and_proves_full_balanced_coverage(
    tmp_path,
):
    parent_plan, parent_manifest, parent_root = aborted_parent(tmp_path)
    continuation_plan, continuation_manifest, continuation_root = (
        complete_continuation(tmp_path, parent_plan, parent_manifest)
    )

    manifest, selection, report = build_mass_completion_selection(
        parent_plan,
        parent_manifest,
        continuation_plan,
        continuation_manifest,
        parent_episodes_root=parent_root,
        continuation_episodes_root=continuation_root,
        collection_id="mass_completion50",
    )

    assert report["status"] == "PASS"
    assert report["selected_successes"] == 50
    assert len(report["balance"]["workspace_cells"]) == 25
    assert report["balance"]["sizes"] == {"size_0": 25, "size_1": 25}
    assert report["balance"]["colors"] == {"color_0": 25, "color_1": 25}
    assert report["balance"]["splits"] == {
        "test": 5,
        "train": 40,
        "validation": 5,
    }
    assert manifest["execution_status"] == "FINISHED"
    assert manifest["quality_status"] == "PASS"
    assert len(manifest["selected_variations"]) == 50
    assert len(selection["episodes"]) == 50
    assert len({row["episode_dir"] for row in selection["episodes"]}) == 50


def test_completion_detects_parent_evidence_changed_after_plan_freeze(
    tmp_path,
):
    parent_plan, parent_manifest, parent_root = aborted_parent(tmp_path)
    continuation_plan, continuation_manifest, continuation_root = (
        complete_continuation(tmp_path, parent_plan, parent_manifest)
    )
    parent_manifest["abort_reason"] = "modified"

    report = build_mass_completion_report(
        parent_plan,
        parent_manifest,
        continuation_plan,
        continuation_manifest,
        parent_episodes_root=parent_root,
        continuation_episodes_root=continuation_root,
    )

    assert report["status"] == "INVALID_EVIDENCE"
    assert "continuation_parent_manifest_hash_mismatch" in report[
        "evidence_errors"
    ]
