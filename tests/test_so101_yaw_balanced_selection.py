import copy
import hashlib
import json

import pytest

from farpoint.so101_yaw_balanced_selection import (
    COLUMN_TARGET,
    MASS_COLOR_TARGET,
    MISSING_CELLS,
    ROW_TARGET,
    SELECTION_SEED,
    SPLIT_TARGET,
    build_artifacts,
    select_balanced30,
    selection_stats,
    validate_aborted_source,
    validate_balance,
)


def _trial(row, column, mass, color, split):
    cell = f"r{row:02d}_c{column:02d}"
    mass_grams = int(round(mass * 1000))
    trial_id = f"cube_{cell}_s0_k{color}_yaw00000_m{mass_grams:03d}g"
    return {
        "trial_id": trial_id,
        "variation_id": trial_id,
        "cell_id": cell,
        "split": split,
        "object_yaw_degrees": 0.0,
        "seed_material": {"size_index": 0, "color_index": color},
        "resolved": {"mass_kg": mass},
    }


def _source():
    trials = []
    for row in range(5):
        for column in range(5):
            for color in range(2):
                mass = 0.03 if (row + column + color) % 2 == 0 else 0.04
                split = "train"
                if row == 4 and column == 2 and color == 0:
                    split = "validation"
                if (row, column, color) in {
                    (0, 0, 0),
                    (0, 1, 0),
                    (4, 2, 1),
                    (4, 3, 1),
                    (4, 4, 1),
                }:
                    split = "test"
                trials.append(_trial(row, column, mass, color, split))
    plan = {
        "plan_id": "yaw-plan",
        "plan_sha256": "a" * 64,
        "collection": {
            "kind": "balanced_yaw_success_collection",
            "yaw_degrees": 0.0,
            "cube_size_m": 0.03,
        },
        "trials": trials,
    }
    eligible_trials = [
        trial for trial in trials if trial["cell_id"] not in MISSING_CELLS
    ]
    attempts = []
    for index, trial in enumerate(eligible_trials):
        attempts.append(
            {
                "attempt_id": f"{trial['trial_id']}__attempt00",
                "trial_id": trial["trial_id"],
                "variation_id": trial["variation_id"],
                "episode_id": f"episode_{trial['trial_id']}",
                "split": trial["split"],
                "success": True,
                "dataset_valid": True,
                "finished_at": f"2026-08-08T00:{index:02d}:00+00:00",
            }
        )
    manifest = {
        "collection_id": "aborted-yaw",
        "task_id": "so101_cube_pick_place",
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "execution_status": "ABORTED",
        "quality_status": "NOT_EVALUATED",
        "abort_reason": "owner_requested_stop_after_balanced30_feasibility",
        "attempts": attempts,
        "selected_variations": {
            row["variation_id"]: row["attempt_id"] for row in attempts
        },
    }
    abort = {
        "collection_id": manifest["collection_id"],
        "execution_status": "ABORTED",
        "reason": manifest["abort_reason"],
        "completed_attempt_count": len(attempts),
        "selected_variation_count": len(attempts),
    }
    return plan, manifest, abort


def test_source_contract_requires_matching_aborted_lineage():
    plan, manifest, abort = _source()
    assert validate_aborted_source(manifest, plan, abort) == []
    changed = copy.deepcopy(manifest)
    changed["execution_status"] = "RUNNING"
    assert "source_execution_status_not_aborted" in validate_aborted_source(
        changed, plan, abort
    )


def test_balanced30_is_deterministic_and_satisfies_frozen_contract():
    plan, manifest, _ = _source()
    first, stats = select_balanced30(manifest, plan, seed=SELECTION_SEED)
    second, repeated_stats = select_balanced30(manifest, plan, seed=SELECTION_SEED)
    assert [row["attempt_id"] for row in first] == [
        row["attempt_id"] for row in second
    ]
    assert stats == repeated_stats
    assert validate_balance(stats) == []
    assert stats["splits"] == SPLIT_TARGET
    assert stats["workspace_rows"] == ROW_TARGET
    assert stats["workspace_columns"] == COLUMN_TARGET
    assert stats["mass_color"] == MASS_COLOR_TARGET
    assert stats["missing_cells"] == sorted(MISSING_CELLS)


def test_balance_rejects_split_relabeling():
    plan, manifest, _ = _source()
    selected, _ = select_balanced30(manifest, plan)
    selected[0]["split"] = "validation"
    errors = validate_balance(selection_stats(selected))
    assert any(error.startswith("splits_mismatch") for error in errors)


def test_selection_rejects_insufficient_frozen_split():
    plan, manifest, _ = _source()
    manifest["attempts"] = [
        row for row in manifest["attempts"] if row["split"] != "validation"
    ]
    with pytest.raises(ValueError, match="split validation"):
        select_balanced30(manifest, plan)


def test_artifacts_bind_source_files_and_preserve_episode_identity(tmp_path):
    plan, manifest, abort = _source()
    selected, stats = select_balanced30(manifest, plan)
    source_digest = hashlib.sha256(b"source manifest bytes").hexdigest()
    abort_digest = hashlib.sha256(b"abort record bytes").hexdigest()
    candidate, export = build_artifacts(
        manifest,
        plan,
        abort,
        selected,
        stats,
        collection_id="balanced30",
        dataset_id="farpoint_so101",
        episodes_root=tmp_path,
        git_commit="b" * 40,
        source_manifest_file_sha256=source_digest,
        abort_record_file_sha256=abort_digest,
    )
    assert candidate["source_collection"]["manifest_sha256"] == source_digest
    assert candidate["source_collection"]["abort_record_sha256"] == abort_digest
    assert candidate["execution_status"] == "FINISHED"
    assert candidate["quality_status"] == "PASS"
    assert len(candidate["attempts"]) == 30
    assert export["episodes"][0]["source_attempt_id"] == selected[0]["attempt_id"]
    assert export["episodes"][0]["episode_dir"] == str(
        (tmp_path / selected[0]["episode_id"]).resolve()
    )


def test_stats_are_json_serializable():
    plan, manifest, _ = _source()
    _, stats = select_balanced30(manifest, plan)
    json.dumps(stats)
