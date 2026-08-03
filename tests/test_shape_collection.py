from collections import Counter
from pathlib import Path

from farpoint.contracts import validate_contract, validate_shape_collection_semantics
from farpoint.shape_collection import (
    acceptance_snapshot,
    append_shape_attempt,
    build_shape_collection_manifest,
    build_shape_collection_selection,
    finish_shape_collection,
    impossible_reason,
    new_shape_collection_state,
    scheduled_shape_trials,
)
from farpoint.shape_position import generate_shape_position_plan, load_shape_position_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/variations/farpoint_v0_0_1_cylinder_position.json"
GIT_SHA = "a" * 40


def fixtures():
    plan = generate_shape_position_plan(load_shape_position_config(CONFIG))
    policy = {
        "policy_id": "cylinder",
        "display_name": "UR10e Cylinder Position Collection",
        "task_id": plan["task_id"],
        "object_shape": "cylinder",
        "position_plan_sha256": plan["plan_sha256"],
        "config_sha256": plan["config_sha256"],
        "simulator_image_digest": "sha256:" + "b" * 64,
        "simulator_payload": {"sha256": "c" * 64},
        "acceptance": {
            "required_cells": 25,
            "selected_episodes": 25,
            "selected_per_cell": 1,
            "maximum_task_attempts": 150,
            "maximum_candidates_per_cell": 6,
            "splits": {"train": 17, "validation": 4, "test": 4},
        },
    }
    state = new_shape_collection_state(
        collection_id="collection", git_commit=GIT_SHA, policy=policy, policy_sha256="d" * 64
    )
    return plan, policy, state


def audited(trial, success=True):
    return {
        "episode_id": "episode_" + trial["trial_id"],
        "success": success,
        "dataset_valid": True,
        "failure_category": None if success else "pickup",
        "failure_reason": None if success else "task_failure",
    }


def test_schedule_is_slot_round_coverage_first():
    plan, _, state = fixtures()
    rows = scheduled_shape_trials(state, plan)
    assert len(rows) == 150
    assert {row["slot"] for row in rows[:25]} == {0}
    assert {row["cell_id"] for row in rows[:25]} == {f"r{r:02d}_c{c:02d}" for r in range(5) for c in range(5)}


def test_task_failure_advances_seed_and_success_stops_cell():
    plan, policy, state = fixtures()
    first = scheduled_shape_trials(state, plan)[0]
    append_shape_attempt(state, first, audited(first, False), policy)
    rows = scheduled_shape_trials(state, plan)
    assert first["trial_id"] not in {row["trial_id"] for row in rows}
    second = next(row for row in rows if row["cell_id"] == first["cell_id"])
    assert second["slot"] == 1
    assert second["seed"] != first["seed"]
    append_shape_attempt(state, second, audited(second), policy)
    assert first["cell_id"] not in {row["cell_id"] for row in scheduled_shape_trials(state, plan)}


def test_collection_accepts_coverage_without_minimum_yield():
    plan, policy, state = fixtures()
    by_cell = {}
    for row in plan["trials"]:
        by_cell.setdefault(row["cell_id"], []).append(row)
    for cell in sorted(by_cell):
        rows = sorted(by_cell[cell], key=lambda row: row["slot"])
        for row in rows[:5]:
            append_shape_attempt(state, row, audited(row, False), policy)
        append_shape_attempt(state, rows[5], audited(rows[5]), policy)
    assert state["task_yield"] == 25 / 150
    assert acceptance_snapshot(state, policy)["accepted"] is True
    finish_shape_collection(state, policy)
    manifest = build_shape_collection_manifest(state)
    selection = build_shape_collection_selection(manifest, dataset_id="dataset")
    assert manifest["quality_status"] == "PASS"
    assert Counter(row["dataset_split"] for row in manifest["attempts"] if row["selected_for_dataset"]) == {"train": 17, "validation": 4, "test": 4}
    assert len(selection["episodes"]) == 25
    assert validate_contract(manifest) == []
    assert validate_shape_collection_semantics(manifest) == []


def test_six_failures_make_cell_impossible():
    plan, policy, state = fixtures()
    cell = "r00_c00"
    for row in sorted((row for row in plan["trials"] if row["cell_id"] == cell), key=lambda row: row["slot"]):
        append_shape_attempt(state, row, audited(row, False), policy)
    assert impossible_reason(state, plan, policy) == f"cell_candidate_quota_exhausted:{cell}"
