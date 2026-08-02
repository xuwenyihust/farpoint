import copy
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from farpoint.contracts import validate_collection_semantics, validate_contract
from farpoint.position_collection import (
    acceptance_snapshot,
    append_new_attempt,
    build_collection_manifest,
    build_collection_selection,
    complete_import_pilot,
    dataset_split,
    finish_collection,
    import_source_attempts,
    impossible_reason,
    load_collection_policy,
    new_collection_state,
    scheduled_trials,
    validate_collection_policy,
    validate_resume_state,
)
from farpoint.position_plan import load_position_plan


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_position_benchmark  # noqa: E402


POLICY_PATH = (
    ROOT / "configs/collections/farpoint_v1_3_cube_position_balanced.json"
)
PLAN_PATH = ROOT / "configs/plans/farpoint_v1_3_cube_position_expanded_candidate.json"
GIT_COMMIT = "f" * 40


def policy_and_plan():
    policy = load_collection_policy(POLICY_PATH)
    plan = load_position_plan(PLAN_PATH)
    validate_collection_policy(policy, plan, ROOT)
    return policy, plan


def source_state():
    policy, plan = policy_and_plan()
    failures = {
        "primary_r00_c02_s00",
        "primary_r00_c04_s00",
        "primary_r01_c00_s02",
        "primary_r01_c01_s01",
    }
    trials = []
    for index, trial in enumerate(plan["trials"][:27]):
        success = trial["trial_id"] not in failures
        trials.append(
            {
                **trial,
                "episode_id": f"episode-{index:03d}",
                "success": success,
                "dataset_valid": True,
                "accepted": success,
                "checks": {"task": success, "dataset": True},
                "failure_category": None if success else "pickup",
                "failure_reason": None if success else "task_failure",
            }
        )
    return {
        "benchmark_id": policy["source"]["run_id"],
        "git_commit": policy["source"]["git_commit"],
        "execution_status": "ABORTED",
        "completed_trials": 27,
        "passed_trials": 23,
        "position_plan_sha256": policy["position_plan_sha256"],
        "config_sha256": policy["config_sha256"],
        "image_digest": policy["simulator_image_digest"],
        "trials": trials,
    }


def imported_state():
    policy, plan = policy_and_plan()
    imported = import_source_attempts(source_state(), policy, plan)
    state = new_collection_state(
        collection_id="collection",
        git_commit=GIT_COMMIT,
        policy=policy,
        policy_sha256="a" * 64,
        imported_attempts=imported,
    )
    return policy, plan, imported, state


def audited(trial, success=True):
    return {
        "episode_id": "episode-" + trial["trial_id"],
        "success": success,
        "dataset_valid": True,
        "accepted": success,
        "failure_category": None if success else "pickup",
        "failure_reason": None if success else "task_failure",
    }


def test_frozen_policy_payload_and_source_import_are_exact():
    policy, plan = policy_and_plan()
    attempts = import_source_attempts(source_state(), policy, plan)
    selected = [row for row in attempts if row["selected_for_dataset"]]

    assert len(attempts) == 27
    assert sum(row["outcome_success"] for row in attempts) == 23
    assert len(selected) == 18
    assert Counter(row["cell_id"] for row in selected) == {
        f"r{row:02d}_c{column:02d}": 2
        for row, columns in ((0, range(5)), (1, range(4)))
        for column in columns
    }


def test_source_import_refuses_identity_or_gate_drift():
    policy, plan = policy_and_plan()
    changed = source_state()
    changed["image_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="source run identity mismatch"):
        import_source_attempts(changed, policy, plan)

    changed = source_state()
    successful = next(row for row in changed["trials"] if row["success"])
    successful["checks"]["dataset"] = False
    with pytest.raises(ValueError, match="failed strict gates"):
        import_source_attempts(changed, policy, plan)

    changed = source_state()
    changed["trials"][0]["seed"] += 1
    with pytest.raises(ValueError, match="source trial seed mismatch"):
        import_source_attempts(changed, policy, plan)

    changed = source_state()
    changed["trials"][1]["trial_id"] = changed["trials"][0]["trial_id"]
    with pytest.raises(ValueError, match="trial ids must be unique"):
        import_source_attempts(changed, policy, plan)


def test_schedule_is_coverage_first_and_skips_saturated_imported_cells():
    policy, plan, _, state = imported_state()
    scheduled = scheduled_trials(state, plan, policy)

    assert len(scheduled) == 48
    assert {row["cell_id"] for row in scheduled[:16]} == {
        "r01_c04",
        *(f"r{row:02d}_c{column:02d}" for row in range(2, 5) for column in range(5)),
    }
    assert {row["slot"] for row in scheduled[:16]} == {0}
    assert {row["slot"] for row in scheduled[16:32]} == {1}
    assert {row["slot"] for row in scheduled[32:]} == {2}


def test_balanced_collection_accepts_exactly_fifty_with_34_8_8_splits():
    policy, plan, _, state = imported_state()
    remaining_cells = sorted(
        {trial["cell_id"] for trial in plan["trials"]}
        - set(state["selected_per_cell"])
    )
    by_cell = {
        cell: sorted(
            [trial for trial in plan["trials"] if trial["cell_id"] == cell],
            key=lambda trial: trial["slot"],
        )
        for cell in remaining_cells
    }
    for cell in remaining_cells:
        for trial in by_cell[cell][:2]:
            append_new_attempt(state, trial, audited(trial), policy)

    finish_collection(state, policy)
    manifest = build_collection_manifest(state, policy)
    selection = build_collection_selection(manifest, dataset_id="dataset")

    assert state["task_attempts"] == 59
    assert state["task_successes"] == 55
    assert state["selected_episodes"] == 50
    assert state["selected_splits"] == {"train": 34, "validation": 8, "test": 8}
    assert set(state["selected_per_cell"].values()) == {2}
    assert acceptance_snapshot(state, policy)["accepted"] is True
    assert validate_contract(manifest) == []
    assert validate_collection_semantics(manifest) == []
    assert len(selection["episodes"]) == 50
    assert selection["collection_id"] == "collection"


def test_failures_remain_in_yield_and_task_failure_advances_candidate():
    policy, plan, _, state = imported_state()
    trial = next(row for row in plan["trials"] if row["cell_id"] == "r01_c04")
    append_new_attempt(state, trial, audited(trial, success=False), policy)

    assert state["task_attempts"] == 28
    assert state["task_successes"] == 23
    assert state["task_yield"] == 23 / 28
    assert trial["trial_id"] not in {row["trial_id"] for row in scheduled_trials(state, plan, policy)}


def test_cell_quota_exhaustion_is_detected_before_wasting_last_candidate():
    policy, plan, _, state = imported_state()
    trials = sorted(
        [row for row in plan["trials"] if row["cell_id"] == "r01_c04"],
        key=lambda row: row["slot"],
    )
    append_new_attempt(state, trials[0], audited(trials[0], success=False), policy)
    append_new_attempt(state, trials[1], audited(trials[1], success=False), policy)

    assert impossible_reason(state, plan, policy) == "cell_candidate_quota_exhausted:r01_c04"


def test_attempt_cap_stops_before_a_seventy_fourth_task_attempt():
    policy, plan, _, state = imported_state()
    state["task_attempts"] = policy["acceptance"]["maximum_task_attempts"]
    state["acceptance"] = acceptance_snapshot(state, policy)

    assert impossible_reason(state, plan, policy) == "maximum_task_attempts_reached"


def test_resume_requires_the_same_collection_commit_and_policy():
    _, _, _, state = imported_state()
    validate_resume_state(
        state,
        collection_id="collection",
        git_commit=GIT_COMMIT,
        policy_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="git_commit, policy_sha256"):
        validate_resume_state(
            state,
            collection_id="collection",
            git_commit="0" * 40,
            policy_sha256="b" * 64,
        )


def test_import_pilot_finishes_without_claiming_collection_acceptance():
    policy, _, _, state = imported_state()

    complete_import_pilot(state, policy)

    assert state["execution_status"] == "PILOT_COMPLETE"
    assert state["quality_status"] == "PASS"
    assert state["accepted"] is False
    assert state["acceptance"]["accepted"] is False


def test_episode_audit_uses_the_explicit_source_episode_root(tmp_path, monkeypatch):
    policy, plan = policy_and_plan()
    trial = plan["trials"][0]
    source_root = tmp_path / "source-episodes"
    episode = source_root / "episode"
    episode.mkdir(parents=True)
    (episode / "metadata.json").write_text("{}", encoding="utf-8")
    (episode / "metrics.json").write_text(
        json.dumps({"dataset_valid": True}), encoding="utf-8"
    )
    observed = {}

    def fake_pilot_audit(_episode, _trial, *, plan_sha256, episode_root):
        observed["episode_root"] = episode_root
        return {"checks": {}, "errors": [], "episode_id": "episode"}

    monkeypatch.setattr(
        run_position_benchmark, "audit_pilot_episode", fake_pilot_audit
    )
    monkeypatch.setattr(
        run_position_benchmark,
        "normalize_episode_metadata_v2",
        lambda *_args, **_kwargs: {
            "provenance": {
                "git_commit": policy["source"]["git_commit"],
                "config_sha256": policy["config_sha256"],
                "simulator_image_digest": policy["simulator_image_digest"],
            },
            "variation": {
                "variation_id": trial["variation_id"],
                "cell_id": trial["cell_id"],
                "slot": trial["slot"],
            },
        },
    )

    result = run_position_benchmark.audit_episode(
        episode,
        trial,
        plan=plan,
        git_commit=policy["source"]["git_commit"],
        simulator_image_digest=policy["simulator_image_digest"],
        dataset_episode_index=0,
        episode_root=source_root,
    )

    assert result["success"] is True
    assert observed["episode_root"] == source_root


def test_split_assignment_is_stable_by_cell_and_success_rank():
    counts = Counter()
    for row in range(5):
        for column in range(5):
            cell = f"r{row:02d}_c{column:02d}"
            counts[dataset_split(cell, 1)] += 1
            counts[dataset_split(cell, 2)] += 1
    assert counts == {"train": 34, "validation": 8, "test": 8}


def test_collection_semantics_detect_tampered_acceptance():
    policy, plan, _, state = imported_state()
    state["execution_status"] = "FINISHED"
    state["quality_status"] = "FAIL"
    manifest = {
        "schema_version": "farpoint.collection.v1",
        "collection_id": state["collection_id"],
        "task_id": state["task_id"],
        "git_commit": state["git_commit"],
        "policy_id": state["policy_id"],
        "policy_sha256": state["policy_sha256"],
        "position_plan_sha256": state["position_plan_sha256"],
        "config_sha256": state["config_sha256"],
        "simulator_image_digest": state["simulator_image_digest"],
        "simulator_payload_sha256": state["simulator_payload_sha256"],
        "execution_status": "FINISHED",
        "quality_status": "FAIL",
        "failure_reason": "incomplete",
        "attempts": copy.deepcopy(state["attempts"]),
        "acceptance": acceptance_snapshot(state, policy),
    }
    manifest["acceptance"]["observed_task_attempts"] = 1
    assert any(
        "observed_task_attempts" in error
        for error in validate_collection_semantics(manifest)
    )
