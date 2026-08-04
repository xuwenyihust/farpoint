from __future__ import annotations

from pathlib import Path

from farpoint.contracts import validate_contract, validate_yaw_collection_semantics
from farpoint.yaw_collection import (
    acceptance_snapshot,
    append_attempt,
    finish_collection,
    load_plan_for_policy,
    new_collection_state,
    scheduled_trials,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/collections/farpoint_v0_0_1_cube_yaw_aware.json"
DIGEST = "sha256:" + "0" * 64


def policy_and_plan():
    return load_plan_for_policy(POLICY)


def audited():
    return {
        "success": True,
        "dataset_valid": True,
        "yaw_aware": {
            "control_source": "rgbd_cube_yaw",
            "alignment_stable": True,
            "audit_error_degrees": 1.0,
        },
        "object_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "object_spec": {"variant": "cube_55mm_red_v1"},
        "failure_category": None,
        "failure_reason": None,
    }


def test_schedule_covers_100_unique_yaw_conditions_with_reserves():
    policy, plan = policy_and_plan()
    state = new_collection_state("yaw_test", "a" * 40, policy, plan, DIGEST)
    candidates = scheduled_trials(state, plan)
    assert len(candidates) == 100
    assert {row["condition_id"] for row in candidates} == {row["trial_id"] for row in plan["trials"]}
    first = candidates[0]
    append_attempt(state, first, "episode_0000", {**audited(), "success": False}, policy)
    retry = next(row for row in scheduled_trials(state, plan) if row["trial_id"] == first["trial_id"])
    assert retry["reserve_index"] == 1


def test_v2_acceptance_requires_all_conditions_and_fixed_splits():
    policy, plan = policy_and_plan()
    state = new_collection_state("yaw_test", "a" * 40, policy, plan, DIGEST)
    for index, trial in enumerate(scheduled_trials(state, plan)):
        append_attempt(state, trial, f"episode_{index:04d}", audited(), policy)
    snapshot = acceptance_snapshot(state, policy)
    assert snapshot["accepted"] is True
    assert snapshot["observed_splits"] == {"train": 68, "validation": 16, "test": 16}
    finish_collection(state, policy)
    from farpoint.yaw_collection import build_manifest
    manifest = build_manifest(state, policy, "b" * 64)
    assert validate_contract(manifest) == []
    assert validate_yaw_collection_semantics(manifest) == []


def test_yaw_telemetry_is_required_for_selected_attempts():
    policy, plan = policy_and_plan()
    state = new_collection_state("yaw_test", "a" * 40, policy, plan, DIGEST)
    trial = scheduled_trials(state, plan)[0]
    invalid = audited()
    invalid["yaw_aware"] = {"control_source": "task_ground_truth", "alignment_stable": True, "audit_error_degrees": 0.0}
    append_attempt(state, trial, "episode_0000", invalid, policy)
    finish_collection(state, policy)
    from farpoint.yaw_collection import build_manifest
    try:
        build_manifest(state, policy, "b" * 64)
    except ValueError as error:
        assert "yaw-aware control" in str(error)
    else:
        raise AssertionError("invalid control source was accepted")
