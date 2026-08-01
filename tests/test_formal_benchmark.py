import copy
import json
from pathlib import Path

import pytest

from farpoint.formal_benchmark import (
    CONTRACT_PILOT_TRIAL_IDS,
    FORMAL_CONFIG_SHA256,
    FORMAL_PLAN_SHA256,
    append_completed_trial,
    append_infrastructure_attempt,
    build_formal_manifest,
    build_release_selection,
    finish_run_state,
    infrastructure_retry_allowed,
    new_run_state,
    selected_trials,
    validate_formal_plan,
    validate_resume_state,
)
from farpoint.position_plan import load_position_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    PROJECT_ROOT
    / "configs"
    / "plans"
    / "farpoint_v1_3_cube_position_expanded_candidate.json"
)
GIT_COMMIT = "a" * 40
IMAGE_DIGEST = "sha256:" + "b" * 64


def frozen_plan():
    return load_position_plan(PLAN_PATH)


def completed_state(successes=75):
    plan = frozen_plan()
    state = new_run_state(
        benchmark_id="cube_position_formal_20260801_aaaaaaa",
        mode="formal",
        git_commit=GIT_COMMIT,
        image="nvcr.io/nvidia/isaac-sim:6.0.0",
        image_digest=IMAGE_DIGEST,
        plan=plan,
    )
    for index, trial in enumerate(plan["trials"]):
        success = index < successes
        append_completed_trial(
            state,
            {
                "trial_id": trial["trial_id"],
                "episode_id": f"episode-{index:04d}",
                "variation_id": trial["variation_id"],
                "split": trial["split"],
                "success": success,
                "dataset_valid": success,
            },
        )
    finish_run_state(state)
    return plan, state


def test_frozen_formal_plan_identity_and_distribution_are_exact():
    plan = frozen_plan()
    validate_formal_plan(plan)
    assert plan["plan_sha256"] == FORMAL_PLAN_SHA256
    assert plan["config_sha256"] == FORMAL_CONFIG_SHA256
    assert len(selected_trials(plan, "formal")) == 75
    assert tuple(trial["trial_id"] for trial in selected_trials(plan, "pilot")) == (
        CONTRACT_PILOT_TRIAL_IDS
    )


def test_formal_manifest_requires_all_trials_and_exact_acceptance_math():
    plan, passing = completed_state(successes=68)
    manifest = build_formal_manifest(passing, plan)
    assert manifest["acceptance"] == {
        "accepted": True,
        "required_success_rate": 0.9,
        "observed_success_rate": 68 / 75,
        "required_successes": 68,
        "observed_successes": 68,
    }

    _, failing = completed_state(successes=67)
    assert build_formal_manifest(failing, plan)["acceptance"]["accepted"] is False

    incomplete = copy.deepcopy(passing)
    incomplete["trials"].pop()
    with pytest.raises(ValueError, match="75 recorded episodes"):
        build_formal_manifest(incomplete, plan)


def test_resume_and_retry_rules_prevent_duplicate_or_seed_drift():
    plan = frozen_plan()
    state = new_run_state(
        benchmark_id="pilot",
        mode="pilot",
        git_commit=GIT_COMMIT,
        image="image",
        image_digest=IMAGE_DIGEST,
        plan=plan,
    )
    validate_resume_state(
        state,
        mode="pilot",
        git_commit=GIT_COMMIT,
        image_digest=IMAGE_DIGEST,
        plan=plan,
    )
    trial = selected_trials(plan, "pilot")[0]
    attempt = append_infrastructure_attempt(
        state, trial, attempt_number=1, run_id="run-1"
    )
    assert attempt["seed"] == trial["seed"]
    state["infrastructure_attempts"].append(
        {"trial_id": trial["trial_id"], "seed": trial["seed"] + 1}
    )
    with pytest.raises(ValueError, match="seed drift"):
        append_infrastructure_attempt(state, trial, attempt_number=2, run_id="run-2")

    completed = {"trial_id": trial["trial_id"], "success": False}
    append_completed_trial(state, completed)
    with pytest.raises(ValueError, match="already recorded"):
        append_completed_trial(state, completed)
    assert infrastructure_retry_allowed(None) is True
    assert infrastructure_retry_allowed("episode-task-failure") is False


def test_release_selection_is_successful_only_relative_and_keeps_identity():
    plan, state = completed_state(successes=68)
    manifest = build_formal_manifest(state, plan)
    selection = build_release_selection(
        manifest,
        dataset_id="farpoint_ur10e_robotiq_2f85",
    )
    assert len(selection["episodes"]) == 68
    assert all(not Path(row["episode_dir"]).is_absolute() for row in selection["episodes"])
    assert all(row["episode_dir"].startswith("outputs/episodes/") for row in selection["episodes"])
    assert all("reserve" not in row["variation_id"] for row in selection["episodes"])
    assert selection["episodes"][0]["trial_id"] == manifest["trials"][0]["trial_id"]

    rejected = copy.deepcopy(manifest)
    rejected["acceptance"]["accepted"] = False
    with pytest.raises(ValueError, match="invalid benchmark manifest"):
        build_release_selection(rejected, dataset_id="dataset")


def test_committed_plan_has_no_hidden_primary_reserve_index():
    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert {trial.get("reserve_index", 0) for trial in payload["trials"]} == {0}
