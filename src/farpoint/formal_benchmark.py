"""Release-grade cube position benchmark contracts and state transitions."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from farpoint.contracts import validate_benchmark_semantics, validate_contract
from farpoint.position_plan import validate_position_plan


FORMAL_PLAN_SHA256 = "f13bb891d6044145a0e2c5b65982f91810298f0a8387328cc933fb51bd0da8db"
FORMAL_CONFIG_SHA256 = "597ddb66cfb45d86f0c9dfab5340eb5dd099924bec51bf54a5fcf5ef5a9e7412"
FORMAL_TRIAL_COUNT = 75
FORMAL_REQUIRED_SUCCESS_RATE = 0.90
FORMAL_REQUIRED_SUCCESSES = 68
FORMAL_SPLITS = {"train": 50, "validation": 13, "test": 12}
CONTRACT_PILOT_TRIAL_IDS = (
    "primary_r00_c00_s00",
    "primary_r02_c02_s00",
    "primary_r04_c04_s00",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_formal_plan(plan: dict[str, Any]) -> None:
    validate_position_plan(plan)
    if plan.get("plan_sha256") != FORMAL_PLAN_SHA256:
        raise ValueError("formal benchmark plan SHA does not match the frozen v1.3 candidate")
    if plan.get("config_sha256") != FORMAL_CONFIG_SHA256:
        raise ValueError("formal benchmark config SHA does not match the frozen v1.3 candidate")
    if len(plan["trials"]) != FORMAL_TRIAL_COUNT:
        raise ValueError("formal benchmark must contain exactly 75 primary trials")
    if Counter(trial["split"] for trial in plan["trials"]) != FORMAL_SPLITS:
        raise ValueError("formal benchmark split counts must be 50/13/12")
    for trial in plan["trials"]:
        if trial.get("reserve_index", 0) != 0 or "reserve" in trial["variation_id"]:
            raise ValueError("formal benchmark may use primary trials only")


def selected_trials(
    plan: dict[str, Any],
    mode: str,
    pilot_trial_ids: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    validate_formal_plan(plan)
    if mode == "formal":
        if pilot_trial_ids:
            raise ValueError("formal mode cannot filter trials")
        return list(plan["trials"])
    if mode != "pilot":
        raise ValueError("mode must be pilot or formal")
    by_id = {trial["trial_id"]: trial for trial in plan["trials"]}
    requested_ids = tuple(pilot_trial_ids or CONTRACT_PILOT_TRIAL_IDS)
    if not requested_ids:
        raise ValueError("pilot trial selection cannot be empty")
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("pilot trial selection contains duplicate IDs")
    unknown = [trial_id for trial_id in requested_ids if trial_id not in by_id]
    if unknown:
        raise ValueError("unknown pilot trial IDs: " + ", ".join(unknown))
    return [by_id[trial_id] for trial_id in requested_ids]


def new_run_state(
    *,
    benchmark_id: str,
    mode: str,
    git_commit: str,
    image: str,
    image_digest: str,
    plan: dict[str, Any],
    pilot_trial_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    trials = selected_trials(plan, mode, pilot_trial_ids)
    required_rate = FORMAL_REQUIRED_SUCCESS_RATE if mode == "formal" else 1.0
    required_successes = (
        FORMAL_REQUIRED_SUCCESSES if mode == "formal" else len(trials)
    )
    return {
        "schema_version": "farpoint.benchmark-run.v1",
        "benchmark_id": benchmark_id,
        "mode": mode,
        "task_id": plan["task_id"],
        "task_name": plan["task_id"],
        "task_type": "cube_position_formal" if mode == "formal" else "cube_position_contract_pilot",
        "execution_status": "RUNNING",
        "quality_status": "NOT_EVALUATED",
        "release_status": "CANDIDATE" if mode == "formal" else "PILOT",
        "git_commit": git_commit,
        "config_sha256": plan["config_sha256"],
        "position_plan_id": plan["plan_id"],
        "position_plan_sha256": plan["plan_sha256"],
        "image": image,
        "image_digest": image_digest,
        "created_at": utc_now(),
        "planned_trials": len(trials),
        "planned_trial_ids": [trial["trial_id"] for trial in trials],
        "completed_trials": 0,
        "passed_trials": 0,
        "success_rate": 0.0,
        "accepted": False,
        "acceptance": {
            "min_success_rate": required_rate,
            "required_success_rate": required_rate,
            "required_successes": required_successes,
            "contact_only": True,
            "max_perception_xy_error": 0.02,
            "min_object_lift_height": 0.15,
            "min_bilateral_contact_frames": 20,
            "min_transport_contact_frames": 120,
            "max_final_target_xy_distance": 0.05,
            "min_release_settle_frames": 120,
            "require_dataset": True,
        },
        "infrastructure_attempts": [],
        "trials": [],
    }


def validate_resume_state(
    state: dict[str, Any],
    *,
    mode: str,
    git_commit: str,
    image_digest: str,
    plan: dict[str, Any],
    pilot_trial_ids: list[str] | tuple[str, ...] | None = None,
) -> None:
    expected = {
        "mode": mode,
        "git_commit": git_commit,
        "config_sha256": plan["config_sha256"],
        "position_plan_sha256": plan["plan_sha256"],
        "image_digest": image_digest,
    }
    mismatches = [key for key, value in expected.items() if state.get(key) != value]
    if mismatches:
        raise ValueError("resume state identity mismatch: " + ", ".join(mismatches))
    expected_ids = [
        trial["trial_id"]
        for trial in selected_trials(plan, mode, pilot_trial_ids)
    ]
    actual_ids = state.get("planned_trial_ids")
    if actual_ids is not None and actual_ids != expected_ids:
        raise ValueError("resume state planned trial IDs do not match")
    if state.get("planned_trials") != len(expected_ids):
        raise ValueError("resume state planned trial count does not match")


def update_run_progress(state: dict[str, Any]) -> None:
    completed = len(state.get("trials", []))
    passed = sum(trial.get("success") is True for trial in state.get("trials", []))
    state["completed_trials"] = completed
    state["passed_trials"] = passed
    state["success_rate"] = passed / completed if completed else 0.0
    state["updated_at"] = utc_now()


def append_completed_trial(state: dict[str, Any], trial: dict[str, Any]) -> None:
    trial_id = trial.get("trial_id")
    if not trial_id:
        raise ValueError("completed benchmark trial must define trial_id")
    if any(existing.get("trial_id") == trial_id for existing in state.get("trials", [])):
        raise ValueError(f"benchmark trial is already recorded: {trial_id}")
    state.setdefault("trials", []).append(trial)
    update_run_progress(state)


def append_infrastructure_attempt(
    state: dict[str, Any],
    trial: dict[str, Any],
    *,
    attempt_number: int,
    run_id: str,
) -> dict[str, Any]:
    if attempt_number < 1:
        raise ValueError("infrastructure attempt number must be positive")
    previous = [
        attempt
        for attempt in state.get("infrastructure_attempts", [])
        if attempt.get("trial_id") == trial["trial_id"]
    ]
    if any(attempt.get("seed") != trial["seed"] for attempt in previous):
        raise ValueError("infrastructure retry seed drift detected")
    attempt = {
        "trial_id": trial["trial_id"],
        "seed": trial["seed"],
        "attempt": attempt_number,
        "run_id": run_id,
        "started_at": utc_now(),
    }
    state.setdefault("infrastructure_attempts", []).append(attempt)
    return attempt


def infrastructure_retry_allowed(episode_id: str | None) -> bool:
    """Task failures produce an episode and therefore can never be retried."""
    return episode_id is None


def finish_run_state(state: dict[str, Any]) -> None:
    update_run_progress(state)
    required = state["acceptance"]["required_successes"]
    required_rate = state["acceptance"]["required_success_rate"]
    complete = state["completed_trials"] == state["planned_trials"]
    accepted = (
        complete
        and state["passed_trials"] >= required
        and state["success_rate"] >= required_rate
    )
    state["execution_status"] = "FINISHED"
    state["quality_status"] = "PASS" if accepted else "FAIL"
    state["accepted"] = accepted
    state["finished_at"] = utc_now()


def abort_run_state(state: dict[str, Any], reason: str) -> None:
    update_run_progress(state)
    state["execution_status"] = "ABORTED"
    state["quality_status"] = "NOT_EVALUATED"
    state["accepted"] = False
    state["failure_reason"] = reason
    state["finished_at"] = utc_now()


def build_formal_manifest(
    state: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    validate_formal_plan(plan)
    if state.get("mode") != "formal":
        raise ValueError("only a formal run can produce a formal benchmark manifest")
    if state.get("execution_status") != "FINISHED":
        raise ValueError("formal benchmark must finish before its manifest is created")
    if len(state.get("trials", [])) != FORMAL_TRIAL_COUNT:
        raise ValueError("formal benchmark manifest requires all 75 recorded episodes")
    expected_ids = {trial["trial_id"] for trial in plan["trials"]}
    actual_ids = {trial["trial_id"] for trial in state["trials"]}
    if actual_ids != expected_ids:
        raise ValueError("formal benchmark trial identities do not match the frozen plan")
    trials = [
        {
            "trial_id": trial["trial_id"],
            "episode_id": trial["episode_id"],
            "variation_id": trial["variation_id"],
            "split": trial["split"],
            "status": "completed",
            "success": bool(trial["success"]),
            "dataset_valid": bool(trial["dataset_valid"]),
        }
        for trial in sorted(state["trials"], key=lambda item: item["trial_id"])
    ]
    observed_successes = sum(trial["success"] for trial in trials)
    observed_rate = observed_successes / FORMAL_TRIAL_COUNT
    accepted = (
        observed_successes >= FORMAL_REQUIRED_SUCCESSES
        and observed_rate >= FORMAL_REQUIRED_SUCCESS_RATE
    )
    manifest = {
        "schema_version": "farpoint.benchmark.v2",
        "benchmark_id": state["benchmark_id"],
        "task_id": plan["task_id"],
        "git_commit": state["git_commit"],
        "config_sha256": plan["config_sha256"],
        "simulator_image_digest": state["image_digest"],
        "trials": trials,
        "acceptance": {
            "accepted": accepted,
            "required_success_rate": FORMAL_REQUIRED_SUCCESS_RATE,
            "observed_success_rate": observed_rate,
            "required_successes": FORMAL_REQUIRED_SUCCESSES,
            "observed_successes": observed_successes,
        },
    }
    errors = validate_contract(manifest) + validate_benchmark_semantics(manifest)
    if errors:
        raise ValueError("invalid formal benchmark manifest: " + "; ".join(errors))
    return manifest


def build_release_selection(
    benchmark: dict[str, Any], *, dataset_id: str, episode_root: str = "outputs/episodes"
) -> dict[str, Any]:
    errors = validate_contract(benchmark) + validate_benchmark_semantics(benchmark)
    if errors:
        raise ValueError("invalid benchmark manifest: " + "; ".join(errors))
    if benchmark["acceptance"]["accepted"] is not True:
        raise ValueError("release selection requires an accepted benchmark")
    root = Path(episode_root)
    if root.is_absolute():
        raise ValueError("selection episode root must be repository-relative")
    selected = []
    for trial in sorted(benchmark["trials"], key=lambda item: item["trial_id"]):
        if trial["success"] is not True or trial["dataset_valid"] is not True:
            continue
        if trial["split"] == "reserve" or "reserve" in trial["variation_id"]:
            raise ValueError("release selection may not contain reserve trials")
        selected.append(
            {
                "episode_dir": (root / trial["episode_id"]).as_posix(),
                "trial_id": trial["trial_id"],
                "variation_id": trial["variation_id"],
                "split": trial["split"],
            }
        )
    if not selected:
        raise ValueError("accepted benchmark did not contain release-eligible episodes")
    return {
        "schema_version": "farpoint.export-selection.v1",
        "dataset_id": dataset_id,
        "benchmark_id": benchmark["benchmark_id"],
        "episodes": selected,
    }


def required_successes(trial_count: int, rate: float) -> int:
    return math.ceil(trial_count * rate)
