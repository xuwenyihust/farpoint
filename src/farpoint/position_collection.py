"""Balanced cube-position collection contracts and scheduling."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from farpoint.contracts import validate_collection_semantics, validate_contract
from farpoint.formal_benchmark import validate_formal_plan


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simulator_payload_sha256(project_root: Path, paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        path = project_root / relative
        if not path.is_file():
            raise ValueError(f"simulator payload file is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_collection_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != "farpoint.collection-policy.v1":
        raise ValueError("unsupported collection policy schema")
    return policy


def validate_collection_policy(
    policy: dict[str, Any], plan: dict[str, Any], project_root: Path
) -> None:
    validate_formal_plan(plan)
    if policy.get("position_plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("collection policy position plan SHA mismatch")
    if policy.get("config_sha256") != plan.get("config_sha256"):
        raise ValueError("collection policy config SHA mismatch")
    payload = policy.get("simulator_payload") or {}
    observed = simulator_payload_sha256(project_root, payload.get("paths") or [])
    if observed != payload.get("sha256"):
        raise ValueError("simulator payload SHA mismatch")
    acceptance = policy.get("acceptance") or {}
    expected = {
        "grid_cells": 25,
        "selected_episodes": 50,
        "selected_per_cell": 2,
        "minimum_task_yield": 0.75,
        "maximum_task_attempts": 73,
        "maximum_new_task_attempts": 46,
        "maximum_candidates_per_cell": 3,
        "expected_task_successes": 55,
        "splits": {"train": 34, "validation": 8, "test": 8},
    }
    if acceptance != expected:
        raise ValueError("collection acceptance policy does not match v1.3")


def cell_index(cell_id: str) -> int:
    try:
        row = int(cell_id[1:3])
        column = int(cell_id[5:7])
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid grid cell ID: {cell_id}") from error
    if not (0 <= row < 5 and 0 <= column < 5):
        raise ValueError(f"invalid grid cell ID: {cell_id}")
    return row * 5 + column


def dataset_split(cell_id: str, selection_rank: int) -> str:
    if selection_rank == 1:
        return "train"
    if selection_rank != 2:
        raise ValueError("selection rank must be 1 or 2")
    return ("train", "validation", "test")[cell_index(cell_id) % 3]


def _attempt_from_trial(
    trial: dict[str, Any],
    *,
    origin: str,
    source_run_id: str,
    source_git_commit: str,
) -> dict[str, Any]:
    return {
        "trial_id": trial["trial_id"],
        "episode_id": trial["episode_id"],
        "variation_id": trial["variation_id"],
        "cell_id": trial["cell_id"],
        "slot": int(trial["slot"]),
        "seed": int(trial["seed"]),
        "object_position_xy_m": [float(value) for value in trial["object_position_xy_m"]],
        "source_split": trial["split"],
        "origin": origin,
        "source_run_id": source_run_id,
        "source_git_commit": source_git_commit,
        "outcome_success": bool(trial.get("success")),
        "dataset_valid": bool(trial.get("dataset_valid")),
        "selected_for_dataset": False,
        "failure_category": trial.get("failure_category"),
        "failure_reason": trial.get("failure_reason"),
    }


def _apply_balanced_selection(attempts: list[dict[str, Any]]) -> None:
    for attempt in attempts:
        attempt["selected_for_dataset"] = False
        attempt.pop("selection_rank", None)
        attempt.pop("dataset_split", None)
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        by_cell.setdefault(attempt["cell_id"], []).append(attempt)
    for cell_id, rows in by_cell.items():
        eligible = sorted(
            (
                row
                for row in rows
                if row["outcome_success"] and row["dataset_valid"]
            ),
            key=lambda row: (row["slot"], row["trial_id"]),
        )[:2]
        for rank, row in enumerate(eligible, start=1):
            row["selected_for_dataset"] = True
            row["selection_rank"] = rank
            row["dataset_split"] = dataset_split(cell_id, rank)


def import_source_attempts(
    source_state: dict[str, Any], policy: dict[str, Any], plan: dict[str, Any]
) -> list[dict[str, Any]]:
    source = policy["source"]
    expected = {
        "benchmark_id": source["run_id"],
        "git_commit": source["git_commit"],
        "execution_status": source["execution_status"],
        "completed_trials": source["completed_attempts"],
        "passed_trials": source["successful_attempts"],
        "position_plan_sha256": policy["position_plan_sha256"],
        "config_sha256": policy["config_sha256"],
        "image_digest": policy["simulator_image_digest"],
    }
    mismatches = [key for key, value in expected.items() if source_state.get(key) != value]
    if mismatches:
        raise ValueError("source run identity mismatch: " + ", ".join(mismatches))
    source_trials = source_state.get("trials") or []
    if len(source_trials) != source["completed_attempts"]:
        raise ValueError("source run trial count mismatch")
    source_trial_ids = [trial.get("trial_id") for trial in source_trials]
    if len(set(source_trial_ids)) != len(source_trial_ids):
        raise ValueError("source run trial ids must be unique")
    by_id = {trial["trial_id"]: trial for trial in plan["trials"]}
    attempts = []
    for recorded in source_trials:
        planned = by_id.get(recorded.get("trial_id"))
        if not planned:
            raise ValueError(f"source trial is absent from frozen plan: {recorded.get('trial_id')}")
        if recorded.get("episode_id") is None:
            raise ValueError(f"source trial has no episode: {recorded['trial_id']}")
        for key in (
            "variation_id",
            "cell_id",
            "slot",
            "seed",
            "split",
            "object_position_xy_m",
        ):
            if recorded.get(key) != planned.get(key):
                raise ValueError(f"source trial {key} mismatch: {recorded['trial_id']}")
        if recorded.get("success") and (
            not recorded.get("dataset_valid")
            or not recorded.get("accepted")
            or not all((recorded.get("checks") or {}).values())
        ):
            raise ValueError(f"source successful trial failed strict gates: {recorded['trial_id']}")
        attempts.append(
            _attempt_from_trial(
                recorded,
                origin="imported",
                source_run_id=source["run_id"],
                source_git_commit=source["git_commit"],
            )
        )
    if sum(row["outcome_success"] for row in attempts) != source["successful_attempts"]:
        raise ValueError("source run successful attempt count mismatch")
    _apply_balanced_selection(attempts)
    selected = [row for row in attempts if row["selected_for_dataset"]]
    covered = {row["cell_id"] for row in selected}
    if len(selected) != source["selected_episodes"] or len(covered) != source["covered_cells"]:
        raise ValueError("source balanced selection mismatch")
    return attempts


def new_collection_state(
    *,
    collection_id: str,
    git_commit: str,
    policy: dict[str, Any],
    policy_sha256: str,
    imported_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    state = {
        "schema_version": "farpoint.collection-run.v1",
        "collection_id": collection_id,
        "task_id": policy["task_id"],
        "git_commit": git_commit,
        "policy_id": policy["policy_id"],
        "policy_sha256": policy_sha256,
        "position_plan_sha256": policy["position_plan_sha256"],
        "config_sha256": policy["config_sha256"],
        "simulator_image": policy["simulator_image"],
        "simulator_image_digest": policy["simulator_image_digest"],
        "simulator_payload_sha256": policy["simulator_payload"]["sha256"],
        "execution_status": "RUNNING",
        "quality_status": "NOT_EVALUATED",
        "release_status": "CANDIDATE",
        "created_at": utc_now(),
        "attempts": deepcopy(imported_attempts),
        "infrastructure_attempts": [],
    }
    update_collection_progress(state, policy)
    return state


def update_collection_progress(state: dict[str, Any], policy: dict[str, Any]) -> None:
    _apply_balanced_selection(state["attempts"])
    attempts = state["attempts"]
    selected = [row for row in attempts if row["selected_for_dataset"]]
    state["task_attempts"] = len(attempts)
    state["task_successes"] = sum(row["outcome_success"] for row in attempts)
    state["task_yield"] = state["task_successes"] / len(attempts) if attempts else 0.0
    state["selected_episodes"] = len(selected)
    state["covered_cells"] = len({row["cell_id"] for row in selected})
    state["selected_per_cell"] = dict(sorted(Counter(row["cell_id"] for row in selected).items()))
    state["selected_splits"] = {
        split: sum(row.get("dataset_split") == split for row in selected)
        for split in ("train", "validation", "test")
    }
    state["updated_at"] = utc_now()
    state["acceptance"] = acceptance_snapshot(state, policy)


def append_new_attempt(
    state: dict[str, Any], trial: dict[str, Any], audited: dict[str, Any], policy: dict[str, Any]
) -> None:
    if any(row["trial_id"] == trial["trial_id"] for row in state["attempts"]):
        raise ValueError(f"collection trial is already recorded: {trial['trial_id']}")
    merged = {**trial, **audited}
    state["attempts"].append(
        _attempt_from_trial(
            merged,
            origin="new",
            source_run_id=state["collection_id"],
            source_git_commit=state["git_commit"],
        )
    )
    update_collection_progress(state, policy)


def scheduled_trials(
    state: dict[str, Any], plan: dict[str, Any], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    attempted = {row["trial_id"] for row in state["attempts"]}
    selected = Counter(
        row["cell_id"] for row in state["attempts"] if row["selected_for_dataset"]
    )
    quota = policy["acceptance"]["selected_per_cell"]
    return sorted(
        (
            trial
            for trial in plan["trials"]
            if trial["trial_id"] not in attempted and selected[trial["cell_id"]] < quota
        ),
        key=lambda trial: (trial["slot"], trial["row"], trial["column"]),
    )


def acceptance_snapshot(state: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    required = policy["acceptance"]
    attempts = state["task_attempts"]
    accepted = (
        attempts <= required["maximum_task_attempts"]
        and state["task_yield"] >= required["minimum_task_yield"]
        and state["selected_episodes"] == required["selected_episodes"]
        and state["covered_cells"] == required["grid_cells"]
        and set(state["selected_per_cell"].values()) == {required["selected_per_cell"]}
        and state["selected_splits"] == required["splits"]
    )
    return {
        "accepted": accepted,
        "required_task_yield": required["minimum_task_yield"],
        "observed_task_yield": state["task_yield"],
        "maximum_task_attempts": required["maximum_task_attempts"],
        "observed_task_attempts": attempts,
        "observed_task_successes": state["task_successes"],
        "required_selected_episodes": required["selected_episodes"],
        "observed_selected_episodes": state["selected_episodes"],
        "required_cells": required["grid_cells"],
        "observed_covered_cells": state["covered_cells"],
        "required_selected_per_cell": required["selected_per_cell"],
        "selected_per_cell": state["selected_per_cell"],
        "required_splits": required["splits"],
        "observed_splits": state["selected_splits"],
    }


def impossible_reason(
    state: dict[str, Any], plan: dict[str, Any], policy: dict[str, Any]
) -> str | None:
    required = policy["acceptance"]
    if state["task_attempts"] >= required["maximum_task_attempts"] and not acceptance_snapshot(state, policy)["accepted"]:
        return "maximum_task_attempts_reached"
    selected = Counter(
        row["cell_id"] for row in state["attempts"] if row["selected_for_dataset"]
    )
    attempted = {row["trial_id"] for row in state["attempts"]}
    remaining = Counter(
        trial["cell_id"] for trial in plan["trials"] if trial["trial_id"] not in attempted
    )
    for cell_id in sorted({trial["cell_id"] for trial in plan["trials"]}):
        if selected[cell_id] + remaining[cell_id] < required["selected_per_cell"]:
            return f"cell_candidate_quota_exhausted:{cell_id}"
    return None


def finish_collection(
    state: dict[str, Any], policy: dict[str, Any], *, failure_reason: str | None = None
) -> None:
    update_collection_progress(state, policy)
    snapshot = acceptance_snapshot(state, policy)
    state["execution_status"] = "FINISHED"
    state["quality_status"] = "PASS" if snapshot["accepted"] else "FAIL"
    state["accepted"] = snapshot["accepted"]
    state["failure_reason"] = None if snapshot["accepted"] else failure_reason
    state["finished_at"] = utc_now()


def abort_collection(state: dict[str, Any], policy: dict[str, Any], reason: str) -> None:
    update_collection_progress(state, policy)
    state["execution_status"] = "ABORTED"
    state["quality_status"] = "NOT_EVALUATED"
    state["accepted"] = False
    state["failure_reason"] = reason
    state["finished_at"] = utc_now()


def complete_import_pilot(state: dict[str, Any], policy: dict[str, Any]) -> None:
    """Record a successful offline import audit without claiming collection acceptance."""
    update_collection_progress(state, policy)
    state["execution_status"] = "PILOT_COMPLETE"
    state["quality_status"] = "PASS"
    state["accepted"] = False
    state["finished_at"] = utc_now()


def build_collection_manifest(state: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    if state.get("execution_status") != "FINISHED":
        raise ValueError("collection must finish before its manifest is created")
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
        "execution_status": state["execution_status"],
        "quality_status": state["quality_status"],
        "failure_reason": state.get("failure_reason"),
        "attempts": deepcopy(state["attempts"]),
        "acceptance": acceptance_snapshot(state, policy),
    }
    errors = validate_contract(manifest) + validate_collection_semantics(manifest)
    if errors:
        raise ValueError("invalid collection manifest: " + "; ".join(errors))
    return manifest


def build_collection_selection(
    manifest: dict[str, Any], *, dataset_id: str, episode_root: str = "outputs/episodes"
) -> dict[str, Any]:
    errors = validate_contract(manifest) + validate_collection_semantics(manifest)
    if errors:
        raise ValueError("invalid collection manifest: " + "; ".join(errors))
    if manifest["acceptance"]["accepted"] is not True:
        raise ValueError("release selection requires an accepted collection")
    root = Path(episode_root)
    if root.is_absolute():
        raise ValueError("selection episode root must be repository-relative")
    selected = []
    for attempt in manifest["attempts"]:
        if not attempt["selected_for_dataset"]:
            continue
        selected.append(
            {
                "episode_dir": (root / attempt["episode_id"]).as_posix(),
                "trial_id": attempt["trial_id"],
                "variation_id": attempt["variation_id"],
                "split": attempt["dataset_split"],
            }
        )
    return {
        "schema_version": "farpoint.export-selection.v1",
        "dataset_id": dataset_id,
        "collection_id": manifest["collection_id"],
        "episodes": sorted(selected, key=lambda row: row["trial_id"]),
    }


def validate_resume_state(
    state: dict[str, Any], *, collection_id: str, git_commit: str, policy_sha256: str
) -> None:
    expected = {
        "collection_id": collection_id,
        "git_commit": git_commit,
        "policy_sha256": policy_sha256,
    }
    mismatches = [key for key, value in expected.items() if state.get(key) != value]
    if mismatches:
        raise ValueError("collection resume identity mismatch: " + ", ".join(mismatches))


def minimum_required_successes(attempts: int, rate: float) -> int:
    return math.ceil(attempts * rate)
