"""Coverage-first collection state for shape-position datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from farpoint.contracts import validate_contract, validate_shape_collection_semantics
from farpoint.shape_position import validate_shape_position_plan


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
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_shape_collection_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != "farpoint.collection-policy.v3":
        raise ValueError("unsupported shape collection policy")
    return policy


def validate_shape_collection_policy(
    policy: dict[str, Any], plan: dict[str, Any], project_root: Path
) -> None:
    validate_shape_position_plan(plan)
    expected = {
        "required_cells": 25,
        "selected_episodes": 25,
        "selected_per_cell": 1,
        "maximum_task_attempts": 150,
        "maximum_candidates_per_cell": 6,
        "splits": {"train": 17, "validation": 4, "test": 4},
    }
    if policy.get("acceptance") != expected:
        raise ValueError("shape collection acceptance policy is not frozen")
    if policy.get("position_plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("collection policy position plan SHA mismatch")
    if policy.get("config_sha256") != plan.get("config_sha256"):
        raise ValueError("collection policy config SHA mismatch")
    if policy.get("task_id") != plan.get("task_id"):
        raise ValueError("collection task ID mismatch")
    payload = policy.get("simulator_payload") or {}
    if simulator_payload_sha256(project_root, payload.get("paths") or []) != payload.get("sha256"):
        raise ValueError("simulator payload SHA mismatch")


def new_shape_collection_state(
    *, collection_id: str, git_commit: str, policy: dict[str, Any], policy_sha256: str
) -> dict[str, Any]:
    state = {
        "schema_version": "farpoint.collection-state.v2",
        "record_type": "COLLECTION",
        "collection_id": collection_id,
        "display_name": policy["display_name"],
        "task_id": policy["task_id"],
        "object_shape": policy["object_shape"],
        "git_commit": git_commit,
        "policy_id": policy["policy_id"],
        "policy_sha256": policy_sha256,
        "position_plan_sha256": policy["position_plan_sha256"],
        "config_sha256": policy["config_sha256"],
        "simulator_image_digest": policy["simulator_image_digest"],
        "simulator_payload_sha256": policy["simulator_payload"]["sha256"],
        "execution_status": "RUNNING",
        "quality_status": "NOT_EVALUATED",
        "started_at": utc_now(),
        "attempts": [],
        "infrastructure_attempts": [],
    }
    update_shape_collection_progress(state, policy)
    return state


def _select_first_success_per_cell(attempts: list[dict[str, Any]]) -> None:
    for attempt in attempts:
        attempt["selected_for_dataset"] = False
        attempt.pop("selection_rank", None)
        attempt.pop("dataset_split", None)
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        by_cell.setdefault(attempt["cell_id"], []).append(attempt)
    for rows in by_cell.values():
        eligible = sorted(
            (row for row in rows if row["outcome_success"] and row["dataset_valid"]),
            key=lambda row: (row["slot"], row["trial_id"]),
        )
        if eligible:
            eligible[0]["selected_for_dataset"] = True
            eligible[0]["selection_rank"] = 1
            eligible[0]["dataset_split"] = eligible[0]["source_split"]


def acceptance_snapshot(state: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    required = policy["acceptance"]
    selected = [row for row in state["attempts"] if row["selected_for_dataset"]]
    per_cell = dict(sorted(Counter(row["cell_id"] for row in selected).items()))
    splits = {
        split: sum(row.get("dataset_split") == split for row in selected)
        for split in ("train", "validation", "test")
    }
    accepted = (
        len(state["attempts"]) <= required["maximum_task_attempts"]
        and len(selected) == required["selected_episodes"]
        and len(per_cell) == required["required_cells"]
        and set(per_cell.values()) == {required["selected_per_cell"]}
        and splits == required["splits"]
    )
    successes = sum(row["outcome_success"] for row in state["attempts"])
    return {
        "accepted": accepted,
        "observed_task_yield": successes / len(state["attempts"]) if state["attempts"] else 0.0,
        "maximum_task_attempts": required["maximum_task_attempts"],
        "observed_task_attempts": len(state["attempts"]),
        "observed_task_successes": successes,
        "required_selected_episodes": required["selected_episodes"],
        "observed_selected_episodes": len(selected),
        "required_cells": required["required_cells"],
        "observed_covered_cells": len(per_cell),
        "required_selected_per_cell": required["selected_per_cell"],
        "selected_per_cell": per_cell,
        "required_splits": required["splits"],
        "observed_splits": splits,
    }


def update_shape_collection_progress(state: dict[str, Any], policy: dict[str, Any]) -> None:
    _select_first_success_per_cell(state["attempts"])
    snapshot = acceptance_snapshot(state, policy)
    state.update(
        {
            "task_attempts": snapshot["observed_task_attempts"],
            "task_successes": snapshot["observed_task_successes"],
            "task_yield": snapshot["observed_task_yield"],
            "selected_episodes": snapshot["observed_selected_episodes"],
            "covered_cells": snapshot["observed_covered_cells"],
            "selected_per_cell": snapshot["selected_per_cell"],
            "selected_splits": snapshot["observed_splits"],
            "acceptance": snapshot,
            "updated_at": utc_now(),
        }
    )


def append_shape_attempt(
    state: dict[str, Any], trial: dict[str, Any], audited: dict[str, Any], policy: dict[str, Any]
) -> None:
    if any(row["trial_id"] == trial["trial_id"] for row in state["attempts"]):
        raise ValueError(f"collection attempt is already recorded: {trial['trial_id']}")
    state["attempts"].append(
        {
            "trial_id": trial["trial_id"],
            "episode_id": audited["episode_id"],
            "variation_id": trial["variation_id"],
            "cell_id": trial["cell_id"],
            "slot": trial["slot"],
            "seed": trial["seed"],
            "object_position_xy_m": trial["object_position_xy_m"],
            "source_split": trial["split"],
            "source_run_id": state["collection_id"],
            "source_git_commit": state["git_commit"],
            "outcome_success": bool(audited["success"]),
            "dataset_valid": bool(audited["dataset_valid"]),
            "selected_for_dataset": False,
            "failure_category": audited.get("failure_category"),
            "failure_reason": audited.get("failure_reason"),
        }
    )
    update_shape_collection_progress(state, policy)


def scheduled_shape_trials(
    state: dict[str, Any], plan: dict[str, Any]
) -> list[dict[str, Any]]:
    completed = {row["trial_id"] for row in state["attempts"]}
    covered = {row["cell_id"] for row in state["attempts"] if row["selected_for_dataset"]}
    return sorted(
        (
            deepcopy(row)
            for row in plan["trials"]
            if row["trial_id"] not in completed and row["cell_id"] not in covered
        ),
        key=lambda row: (row["slot"], row["row"], row["column"]),
    )


def impossible_reason(state: dict[str, Any], plan: dict[str, Any], policy: dict[str, Any]) -> str | None:
    if state["task_attempts"] >= policy["acceptance"]["maximum_task_attempts"]:
        return "maximum_task_attempts_reached"
    covered = {row["cell_id"] for row in state["attempts"] if row["selected_for_dataset"]}
    remaining = Counter(row["cell_id"] for row in scheduled_shape_trials(state, plan))
    for cell in sorted({row["cell_id"] for row in plan["trials"]}):
        if cell not in covered and remaining[cell] == 0:
            return f"cell_candidate_quota_exhausted:{cell}"
    return None


def finish_shape_collection(
    state: dict[str, Any], policy: dict[str, Any], failure_reason: str | None = None
) -> None:
    update_shape_collection_progress(state, policy)
    state["execution_status"] = "FINISHED"
    state["quality_status"] = "PASS" if state["acceptance"]["accepted"] else "FAIL"
    state["accepted"] = state["acceptance"]["accepted"]
    state["failure_reason"] = None if state["accepted"] else failure_reason
    state["finished_at"] = utc_now()


def abort_shape_collection(state: dict[str, Any], policy: dict[str, Any], reason: str) -> None:
    update_shape_collection_progress(state, policy)
    state.update(
        {
            "execution_status": "ABORTED",
            "quality_status": "NOT_EVALUATED",
            "accepted": False,
            "failure_reason": reason,
            "finished_at": utc_now(),
        }
    )


def build_shape_collection_manifest(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("execution_status") != "FINISHED":
        raise ValueError("collection must finish before its manifest is created")
    keys = (
        "collection_id", "display_name", "task_id", "object_shape", "git_commit", "policy_id",
        "policy_sha256", "position_plan_sha256", "config_sha256", "simulator_image_digest",
        "simulator_payload_sha256", "execution_status", "quality_status", "failure_reason",
        "attempts", "acceptance",
    )
    manifest = {"schema_version": "farpoint.collection.v2", **{key: deepcopy(state.get(key)) for key in keys}}
    errors = validate_contract(manifest) + validate_shape_collection_semantics(manifest)
    if errors:
        raise ValueError("invalid shape collection manifest: " + "; ".join(errors))
    return manifest


def build_shape_collection_selection(
    manifest: dict[str, Any], *, dataset_id: str, episode_root: str = "outputs/episodes"
) -> dict[str, Any]:
    if manifest.get("acceptance", {}).get("accepted") is not True:
        raise ValueError("release selection requires an accepted collection")
    episodes = [
        {
            "episode_dir": (Path(episode_root) / row["episode_id"]).as_posix(),
            "trial_id": row["trial_id"],
            "variation_id": row["variation_id"],
            "split": row["dataset_split"],
        }
        for row in manifest["attempts"]
        if row["selected_for_dataset"]
    ]
    return {
        "schema_version": "farpoint.export-selection.v1",
        "dataset_id": dataset_id,
        "collection_id": manifest["collection_id"],
        "episodes": sorted(episodes, key=lambda row: row["trial_id"]),
    }
