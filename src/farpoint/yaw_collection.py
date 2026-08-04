"""Collection v2 scheduling and evidence for the v0.0.1 yaw expansion."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from farpoint.contracts import validate_contract, validate_yaw_collection_semantics
from farpoint.yaw_plan import load_yaw_plan, resolve_yaw_trial

SPLITS = ("train", "validation", "test")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def condition_id(trial: dict[str, Any]) -> str:
    return f"{trial['trial_id']}"


def load_yaw_collection_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "policy_id", "task_id", "yaw_plan", "target_successes",
        "selected_per_condition", "maximum_task_attempts", "minimum_task_yield",
        "release_profile", "release_includes_prior_position_baseline", "pilot",
    }
    if policy.get("schema_version") != "farpoint.yaw-collection-policy.v1" or required - set(policy):
        raise ValueError("invalid yaw collection policy")
    if (policy["target_successes"], policy["selected_per_condition"], policy["maximum_task_attempts"], policy["minimum_task_yield"]) != (100, 1, 134, 0.75):
        raise ValueError("yaw collection policy does not match v0.0.1 acceptance")
    if policy["release_includes_prior_position_baseline"] is not False:
        raise ValueError("v0.0.1 must not import fixed-yaw baseline episodes")
    return policy


def new_collection_state(collection_id: str, git_commit: str, policy: dict[str, Any], plan: dict[str, Any], image_digest: str) -> dict[str, Any]:
    state = {
        "schema_version": "farpoint.yaw-collection-run.v1", "collection_id": collection_id,
        "task_id": policy["task_id"], "git_commit": git_commit, "policy_id": policy["policy_id"],
        "policy_sha256": file_sha256(Path(policy["_path"])), "variation_plan_sha256": plan["plan_sha256"],
        "config_sha256": plan["config_sha256"], "simulator_image_digest": image_digest,
        "execution_status": "RUNNING", "quality_status": "NOT_EVALUATED", "attempts": [],
        "infrastructure_attempts": [], "created_at": datetime.now(timezone.utc).isoformat(),
    }
    update_progress(state, policy)
    return state


def _select(attempts: list[dict[str, Any]]) -> None:
    for row in attempts:
        row["selected_for_dataset"] = False
        row.pop("dataset_split", None)
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in attempts:
        by_condition.setdefault(row["condition_id"], []).append(row)
    for rows in by_condition.values():
        eligible = sorted((row for row in rows if row["outcome_success"] and row["dataset_valid"]), key=lambda row: (row["reserve_index"], row["attempt_id"]))
        if eligible:
            eligible[0]["selected_for_dataset"] = True
            eligible[0]["dataset_split"] = eligible[0]["source_split"]


def update_progress(state: dict[str, Any], policy: dict[str, Any]) -> None:
    _select(state["attempts"])
    attempts = state["attempts"]
    selected = [row for row in attempts if row["selected_for_dataset"]]
    state["task_attempts"] = len(attempts)
    state["task_successes"] = sum(row["outcome_success"] for row in attempts)
    state["task_yield"] = state["task_successes"] / len(attempts) if attempts else 0.0
    state["selected_episodes"] = len(selected)
    state["covered_conditions"] = len({row["condition_id"] for row in selected})
    state["selected_splits"] = {split: sum(row.get("dataset_split") == split for row in selected) for split in SPLITS}


def acceptance_snapshot(state: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    accepted = (
        state["task_attempts"] <= policy["maximum_task_attempts"]
        and state["task_yield"] >= policy["minimum_task_yield"]
        and state["selected_episodes"] == policy["target_successes"]
        and state["covered_conditions"] == policy["target_successes"]
        and state["selected_splits"] == {"train": 68, "validation": 16, "test": 16}
    )
    return {"accepted": accepted, "required_task_yield": policy["minimum_task_yield"], "observed_task_yield": state["task_yield"], "maximum_task_attempts": policy["maximum_task_attempts"], "observed_task_attempts": state["task_attempts"], "observed_task_successes": state["task_successes"], "required_selected_episodes": policy["target_successes"], "observed_selected_episodes": state["selected_episodes"], "required_conditions": policy["target_successes"], "observed_covered_conditions": state["covered_conditions"], "required_splits": {"train": 68, "validation": 16, "test": 16}, "observed_splits": state["selected_splits"]}


def scheduled_trials(state: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    selected = {row["condition_id"] for row in state["attempts"] if row["selected_for_dataset"]}
    attempted = {(row["trial_id"], row["reserve_index"]) for row in state["attempts"]}
    candidates = []
    for trial in plan["trials"]:
        if condition_id(trial) in selected:
            continue
        for reserve_index in range(3):
            if (trial["trial_id"], reserve_index) not in attempted:
                resolved = resolve_yaw_trial(plan, trial["trial_id"], reserve_index=reserve_index)
                candidates.append({**trial, "condition_id": condition_id(trial), "reserve_index": reserve_index, "seed": resolved["seed"], "variation_id": resolved["variation"]["variation_id"]})
                break
    return sorted(candidates, key=lambda row: (row["reserve_index"], row["object_yaw_degrees"], row["row"], row["column"]))


def append_attempt(state: dict[str, Any], trial: dict[str, Any], episode_id: str, audited: dict[str, Any], policy: dict[str, Any]) -> None:
    attempt_id = f"{trial['trial_id']}__reserve{trial['reserve_index']}"
    if any(row["attempt_id"] == attempt_id for row in state["attempts"]):
        raise ValueError("yaw attempt is already recorded")
    state["attempts"].append({"attempt_id": attempt_id, "trial_id": trial["trial_id"], "condition_id": trial["condition_id"], "episode_id": episode_id, "variation_id": trial["variation_id"], "cell_id": trial["cell_id"], "seed": trial["seed"], "reserve_index": trial["reserve_index"], "object_position_xy_m": trial["object_position_xy_m"], "object_yaw_degrees": trial["object_yaw_degrees"], "object_orientation_xyzw": audited["object_orientation_xyzw"], "object_spec": audited["object_spec"], "source_split": trial["split"], "outcome_success": audited["success"], "dataset_valid": audited["dataset_valid"], "selected_for_dataset": False, "yaw_aware": audited["yaw_aware"], "failure_category": audited.get("failure_category"), "failure_reason": audited.get("failure_reason")})
    update_progress(state, policy)


def finish_collection(state: dict[str, Any], policy: dict[str, Any], reason: str | None = None) -> None:
    update_progress(state, policy)
    state["execution_status"] = "FINISHED"
    state["acceptance"] = acceptance_snapshot(state, policy)
    state["quality_status"] = "PASS" if state["acceptance"]["accepted"] else "FAIL"
    state["failure_reason"] = None if state["acceptance"]["accepted"] else reason


def build_manifest(state: dict[str, Any], policy: dict[str, Any], simulator_payload_sha256: str) -> dict[str, Any]:
    if state.get("execution_status") != "FINISHED":
        raise ValueError("yaw collection must finish before manifest creation")
    manifest = {key: deepcopy(state[key]) for key in ("collection_id", "task_id", "git_commit", "policy_id", "policy_sha256", "variation_plan_sha256", "config_sha256", "simulator_image_digest", "execution_status", "quality_status", "failure_reason", "attempts", "acceptance")}
    manifest.update({"schema_version": "farpoint.collection.v2", "simulator_payload_sha256": simulator_payload_sha256})
    errors = validate_contract(manifest) + validate_yaw_collection_semantics(manifest)
    if errors:
        raise ValueError("invalid yaw collection manifest: " + "; ".join(errors))
    return manifest


def load_plan_for_policy(policy_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = load_yaw_collection_policy(policy_path)
    policy["_path"] = str(policy_path)
    plan = load_yaw_plan(policy_path.parents[2] / policy["yaw_plan"])
    return policy, plan
