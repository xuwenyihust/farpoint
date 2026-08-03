#!/usr/bin/env python3
"""Run a coverage-first Farpoint shape-position collection."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from farpoint.formal_benchmark import append_infrastructure_attempt, infrastructure_retry_allowed  # noqa: E402
from farpoint.position_pilot import find_episode  # noqa: E402
from farpoint.release_spec import load_release_spec  # noqa: E402
from farpoint.shape_collection import (  # noqa: E402
    abort_shape_collection,
    acceptance_snapshot,
    append_shape_attempt,
    build_shape_collection_manifest,
    build_shape_collection_selection,
    file_sha256,
    finish_shape_collection,
    impossible_reason,
    load_shape_collection_policy,
    new_shape_collection_state,
    scheduled_shape_trials,
    update_shape_collection_progress,
    validate_shape_collection_policy,
)
from farpoint.shape_position import load_shape_position_plan  # noqa: E402
from run_position_benchmark import (  # noqa: E402
    audit_episode,
    checked_git_revision,
    image_digest,
    prepare_container_output_directories,
    read_json,
    refresh_reports,
    runtime_directory,
    write_json,
)


DEFAULT_POLICY = PROJECT_ROOT / "configs/collections/farpoint_v0_0_1_cylinder_position.json"
BENCHMARKS_ROOT = PROJECT_ROOT / "outputs/benchmarks"
EPISODES_ROOT = PROJECT_ROOT / "outputs/episodes"
EXAMPLE_PATH = "examples/isaac_perception_contact_scene"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_resume(state: dict, collection_id: str, git_commit: str, policy_sha: str) -> None:
    expected = {"collection_id": collection_id, "git_commit": git_commit, "policy_sha256": policy_sha}
    mismatches = [key for key, value in expected.items() if state.get(key) != value]
    if mismatches:
        raise ValueError("collection resume identity mismatch: " + ", ".join(mismatches))


def recorded_episode(state: dict, trial: dict) -> Path | None:
    for attempt in reversed(state.get("infrastructure_attempts", [])):
        if attempt.get("trial_id") != trial["trial_id"]:
            continue
        episode_id = attempt.get("episode_id")
        if episode_id and (EPISODES_ROOT / episode_id).is_dir():
            return EPISODES_ROOT / episode_id
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--collection-id")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cooldown-seconds", type=int, default=10)
    parser.add_argument("--retry-cooldown-seconds", type=int, default=180)
    parser.add_argument("--run-timeout-seconds", type=int, default=900)
    parser.add_argument("--startup-timeout-seconds", type=int, default=300)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=3)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path.home() / ".cache/farpoint/isaac-sim/shape-collection-runs",
    )
    args = parser.parse_args()
    if min(args.run_timeout_seconds, args.startup_timeout_seconds, args.max_infrastructure_attempts) < 1:
        parser.error("timeouts and infrastructure attempts must be positive")
    if min(args.cooldown_seconds, args.retry_cooldown_seconds) < 0:
        parser.error("cooldown values cannot be negative")

    checked_git_revision(args.git_commit)
    policy = load_shape_collection_policy(args.policy)
    policy_sha = file_sha256(args.policy)
    plan_path = PROJECT_ROOT / policy["position_plan"]
    plan = load_shape_position_plan(plan_path)
    validate_shape_collection_policy(policy, plan, PROJECT_ROOT)
    digest = image_digest(policy["simulator_image"])
    if digest != policy["simulator_image_digest"]:
        raise ValueError("local simulator image digest does not match collection policy")

    collection_id = args.collection_id or (
        "farpoint_cylinder_position_collection_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d')}_{args.git_commit[:7]}"
    )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", collection_id):
        parser.error("collection ID contains unsupported characters")
    collection_dir = BENCHMARKS_ROOT / collection_id
    state_path = collection_dir / "run-state.json"
    manifest_path = collection_dir / "manifest.json"
    if manifest_path.exists():
        print(f"COLLECTION_ALREADY_FINAL: {manifest_path}")
        return 0
    if state_path.exists():
        if not args.resume:
            parser.error(f"{collection_id} already exists; pass --resume")
        state = read_json(state_path)
        validate_resume(state, collection_id, args.git_commit, policy_sha)
        state.update({"execution_status": "RUNNING", "quality_status": "NOT_EVALUATED"})
        state.pop("failure_reason", None)
        state.pop("finished_at", None)
        update_shape_collection_progress(state, policy)
    else:
        state = new_shape_collection_state(
            collection_id=collection_id,
            git_commit=args.git_commit,
            policy=policy,
            policy_sha256=policy_sha,
        )

    prepare_container_output_directories()
    collection_dir.mkdir(parents=True, exist_ok=True)
    write_json(state_path, state)
    refresh_reports(state_path)
    relative_plan = plan_path.resolve().relative_to(PROJECT_ROOT.resolve())

    while not acceptance_snapshot(state, policy)["accepted"]:
        reason = impossible_reason(state, plan, policy)
        if reason:
            finish_shape_collection(state, policy, reason)
            break
        candidates = scheduled_shape_trials(state, plan)
        if not candidates:
            finish_shape_collection(state, policy, "candidate_schedule_exhausted")
            break
        trial = candidates[0]
        previous = find_episode(EPISODES_ROOT, collection_id, trial["trial_id"])
        episode = recorded_episode(state, trial) or previous
        if episode:
            print(f"ADOPT {trial['trial_id']}: {episode.name}", flush=True)
        else:
            for attempt_number in range(1, args.max_infrastructure_attempts + 1):
                if state["task_attempts"] or attempt_number > 1:
                    delay = args.retry_cooldown_seconds if attempt_number > 1 else args.cooldown_seconds
                    if delay:
                        time.sleep(delay)
                runtime = runtime_directory(args.runtime_root, collection_id, trial["trial_id"], attempt_number)
                infrastructure = append_infrastructure_attempt(
                    state, trial, attempt_number=attempt_number, run_id=runtime.name
                )
                write_json(state_path, state)
                print(
                    f"RUN {trial['trial_id']} seed={trial['seed']} attempt={attempt_number}",
                    flush=True,
                )
                command = [
                    "bash", "scripts/run_remote_isaac_example.sh", EXAMPLE_PATH,
                    policy["simulator_image"], str(runtime), runtime.name, str(trial["seed"]),
                    collection_id, "0", "", str(relative_plan), trial["trial_id"], "0",
                    args.git_commit, policy["config_sha256"], digest,
                ]
                env = os.environ.copy()
                env["FARPOINT_RUN_TIMEOUT_SECONDS"] = str(args.run_timeout_seconds)
                env["FARPOINT_STARTUP_TIMEOUT_SECONDS"] = str(args.startup_timeout_seconds)
                result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
                latest = find_episode(EPISODES_ROOT, collection_id, trial["trial_id"])
                episode = latest if latest and (previous is None or latest.name != previous.name) else None
                infrastructure.update(
                    {
                        "finished_at": utc_now(),
                        "return_code": result.returncode,
                        "episode_id": episode.name if episode else None,
                        "status": "EPISODE_RECORDED" if episode else "INFRASTRUCTURE_FAILURE",
                    }
                )
                write_json(state_path, state)
                if not infrastructure_retry_allowed(episode.name if episode else None):
                    break
        if episode is None:
            abort_shape_collection(state, policy, f"infrastructure attempts exhausted for {trial['trial_id']}")
            write_json(state_path, state)
            refresh_reports(state_path)
            return 2
        audited = audit_episode(
            episode,
            trial,
            plan=plan,
            git_commit=args.git_commit,
            simulator_image_digest=digest,
            dataset_episode_index=state["selected_episodes"],
        )
        append_shape_attempt(state, trial, audited, policy)
        write_json(state_path, state)
        refresh_reports(state_path, episode)
        print(f"TRIAL_{'PASS' if audited['success'] else 'FAIL'} {trial['trial_id']} {episode.name}", flush=True)

    if state["execution_status"] == "RUNNING":
        finish_shape_collection(state, policy, impossible_reason(state, plan, policy))
    write_json(state_path, state)
    manifest = build_shape_collection_manifest(state)
    write_json(manifest_path, manifest)
    if manifest["acceptance"]["accepted"]:
        selection = build_shape_collection_selection(manifest, dataset_id=load_release_spec()["dataset_id"])
        write_json(collection_dir / "release-selection.json", selection)
    refresh_reports(manifest_path)
    print(
        f"SHAPE_COLLECTION {'PASS' if state['accepted'] else 'FAIL'} "
        f"selected={state['selected_episodes']} attempts={state['task_attempts']} "
        f"yield={state['task_yield']:.1%} {collection_dir}",
        flush=True,
    )
    return 0 if state["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
