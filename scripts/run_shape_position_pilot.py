#!/usr/bin/env python3
"""Run the frozen five-cell Farpoint shape-position readiness pilot."""

from __future__ import annotations

import argparse
import os
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
from farpoint.shape_collection import (  # noqa: E402
    file_sha256,
    load_shape_collection_policy,
    validate_shape_collection_policy,
)
from farpoint.shape_pilot import (  # noqa: E402
    impossible_pilot_cell,
    pilot_acceptance,
    scheduled_pilot_trials,
)
from farpoint.shape_position import load_shape_position_plan  # noqa: E402
from run_position_benchmark import (  # noqa: E402
    audit_episode,
    checked_git_revision,
    image_digest,
    prepare_container_output_directories,
    refresh_reports,
    runtime_directory,
    write_json,
)


DEFAULT_POLICY = PROJECT_ROOT / "configs/collections/farpoint_v0_0_1_cylinder_position.json"
EPISODES_ROOT = PROJECT_ROOT / "outputs/episodes"
BENCHMARKS_ROOT = PROJECT_ROOT / "outputs/benchmarks"
EXAMPLE_PATH = "examples/isaac_cylinder_contact_scene"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--pilot-id")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--cooldown-seconds", type=int, default=10)
    parser.add_argument("--retry-cooldown-seconds", type=int, default=180)
    parser.add_argument("--run-timeout-seconds", type=int, default=900)
    parser.add_argument("--startup-timeout-seconds", type=int, default=300)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=3)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path.home() / ".cache/farpoint/isaac-sim/shape-pilot-runs",
    )
    args = parser.parse_args()
    checked_git_revision(args.git_commit)
    policy = load_shape_collection_policy(args.policy)
    plan_path = PROJECT_ROOT / policy["position_plan"]
    plan = load_shape_position_plan(plan_path)
    validate_shape_collection_policy(policy, plan, PROJECT_ROOT)
    digest = image_digest(policy["simulator_image"])
    if digest != policy["simulator_image_digest"]:
        raise ValueError("local simulator image digest does not match pilot policy")
    pilot_id = args.pilot_id or (
        f"cylinder_position_readiness_pilot_{datetime.now(timezone.utc).strftime('%Y%m%d')}_"
        f"{args.git_commit[:7]}"
    )
    pilot_dir = BENCHMARKS_ROOT / pilot_id
    state_path = pilot_dir / "run-state.json"
    if state_path.exists():
        raise FileExistsError(f"pilot already exists: {pilot_id}")
    state = {
        "schema_version": "farpoint.shape-position-pilot.v1",
        "benchmark_id": pilot_id,
        "display_name": "UR10e Cylinder Position Readiness Pilot",
        "task_id": plan["task_id"],
        "task_type": "cylinder_position_pilot",
        "git_commit": args.git_commit,
        "policy_sha256": file_sha256(args.policy),
        "position_plan_sha256": plan["plan_sha256"],
        "config_sha256": plan["config_sha256"],
        "simulator_image_digest": digest,
        "execution_status": "RUNNING",
        "planned_trials": 15,
        "completed_trials": 0,
        "passed_trials": 0,
        "success_rate": 0.0,
        "accepted": False,
        "started_at": utc_now(),
        "attempts": [],
        "infrastructure_attempts": [],
    }
    pilot_dir.mkdir(parents=True)
    prepare_container_output_directories()
    write_json(state_path, state)
    refresh_reports(state_path)
    relative_plan = plan_path.resolve().relative_to(PROJECT_ROOT.resolve())

    while not pilot_acceptance(state["attempts"])["accepted"]:
        exhausted = impossible_pilot_cell(plan, state["attempts"])
        if exhausted:
            state.update(
                {
                    "execution_status": "FINISHED",
                    "quality_status": "FAIL",
                    "accepted": False,
                    "failure_reason": f"pilot_cell_candidates_exhausted:{exhausted}",
                }
            )
            break
        trial = scheduled_pilot_trials(plan, state["attempts"])[0]
        previous = find_episode(EPISODES_ROOT, pilot_id, trial["trial_id"])
        episode = None
        for attempt_number in range(1, args.max_infrastructure_attempts + 1):
            if state["attempts"] or attempt_number > 1:
                delay = args.retry_cooldown_seconds if attempt_number > 1 else args.cooldown_seconds
                if delay:
                    time.sleep(delay)
            runtime = runtime_directory(args.runtime_root, pilot_id, trial["trial_id"], attempt_number)
            infrastructure = append_infrastructure_attempt(
                state, trial, attempt_number=attempt_number, run_id=runtime.name
            )
            write_json(state_path, state)
            command = [
                "bash", "scripts/run_remote_isaac_example.sh", EXAMPLE_PATH,
                policy["simulator_image"], str(runtime), runtime.name, str(trial["seed"]),
                pilot_id, "0", "", str(relative_plan), trial["trial_id"], "0",
                args.git_commit, policy["config_sha256"], digest,
            ]
            env = os.environ.copy()
            env["FARPOINT_RUN_TIMEOUT_SECONDS"] = str(args.run_timeout_seconds)
            env["FARPOINT_STARTUP_TIMEOUT_SECONDS"] = str(args.startup_timeout_seconds)
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
            latest = find_episode(EPISODES_ROOT, pilot_id, trial["trial_id"])
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
            state.update(
                {
                    "execution_status": "ABORTED",
                    "quality_status": "NOT_EVALUATED",
                    "accepted": False,
                    "failure_reason": f"infrastructure attempts exhausted for {trial['trial_id']}",
                }
            )
            break
        audited = audit_episode(
            episode,
            trial,
            plan=plan,
            git_commit=args.git_commit,
            simulator_image_digest=digest,
            dataset_episode_index=len(state["attempts"]),
        )
        state["attempts"].append({**trial, **audited})
        state["acceptance"] = pilot_acceptance(state["attempts"])
        state["completed_trials"] = len(state["attempts"])
        state["passed_trials"] = state["acceptance"]["successful_episodes"]
        state["success_rate"] = state["passed_trials"] / state["completed_trials"]
        write_json(state_path, state)
        refresh_reports(state_path, episode)

    if pilot_acceptance(state["attempts"])["accepted"]:
        state.update(
            {"execution_status": "FINISHED", "quality_status": "PASS", "accepted": True, "failure_reason": None}
        )
    state["acceptance"] = pilot_acceptance(state["attempts"])
    state["finished_at"] = utc_now()
    write_json(state_path, state)
    write_json(pilot_dir / "manifest.json", state)
    refresh_reports(state_path)
    print(
        f"SHAPE_PILOT {'PASS' if state['accepted'] else 'FAIL'} "
        f"successes={state['acceptance']['successful_episodes']} "
        f"attempts={state['acceptance']['task_attempts']} {pilot_dir}",
        flush=True,
    )
    return 0 if state["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
