#!/usr/bin/env python3
"""Run and audit the nine-episode v1.3 cube position pilot on a GPU host."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.position_pilot import (  # noqa: E402
    audit_pilot_episode,
    find_episode,
    pilot_kind,
    pilot_trials,
    workspace_coverage,
)
from farpoint.position_plan import load_position_plan  # noqa: E402


DEFAULT_PLAN = PROJECT_ROOT / "configs" / "plans" / "farpoint_v1_3_cube_position_baseline.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--pilot-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--image", default=os.environ.get("ISAAC_SIM_IMAGE", "nvcr.io/nvidia/isaac-sim:6.0.0"))
    parser.add_argument("--runtime-root", type=Path, default=Path.home() / ".cache" / "farpoint" / "isaac-sim" / "pilot-runs")
    parser.add_argument("--episode-root", type=Path, default=PROJECT_ROOT / "outputs" / "episodes")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cooldown-seconds", type=int, default=10)
    parser.add_argument("--run-timeout-seconds", type=int, default=900)
    parser.add_argument("--startup-timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    try:
        task_type = pilot_kind(args.pilot_id)
    except ValueError as error:
        parser.error(str(error))
    if not re.fullmatch(r"[0-9a-f]{40}", args.git_commit):
        parser.error("--git-commit must be a full 40-character SHA")
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds cannot be negative")
    if args.run_timeout_seconds < 1 or args.startup_timeout_seconds < 1:
        parser.error("pilot timeouts must be positive")

    image_digest_result = subprocess.run(
        ["docker", "image", "inspect", args.image, "--format", "{{index .RepoDigests 0}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    image_digest = image_digest_result.stdout.strip().partition("@")[-1]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest):
        parser.error(f"could not resolve an immutable image digest for {args.image}")

    plan = load_position_plan(args.plan)
    selected = pilot_trials(plan)
    coverage = workspace_coverage(selected)
    is_workspace_feasibility = task_type == "cube_position_workspace_feasibility"
    output = PROJECT_ROOT / "outputs" / "benchmarks" / args.pilot_id / "manifest.json"
    if output.exists() and not (args.resume or args.audit_only):
        parser.error(f"{args.pilot_id} already exists; use --resume or --audit-only")
    new_manifest = {
        "schema_version": "farpoint.position-pilot.v1",
        "pilot_id": args.pilot_id,
        "benchmark_id": args.pilot_id,
        "execution_status": "RUNNING",
        "quality_status": "NOT_EVALUATED",
        "release_status": "PILOT",
        "git_commit": args.git_commit,
        "position_plan_id": plan["plan_id"],
        "position_plan_sha256": plan["plan_sha256"],
        "task_id": plan["task_id"],
        "task_name": plan["task_id"],
        "task_type": task_type,
        "image": args.image,
        "image_digest": image_digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "planned_trials": 9,
        "workspace_bounds_m": {
            "x": plan["grid"]["x_bounds_m"],
            "y": plan["grid"]["y_bounds_m"],
        },
        "workspace_coverage": coverage,
        "acceptance": {
            "required_successes": 9,
            "min_success_rate": 1.0,
            "contact_only": True,
            "require_contact_only": True,
            "max_perception_xy_error_m": 0.02,
            "max_perception_xy_error": 0.02,
            "min_lift_height_m": 0.15,
            "min_object_lift_height": 0.15,
            "min_bilateral_contact_frames": 20,
            "min_transport_contact_frames": 120,
            "max_final_target_xy_error_m": 0.05,
            "max_final_target_xy_distance": 0.05,
            "min_settle_frames": 120,
            "min_release_settle_frames": 120,
            "require_dataset": True,
            "require_visual_replay_source": True,
            "require_preview": True,
            "require_telemetry": True,
            **(
                {
                    "min_selected_x_span_m": coverage["min_x_span_m"],
                    "min_selected_y_span_m": coverage["min_y_span_m"],
                }
                if is_workspace_feasibility
                else {}
            ),
        },
        "infrastructure_attempts": [],
        "trials": [],
    }
    if output.exists():
        manifest = json.loads(output.read_text(encoding="utf-8"))
        if manifest.get("git_commit") != args.git_commit:
            parser.error("existing pilot Git commit does not match --git-commit")
        if manifest.get("position_plan_sha256") != plan["plan_sha256"]:
            parser.error("existing pilot position plan does not match --plan")
    else:
        manifest = new_manifest
    manifest["benchmark_id"] = args.pilot_id
    manifest["task_name"] = plan["task_id"]
    manifest["task_type"] = task_type
    manifest["workspace_bounds_m"] = new_manifest["workspace_bounds_m"]
    manifest["workspace_coverage"] = coverage
    manifest["acceptance"] = {
        **new_manifest["acceptance"],
        **manifest.get("acceptance", {}),
    }
    write_json(output, manifest)

    relative_plan = args.plan.resolve().relative_to(PROJECT_ROOT.resolve())
    if not args.audit_only:
        for ordinal, trial in enumerate(selected, start=1):
            existing = find_episode(args.episode_root, args.pilot_id, trial["trial_id"])
            if existing and args.resume:
                print(f"SKIP {trial['trial_id']}: {existing.name}", flush=True)
                continue
            if ordinal > 1 and args.cooldown_seconds:
                time.sleep(args.cooldown_seconds)
            run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{trial['cell_id']}"
            runtime = args.runtime_root / run_id
            for directory in ("cache", "compute", "config", "data", "logs", "pkg", "hub"):
                path = runtime / directory
                path.mkdir(parents=True, exist_ok=True)
                path.chmod(0o777)
            command = [
                "bash", "scripts/run_remote_isaac_example.sh",
                "examples/isaac_perception_contact_scene", args.image, str(runtime), run_id,
                str(trial["seed"]), args.pilot_id, "0", "", str(relative_plan), trial["trial_id"], "0",
                args.git_commit, plan["config_sha256"], image_digest,
            ]
            print(f"RUN {ordinal}/9 {trial['trial_id']} xy={trial['object_position_xy_m']}", flush=True)
            run_env = os.environ.copy()
            run_env["FARPOINT_RUN_TIMEOUT_SECONDS"] = str(args.run_timeout_seconds)
            run_env["FARPOINT_STARTUP_TIMEOUT_SECONDS"] = str(args.startup_timeout_seconds)
            attempt = {
                "trial_id": trial["trial_id"],
                "seed": trial["seed"],
                "run_id": run_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest.setdefault("infrastructure_attempts", []).append(attempt)
            write_json(output, manifest)
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=run_env, check=False)
            episode = find_episode(args.episode_root, args.pilot_id, trial["trial_id"])
            attempt["finished_at"] = datetime.now(timezone.utc).isoformat()
            attempt["return_code"] = result.returncode
            attempt["episode_id"] = episode.name if episode else None
            attempt["status"] = "EPISODE_RECORDED" if episode else "INFRASTRUCTURE_FAILURE"
            write_json(output, manifest)
            if episode is None:
                print(
                    f"FAIL_FAST {trial['trial_id']}: runner produced no episode",
                    flush=True,
                )
                break

    audited = []
    for trial in selected:
        episode = find_episode(args.episode_root, args.pilot_id, trial["trial_id"])
        audited.append(
            audit_pilot_episode(
                episode,
                trial,
                plan_sha256=plan["plan_sha256"],
                episode_root=args.episode_root,
            )
        )
    passed = sum(item["accepted"] for item in audited)
    manifest["trials"] = audited
    manifest["completed_trials"] = sum(item["episode_id"] is not None for item in audited)
    manifest["passed_trials"] = passed
    manifest["success_rate"] = passed / len(selected)
    manifest["execution_status"] = "FINISHED"
    coverage_accepted = coverage["accepted"] if is_workspace_feasibility else True
    manifest["quality_status"] = "PASS" if passed == 9 and coverage_accepted else "FAIL"
    manifest["accepted"] = passed == 9 and coverage_accepted
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output, manifest)
    print(f"POSITION_PILOT {'PASS' if manifest['accepted'] else 'FAIL'} {passed}/9 {output}")
    return 0 if manifest["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
