#!/usr/bin/env python3
"""Run and audit the twelve deterministic v0.0.1 cube-yaw pilot episodes."""

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

from farpoint.episode_metadata import normalize_episode_metadata_v2  # noqa: E402
from farpoint.yaw_pilot import pilot_trials, yaw_audit_accepted  # noqa: E402
from farpoint.yaw_plan import load_yaw_plan  # noqa: E402
from run_position_benchmark import (  # noqa: E402
    checked_git_revision, image_digest, prepare_container_output_directories,
    read_json, runtime_directory, write_json,
)


def find_episode(root: Path, benchmark_id: str, trial_id: str) -> Path | None:
    matches = []
    for metadata_path in root.glob("episode_*/metadata.json"):
        metadata = read_json(metadata_path)
        if metadata.get("benchmark_id") == benchmark_id and metadata.get("trial_id") == trial_id:
            matches.append(metadata_path.parent)
    return sorted(matches)[-1] if matches else None


def audit(episode: Path | None, trial: dict, git_commit: str, digest: str, plan: dict) -> dict:
    errors = []
    if episode is None:
        return {"trial_id": trial["trial_id"], "episode_id": None, "accepted": False, "errors": ["episode_missing"]}
    try:
        metadata, metrics = read_json(episode / "metadata.json"), read_json(episode / "metrics.json")
        normalized = normalize_episode_metadata_v2(metadata, metrics, split=trial["split"], dataset_episode_index=0, trial_id=trial["trial_id"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"trial_id": trial["trial_id"], "episode_id": episode.name, "accepted": False, "errors": [f"metadata:{error}"]}
    checks = {
        "simulation_success": metrics.get("success") is True,
        "dataset_valid": metrics.get("dataset_valid") is True,
        "trial_identity": metadata.get("trial_id") == trial["trial_id"],
        "variation_identity": normalized["variation"]["variation_id"] == trial["variation_id"],
        "plan_identity": metadata.get("variation_plan_sha256") == plan["plan_sha256"],
        "provenance": normalized["provenance"]["git_commit"] == git_commit and normalized["provenance"]["config_sha256"] == plan["config_sha256"] and normalized["provenance"]["simulator_image_digest"] == digest,
        "yaw_audit": yaw_audit_accepted(metrics),
        "contact_only": metrics.get("temporary_grasp_joint_created") is False,
    }
    errors.extend(name for name, passed in checks.items() if not passed)
    return {"trial_id": trial["trial_id"], "episode_id": episode.name, "accepted": not errors, "checks": checks, "errors": errors, "success": metrics.get("success") is True, "dataset_valid": metrics.get("dataset_valid") is True, "yaw_aware": metrics.get("yaw_aware") or {}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "configs/plans/farpoint_v0_0_1_cube_yaw_aware.json")
    parser.add_argument("--pilot-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--image", default=os.environ.get("ISAAC_SIM_IMAGE", "nvcr.io/nvidia/isaac-sim:6.0.0"))
    parser.add_argument("--episode-root", type=Path, default=PROJECT_ROOT / "outputs/episodes")
    parser.add_argument("--runtime-root", type=Path, default=Path.home() / ".cache/farpoint/isaac-sim/yaw-pilot-runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--cooldown-seconds", type=int, default=10)
    parser.add_argument("--run-timeout-seconds", type=int, default=900)
    parser.add_argument("--startup-timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.git_commit):
        parser.error("--git-commit must be a full lowercase SHA")
    if not re.fullmatch(r"cube_yaw_pilot_[0-9]{8}_[0-9a-f]{7,40}", args.pilot_id):
        parser.error("pilot ID must be cube_yaw_pilot_YYYYMMDD_SHA")
    checked_git_revision(args.git_commit)
    plan, trials, digest = load_yaw_plan(args.plan), None, image_digest(args.image)
    trials = pilot_trials(plan)
    output = PROJECT_ROOT / "outputs/benchmarks" / args.pilot_id / "manifest.json"
    if output.exists() and not (args.resume or args.audit_only):
        parser.error("pilot already exists; pass --resume or --audit-only")
    manifest = read_json(output) if output.exists() else {"schema_version": "farpoint.yaw-pilot.v1", "pilot_id": args.pilot_id, "benchmark_id": args.pilot_id, "git_commit": args.git_commit, "variation_plan_sha256": plan["plan_sha256"], "config_sha256": plan["config_sha256"], "image_digest": digest, "execution_status": "RUNNING", "quality_status": "NOT_EVALUATED", "release_status": "PILOT", "infrastructure_attempts": []}
    prepare_container_output_directories()
    relative_plan = args.plan.resolve().relative_to(PROJECT_ROOT.resolve())
    if not args.audit_only:
        for ordinal, trial in enumerate(trials, start=1):
            if find_episode(args.episode_root, args.pilot_id, trial["trial_id"]) and args.resume:
                continue
            if ordinal > 1 and args.cooldown_seconds:
                time.sleep(args.cooldown_seconds)
            runtime = runtime_directory(args.runtime_root, args.pilot_id, trial["trial_id"], 1)
            command = ["bash", "scripts/run_remote_isaac_example.sh", "examples/isaac_perception_contact_scene", args.image, str(runtime), runtime.name, str(trial["seed"]), args.pilot_id, "0", "", "", trial["trial_id"], "0", args.git_commit, plan["config_sha256"], digest, str(relative_plan)]
            result = subprocess.run(command, cwd=PROJECT_ROOT, env={**os.environ, "FARPOINT_RUN_TIMEOUT_SECONDS": str(args.run_timeout_seconds), "FARPOINT_STARTUP_TIMEOUT_SECONDS": str(args.startup_timeout_seconds)}, check=False)
            manifest["infrastructure_attempts"].append({"trial_id": trial["trial_id"], "return_code": result.returncode, "finished_at": datetime.now(timezone.utc).isoformat()})
            write_json(output, manifest)
            if result.returncode != 0:
                break
    manifest["trials"] = [audit(find_episode(args.episode_root, args.pilot_id, trial["trial_id"]), trial, args.git_commit, digest, plan) for trial in trials]
    manifest["completed_trials"] = sum(row["episode_id"] is not None for row in manifest["trials"])
    manifest["passed_trials"] = sum(row["accepted"] for row in manifest["trials"])
    manifest["execution_status"] = "FINISHED"
    manifest["quality_status"] = "PASS" if manifest["passed_trials"] == 12 else "FAIL"
    manifest["accepted"] = manifest["passed_trials"] == 12
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output, manifest)
    print(f"YAW_PILOT {'PASS' if manifest['accepted'] else 'FAIL'} {manifest['passed_trials']}/12 {output}")
    return 0 if manifest["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
