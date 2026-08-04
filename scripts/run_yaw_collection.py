#!/usr/bin/env python3
"""Collect the frozen v0.0.1 yaw-aware cube conditions after pilot approval."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from farpoint.episode_metadata import normalize_episode_metadata_v2  # noqa: E402
from farpoint.position_collection import simulator_payload_sha256  # noqa: E402
from farpoint.yaw_collection import (  # noqa: E402
    acceptance_snapshot, append_attempt, build_manifest, finish_collection,
    load_plan_for_policy, new_collection_state, scheduled_trials,
)
from farpoint.yaw_pilot import yaw_audit_accepted  # noqa: E402
from run_position_benchmark import checked_git_revision, image_digest, read_json, runtime_directory, write_json  # noqa: E402


def find_episode(root: Path, collection_id: str, trial_id: str) -> Path | None:
    matches = [path.parent for path in root.glob("episode_*/metadata.json") if (lambda metadata: metadata.get("benchmark_id") == collection_id and metadata.get("trial_id") == trial_id)(read_json(path))]
    return sorted(matches)[-1] if matches else None


def audit(episode: Path, trial: dict, plan: dict, git_commit: str, digest: str) -> dict:
    metadata, metrics = read_json(episode / "metadata.json"), read_json(episode / "metrics.json")
    try:
        normalized = normalize_episode_metadata_v2(metadata, metrics, split=trial["split"], dataset_episode_index=0, trial_id=trial["trial_id"])
        metadata_ok = normalized["variation"]["variation_id"] == trial["variation_id"] and metadata.get("variation_plan_sha256") == plan["plan_sha256"]
        provenance_ok = normalized["provenance"]["git_commit"] == git_commit and normalized["provenance"]["simulator_image_digest"] == digest
    except (KeyError, TypeError, ValueError):
        metadata_ok = provenance_ok = False
    success = metrics.get("success") is True and metrics.get("dataset_valid") is True and metadata_ok and provenance_ok and yaw_audit_accepted(metrics)
    resolved = (normalized.get("variation") or {}).get("resolved") if metadata_ok else {}
    return {"success": success, "dataset_valid": metrics.get("dataset_valid") is True, "yaw_aware": metrics.get("yaw_aware") or {}, "object_orientation_xyzw": resolved.get("object_orientation_xyzw", [0.0, 0.0, 0.0, 1.0]), "object_spec": {"shape": (normalized.get("scene") or {}).get("object", {}).get("shape"), "dimensions_m": resolved.get("object_dimensions_m"), "variant": resolved.get("object_variant_id")}, "failure_category": metrics.get("failure_category"), "failure_reason": metrics.get("failure_reason")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=PROJECT_ROOT / "configs/collections/farpoint_v0_0_1_cube_yaw_aware.json")
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--image", default=os.environ.get("ISAAC_SIM_IMAGE", "nvcr.io/nvidia/isaac-sim:6.0.0"))
    parser.add_argument("--episode-root", type=Path, default=PROJECT_ROOT / "outputs/episodes")
    parser.add_argument("--runtime-root", type=Path, default=Path.home() / ".cache/farpoint/isaac-sim/yaw-collection-runs")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.git_commit):
        parser.error("--git-commit must be a full lowercase SHA")
    checked_git_revision(args.git_commit)
    policy, plan = load_plan_for_policy(args.policy)
    digest = image_digest(args.image)
    state_path = PROJECT_ROOT / "outputs/benchmarks" / args.collection_id / "run-state.json"
    state = read_json(state_path) if state_path.exists() and args.resume else new_collection_state(args.collection_id, args.git_commit, policy, plan, digest)
    relative_plan = (PROJECT_ROOT / policy["yaw_plan"]).resolve().relative_to(PROJECT_ROOT.resolve())
    while not acceptance_snapshot(state, policy)["accepted"]:
        candidates = scheduled_trials(state, plan)
        if not candidates or state["task_attempts"] >= policy["maximum_task_attempts"]:
            break
        trial = candidates[0]
        runtime = runtime_directory(args.runtime_root, args.collection_id, trial["trial_id"], trial["reserve_index"] + 1)
        command = ["bash", "scripts/run_remote_isaac_example.sh", "examples/isaac_perception_contact_scene", args.image, str(runtime), runtime.name, str(trial["seed"]), args.collection_id, "0", "", "", trial["trial_id"], str(trial["reserve_index"]), args.git_commit, plan["config_sha256"], digest, str(relative_plan)]
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        episode = find_episode(args.episode_root, args.collection_id, trial["trial_id"])
        if episode is None:
            raise RuntimeError(f"infrastructure failure for {trial['trial_id']}: {result.returncode}")
        append_attempt(state, trial, episode.name, audit(episode, trial, plan, args.git_commit, digest), policy)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)
    finish_collection(state, policy, "candidate_schedule_exhausted")
    payload_paths = ["configs/plans/farpoint_v0_0_1_cube_yaw_aware.json", "configs/variations/farpoint_v0_0_1_cube_yaw.json", "examples/isaac_perception_contact_scene/task.yaml", "examples/isaac_ur10e_robotiq_scene/scene.py", "scripts/run_remote_isaac_example.sh", "src/farpoint/perception.py", "src/farpoint/yaw_plan.py"]
    manifest = build_manifest(state, policy, simulator_payload_sha256(PROJECT_ROOT, payload_paths))
    write_json(state_path.parent / "manifest.json", manifest)
    print(f"YAW_COLLECTION {'PASS' if manifest['acceptance']['accepted'] else 'FAIL'} {state['selected_episodes']}/100")
    return 0 if manifest["acceptance"]["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
