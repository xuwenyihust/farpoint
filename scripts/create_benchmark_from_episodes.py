#!/usr/bin/env python3
"""Create a benchmark manifest from already-recorded Farpoint episodes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_manifest(
    episode_root: Path,
    benchmark_id: str,
    episode_ids: list[str],
    task_name: str,
    task_type: str,
    min_success_rate: float,
) -> dict:
    trials = []
    for repeat, episode_id in enumerate(episode_ids):
        episode_dir = episode_root / episode_id
        metadata = read_json(episode_dir / "metadata.json")
        metrics = read_json(episode_dir / "metrics.json")
        trials.append(
            {
                "seed": int(metadata.get("episode_seed", repeat)),
                "repeat": repeat,
                "episode_id": episode_id,
                "return_code": 0 if metrics.get("success") else 1,
                "success": bool(metrics.get("success")),
                "failure_category": metrics.get("failure_category"),
                "failure_reason": metrics.get("failure_reason"),
                "final_target_xy_distance": metrics.get("final_target_xy_distance"),
                "object_lift_height": metrics.get("object_lift_height"),
                "release_settle_frames": metrics.get("release_settle_frames"),
                "elapsed_seconds": metrics.get("elapsed_seconds"),
                "initial_object_perception_xy_error": metrics.get(
                    "initial_object_perception_xy_error"
                ),
                "bilateral_contact_frames": metrics.get("bilateral_contact_frames"),
                "transport_contact_frames": metrics.get("transport_contact_frames"),
                "temporary_grasp_joint_created": metrics.get(
                    "temporary_grasp_joint_created"
                ),
                "dataset_valid": metrics.get("dataset_valid"),
                "dataset_observation_count": metrics.get("dataset_observation_count"),
                "variation_id": metadata.get("variation_id")
                or (metadata.get("variation") or {}).get("variation_id"),
            }
        )

    passed = sum(1 for trial in trials if trial["success"])
    completed = len(trials)
    return {
        "schema_version": "benchmark.v1",
        "benchmark_id": benchmark_id,
        "task_name": task_name,
        "task_type": task_type,
        "example_path": "examples/isaac_perception_contact_scene",
        "created_at": utc_now(),
        "finished_at": utc_now(),
        "seeds": [trial["seed"] for trial in trials],
        "repeats": 1,
        "planned_trials": completed,
        "completed_trials": completed,
        "passed_trials": passed,
        "success_rate": passed / completed if completed else 0.0,
        "accepted": completed > 0 and passed / completed >= min_success_rate,
        "acceptance": {
            "min_success_rate": min_success_rate,
            "max_final_target_xy_distance": 0.05,
            "min_object_lift_height": 0.15,
            "min_release_settle_frames": 120,
            "require_dataset": True,
        },
        "provenance": {
            "type": "retroactive_episode_group",
            "source": "existing_episode_artifacts",
        },
        "trials": trials,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--episode-ids", nargs="+", required=True)
    parser.add_argument("--task-name", default="isaac_perception_contact_scene")
    parser.add_argument("--task-type", default="variation_expansion_v1")
    parser.add_argument("--min-success-rate", type=float, default=0.90)
    args = parser.parse_args()
    manifest = build_manifest(
        args.episode_root,
        args.benchmark_id,
        args.episode_ids,
        args.task_name,
        args.task_type,
        args.min_success_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"BENCHMARK_MANIFEST {args.benchmark_id} "
        f"{manifest['passed_trials']}/{manifest['completed_trials']} "
        f"({manifest['success_rate']:.1%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
