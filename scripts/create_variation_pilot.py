#!/usr/bin/env python3
"""Select and audit a reproducible two-seed pilot for every variation profile."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FILES = (
    "metadata.json",
    "metrics.json",
    "observations.jsonl",
    "trajectory.jsonl",
    "labels.jsonl",
    "phase_events.jsonl",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def variation_id(metadata: dict) -> str | None:
    return metadata.get("variation_id") or (metadata.get("variation") or {}).get("variation_id")


def audit_episode(episode_dir: Path) -> dict:
    metadata = read_json(episode_dir / "metadata.json") if (episode_dir / "metadata.json").is_file() else {}
    metrics = read_json(episode_dir / "metrics.json") if (episode_dir / "metrics.json").is_file() else {}
    success = bool(metrics.get("success"))
    failure_reason = metrics.get("failure_reason")
    required_files = REQUIRED_FILES if success else ("metadata.json", "metrics.json", "phase_events.jsonl")
    errors = [name for name in required_files if not (episode_dir / name).is_file()]
    preview_count = len(list((episode_dir / "preview").glob("*.png")))
    if preview_count < 10:
        errors.append(f"preview_images<{10}")
    run_id = metadata.get("run_id")
    resource_root = episode_dir.parent / "_resources"
    resource_files = list(resource_root.glob(f"*{run_id}*")) if run_id else []
    if not resource_files:
        errors.append("resource_telemetry_missing")
    if not success and not failure_reason:
        errors.append("failed_episode_missing_failure_reason")
    return {
        "complete": not errors,
        "errors": errors,
        "preview_count": preview_count,
        "resource_files": [str(path.relative_to(episode_dir.parent.parent)) for path in resource_files],
        "frame_count": int(metrics.get("recorded_frame_count") or metrics.get("dataset_observation_count") or 0),
        "success": success,
        "failure_category": metrics.get("failure_category"),
        "failure_reason": failure_reason,
    }


def select_pilot(episode_root: Path, config: dict, seeds: list[int]) -> tuple[list[dict], list[str]]:
    candidates: dict[tuple[str, int], list[tuple[str, Path, dict]]] = {}
    for metadata_path in episode_root.glob("episode_*/metadata.json"):
        try:
            metadata = read_json(metadata_path)
            key = (variation_id(metadata), int(metadata.get("episode_seed")))
        except (OSError, TypeError, ValueError):
            continue
        if key[0] in {profile["variation_id"] for profile in config["profiles"]} and key[1] in seeds:
            candidates.setdefault(key, []).append((metadata.get("finished_at", ""), metadata_path.parent, metadata))

    trials = []
    missing = []
    for profile in config["profiles"]:
        profile_id = profile["variation_id"]
        for seed in seeds:
            matches = sorted(candidates.get((profile_id, seed), []), key=lambda item: item[0])
            if not matches:
                missing.append(f"{profile_id}:seed={seed}")
                continue
            _, episode_dir, metadata = matches[-1]
            audit = audit_episode(episode_dir)
            metrics = read_json(episode_dir / "metrics.json")
            trials.append({
                "variation_id": profile_id,
                "profile": profile,
                "seed": seed,
                "episode_id": metadata.get("episode_id", episode_dir.name),
                "episode_dir": str(episode_dir),
                "success": bool(metrics.get("success")),
                "failure_category": metrics.get("failure_category"),
                "failure_reason": metrics.get("failure_reason"),
                "artifacts": audit,
            })
    return trials, missing


def build_pilot(episode_root: Path, config_path: Path, output: Path, benchmark_id: str, seeds: list[int]) -> dict:
    config = read_json(config_path)
    trials, missing = select_pilot(episode_root, config, seeds)
    expected = len(config["profiles"]) * len(seeds)
    complete = all(trial["artifacts"]["complete"] for trial in trials) and not missing and len(trials) == expected
    manifest = {
        "schema_version": "benchmark.v1",
        "benchmark_id": benchmark_id,
        "task_name": config["task_name"],
        "task_type": "intra_task_diversity_pilot_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "planned_trials": expected,
        "completed_trials": len(trials),
        "passed_trials": sum(1 for trial in trials if trial["success"]),
        "success_rate": sum(1 for trial in trials if trial["success"]) / len(trials) if trials else 0.0,
        "artifact_complete": complete,
        "accepted": complete and (
            sum(1 for trial in trials if trial["success"]) / len(trials) >= 0.90
            if trials else False
        ),
        "acceptance": {
            "profiles": [profile["variation_id"] for profile in config["profiles"]],
            "seeds": seeds,
            "min_success_rate": 0.90,
            "require_complete_artifacts": True,
            "quality_gate_deferred_to": "subgoal_3",
        },
        "provenance": {"type": "variation_pilot_selection", "config": str(config_path)},
        "missing": missing,
        "trials": trials,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--seeds", default="0,1")
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    manifest = build_pilot(args.episode_root, args.config, args.output, args.benchmark_id, seeds)
    print(
        f"VARIATION_PILOT {manifest['benchmark_id']} "
        f"{manifest['completed_trials']}/{manifest['planned_trials']} complete "
        f"success_rate={manifest['success_rate']:.1%}"
    )
    return 0 if manifest["artifact_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
