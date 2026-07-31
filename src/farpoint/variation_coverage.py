"""Offline coverage audit for Farpoint variation pilots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


REQUIRED_FILES = ("metadata.json", "metrics.json", "trajectory.jsonl", "observations.jsonl")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _episode_record(episode_dir: Path) -> dict | None:
    metadata = _read_json(episode_dir / "metadata.json")
    metrics = _read_json(episode_dir / "metrics.json")
    variation = metadata.get("variation") or {}
    variation_id = metadata.get("variation_id") or variation.get("variation_id")
    if not variation_id:
        return None
    required_files = {name: (episode_dir / name).is_file() for name in REQUIRED_FILES}
    previews = sorted(episode_dir.glob("preview/*.png"))
    success = bool(metrics.get("success"))
    checks = metrics.get("success_checks")
    if isinstance(checks, dict) and checks and not all(bool(v) for v in checks.values()):
        success = False
    return {
        "episode_id": metadata.get("episode_id") or episode_dir.name,
        "path": str(episode_dir),
        "variation_id": variation_id,
        "object_type": variation.get("object_type"),
        "seed": metadata.get("episode_seed", variation.get("seed")),
        "task_name": metadata.get("task_name"),
        "success": success,
        "failure_reason": metrics.get("failure_reason"),
        "recorded_frames": metrics.get("recorded_frames"),
        "dataset_valid": bool(metrics.get("dataset_valid")),
        "preview_count": len(previews),
        "required_files": required_files,
        "artifact_complete": (
            all(required_files.values())
            and bool(previews)
            and bool(metrics.get("dataset_valid"))
        ),
    }


def audit_variation_coverage(
    episode_root: str | Path,
    expected_variation_ids: Iterable[str],
    expected_seeds: Iterable[int] = (0, 1),
    min_passes: int = 10,
    task_name: str | None = None,
) -> dict:
    """Return a JSON-serializable pilot gate report."""
    root = Path(episode_root)
    expected_ids = list(expected_variation_ids)
    expected_seed_set = {int(seed) for seed in expected_seeds}
    episodes = []
    if root.is_dir():
        for directory in sorted(root.iterdir()):
            if directory.is_dir():
                record = _episode_record(directory)
                if (
                    record
                    and record["variation_id"] in expected_ids
                    and (task_name is None or record["task_name"] == task_name)
                ):
                    episodes.append(record)

    passing = [item for item in episodes if item["success"] and item["artifact_complete"]]
    profile_summary = []
    missing_profiles = []
    for variation_id in expected_ids:
        profile_records = [item for item in episodes if item["variation_id"] == variation_id]
        profile_passes = [item for item in passing if item["variation_id"] == variation_id]
        missing_seeds = sorted(expected_seed_set - {item["seed"] for item in profile_records})
        profile_summary.append({
            "variation_id": variation_id,
            "episodes": len(profile_records),
            "passing_episodes": len(profile_passes),
            "missing_seeds": missing_seeds,
            "has_passing_seed": bool(profile_passes),
        })
        if not profile_passes:
            missing_profiles.append(variation_id)

    return {
        "episode_root": str(root),
        "expected_variations": expected_ids,
        "expected_seeds": sorted(expected_seed_set),
        "episode_count": len(episodes),
        "passing_episode_count": len(passing),
        "minimum_passing_episode_count": int(min_passes),
        "all_profiles_have_passing_seed": not missing_profiles,
        "missing_passing_profiles": missing_profiles,
        "gate_passed": len(passing) >= int(min_passes) and not missing_profiles,
        "profiles": profile_summary,
        "episodes": episodes,
        "duplicate_keys": sorted(
            f"{variation_id}:{seed}"
            for variation_id in expected_ids
            for seed in expected_seed_set
            if sum(1 for item in episodes if item["variation_id"] == variation_id and item["seed"] == seed) > 1
        ),
    }
