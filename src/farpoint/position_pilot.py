"""Acceptance audit helpers for the v1.3 cube position pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PILOT_COORDINATES = (0, 2, 4)
REQUIRED_EPISODE_FILES = (
    "metadata.json",
    "metrics.json",
    "observations.jsonl",
    "trajectory.jsonl",
    "labels.jsonl",
    "phase_events.jsonl",
)


def pilot_trials(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Select slot zero at the 3x3 cross-product of edge/center grid cells."""
    selected = [
        trial
        for trial in plan["trials"]
        if trial["row"] in PILOT_COORDINATES
        and trial["column"] in PILOT_COORDINATES
        and trial["slot"] == 0
    ]
    if len(selected) != 9:
        raise ValueError(f"pilot selection must contain 9 trials, found {len(selected)}")
    return selected


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_episode(episode_root: Path, pilot_id: str, trial_id: str) -> Path | None:
    matches = []
    for metadata_path in episode_root.glob("episode_*/metadata.json"):
        try:
            metadata = _read_json(metadata_path)
        except (OSError, ValueError):
            continue
        if metadata.get("benchmark_id") == pilot_id and metadata.get("trial_id") == trial_id:
            matches.append((metadata.get("finished_at", ""), metadata_path.parent))
    return sorted(matches)[-1][1] if matches else None


def audit_pilot_episode(
    episode_dir: Path | None,
    trial: dict[str, Any],
    *,
    plan_sha256: str,
    episode_root: Path,
) -> dict[str, Any]:
    errors = []
    if episode_dir is None:
        return {
            "trial_id": trial["trial_id"],
            "episode_id": None,
            "accepted": False,
            "errors": ["episode_missing"],
        }
    missing = [name for name in REQUIRED_EPISODE_FILES if not (episode_dir / name).is_file()]
    errors.extend(f"artifact_missing:{name}" for name in missing)
    try:
        metadata = _read_json(episode_dir / "metadata.json")
        metrics = _read_json(episode_dir / "metrics.json")
    except (OSError, ValueError) as error:
        return {
            "trial_id": trial["trial_id"],
            "episode_id": episode_dir.name,
            "accepted": False,
            "errors": [*errors, f"invalid_json:{error}"],
        }

    preview_count = len(list((episode_dir / "preview").glob("*.png")))
    observation_rgb_count = len(list((episode_dir / "observations" / "rgb").glob("*.png")))
    observation_depth_count = len(list((episode_dir / "observations" / "depth").glob("*.npy")))
    expected_observations = int(metrics.get("dataset_observation_count") or 0)
    run_id = metadata.get("run_id")
    resource_files = list((episode_root / "_resources").glob(f"*{run_id}*")) if run_id else []

    checks = {
        "simulation_success": metrics.get("success") is True,
        "trial_identity": metadata.get("trial_id") == trial["trial_id"],
        "plan_identity": metadata.get("position_plan_sha256") == plan_sha256,
        "position_identity": (metadata.get("variation") or {}).get("resolved", {}).get("object_position_m", [])[:2]
        == trial["object_position_xy_m"],
        "contact_only": metrics.get("temporary_grasp_joint_created") is False,
        "perception_xy_error": float(metrics.get("initial_object_perception_xy_error", float("inf"))) <= 0.02,
        "lift_height": float(metrics.get("object_lift_height", float("-inf"))) >= 0.15,
        "bilateral_contact": int(metrics.get("bilateral_contact_frames") or 0) >= 20,
        "transport_contact": int(
            metrics.get("max_continuous_transport_contact_frames", metrics.get("transport_contact_frames", 0))
            or 0
        ) >= 120,
        "target_xy_error": float(metrics.get("final_target_xy_distance", float("inf"))) <= 0.05,
        "settle_frames": int(metrics.get("release_settle_frames") or 0) >= 120,
        "dataset_valid": metrics.get("dataset_valid") is True,
        "preview_valid": preview_count >= 10,
        "video_source_valid": expected_observations > 0
        and observation_rgb_count == expected_observations,
        "depth_valid": expected_observations > 0
        and observation_depth_count == expected_observations,
        "telemetry_valid": len(resource_files) >= 2,
        "required_files": not missing,
    }
    errors.extend(name for name, passed in checks.items() if not passed)
    return {
        "trial_id": trial["trial_id"],
        "variation_id": trial["variation_id"],
        "cell_id": trial["cell_id"],
        "slot": trial["slot"],
        "split": trial["split"],
        "seed": trial["seed"],
        "episode_id": metadata.get("episode_id", episode_dir.name),
        "episode_path": str(episode_dir),
        "accepted": not errors,
        "checks": checks,
        "errors": errors,
        "quality": {
            "perception_xy_error_m": metrics.get("initial_object_perception_xy_error"),
            "lift_height_m": metrics.get("object_lift_height"),
            "bilateral_contact_frames": metrics.get("bilateral_contact_frames"),
            "transport_contact_frames": metrics.get("max_continuous_transport_contact_frames", metrics.get("transport_contact_frames")),
            "final_target_xy_error_m": metrics.get("final_target_xy_distance"),
            "settle_frames": metrics.get("release_settle_frames"),
            "dataset_observations": expected_observations,
            "preview_frames": preview_count,
        },
    }
