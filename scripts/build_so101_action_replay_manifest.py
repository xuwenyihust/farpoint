#!/usr/bin/env python3
"""Build a frozen calibrated-action replay manifest from Farpoint episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from farpoint.policy_training import file_sha256
from farpoint.so101 import USD_MAX_DEGREES, USD_MIN_DEGREES, radians_to_lerobot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_manifest(source_scenes: dict) -> dict:
    scenes = []
    for source in source_scenes["scenes"]:
        episode_root = Path(source["source_training_episode_path"])
        metadata_path = episode_root / "metadata.json"
        observations_path = episode_root / "observations.jsonl"
        if file_sha256(metadata_path) != source["source_metadata_sha256"]:
            raise ValueError(f"source metadata hash mismatch: {metadata_path}")
        actions = []
        phases = []
        clipped_source_values = 0
        for line in observations_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            radians = np.asarray(row["action_joint_positions"], dtype=np.float64)
            if radians.shape != (6,) or not np.isfinite(radians).all():
                raise ValueError(f"invalid source action in {observations_path}")
            degrees = np.rad2deg(radians)
            clipped_source_values += int(
                np.count_nonzero((degrees < USD_MIN_DEGREES) | (degrees > USD_MAX_DEGREES))
            )
            # Match the published LeRobot exporter exactly. The diagnostic replays
            # the actions ACT was trained on, not unbounded internal Oracle targets.
            actions.append(radians_to_lerobot(radians, clip=True).tolist())
            phases.append(str(row.get("phase", "unknown")))
        if not actions:
            raise ValueError(f"source episode has no observations: {episode_root}")
        scenes.append(
            {
                "scene_id": source["scene_id"],
                "source_training_episode_id": source["source_training_episode_id"],
                "source_metadata_sha256": source["source_metadata_sha256"],
                "source_observations_sha256": file_sha256(observations_path),
                "actions_calibrated": actions,
                "phases": phases,
                "source_values_clipped_by_exporter": clipped_source_values,
            }
        )
    return {
        "schema_version": "farpoint.expert-action-replay.v1",
        "dataset_revision": source_scenes.get("dataset_tag", "unknown"),
        "camera_features": [
            "observation.images.front",
            "observation.images.wrist",
        ],
        "action_conversion": {
            "source_unit": "radian",
            "output_unit": "so101_calibrated_position",
            "clip_to_calibrated_range": True,
        },
        "scenes": scenes,
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = build_manifest(json.loads(args.source_scenes.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": file_sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
