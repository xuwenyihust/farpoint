#!/usr/bin/env python3
"""Convert one Farpoint episode into a LeRobot Dataset v3 directory.

The exporter intentionally runs outside Isaac Sim. It consumes the raw
episode artifacts written by the production scene and uses the installed
LeRobot writer for Parquet/MP4 encoding and metadata generation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
TASK_NAME = "ur10e_robotiq_single_cube_pick_place"
TASK_INSTRUCTION = "Pick up the cube and place it in the target zone."


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def infer_fps(rows: list[dict]) -> int:
    """Infer a stable integer recording rate from source timestamps."""
    if len(rows) < 2:
        return 1
    timestamps = np.asarray(
        [float(row["timestamp_seconds"]) for row in rows], dtype=np.float64
    )
    deltas = np.diff(timestamps)
    if np.any(deltas <= 0):
        raise ValueError("source observation timestamps must be strictly increasing")
    fps = int(round(1.0 / float(np.median(deltas))))
    if fps <= 0:
        raise ValueError("could not infer a positive recording fps")
    expected = 1.0 / fps
    if np.max(np.abs(deltas - expected)) > max(0.01, expected * 0.05):
        raise ValueError(
            "source observation timestamps are not close to a uniform recording rate"
        )
    return fps


def resolve_controlled_joint_names(metadata: dict, first_row: dict) -> list[str]:
    names = metadata.get("controlled_joint_names")
    if not names:
        names = first_row.get("controlled_joint_names")
    if not names:
        names = list(first_row.get("joint_names", []))
    names = list(names)
    arm_names = [name for name in ARM_JOINT_NAMES if name in names]
    if len(arm_names) != 6:
        raise ValueError(f"could not resolve six UR10e arm joints from {names!r}")
    gripper_names = [
        name
        for name in names
        if "finger" in name.lower() and "mimic" not in name.lower()
    ]
    if not gripper_names:
        raise ValueError(f"could not resolve an actuated Robotiq finger joint from {names!r}")
    return [*arm_names, gripper_names[0]]


def select_joint_values(row: dict, selected_names: list[str], field: str) -> np.ndarray:
    names = list(row.get("joint_names", []))
    values = list(row.get(field, []))
    if len(names) != len(values):
        raise ValueError(f"{field} length does not match joint_names at frame {row.get('frame')}")
    positions = {name: index for index, name in enumerate(names)}
    missing = [name for name in selected_names if name not in positions]
    if missing:
        raise ValueError(f"{field} is missing controlled joints: {missing!r}")
    return np.asarray([values[positions[name]] for name in selected_names], dtype=np.float32)


def build_sidecar(
    dataset_id: str,
    metadata: dict,
    image_width: int,
    image_height: int,
    fps: int,
    selected_names: list[str],
) -> dict:
    return {
        "schema_version": "farpoint.dataset.v1",
        "dataset_id": dataset_id,
        "format": "lerobot",
        "format_version": "v3",
        "split": "train",
        "task": {
            "name": TASK_NAME,
            "instruction": metadata.get("language_instruction", TASK_INSTRUCTION),
        },
        "robot": {
            "name": "ur10e",
            "gripper": "robotiq_2f85",
            "arm_dof": 6,
            "gripper_dof": 1,
        },
        "simulation": {
            "simulator": metadata.get("simulator", "Isaac Sim"),
            "image": metadata.get("image", "nvcr.io/nvidia/isaac-sim:6.0.0"),
            "physics": "PhysX",
            "control_mode": "articulation_drive",
        },
        "recording": {
            "fps": fps,
            "cameras": ["observation.images.front"],
            "image_width": image_width,
            "image_height": image_height,
        },
    }


def export_episode(episode_dir: Path, output_dir: Path, dataset_id: str) -> Path:
    metadata_path = episode_dir / "metadata.json"
    observations_path = episode_dir / "observations.jsonl"
    required = [metadata_path, observations_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"episode is missing required artifacts: {missing}")

    metadata = read_json(metadata_path)
    observations = read_jsonl(observations_path)
    if not observations:
        raise ValueError("episode contains no observations")

    selected_names = resolve_controlled_joint_names(metadata, observations[0])
    fps = infer_fps(observations)
    first_image_path = episode_dir / observations[0]["rgb_path"]
    from PIL import Image

    with Image.open(first_image_path) as image:
        image = image.convert("RGB")
        image_width, image_height = image.size

    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": selected_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": selected_names,
        },
        "observation.images.front": {
            # LeRobot's ``video`` dtype is what triggers MP4 encoding when
            # ``use_videos=True``; ``image`` would store PNG-like frames only.
            "dtype": "video",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channel"],
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["done"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=dataset_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type="ur10e_robotiq_2f85",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=2,
        video_backend="pyav",
    )
    try:
        for index, row in enumerate(observations):
            rgb_path = episode_dir / row["rgb_path"]
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            state = select_joint_values(row, selected_names, "joint_positions")
            action = select_joint_values(row, selected_names, "action_joint_positions")
            dataset.add_frame(
                {
                    "observation.state": state,
                    "action": action,
                    "observation.images.front": rgb,
                    "next.done": np.asarray([index == len(observations) - 1], dtype=np.bool_),
                    "task": TASK_INSTRUCTION,
                }
            )
        dataset.save_episode(parallel_encoding=False)
        dataset.finalize()
    except Exception:
        # LeRobot creates temporary files while writing; leave no misleading
        # half-exported dataset at the requested destination.
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    sidecar = build_sidecar(
        dataset_id,
        metadata,
        image_width,
        image_height,
        fps,
        selected_names,
    )
    (output_dir / "meta" / "farpoint_v1.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "meta" / "source_episode.json").write_text(
        json.dumps(
            {
                "episode_dir": str(episode_dir.resolve()),
                "episode_id": metadata.get("episode_id"),
                "episode_seed": metadata.get("episode_seed"),
                "observation_count": len(observations),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-id", default="farpoint_v1_episode_0000")
    args = parser.parse_args()
    output = export_episode(args.episode_dir.resolve(), args.output_dir.resolve(), args.dataset_id)
    print(f"Farpoint V1 LeRobot dataset written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
