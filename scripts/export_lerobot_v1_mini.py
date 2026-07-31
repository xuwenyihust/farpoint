#!/usr/bin/env python3
"""Export a fixed set of Farpoint episodes as one LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from export_lerobot_v1_episode import (
    TASK_INSTRUCTION,
    build_sidecar,
    infer_fps,
    read_json,
    read_jsonl,
    resolve_controlled_joint_names,
    select_joint_values,
)


def load_episode(episode_dir: Path) -> tuple[dict, list[dict]]:
    metadata = read_json(episode_dir / "metadata.json")
    observations = read_jsonl(episode_dir / "observations.jsonl")
    if not observations:
        raise ValueError(f"episode has no observations: {episode_dir}")
    return metadata, observations


def export_mini(episode_dirs: list[Path], output_dir: Path, dataset_id: str) -> Path:
    if not episode_dirs:
        raise ValueError("at least one episode is required")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    loaded = [(path.resolve(), *load_episode(path.resolve())) for path in episode_dirs]
    from PIL import Image

    first_dir, first_metadata, first_rows = loaded[0]
    selected_names = resolve_controlled_joint_names(first_metadata, first_rows[0])
    fps = infer_fps(first_rows)
    first_rgb = first_dir / first_rows[0]["rgb_path"]
    with Image.open(first_rgb) as image:
        image_width, image_height = image.size

    for episode_dir, metadata, rows in loaded:
        if infer_fps(rows) != fps:
            raise ValueError(f"episode has a different recording rate: {episode_dir}")
        with Image.open(episode_dir / rows[0]["rgb_path"]) as image:
            if image.size != (image_width, image_height):
                raise ValueError(f"episode has a different camera resolution: {episode_dir}")
        if metadata.get("task_name") != first_metadata.get("task_name"):
            raise ValueError(f"episode has a different task: {episode_dir}")

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
    source_records = []
    try:
        for episode_index, (episode_dir, metadata, rows) in enumerate(loaded):
            for index, row in enumerate(rows):
                with Image.open(episode_dir / row["rgb_path"]) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                dataset.add_frame(
                    {
                        "observation.state": select_joint_values(
                            row, selected_names, "joint_positions"
                        ),
                        "action": select_joint_values(
                            row, selected_names, "action_joint_positions"
                        ),
                        "observation.images.front": rgb,
                        "next.done": np.asarray(
                            [index == len(rows) - 1], dtype=np.bool_
                        ),
                        "task": TASK_INSTRUCTION,
                    }
                )
            dataset.save_episode(parallel_encoding=False)
            metrics_path = episode_dir / "metrics.json"
            metrics = read_json(metrics_path) if metrics_path.exists() else {}
            source_records.append(
                {
                    "dataset_episode_index": episode_index,
                    "source_episode_id": metadata.get("episode_id", episode_dir.name),
                    "source_episode_dir": str(episode_dir),
                    "seed": metadata.get("episode_seed"),
                    "success": bool(metrics.get("success")),
                    "failure_category": metrics.get("failure_category"),
                    "frame_count": len(rows),
                }
            )
        dataset.finalize()
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    sidecar = build_sidecar(
        dataset_id,
        first_metadata,
        image_width,
        image_height,
        fps,
        selected_names,
    )
    (output_dir / "meta" / "farpoint_v1.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "meta" / "source_episodes.json").write_text(
        json.dumps(source_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("episode_dirs", nargs="+", type=Path)
    parser.add_argument("--dataset-id", default="farpoint_v1_mini")
    args = parser.parse_args()
    output = export_mini(args.episode_dirs, args.output_dir.resolve(), args.dataset_id)
    print(f"Farpoint V1 mini dataset written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
