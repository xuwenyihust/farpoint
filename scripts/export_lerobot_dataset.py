#!/usr/bin/env python3
"""Export a manifest-selected, multi-task Farpoint dataset as LeRobot v3."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from export_lerobot_v1_episode import (  # noqa: E402
    infer_fps,
    read_json,
    read_jsonl,
    resolve_controlled_joint_names,
    select_joint_values,
)
from farpoint.contracts import SPLITS, validate_contract  # noqa: E402
from farpoint.episode_metadata import normalize_episode_metadata_v2  # noqa: E402


SELECTION_SCHEMA_VERSION = "farpoint.export-selection.v1"


def load_selection_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    if manifest.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError(f"selection manifest must use {SELECTION_SCHEMA_VERSION}")
    if not manifest.get("dataset_id"):
        raise ValueError("selection manifest must define dataset_id")
    sources = [name for name in ("benchmark_id", "collection_id") if manifest.get(name)]
    if len(sources) > 1:
        raise ValueError("selection manifest cannot define both benchmark_id and collection_id")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("selection manifest must contain at least one episode")
    for index, entry in enumerate(episodes):
        if entry.get("split") not in SPLITS:
            raise ValueError(f"selection episode {index} has an invalid split")
        if not entry.get("episode_dir") or not entry.get("trial_id"):
            raise ValueError(f"selection episode {index} must define episode_dir and trial_id")
    return manifest


def order_selection(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make LeRobot episode indexes contiguous within each public split."""
    split_order = {name: index for index, name in enumerate(SPLITS)}
    return sorted(entries, key=lambda item: (split_order[item["split"]], item["trial_id"]))


def lerobot_split_ranges(split_counts: dict[str, int]) -> dict[str, str]:
    ranges = {}
    start = 0
    for split in SPLITS:
        stop = start + split_counts.get(split, 0)
        ranges[split] = f"{start}:{stop}"
        start = stop
    return ranges


def build_dataset_sidecar(
    dataset_id: str,
    records: list[dict[str, Any]],
    *,
    fps: int,
    image_width: int,
    image_height: int,
    selected_names: list[str],
    cameras: list[str] | None = None,
    source_contract: str = "benchmark",
    selection_policy: str | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot build a dataset sidecar without episodes")
    cameras = cameras or ["observation.images.front"]
    tasks = {}
    task_id_by_instruction = {}
    split_counts = Counter()
    first = records[0]
    for record in records:
        task = record["task"]
        previous = tasks.setdefault(task["task_id"], task)
        if previous != task:
            raise ValueError(f"task id has conflicting definitions: {task['task_id']}")
        existing_task_id = task_id_by_instruction.setdefault(task["instruction"], task["task_id"])
        if existing_task_id != task["task_id"]:
            raise ValueError("different task ids cannot share one LeRobot instruction")
        split_counts[record["identity"]["split"]] += 1
        for key in ("robot", "gripper", "arm_dof", "gripper_dof"):
            if record["embodiment"][key] != first["embodiment"][key]:
                raise ValueError(f"episodes have different embodiment.{key} values")
        for key in ("simulator", "physics_engine", "simulator_image", "simulator_image_digest"):
            if record["provenance"][key] != first["provenance"][key]:
                raise ValueError(f"episodes have different {key} values")
        recording = record["recording"]
        if (
            recording["fps"] != fps
            or recording["image_width"] != image_width
            or recording["image_height"] != image_height
        ):
            raise ValueError("episode recording metadata does not match the export")
    sidecar = {
        "schema_version": "farpoint.dataset.v2",
        "dataset_id": dataset_id,
        "format": "lerobot",
        "format_version": "v3",
        "demonstration_policy": "successful_only",
        "splits": {split: split_counts[split] for split in SPLITS},
        "tasks": [tasks[key] for key in sorted(tasks)],
        "robot": {
            "name": first["embodiment"]["robot"],
            "gripper": first["embodiment"]["gripper"],
            "arm_dof": first["embodiment"]["arm_dof"],
            "gripper_dof": first["embodiment"]["gripper_dof"],
        },
        "simulation": {
            "simulator": first["provenance"]["simulator"],
            "image": first["provenance"]["simulator_image"],
            "image_digest": first["provenance"]["simulator_image_digest"],
            "physics": first["provenance"]["physics_engine"],
        },
        "recording": {
            "fps": fps,
            "cameras": cameras,
            "image_width": image_width,
            "image_height": image_height,
            "state_features": selected_names,
            "action_features": selected_names,
        },
        "contracts": {
            "episode": "farpoint.episode.v2",
            "variation": "farpoint.variation.v2",
            source_contract: (
                "farpoint.collection.v1"
                if source_contract == "collection"
                else "farpoint.benchmark.v2"
            ),
        },
    }
    if selection_policy:
        sidecar["selection_policy"] = selection_policy
    errors = validate_contract(sidecar)
    if errors:
        raise ValueError("invalid farpoint.dataset.v2 sidecar: " + "; ".join(errors))
    return sidecar


def observation_rgb_path(row: dict[str, Any], camera: str) -> str:
    """Return the recorded RGB artifact for a LeRobot camera feature."""
    if camera == "observation.images.front":
        return row["rgb_path"]
    key = camera.removeprefix("observation.images.") + "_rgb_path"
    path = row.get(key)
    if not path:
        raise ValueError(f"observation is missing {key} for {camera}")
    return path


def _write_split_ranges(output_dir: Path, split_counts: dict[str, int]) -> None:
    info_path = output_dir / "meta" / "info.json"
    info = read_json(info_path)
    info["splits"] = lerobot_split_ranges(split_counts)
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_dataset(
    manifest_path: Path,
    output_dir: Path,
    *,
    dataset_class: type | None = None,
) -> Path:
    """Export all successful manifest entries and write canonical v2 metadata."""
    manifest = load_selection_manifest(manifest_path)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    entries = order_selection(manifest["episodes"])
    loaded = []
    for entry in entries:
        episode_dir = Path(entry["episode_dir"]).expanduser().resolve()
        metadata = read_json(episode_dir / "metadata.json")
        observations = read_jsonl(episode_dir / "observations.jsonl")
        metrics = read_json(episode_dir / "metrics.json")
        if not observations:
            raise ValueError(f"episode has no observations: {episode_dir}")
        if not metrics.get("success") or not metrics.get("dataset_valid"):
            raise ValueError(f"release selection contains an unsuccessful episode: {episode_dir}")
        loaded.append((entry, episode_dir, metadata, observations, metrics))

    from PIL import Image

    _, first_dir, first_metadata, first_rows, _ = loaded[0]
    selected_names = resolve_controlled_joint_names(first_metadata, first_rows[0])
    fps = infer_fps(first_rows)
    with Image.open(first_dir / first_rows[0]["rgb_path"]) as image:
        image_width, image_height = image.size
    cameras = list((first_metadata.get("recording") or {}).get("cameras") or [])
    if not cameras:
        cameras = ["observation.images.front"]

    normalized_records = []
    for index, (entry, episode_dir, metadata, rows, metrics) in enumerate(loaded):
        if infer_fps(rows) != fps:
            raise ValueError(f"episode has a different recording rate: {episode_dir}")
        with Image.open(episode_dir / rows[0]["rgb_path"]) as image:
            if image.size != (image_width, image_height):
                raise ValueError(f"episode has a different camera resolution: {episode_dir}")
        if list((metadata.get("recording") or {}).get("cameras") or ["observation.images.front"]) != cameras:
            raise ValueError("episodes have different recording camera contracts")
        for camera in cameras:
            with Image.open(episode_dir / observation_rgb_path(rows[0], camera)) as image:
                if image.size != (image_width, image_height):
                    raise ValueError(f"episode camera has a different resolution: {episode_dir} ({camera})")
        enriched = dict(metadata)
        enriched["recording"] = {
            **metadata.get("recording", {}),
            "fps": fps,
            "cameras": cameras,
            "image_width": image_width,
            "image_height": image_height,
            "frame_count": len(rows),
        }
        normalized_records.append(
            normalize_episode_metadata_v2(
                enriched,
                metrics,
                split=entry["split"],
                dataset_episode_index=index,
                trial_id=entry["trial_id"],
            )
        )

    if dataset_class is None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset_class = LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": selected_names},
        "action": {"dtype": "float32", "shape": (7,), "names": selected_names},
        "next.done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
    }
    features.update({
        camera: {
            "dtype": "video",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channel"],
        }
        for camera in cameras
    })
    dataset = dataset_class.create(
        repo_id=manifest["dataset_id"],
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
        for (_, episode_dir, _, rows, _), record in zip(loaded, normalized_records):
            instruction = record["task"]["instruction"]
            for frame_index, row in enumerate(rows):
                frame = {
                        "observation.state": select_joint_values(
                            row, selected_names, "joint_positions"
                        ),
                        "action": select_joint_values(
                            row, selected_names, "action_joint_positions"
                        ),
                        "next.done": np.asarray([frame_index == len(rows) - 1], dtype=np.bool_),
                        "task": instruction,
                    }
                for camera in cameras:
                    with Image.open(episode_dir / observation_rgb_path(row, camera)) as image:
                        frame[camera] = np.asarray(image.convert("RGB"), dtype=np.uint8)
                dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=False)
        dataset.finalize()
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    sidecar = build_dataset_sidecar(
        manifest["dataset_id"],
        normalized_records,
        fps=fps,
        image_width=image_width,
        image_height=image_height,
        selected_names=selected_names,
        cameras=cameras,
        source_contract="collection" if manifest.get("collection_id") else "benchmark",
        selection_policy=manifest.get("selection_policy"),
    )
    meta_dir = output_dir / "meta"
    (meta_dir / "farpoint_v2.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (meta_dir / "episode_metadata.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in normalized_records),
        encoding="utf-8",
    )
    _write_split_ranges(output_dir, sidecar["splits"])
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = export_dataset(args.manifest.resolve(), args.output_dir.resolve())
    print(f"Farpoint dataset written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
