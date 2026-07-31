#!/usr/bin/env python3
"""Validate the structural Farpoint V1 / LeRobot dataset contract."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "farpoint_v1.schema.json"
LEGACY_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "robotsim_v1.schema.json"
PRIMARY_SIDECAR = "farpoint_v1.json"
LEGACY_SIDECAR = "robotsim_v1.json"
REQUIRED_FEATURES = {
    "observation.state",
    "action",
    "observation.images.front",
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
    "next.done",
}


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def add_error(result, message):
    result["errors"].append(message)


def validate_sidecar(sidecar, result, sidecar_name):
    legacy = sidecar_name == LEGACY_SIDECAR
    expected_schema_version = "robotsim.dataset.v1" if legacy else "farpoint.dataset.v1"
    if not isinstance(sidecar, dict):
        add_error(result, f"meta/{sidecar_name} must contain a JSON object")
        return
    required = {
        "schema_version",
        "dataset_id",
        "format",
        "format_version",
        "split",
        "task",
        "robot",
        "simulation",
        "recording",
    }
    for key in sorted(required.difference(sidecar)):
        add_error(result, f"sidecar is missing required field: {key}")
    if sidecar.get("schema_version") != expected_schema_version:
        add_error(result, f"sidecar schema_version must be {expected_schema_version}")
    if sidecar.get("format") != "lerobot":
        add_error(result, "sidecar format must be lerobot")
    if sidecar.get("format_version") != "v3":
        add_error(result, "sidecar format_version must be v3")
    task = sidecar.get("task", {})
    if not isinstance(task, dict) or not task.get("name") or not task.get("instruction"):
        add_error(result, "sidecar task must include name and instruction")
    robot = sidecar.get("robot", {})
    if not isinstance(robot, dict):
        add_error(result, "sidecar robot must be an object")
    else:
        for key, expected in (("name", "ur10e"), ("gripper", "robotiq_2f85"), ("arm_dof", 6), ("gripper_dof", 1)):
            if robot.get(key) != expected:
                add_error(result, f"sidecar robot.{key} must be {expected!r}")
    recording = sidecar.get("recording", {})
    if not isinstance(recording, dict):
        add_error(result, "sidecar recording must be an object")
    else:
        cameras = recording.get("cameras", [])
        if "observation.images.front" not in cameras:
            add_error(result, "sidecar recording must include observation.images.front")
        for key in ("fps", "image_width", "image_height"):
            if recording.get(key, 0) <= 0:
                add_error(result, f"sidecar recording.{key} must be positive")


def validate_info(info, result):
    if not isinstance(info, dict):
        add_error(result, "meta/info.json must contain a JSON object")
        return
    features = info.get("features")
    if not isinstance(features, dict):
        add_error(result, "meta/info.json must contain a features object")
        return
    missing = sorted(REQUIRED_FEATURES.difference(features))
    for feature in missing:
        add_error(result, f"meta/info.json is missing feature: {feature}")
    result["checks"]["required_features"] = not missing
    for feature in sorted(REQUIRED_FEATURES.intersection(features)):
        definition = features[feature]
        if not isinstance(definition, dict):
            add_error(result, f"feature definition must be an object: {feature}")
            continue
        if "dtype" not in definition:
            add_error(result, f"feature is missing dtype: {feature}")
        if feature not in {"timestamp", "frame_index", "episode_index", "task_index", "next.done"} and "shape" not in definition:
            add_error(result, f"feature is missing shape: {feature}")


def validate_layout(root, result, sidecar_name):
    meta = root / "meta"
    data = root / "data"
    videos = root / "videos"
    required_files = [meta / sidecar_name, meta / "info.json", meta / "stats.json"]
    for path in required_files:
        if not path.is_file():
            add_error(result, f"missing required file: {path.relative_to(root)}")
    if not list(data.rglob("*.parquet")):
        add_error(result, "data/ must contain at least one Parquet shard")
    if not list((meta / "episodes").rglob("*.parquet")):
        add_error(result, "meta/episodes/ must contain at least one Parquet shard")
    task_files = list(meta.glob("tasks.parquet")) + list(meta.glob("tasks.jsonl"))
    if not task_files:
        add_error(result, "meta/ must contain tasks.parquet or tasks.jsonl")
    camera_dir = videos / "observation.images.front"
    if not list(camera_dir.rglob("*.mp4")):
        add_error(result, "front camera directory must contain at least one MP4 shard")
    result["checks"]["layout"] = not result["errors"]


def validate_dataset(root):
    root = Path(root).resolve()
    result = {
        "valid": False,
        "schema_version": "farpoint.dataset.v1",
        "dataset_id": None,
        "errors": [],
        "warnings": [],
        "checks": {},
    }
    if not root.is_dir():
        add_error(result, f"dataset root does not exist: {root}")
        return result
    primary_sidecar = root / "meta" / PRIMARY_SIDECAR
    legacy_sidecar = root / "meta" / LEGACY_SIDECAR
    sidecar_path = primary_sidecar if primary_sidecar.is_file() else legacy_sidecar
    sidecar_name = sidecar_path.name
    result["compatibility_mode"] = "legacy" if sidecar_name == LEGACY_SIDECAR else "current"
    result["schema_version"] = (
        "robotsim.dataset.v1" if sidecar_name == LEGACY_SIDECAR else "farpoint.dataset.v1"
    )
    info_path = root / "meta" / "info.json"
    if sidecar_path.is_file():
        try:
            sidecar = read_json(sidecar_path)
            result["dataset_id"] = sidecar.get("dataset_id")
            validate_sidecar(sidecar, result, sidecar_name)
            result["checks"]["sidecar"] = not result["errors"]
        except (OSError, json.JSONDecodeError) as error:
            add_error(result, f"cannot read sidecar: {error}")
    if info_path.is_file():
        try:
            validate_info(read_json(info_path), result)
        except (OSError, json.JSONDecodeError) as error:
            add_error(result, f"cannot read info.json: {error}")
    validate_layout(root, result, sidecar_name)
    result["valid"] = not result["errors"]
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate a Farpoint V1 LeRobot dataset layout.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = validate_dataset(args.dataset_root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
