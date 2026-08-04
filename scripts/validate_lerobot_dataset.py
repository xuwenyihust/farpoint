#!/usr/bin/env python3
"""Validate Farpoint LeRobot datasets, including v2 cross-file contracts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.contracts import (  # noqa: E402
    SPLITS,
    validate_benchmark_episode_links,
    validate_benchmark_semantics,
    validate_collection_episode_links,
    validate_collection_semantics,
    validate_contract,
    validate_episode_semantics,
)


V2_SIDECAR = "farpoint_v2.json"
V3_SIDECAR = "farpoint_v3.json"
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


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def add_error(result: dict[str, Any], message: str) -> None:
    result["errors"].append(message)


def validate_v1_sidecar(sidecar: dict[str, Any], result: dict[str, Any], name: str) -> None:
    legacy = name == LEGACY_SIDECAR
    expected = "robotsim.dataset.v1" if legacy else "farpoint.dataset.v1"
    required = {
        "schema_version", "dataset_id", "format", "format_version", "split", "task",
        "robot", "simulation", "recording",
    }
    for key in sorted(required.difference(sidecar)):
        add_error(result, f"sidecar is missing required field: {key}")
    if sidecar.get("schema_version") != expected:
        add_error(result, f"sidecar schema_version must be {expected}")
    if sidecar.get("format") != "lerobot" or sidecar.get("format_version") != "v3":
        add_error(result, "sidecar must use LeRobot v3")
    task = sidecar.get("task", {})
    if not isinstance(task, dict) or not task.get("name") or not task.get("instruction"):
        add_error(result, "sidecar task must include name and instruction")
    robot = sidecar.get("robot", {})
    for key, expected_value in (
        ("name", "ur10e"), ("gripper", "robotiq_2f85"), ("arm_dof", 6), ("gripper_dof", 1)
    ):
        if not isinstance(robot, dict) or robot.get(key) != expected_value:
            add_error(result, f"sidecar robot.{key} must be {expected_value!r}")
    recording = sidecar.get("recording", {})
    if "observation.images.front" not in recording.get("cameras", []):
        add_error(result, "sidecar recording must include observation.images.front")
    for key in ("fps", "image_width", "image_height"):
        if not isinstance(recording, dict) or recording.get(key, 0) <= 0:
            add_error(result, f"sidecar recording.{key} must be positive")


def validate_v3_sidecar(sidecar: dict[str, Any], result: dict[str, Any]) -> None:
    for error in validate_contract(sidecar):
        add_error(result, f"dataset contract: {error}")
    if sidecar.get("format") != "lerobot" or sidecar.get("format_version") != "v3":
        add_error(result, "SO-101 sidecar must use LeRobot v3")
    robot = sidecar.get("robot") or {}
    for key, expected in (("name", "so101"), ("gripper", "so101_jaw"), ("arm_dof", 5), ("gripper_dof", 1)):
        if robot.get(key) != expected:
            add_error(result, f"sidecar robot.{key} must be {expected!r}")
    recording = sidecar.get("recording") or {}
    for camera in ("observation.images.front", "observation.images.wrist"):
        if camera not in recording.get("cameras", []):
            add_error(result, f"sidecar recording must include {camera}")
    for key in ("fps", "image_width", "image_height"):
        if not isinstance(recording.get(key), (int, float)) or recording[key] <= 0:
            add_error(result, f"sidecar recording.{key} must be positive")


def validate_info(info: dict[str, Any], result: dict[str, Any]) -> None:
    features = info.get("features") if isinstance(info, dict) else None
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


def validate_layout(root: Path, result: dict[str, Any], sidecar_name: str) -> None:
    meta = root / "meta"
    required_files = [meta / sidecar_name, meta / "info.json", meta / "stats.json"]
    for path in required_files:
        if not path.is_file():
            add_error(result, f"missing required file: {path.relative_to(root)}")
    if not list((root / "data").rglob("*.parquet")):
        add_error(result, "data/ must contain at least one Parquet shard")
    if not list((meta / "episodes").rglob("*.parquet")):
        add_error(result, "meta/episodes/ must contain at least one Parquet shard")
    if not (meta / "tasks.parquet").is_file() and not (meta / "tasks.jsonl").is_file():
        add_error(result, "meta/ must contain tasks.parquet or tasks.jsonl")
    if not list((root / "videos" / "observation.images.front").rglob("*.mp4")):
        add_error(result, "front camera directory must contain at least one MP4 shard")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - data dependency guard
        raise RuntimeError("pyarrow is required to validate Parquet metadata") from error
    return pq.read_table(path).to_pylist()


def read_episode_metadata(root: Path) -> list[dict[str, Any]]:
    jsonl = root / "meta" / "episode_metadata.jsonl"
    parquet = root / "meta" / "episode_metadata.parquet"
    if jsonl.is_file():
        return _read_jsonl(jsonl)
    if parquet.is_file():
        return _read_parquet(parquet)
    raise FileNotFoundError("missing normalized episode metadata")


def read_tasks(root: Path) -> list[dict[str, Any]]:
    jsonl = root / "meta" / "tasks.jsonl"
    parquet = root / "meta" / "tasks.parquet"
    if jsonl.is_file():
        return _read_jsonl(jsonl)
    return _read_parquet(parquet)


def task_instruction(row: dict[str, Any]) -> str | None:
    """Read task text from JSONL or the indexed LeRobot v3 Parquet layout."""
    return row.get("task") or row.get("instruction") or row.get("__index_level_0__")


def _read_required_parquet_rows(
    paths: list[Path], root: Path, label: str, result: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        relative = path.relative_to(root)
        if path.stat().st_size == 0:
            add_error(result, f"{label} Parquet is empty: {relative}")
            continue
        try:
            shard_rows = _read_parquet(path)
        except (OSError, ValueError, RuntimeError) as error:
            add_error(result, f"cannot read {label} Parquet {relative}: {error}")
            continue
        if not shard_rows:
            add_error(result, f"{label} Parquet has no rows: {relative}")
            continue
        rows.extend(shard_rows)
    return rows


def _as_done(value: Any) -> bool | None:
    if isinstance(value, (list, tuple)):
        return bool(value[0]) if len(value) == 1 else None
    if isinstance(value, bool):
        return value
    return None


def _finite_vector(value: Any, expected_length: int) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == expected_length
        and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
    )


def _feature_length(info: dict[str, Any], feature: str) -> int:
    definition = (info.get("features") or {}).get(feature) or {}
    names = definition.get("names")
    if isinstance(names, (list, tuple)) and names:
        return len(names)
    shape = definition.get("shape")
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], int):
        return shape[0]
    return 0


def _decode_video_frames(path: Path) -> int:
    try:
        import av
    except ImportError as error:  # pragma: no cover - data dependency guard
        raise RuntimeError("PyAV is required to validate dataset videos") from error
    with av.open(str(path)) as container:
        streams = list(container.streams.video)
        if not streams:
            raise ValueError("file contains no video stream")
        return sum(1 for _ in container.decode(streams[0]))


def validate_camera_video_frames(
    root: Path,
    camera: str,
    expected_frames: int,
    result: dict[str, Any],
) -> None:
    """Decode every shard for one camera and match it to the data table."""
    paths = sorted((root / "videos" / camera).rglob("*.mp4"))
    if not paths:
        add_error(result, f"{camera} directory must contain at least one MP4 shard")
        return
    decoded_frames = 0
    for path in paths:
        relative = path.relative_to(root)
        if path.stat().st_size == 0:
            add_error(result, f"video is empty: {relative}")
            continue
        try:
            decoded = _decode_video_frames(path)
        except (OSError, ValueError, RuntimeError) as error:
            add_error(result, f"cannot decode video {relative}: {error}")
            continue
        if decoded == 0:
            add_error(result, f"video has no decodable frames: {relative}")
        decoded_frames += decoded
    if decoded_frames != expected_frames:
        add_error(result, f"{camera} decoded frame count does not match data Parquet")


def validate_v2_artifacts(
    root: Path,
    episodes: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    info: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Read and cross-check the LeRobot tables and videos used by consumers."""
    task_indexes = [row.get("task_index") for row in tasks]
    task_names = [task_instruction(row) for row in tasks]
    if any(not isinstance(index, int) or index < 0 for index in task_indexes):
        add_error(result, "LeRobot tasks must have non-negative integer task_index values")
    if len(set(task_indexes)) != len(task_indexes):
        add_error(result, "LeRobot task indexes must be unique")
    if len(set(task_names)) != len(task_names) or any(not name for name in task_names):
        add_error(result, "LeRobot task instructions must be present and unique")
    task_index_by_instruction = dict(zip(task_names, task_indexes, strict=False))

    data_rows = _read_required_parquet_rows(
        sorted((root / "data").rglob("*.parquet")), root, "data", result
    )
    standard_episodes = _read_required_parquet_rows(
        sorted((root / "meta" / "episodes").rglob("*.parquet")),
        root,
        "episode metadata",
        result,
    )
    if not data_rows or not standard_episodes:
        return

    required_columns = {
        "observation.state", "action", "timestamp", "frame_index", "episode_index",
        "task_index", "next.done",
    }
    for field in sorted(required_columns):
        if any(field not in row for row in data_rows):
            add_error(result, f"data Parquet is missing values for required field: {field}")

    rows_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(data_rows):
        episode_index = row.get("episode_index")
        if not isinstance(episode_index, int):
            add_error(result, f"data row {row_index} has an invalid episode_index")
            continue
        rows_by_episode[episode_index].append(row)

    expected_episode_indexes = set(range(len(episodes)))
    if set(rows_by_episode) != expected_episode_indexes:
        add_error(result, "data Parquet episode indexes do not match normalized metadata")
    state_length = _feature_length(info, "observation.state")
    action_length = _feature_length(info, "action")
    if state_length <= 0:
        add_error(result, "observation.state must declare a positive vector length")
    if action_length <= 0:
        add_error(result, "action must declare a positive vector length")

    for episode_index, episode in enumerate(episodes):
        rows = rows_by_episode.get(episode_index, [])
        expected_frames = (episode.get("recording") or {}).get("frame_count")
        if len(rows) != expected_frames:
            add_error(result, f"episode {episode_index} frame count does not match recording metadata")
        if [row.get("frame_index") for row in rows] != list(range(len(rows))):
            add_error(result, f"episode {episode_index} frame indexes are not contiguous")
        timestamps = [row.get("timestamp") for row in rows]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in timestamps):
            add_error(result, f"episode {episode_index} has invalid timestamps")
        elif any(right <= left for left, right in zip(timestamps, timestamps[1:], strict=False)):
            add_error(result, f"episode {episode_index} timestamps are not strictly increasing")
        done_values = [_as_done(row.get("next.done")) for row in rows]
        if not rows or done_values != [False] * (len(rows) - 1) + [True]:
            add_error(result, f"episode {episode_index} must end with exactly one terminal frame")
        if any(not _finite_vector(row.get("observation.state"), state_length) for row in rows):
            add_error(result, f"episode {episode_index} has invalid observation.state values")
        if any(not _finite_vector(row.get("action"), action_length) for row in rows):
            add_error(result, f"episode {episode_index} has invalid action values")
        instruction = (episode.get("task") or {}).get("instruction")
        expected_task_index = task_index_by_instruction.get(instruction)
        if expected_task_index is None:
            add_error(result, f"episode {episode_index} task instruction has no task index")
        elif any(row.get("task_index") != expected_task_index for row in rows):
            add_error(result, f"episode {episode_index} data rows use the wrong task_index")

    standard_by_index = {row.get("episode_index"): row for row in standard_episodes}
    if len(standard_by_index) != len(standard_episodes):
        add_error(result, "LeRobot episode metadata indexes must be unique")
    if set(standard_by_index) != expected_episode_indexes:
        add_error(result, "LeRobot episode metadata indexes do not match normalized metadata")
    offset = 0
    for episode_index, episode in enumerate(episodes):
        row = standard_by_index.get(episode_index) or {}
        length = (episode.get("recording") or {}).get("frame_count")
        if row.get("length") != length:
            add_error(result, f"LeRobot episode {episode_index} length mismatch")
        if row.get("dataset_from_index") != offset or row.get("dataset_to_index") != offset + length:
            add_error(result, f"LeRobot episode {episode_index} frame offsets are invalid")
        offset += length
    if info.get("total_episodes") is not None and info.get("total_episodes") != len(episodes):
        add_error(result, "meta/info.json total_episodes does not match metadata")
    if info.get("total_frames") is not None and info.get("total_frames") != len(data_rows):
        add_error(result, "meta/info.json total_frames does not match data Parquet")

    validate_camera_video_frames(
        root, "observation.images.front", len(data_rows), result
    )


def validate_v2_dataset(
    root: Path,
    sidecar: dict[str, Any],
    info: dict[str, Any],
    result: dict[str, Any],
    evidence_path: Path | None,
) -> None:
    for error in validate_contract(sidecar):
        add_error(result, f"dataset contract: {error}")
    try:
        episodes = read_episode_metadata(root)
    except (OSError, ValueError, RuntimeError) as error:
        add_error(result, f"cannot read episode metadata: {error}")
        return

    identities = []
    for index, episode in enumerate(episodes):
        for error in validate_contract(episode):
            add_error(result, f"episode metadata row {index}: {error}")
        for error in validate_episode_semantics(episode):
            add_error(result, f"episode metadata row {index}: {error}")
        identities.append(episode.get("identity") or {})

    episode_ids = [identity.get("episode_id") for identity in identities]
    trial_ids = [identity.get("trial_id") for identity in identities]
    indexes = [identity.get("dataset_episode_index") for identity in identities]
    if len(set(episode_ids)) != len(episode_ids):
        add_error(result, "episode ids must be unique")
    if len(set(trial_ids)) != len(trial_ids):
        add_error(result, "trial ids must be unique")
    if indexes != list(range(len(episodes))):
        add_error(result, "dataset episode indexes must be contiguous and ordered")

    observed_splits = Counter(identity.get("split") for identity in identities)
    for split in SPLITS:
        if observed_splits[split] != sidecar.get("splits", {}).get(split):
            add_error(result, f"split count mismatch for {split}")
    expected_ranges = {}
    start = 0
    for split in SPLITS:
        stop = start + observed_splits[split]
        expected_ranges[split] = f"{start}:{stop}"
        start = stop
    if info.get("splits") != expected_ranges:
        add_error(result, "meta/info.json split ranges do not match episode metadata")

    expected_split_sequence = [
        split for split in SPLITS for _ in range(observed_splits[split])
    ]
    if [identity.get("split") for identity in identities] != expected_split_sequence:
        add_error(result, "episode metadata must be contiguous in train/validation/test order")

    dataset_tasks = sidecar.get("tasks", [])
    task_by_id = {task.get("task_id"): task for task in dataset_tasks}
    if len(task_by_id) != len(dataset_tasks):
        add_error(result, "dataset task ids must be unique")
    dataset_task_instructions = [task.get("instruction") for task in dataset_tasks]
    if len(set(dataset_task_instructions)) != len(dataset_task_instructions):
        add_error(result, "dataset task instructions must map to unique task ids")
    for index, episode in enumerate(episodes):
        task = episode.get("task") or {}
        if task_by_id.get(task.get("task_id")) != task:
            add_error(result, f"episode metadata row {index} does not resolve to a dataset task")
        if not (episode.get("outcome") or {}).get("success"):
            add_error(result, f"episode metadata row {index} violates successful_only policy")
        if not (episode.get("outcome") or {}).get("dataset_valid"):
            add_error(result, f"episode metadata row {index} is not dataset-valid")
        embodiment = episode.get("embodiment") or {}
        dataset_robot = sidecar.get("robot") or {}
        for episode_field, dataset_field in (
            ("robot", "name"),
            ("gripper", "gripper"),
            ("arm_dof", "arm_dof"),
            ("gripper_dof", "gripper_dof"),
        ):
            if embodiment.get(episode_field) != dataset_robot.get(dataset_field):
                add_error(
                    result,
                    f"episode metadata row {index} embodiment.{episode_field} mismatch",
                )
        provenance = episode.get("provenance") or {}
        simulation = sidecar.get("simulation") or {}
        if provenance.get("simulator") != simulation.get("simulator"):
            add_error(result, f"episode metadata row {index} has a different simulator")
        if provenance.get("physics_engine") != simulation.get("physics"):
            add_error(result, f"episode metadata row {index} has a different physics engine")
        if provenance.get("simulator_image") != simulation.get("image"):
            add_error(result, f"episode metadata row {index} has a different simulator image")
        if provenance.get("simulator_image_digest") != simulation.get("image_digest"):
            add_error(result, f"episode metadata row {index} has a different image digest")
        recording = episode.get("recording") or {}
        dataset_recording = sidecar.get("recording") or {}
        for field in ("fps", "cameras", "image_width", "image_height"):
            if recording.get(field) != dataset_recording.get(field):
                add_error(result, f"episode metadata row {index} recording.{field} mismatch")

    try:
        lerobot_tasks = read_tasks(root)
        task_instructions = {
            task_instruction(row)
            for row in lerobot_tasks
            if isinstance(row, dict)
        }
        for task in task_by_id.values():
            if task.get("instruction") not in task_instructions:
                add_error(result, f"task is missing from LeRobot tasks table: {task.get('task_id')}")
        validate_v2_artifacts(root, episodes, lerobot_tasks, info, result)
    except (OSError, ValueError, RuntimeError) as error:
        add_error(result, f"cannot read LeRobot tasks: {error}")

    if evidence_path:
        try:
            evidence = read_json(evidence_path)
            for error in validate_contract(evidence):
                add_error(result, f"release evidence contract: {error}")
            if evidence.get("schema_version") == "farpoint.collection.v1":
                for error in validate_collection_semantics(evidence):
                    add_error(result, f"collection acceptance: {error}")
                for error in validate_collection_episode_links(evidence, episodes):
                    add_error(result, f"collection link: {error}")
            else:
                for error in validate_benchmark_semantics(evidence):
                    add_error(result, f"benchmark acceptance: {error}")
                for error in validate_benchmark_episode_links(evidence, episodes):
                    add_error(result, f"benchmark link: {error}")
        except (OSError, json.JSONDecodeError) as error:
            add_error(result, f"cannot read release evidence manifest: {error}")
    result["checks"]["episode_contracts"] = bool(episodes) and not result["errors"]


def validate_v3_dataset(root: Path, sidecar: dict[str, Any], info: dict[str, Any], result: dict[str, Any]) -> None:
    validate_v3_sidecar(sidecar, result)
    try:
        episodes = read_episode_metadata(root)
        tasks = read_tasks(root)
    except (OSError, ValueError, RuntimeError) as error:
        add_error(result, f"cannot read SO-101 metadata: {error}")
        return
    for index, episode in enumerate(episodes):
        for error in validate_contract(episode):
            add_error(result, f"episode metadata row {index}: {error}")
        for error in validate_episode_semantics(episode):
            add_error(result, f"episode metadata row {index}: {error}")
        outcome = episode.get("outcome") or {}
        if not outcome.get("success") or not outcome.get("dataset_valid"):
            add_error(result, f"episode metadata row {index} is not eligible")
    identities = [episode.get("identity") or {} for episode in episodes]
    indexes = [identity.get("dataset_episode_index") for identity in identities]
    if indexes != list(range(len(episodes))):
        add_error(result, "SO-101 episode indexes must be contiguous")
    observed = Counter(identity.get("split") for identity in identities)
    for split in SPLITS:
        if observed[split] != (sidecar.get("splits") or {}).get(split):
            add_error(result, f"SO-101 split count mismatch for {split}")
    expected_ranges = {}
    start = 0
    for split in SPLITS:
        stop = start + observed[split]
        expected_ranges[split] = f"{start}:{stop}"
        start = stop
    if info.get("splits") != expected_ranges:
        add_error(result, "SO-101 split ranges do not match metadata")
    task_names = {task_instruction(row) for row in tasks}
    for index, episode in enumerate(episodes):
        if (episode.get("task") or {}).get("instruction") not in task_names:
            add_error(result, f"SO-101 episode task is missing at row {index}")
    validate_v2_artifacts(root, episodes, tasks, info, result)
    expected_frames = sum(
        int((episode.get("recording") or {}).get("frame_count") or 0)
        for episode in episodes
    )
    validate_camera_video_frames(
        root, "observation.images.wrist", expected_frames, result
    )
    result["checks"]["episode_contracts"] = bool(episodes) and not result["errors"]


def validate_dataset(root: Path, evidence_path: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    result: dict[str, Any] = {
        "valid": False,
        "schema_version": None,
        "dataset_id": None,
        "errors": [],
        "warnings": [],
        "checks": {},
    }
    if not root.is_dir():
        add_error(result, f"dataset root does not exist: {root}")
        return result
    candidates = [V3_SIDECAR, V2_SIDECAR, PRIMARY_SIDECAR, LEGACY_SIDECAR]
    sidecar_path = next((root / "meta" / name for name in candidates if (root / "meta" / name).is_file()), None)
    if sidecar_path is None:
        add_error(result, "dataset has no supported Farpoint sidecar")
        return result
    sidecar_name = sidecar_path.name
    result["compatibility_mode"] = {
        V3_SIDECAR: "so101-v3",
        V2_SIDECAR: "v2",
        PRIMARY_SIDECAR: "current",
        LEGACY_SIDECAR: "legacy",
    }[sidecar_name]
    try:
        sidecar = read_json(sidecar_path)
        info = read_json(root / "meta" / "info.json")
    except (OSError, json.JSONDecodeError) as error:
        add_error(result, f"cannot read dataset metadata: {error}")
        return result
    result["schema_version"] = sidecar.get("schema_version")
    result["dataset_id"] = sidecar.get("dataset_id")
    validate_info(info, result)
    validate_layout(root, result, sidecar_name)
    if sidecar_name == V3_SIDECAR:
        validate_v3_dataset(root, sidecar, info, result)
    elif sidecar_name == V2_SIDECAR:
        validate_v2_dataset(root, sidecar, info, result, evidence_path)
    else:
        validate_v1_sidecar(sidecar, result, sidecar_name)
    result["checks"]["layout"] = not any(
        error.startswith(("missing required file", "data/", "meta/", "front camera"))
        for error in result["errors"]
    )
    result["checks"]["sidecar"] = not any("sidecar" in error for error in result["errors"])
    result["valid"] = not result["errors"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    evidence = parser.add_mutually_exclusive_group()
    evidence.add_argument("--benchmark-manifest", type=Path)
    evidence.add_argument("--collection-manifest", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = validate_dataset(
        args.dataset_root, args.collection_manifest or args.benchmark_manifest
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
