import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_lerobot_dataset import validate_dataset

from v2_fixtures import GIT_COMMIT, SHA, dataset_sidecar_v2, episode_metadata_v2


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def valid_sidecar():
    return {
        "schema_version": "farpoint.dataset.v1",
        "dataset_id": "farpoint_ur10e_robotiq_2f85",
        "format": "lerobot",
        "format_version": "v3",
        "split": "train",
        "task": {
            "name": "ur10e_robotiq_single_cube_pick_place",
            "instruction": "Pick up the cube and place it in the target zone.",
        },
        "robot": {
            "name": "ur10e",
            "gripper": "robotiq_2f85",
            "arm_dof": 6,
            "gripper_dof": 1,
        },
        "simulation": {
            "simulator": "Isaac Sim",
            "image": "nvcr.io/nvidia/isaac-sim:6.0.0",
            "physics": "PhysX",
            "control_mode": "articulation_drive",
        },
        "recording": {
            "fps": 20,
            "cameras": ["observation.images.front"],
            "image_width": 640,
            "image_height": 360,
        },
    }


def valid_info():
    scalar = {"dtype": "float32", "shape": []}
    return {
        "features": {
            "observation.state": {"dtype": "float32", "shape": [7]},
            "action": {"dtype": "float32", "shape": [7]},
            "observation.images.front": {"dtype": "video", "shape": [360, 640, 3]},
            "timestamp": scalar,
            "frame_index": {"dtype": "int64", "shape": []},
            "episode_index": {"dtype": "int64", "shape": []},
            "task_index": {"dtype": "int64", "shape": []},
            "next.done": {"dtype": "bool", "shape": []},
        }
    }


def make_valid_dataset(root: Path, legacy=False):
    sidecar = valid_sidecar()
    sidecar_name = "robotsim_v1.json" if legacy else "farpoint_v1.json"
    if legacy:
        sidecar["schema_version"] = "robotsim.dataset.v1"
        sidecar["dataset_id"] = "robotsim_v1"
    write_json(root / "meta" / sidecar_name, sidecar)
    write_json(root / "meta" / "info.json", valid_info())
    write_json(root / "meta" / "stats.json", {})
    (root / "meta" / "tasks.parquet").parent.mkdir(parents=True, exist_ok=True)
    (root / "meta" / "tasks.parquet").touch()
    (root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").parent.mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").touch()
    (root / "data" / "chunk-000" / "file-000.parquet").parent.mkdir(parents=True)
    (root / "data" / "chunk-000" / "file-000.parquet").touch()
    (root / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4").parent.mkdir(parents=True)
    (root / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4").touch()


def test_valid_structural_dataset(tmp_path):
    make_valid_dataset(tmp_path)
    result = validate_dataset(tmp_path)
    assert result["valid"] is True
    assert result["dataset_id"] == "farpoint_ur10e_robotiq_2f85"
    assert result["compatibility_mode"] == "current"
    assert result["errors"] == []


def test_legacy_structural_dataset_remains_readable(tmp_path):
    make_valid_dataset(tmp_path, legacy=True)
    result = validate_dataset(tmp_path)
    assert result["valid"] is True
    assert result["dataset_id"] == "robotsim_v1"
    assert result["compatibility_mode"] == "legacy"


def test_missing_required_feature_fails(tmp_path):
    make_valid_dataset(tmp_path)
    info = valid_info()
    del info["features"]["action"]
    write_json(tmp_path / "meta" / "info.json", info)
    result = validate_dataset(tmp_path)
    assert result["valid"] is False
    assert "meta/info.json is missing feature: action" in result["errors"]


def make_valid_v2_dataset(root: Path):
    pytest = __import__("pytest")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    av = pytest.importorskip("av")
    episodes = [
        episode_metadata_v2(),
        episode_metadata_v2(
            episode_id="episode-0001",
            trial_id="trial-0001",
            split="validation",
            dataset_episode_index=1,
            position=[0.55, 0.05, 0.05],
        ),
        episode_metadata_v2(
            episode_id="episode-0002",
            trial_id="trial-0002",
            split="test",
            dataset_episode_index=2,
            position=[0.6, -0.05, 0.05],
        ),
    ]
    write_json(root / "meta" / "farpoint_v2.json", dataset_sidecar_v2(episodes))
    info = valid_info()
    info["splits"] = {"train": "0:1", "validation": "1:2", "test": "2:3"}
    info["total_episodes"] = 3
    info["total_frames"] = 6
    info["total_tasks"] = 1
    write_json(root / "meta" / "info.json", info)
    write_json(root / "meta" / "stats.json", {})
    tasks_path = root / "meta" / "tasks.parquet"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [{"task_index": 0, "task": episodes[0]["task"]["instruction"]}]
        ),
        tasks_path,
    )
    (root / "meta" / "episode_metadata.jsonl").write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes), encoding="utf-8"
    )
    episode_rows = []
    data_rows = []
    offset = 0
    for episode_index, episode in enumerate(episodes):
        length = episode["recording"]["frame_count"]
        episode_rows.append(
            {
                "episode_index": episode_index,
                "tasks": [episode["task"]["instruction"]],
                "length": length,
                "dataset_from_index": offset,
                "dataset_to_index": offset + length,
            }
        )
        for frame_index in range(length):
            data_rows.append(
                {
                    "observation.state": [0.0] * 7,
                    "action": [0.1] * 7,
                    "timestamp": frame_index * 0.05,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "task_index": 0,
                    "next.done": frame_index == length - 1,
                }
            )
        offset += length
    episode_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(episode_rows), episode_path)
    pq.write_table(pa.Table.from_pylist(data_rows), data_path)

    video_path = (
        root
        / "videos"
        / "observation.images.front"
        / "chunk-000"
        / "file-000.mp4"
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("libx264", rate=20)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    for frame_index in range(len(data_rows)):
        array = np.full((16, 16, 3), frame_index, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    benchmark = {
        "schema_version": "farpoint.benchmark.v2",
        "benchmark_id": "farpoint_v1_3_cube_position",
        "task_id": episodes[0]["task"]["task_id"],
        "git_commit": GIT_COMMIT,
        "config_sha256": SHA,
        "simulator_image_digest": f"sha256:{SHA}",
        "trials": [
            {
                "trial_id": episode["identity"]["trial_id"],
                "episode_id": episode["identity"]["episode_id"],
                "variation_id": episode["variation"]["variation_id"],
                "split": episode["identity"]["split"],
                "status": "completed",
                "success": True,
                "dataset_valid": True,
            }
            for episode in episodes
        ],
        "acceptance": {
            "accepted": True,
            "required_success_rate": 0.9,
            "observed_success_rate": 1.0,
            "required_successes": 3,
            "observed_successes": 3,
        },
    }
    benchmark_path = root / "benchmark.json"
    write_json(benchmark_path, benchmark)
    return benchmark_path


def test_v2_dataset_validates_tasks_splits_variations_and_benchmark(tmp_path):
    benchmark = make_valid_v2_dataset(tmp_path)
    result = validate_dataset(tmp_path, benchmark)
    assert result["valid"] is True, result["errors"]
    assert result["compatibility_mode"] == "v2"


def test_v2_dataset_validates_collection_evidence(tmp_path):
    benchmark_path = make_valid_v2_dataset(tmp_path)
    benchmark = json.loads(benchmark_path.read_text())
    sidecar_path = tmp_path / "meta" / "farpoint_v2.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["contracts"].pop("benchmark")
    sidecar["contracts"]["collection"] = "farpoint.collection.v1"
    write_json(sidecar_path, sidecar)
    attempts = []
    splits = {"train": 0, "validation": 0, "test": 0}
    selected_per_cell = {}
    for index, trial in enumerate(benchmark["trials"]):
        cell_id = f"r00_c{index:02d}"
        splits[trial["split"]] += 1
        selected_per_cell[cell_id] = 1
        attempts.append(
            {
                "trial_id": trial["trial_id"],
                "episode_id": trial["episode_id"],
                "variation_id": trial["variation_id"],
                "cell_id": cell_id,
                "slot": 0,
                "seed": index,
                "object_position_xy_m": [0.8 + index * 0.01, 0.2],
                "source_split": trial["split"],
                "dataset_split": trial["split"],
                "selection_rank": 1,
                "origin": "new",
                "source_run_id": "collection",
                "source_git_commit": GIT_COMMIT,
                "outcome_success": True,
                "dataset_valid": True,
                "selected_for_dataset": True,
                "failure_category": None,
                "failure_reason": None,
            }
        )
    collection = {
        "schema_version": "farpoint.collection.v1",
        "collection_id": "collection",
        "task_id": benchmark["task_id"],
        "git_commit": GIT_COMMIT,
        "policy_id": "policy",
        "policy_sha256": SHA,
        "position_plan_sha256": SHA,
        "config_sha256": SHA,
        "simulator_image_digest": f"sha256:{SHA}",
        "simulator_payload_sha256": SHA,
        "execution_status": "FINISHED",
        "quality_status": "PASS",
        "failure_reason": None,
        "attempts": attempts,
        "acceptance": {
            "accepted": True,
            "required_task_yield": 0.75,
            "observed_task_yield": 1.0,
            "maximum_task_attempts": 4,
            "observed_task_attempts": 3,
            "observed_task_successes": 3,
            "required_selected_episodes": 3,
            "observed_selected_episodes": 3,
            "required_cells": 3,
            "observed_covered_cells": 3,
            "required_selected_per_cell": 1,
            "selected_per_cell": selected_per_cell,
            "required_splits": splits,
            "observed_splits": splits,
        },
    }
    collection_path = tmp_path / "collection.json"
    write_json(collection_path, collection)

    result = validate_dataset(tmp_path, collection_path)

    assert result["valid"] is True, result["errors"]


def test_v2_dataset_rejects_split_count_drift(tmp_path):
    make_valid_v2_dataset(tmp_path)
    sidecar = dataset_sidecar_v2([episode_metadata_v2()])
    write_json(tmp_path / "meta" / "farpoint_v2.json", sidecar)
    result = validate_dataset(tmp_path)
    assert result["valid"] is False
    assert any("split count mismatch" in error for error in result["errors"])


def test_v2_dataset_rejects_wrong_task_index(tmp_path):
    make_valid_v2_dataset(tmp_path)
    pytest = __import__("pytest")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    task_path = tmp_path / "meta" / "tasks.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"task_index": 999, "task": "Pick up the cube and place it in the target zone."}]
        ),
        task_path,
    )
    result = validate_dataset(tmp_path)
    assert result["valid"] is False
    assert any("wrong task_index" in error for error in result["errors"])


def test_v2_dataset_rejects_empty_or_unreadable_artifacts(tmp_path):
    make_valid_v2_dataset(tmp_path)
    data_path = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    data_path.write_bytes(b"")
    result = validate_dataset(tmp_path)
    assert result["valid"] is False
    assert any("data Parquet is empty" in error for error in result["errors"])
