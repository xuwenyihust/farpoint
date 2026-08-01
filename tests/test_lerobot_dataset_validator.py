import json
import sys
from pathlib import Path

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
    write_json(root / "meta" / "info.json", info)
    write_json(root / "meta" / "stats.json", {})
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": episodes[0]["task"]["instruction"]}) + "\n",
        encoding="utf-8",
    )
    (root / "meta" / "episode_metadata.jsonl").write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes), encoding="utf-8"
    )
    for relative in (
        "meta/episodes/chunk-000/file-000.parquet",
        "data/chunk-000/file-000.parquet",
        "videos/observation.images.front/chunk-000/file-000.mp4",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
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


def test_v2_dataset_rejects_split_count_drift(tmp_path):
    make_valid_v2_dataset(tmp_path)
    sidecar = dataset_sidecar_v2([episode_metadata_v2()])
    write_json(tmp_path / "meta" / "farpoint_v2.json", sidecar)
    result = validate_dataset(tmp_path)
    assert result["valid"] is False
    assert any("split count mismatch" in error for error in result["errors"])
