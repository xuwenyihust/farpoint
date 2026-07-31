import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_lerobot_dataset import validate_dataset


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def valid_sidecar():
    return {
        "schema_version": "farpoint.dataset.v1",
        "dataset_id": "farpoint_v1",
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
    assert result["dataset_id"] == "farpoint_v1"
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
