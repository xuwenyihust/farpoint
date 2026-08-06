import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_lerobot_dataset import _so101_sidecar, export_dataset
from farpoint.scene_entities import bind_scene_entities
from farpoint.so101 import SIM_JOINT_NAMES, radians_to_lerobot


class FakeLeRobotDataset:
    last_instance = None

    def __init__(self, root):
        self.root = Path(root)
        self.frames = []
        self.saved = 0

    @classmethod
    def create(cls, *, root, **kwargs):
        cls.last_instance = cls(root)
        cls.last_instance.kwargs = kwargs
        return cls.last_instance

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self, **_kwargs):
        self.saved += 1

    def finalize(self):
        meta = self.root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "info.json").write_text(json.dumps({"features": {}}), encoding="utf-8")


def _metadata(
    episode_id="episode-0000",
    split="train",
    *,
    include_wrist=False,
    include_entities=False,
):
    obj = {
        "shape": "cube",
        "asset_id": "procedural_cube",
        "dimensions_m": [0.03, 0.03, 0.03],
        "initial_pose": {"position_m": [0.2, 0.0, 0.047], "orientation_xyzw": [0, 0, 0, 1]},
        "rgba": [0.9, 0.05, 0.05, 1.0],
        "mass_kg": 0.04,
        "static_friction": 1.2,
        "dynamic_friction": 1.0,
        "restitution": 0.0,
    }
    variation_obj = {
        key: value for key, value in obj.items() if key != "initial_pose"
    }
    variation_obj.update(
        {
            "position_m": obj["initial_pose"]["position_m"],
            "orientation_xyzw": obj["initial_pose"]["orientation_xyzw"],
        }
    )
    metadata = {
        "schema_version": "farpoint.episode.v3",
        "identity": {"episode_id": episode_id, "trial_id": episode_id.replace("episode", "trial"), "task_id": "pick_place_cube_v1", "split": split, "episode_seed": 7},
        "provenance": {"simulator": "Isaac Sim", "physics_engine": "PhysX"},
        "task": {"task_id": "pick_place_cube_v1", "instruction": "Pick up the cube and place it in the tray.", "object_shape": "cube", "success_criteria_id": "contact_pick_place_v1"},
        "embodiment": {"robot": "so101", "gripper": "so101_jaw", "arm_dof": 5, "gripper_dof": 1, "controller": "oracle_dls", "control_mode": "joint_position", "grasp_mode": "contact_only", "joint_mapping": {"sim_joint_names": list(SIM_JOINT_NAMES)}},
        "scene": {"coordinate_frame": "world", "object": obj, "target": {"target_id": "tray"}, "cameras": ["front", "wrist"] if include_wrist else ["front"], "lighting_profile_id": "studio_v1"},
        "variation": {"schema_version": "farpoint.variation.v3", "variation_id": "cell_r00_c00_s00_red", "varied_axes": ["position_m"], "frozen_axes": ["shape"], "requested": variation_obj, "resolved": variation_obj, "split": split},
        "recording": {"fps": 30, "control_hz": 120, "recording_stride": 4, "cameras": ["observation.images.front", "observation.images.wrist"] if include_wrist else ["observation.images.front"], "frame_count": 2, "state_features": list(SIM_JOINT_NAMES), "action_features": list(SIM_JOINT_NAMES)},
        "outcome": {"success": True, "dataset_valid": True, "failure_category": None, "failure_reason": None},
    }
    if include_entities:
        object_state = {
            **obj,
            "position_m": obj["initial_pose"]["position_m"],
            "orientation_xyzw": obj["initial_pose"]["orientation_xyzw"],
        }
        object_state.pop("initial_pose")
        target = {
            "target_id": "open_box_v1",
            "asset_id": "custom_usd:open_box_v1",
            "entity_type": "box",
            "representation": "asset",
            "shape": "open_box",
            "position_m": [0.2, 0.1, 0.067],
            "orientation_xyzw": [0, 0, 0, 1],
            "dimensions_m": [0.18, 0.16, 0.07],
            "region_dimensions_m": [0.15, 0.13, 0.055],
            "relation": "inside",
        }
        state = bind_scene_entities(object_state, target)
        metadata["task"].update(
            {
                "manipulated_entity_id": "pick_object",
                "target_entity_id": "placement_target",
                "acceptance_region_id": "placement_region",
            }
        )
        metadata["scene"]["target"] = target
        metadata["scene"]["entities"] = list(state["entities"].values())
        metadata["variation"]["requested"] = state
        metadata["variation"]["resolved"] = state
        metadata["variation"]["varied_axes"] = [
            "entities.placement_target.pose.position_m",
            "entities.placement_target.geometry.dimensions_m",
        ]
    return metadata


def _episode(root, metadata):
    Image = pytest.importorskip("PIL.Image")

    root.mkdir()
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "metrics.json").write_text(json.dumps({"success": True, "dataset_valid": True}), encoding="utf-8")
    rows = []
    for index in range(2):
        cameras = (("front", (255, 0, 0)),)
        if "observation.images.wrist" in metadata["recording"]["cameras"]:
            cameras += (("wrist", (0, 255, 0)),)
        for name, color in cameras:
            Image.new("RGB", (8, 6), color=color).save(root / f"{name}-{index}.png")
        row = {"frame": index, "timestamp_seconds": index / 30, "rgb_path": f"front-{index}.png", "joint_names": list(SIM_JOINT_NAMES), "joint_positions": [0.0] * 6, "action_joint_positions": [0.1] * 6}
        if "observation.images.wrist" in metadata["recording"]["cameras"]:
            row["wrist_rgb_path"] = f"wrist-{index}.png"
        rows.append(row)
    (root / "observations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_export_so101_v0_writes_six_dof_and_front_camera_only(tmp_path):
    metadata = _metadata()
    source = tmp_path / "episode-0000"
    _episode(source, metadata)
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps({"schema_version": "farpoint.export-selection.v1", "dataset_id": "so101-fixture", "episodes": [{"episode_dir": str(source), "trial_id": "trial-0000", "split": "train"}]}), encoding="utf-8")

    output = export_dataset(manifest, tmp_path / "export", dataset_class=FakeLeRobotDataset)
    assert FakeLeRobotDataset.last_instance.saved == 1
    assert FakeLeRobotDataset.last_instance.frames[0]["observation.state"].shape == (6,)
    assert np.allclose(
        FakeLeRobotDataset.last_instance.frames[0]["observation.state"],
        radians_to_lerobot(np.zeros(6)),
    )
    assert FakeLeRobotDataset.last_instance.frames[0]["observation.images.front"].shape == (6, 8, 3)
    assert "observation.images.wrist" not in FakeLeRobotDataset.last_instance.frames[0]
    sidecar = json.loads((output / "meta/farpoint_v3.json").read_text())
    assert sidecar["robot"]["name"] == "so101"
    assert sidecar["recording"]["cameras"] == ["observation.images.front"]
    assert sidecar["recording"]["control_hz"] == 120
    assert sidecar["recording"]["recording_stride"] == 4


def test_export_so101_retains_dual_camera_compatibility(tmp_path):
    metadata = _metadata(include_wrist=True)
    source = tmp_path / "episode-0000"
    _episode(source, metadata)
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps({"schema_version": "farpoint.export-selection.v1", "dataset_id": "so101-dual-camera-fixture", "episodes": [{"episode_dir": str(source), "trial_id": "trial-0000", "split": "train"}]}), encoding="utf-8")

    export_dataset(manifest, tmp_path / "export", dataset_class=FakeLeRobotDataset)
    assert FakeLeRobotDataset.last_instance.frames[0]["observation.images.wrist"].shape == (6, 8, 3)


def test_export_so101_indexes_generic_scene_entities_without_policy_features(tmp_path):
    metadata = _metadata(include_entities=True)
    source = tmp_path / "episode-0000"
    _episode(source, metadata)
    manifest = tmp_path / "selection.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "farpoint.export-selection.v1",
                "dataset_id": "so101-generic-scene-fixture",
                "episodes": [
                    {
                        "episode_dir": str(source),
                        "trial_id": "trial-0000",
                        "split": "train",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = export_dataset(
        manifest, tmp_path / "export", dataset_class=FakeLeRobotDataset
    )
    sidecar = json.loads((output / "meta/farpoint_v3.json").read_text())
    indexed = sidecar["episode_scene_metadata"][0]
    assert indexed["target_entity_id"] == "placement_target"
    assert indexed["acceptance_region_id"] == "placement_region"
    assert indexed["resolved_entities"]["placement_target"]["entity_type"] == "box"
    assert indexed["resolved_entities"]["placement_target"]["regions"][0][
        "relation"
    ] == "inside"
    assert "entities.placement_target.geometry.dimensions_m" in indexed["varied_axes"]
    assert "entities" not in FakeLeRobotDataset.last_instance.kwargs["features"]


def test_so101_sidecar_indexes_entity_variations_without_image_dependencies():
    metadata = _metadata(include_entities=True)
    metadata["identity"]["dataset_episode_index"] = 0

    sidecar = _so101_sidecar(
        "so101-generic-scene-fixture",
        [metadata],
        30,
        640,
        480,
        "one_success_per_stratified_variation",
        ["observation.images.front"],
    )

    indexed = sidecar["episode_scene_metadata"][0]
    assert indexed["entities"] == metadata["scene"]["entities"]
    assert indexed["requested_entities"] == metadata["variation"]["requested"][
        "entities"
    ]
    assert indexed["resolved_entities"] == metadata["variation"]["resolved"][
        "entities"
    ]
