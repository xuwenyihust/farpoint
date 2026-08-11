from copy import deepcopy
import json
from pathlib import Path

import pytest

from farpoint.camera_profiles import (
    build_camera_records,
    camera_cfg_drift_errors,
    camera_profile_sha256,
    load_camera_profile,
    validate_camera_profile,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "configs/cameras/so101_front_wrist_v1.json"


def test_v010_camera_profile_is_frozen_dual_camera_contract():
    profile = load_camera_profile(PROFILE_PATH)
    assert validate_camera_profile(profile) == []
    assert [camera["camera_id"] for camera in profile["cameras"]] == ["front", "wrist"]
    assert len(camera_profile_sha256(profile)) == 64


def test_camera_profile_rejects_missing_wrist_and_wrong_resolution():
    profile = json.loads(PROFILE_PATH.read_text())
    profile["cameras"].pop()
    profile["cameras"][0]["width"] = 320
    errors = validate_camera_profile(profile)
    assert "v0.1.0 camera profile must contain ordered front and wrist cameras" in errors
    assert "camera 'front' must use 640x480 resolution" in errors


def test_resolved_camera_records_bind_runtime_intrinsics_and_mounts():
    profile = load_camera_profile(PROFILE_PATH)
    intrinsics = {
        "front": [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        "wrist": [[450.0, 0.0, 320.0], [0.0, 450.0, 240.0], [0.0, 0.0, 1.0]],
    }
    mounts = {
        camera["camera_id"]: {
            "parent_frame": camera["mount"]["parent_frame"],
            "position_m": deepcopy(camera["mount"]["position_m"]),
            "orientation_xyzw": deepcopy(camera["mount"]["orientation_xyzw"]),
            "convention": "opengl",
        }
        for camera in profile["cameras"]
    }
    records = build_camera_records(
        profile, resolved_intrinsics=intrinsics, resolved_mounts=mounts
    )
    assert [record["feature_key"] for record in records] == [
        "observation.images.front", "observation.images.wrist"
    ]
    assert records[1]["calibration"]["resolved"]["intrinsic_matrix"] == intrinsics["wrist"]
    assert records[0]["config_sha256"] == records[1]["config_sha256"]

    with pytest.raises(ValueError, match="exactly front and wrist"):
        build_camera_records(
            profile,
            resolved_intrinsics={"front": intrinsics["front"]},
            resolved_mounts=mounts,
        )


def test_camera_profile_detects_isaac_scene_config_drift():
    class Value:
        pass

    profile = load_camera_profile(PROFILE_PATH)
    scene = Value()
    for camera in profile["cameras"]:
        cfg = Value()
        cfg.prim_path = camera["prim_path"]
        cfg.width, cfg.height = camera["width"], camera["height"]
        cfg.data_types = tuple(camera["data_types"])
        cfg.spawn, cfg.offset = Value(), Value()
        cfg.spawn.focal_length = camera["focal_length_mm"]
        cfg.spawn.focus_distance = camera["focus_distance_m"]
        cfg.offset.pos = tuple(camera["mount"]["position_m"])
        cfg.offset.rot = tuple(camera["mount"]["orientation_xyzw"])
        cfg.offset.convention = camera["mount"]["convention"]
        setattr(scene, f"{camera['camera_id']}_camera", cfg)
    assert camera_cfg_drift_errors(profile, scene) == []
    scene.wrist_camera.width = 320
    assert "Isaac wrist_camera width drifted" in camera_cfg_drift_errors(profile, scene)[0]
