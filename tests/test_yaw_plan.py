from pathlib import Path

import numpy as np
import pytest

from farpoint.perception import PerceptionError, cube_yaw_error_degrees, estimate_dominant_color_yaw
from farpoint.yaw_plan import (
    apply_yaw_trial,
    generate_yaw_plan,
    load_yaw_config,
    resolve_yaw_trial,
    validate_yaw_plan,
    yaw_quaternion_xyzw,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "variations" / "farpoint_v0_0_1_cube_yaw.json"


def plan():
    return generate_yaw_plan(load_yaw_config(CONFIG))


def test_yaw_plan_is_deterministic_and_has_complete_factorial_coverage():
    first = plan()
    assert first == plan()
    validate_yaw_plan(first)
    assert len(first["trials"]) == 100
    assert {trial["object_yaw_degrees"] for trial in first["trials"]} == {0.0, 15.0, 30.0, 45.0}
    assert {split: sum(row["split"] == split for row in first["trials"]) for split in ("train", "validation", "test")} == {"train": 68, "validation": 16, "test": 16}
    assert all(sum(row["object_yaw_degrees"] == yaw for row in first["trials"]) == 25 for yaw in (0.0, 15.0, 30.0, 45.0))


def test_yaw_trial_applies_pose_object_spec_and_disables_randomization():
    source = {"name": "isaac_perception_contact_scene", "scene": {"pick_object": {"position": [0, 0, 0]}, "target_zone": {"position": [0, 0, 0]}}, "randomization": {"enabled": True}}
    configured, resolved = apply_yaw_trial(source, plan(), "yaw45_r02_c02")
    assert source["randomization"]["enabled"] is True
    assert configured["randomization"]["enabled"] is False
    assert configured["scene"]["pick_object"]["rotation_degrees"] == [0.0, 0.0, 45.0]
    assert resolved["variation"]["resolved"]["object_variant_id"] == "cube_55mm_red_v1"
    assert resolved["variation"]["resolved"]["object_orientation_xyzw"] == yaw_quaternion_xyzw(45)
    assert resolve_yaw_trial(plan(), "yaw15_r00_c00", reserve_index=1)["variation"]["variation_id"].endswith("_reserve1")


def test_cube_yaw_audit_uses_ninety_degree_symmetry():
    assert cube_yaw_error_degrees(0, 90) == 0.0
    assert cube_yaw_error_degrees(44, -46) == 0.0
    assert cube_yaw_error_degrees(0, 45) == 45.0


def test_rgbd_yaw_estimator_reports_or_rejects_low_confidence():
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    rgb[30:50, 10:70, 0] = 255
    depth = np.ones((80, 80), dtype=np.float64)
    intrinsics = np.eye(3)
    world = np.eye(4)
    result = estimate_dominant_color_yaw(rgb, depth, intrinsics, world, "red", min_pixels=20)
    assert result["confidence"] >= 0.15
    assert cube_yaw_error_degrees(result["yaw_degrees"], 0.0) < 1.0
    assert result["orientation_separation"] > 0.0
    ambiguous = np.zeros_like(rgb)
    rows, columns = np.ogrid[:80, :80]
    ambiguous[(rows - 40) ** 2 + (columns - 40) ** 2 <= 14 ** 2, 0] = 255
    with pytest.raises(PerceptionError, match="confidence"):
        estimate_dominant_color_yaw(ambiguous, depth, intrinsics, world, "red", min_pixels=20)
