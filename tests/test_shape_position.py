import copy
from pathlib import Path

import pytest

from farpoint.contracts import validate_contract
from farpoint.shape_position import (
    apply_shape_position_trial,
    generate_shape_position_plan,
    load_shape_position_config,
    resolve_shape_position_trial,
    validate_shape_position_plan,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs/variations/farpoint_v0_0_1_cylinder_position.json"


def plan():
    return generate_shape_position_plan(load_shape_position_config(CONFIG))


def test_cylinder_plan_is_deterministic_and_contract_valid():
    first = plan()
    assert first == plan()
    assert len(first["trials"]) == 150
    assert len({row["seed"] for row in first["trials"]}) == 150
    assert len({tuple(row["object_position_xy_m"]) for row in first["trials"]}) == 150
    assert validate_contract(first) == []


def test_cylinder_plan_has_six_candidates_per_cell_and_frozen_splits():
    manifest = plan()
    assert {
        split: sum(row["split"] == split for row in manifest["trials"])
        for split in ("train", "validation", "test")
    } == {"train": 102, "validation": 24, "test": 24}
    assert set(
        sum(row["cell_id"] == cell for row in manifest["trials"])
        for cell in {row["cell_id"] for row in manifest["trials"]}
    ) == {6}


def test_cylinder_trial_uses_shape_prefixed_identity_and_metadata():
    resolved = resolve_shape_position_trial(plan(), "cylinder_r02_c02_s00")
    assert resolved["variation"]["variation_id"] == "cylinder_position_r02_c02_s00"
    assert resolved["variation"]["resolved"]["object_shape"] == "cylinder"
    assert resolved["variation"]["resolved"]["object_dimensions_m"] == [0.07, 0.07, 0.08]
    assert resolved["split"] == "test"


def test_apply_trial_reuses_template_but_changes_task_identity_and_shape():
    task = {
        "name": "isaac_perception_contact_scene",
        "language_instruction": "cube",
        "scene": {
            "pick_object": {"position": [0, 0, 0], "scale": [1, 1, 1]},
            "target_zone": {"position": [0, 0, 0]},
        },
        "pickup": {"static_friction": 1.8, "dynamic_friction": 1.5, "finger_contact_max_effort": 8.0},
        "randomization": {"enabled": True},
    }
    configured, resolved = apply_shape_position_trial(task, plan(), "cylinder_r00_c00_s00")
    assert configured["name"] == "isaac_perception_contact_cylinder_scene"
    assert "cylinder" in configured["language_instruction"]
    assert configured["scene"]["pick_object"]["cylinder_radius_scale"] == 0.5
    assert configured["pickup"] == {
        "static_friction": 2.5,
        "dynamic_friction": 2.0,
        "finger_contact_max_effort": 20.0,
        "grasp_height_offset_meters": 0.01,
        "grasp_aperture_bias_xy": [-0.0015, 0.001],
        "grasp_tracking_max_xy_error": 0.006,
        "grasp_descent_max_xy_error": 0.006,
        "grasp_tracking_max_finger_z_skew": 0.015,
        "unilateral_recenter_enabled": True,
        "unilateral_recenter_step": 0.0001,
        "unilateral_recenter_max_correction": 0.003,
        "unilateral_recenter_persistence_frames": 12,
        "bilateral_contact_min_finger_position_radians": 0.12,
        "unilateral_force_limit_newtons": 1000.0,
        "unilateral_force_backoff_radians": 0.00001,
        "bilateral_force_limit_newtons": 1000.0,
        "bilateral_hold_max_force_newtons": 1000.0,
        "bilateral_force_backoff_radians": 0.00001,
        "gripper_force_max_preload_error_radians": 0.08,
    }
    assert configured["scene"]["pick_object"]["position"] == resolved["variation"]["resolved"]["object_position_m"]
    assert task["randomization"]["enabled"] is True


def test_plan_hash_detects_mutation():
    manifest = copy.deepcopy(plan())
    manifest["trials"][0]["seed"] += 1
    with pytest.raises(ValueError, match="plan_sha256"):
        validate_shape_position_plan(manifest)
