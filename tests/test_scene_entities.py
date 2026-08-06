from __future__ import annotations

import copy

import pytest

from farpoint.contracts import validate_contract, validate_episode_semantics
from farpoint.scene_entities import (
    ENTITY_SCHEMA_VERSION,
    bind_scene_entities,
    placement_target_entity,
    validate_scene_entities,
    validate_scene_entity,
)
from farpoint.so101 import SIM_JOINT_NAMES


def _object_state(shape: str = "doll"):
    return {
        "shape": shape,
        "asset_id": "custom_usd:doll/rag_doll_v1",
        "dimensions_m": [0.06, 0.035, 0.12],
        "position_m": [0.18, -0.10, 0.092],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "rgba": [0.8, 0.5, 0.3, 1.0],
        "mass_kg": 0.08,
        "static_friction": 1.0,
        "dynamic_friction": 0.8,
        "restitution": 0.0,
    }


def _box_target():
    return {
        "target_id": "open_box_v1",
        "asset_id": "custom_usd:containers/open_box_v1",
        "entity_type": "box",
        "representation": "asset",
        "shape": "open_box",
        "position_m": [0.20, 0.10, 0.067],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "dimensions_m": [0.18, 0.16, 0.07],
        "region_dimensions_m": [0.15, 0.13, 0.055],
        "region_position_m": [0.20, 0.10, 0.075],
        "region_shape": "cuboid",
        "relation": "inside",
        "footprint_margin_m": 0.005,
        "rgba": [0.1, 0.7, 0.2, 1.0],
    }


def _episode():
    state = bind_scene_entities(_object_state(), _box_target())
    obj = {
        "shape": state["shape"],
        "asset_id": state["asset_id"],
        "dimensions_m": state["dimensions_m"],
        "initial_pose": {
            "position_m": state["position_m"],
            "orientation_xyzw": state["orientation_xyzw"],
        },
        "rgba": state["rgba"],
        "mass_kg": state["mass_kg"],
        "static_friction": state["static_friction"],
        "dynamic_friction": state["dynamic_friction"],
        "restitution": state["restitution"],
    }
    return {
        "schema_version": "farpoint.episode.v3",
        "identity": {
            "episode_id": "episode-doll-box-0000",
            "trial_id": "trial-doll-box-0000",
            "task_id": "pick_place_generic_v1",
            "split": "train",
            "episode_seed": 11,
        },
        "provenance": {"simulator": "Isaac Sim"},
        "task": {
            "task_id": "pick_place_generic_v1",
            "instruction": "Pick up the doll and place it inside the box.",
            "object_shape": "doll",
            "success_criteria_id": "entity_region_relation_v1",
            "manipulated_entity_id": "pick_object",
            "target_entity_id": "placement_target",
            "acceptance_region_id": "placement_region",
        },
        "embodiment": {
            "robot": "so101",
            "gripper": "so101_jaw",
            "arm_dof": 5,
            "gripper_dof": 1,
            "controller": "oracle_dls",
            "control_mode": "joint_position",
            "grasp_mode": "contact_only",
            "joint_mapping": {},
        },
        "scene": {
            "coordinate_frame": "isaac_world",
            "object": obj,
            "target": copy.deepcopy(_box_target()),
            "entities": list(copy.deepcopy(state["entities"]).values()),
            "cameras": [{"name": "observation.images.front"}],
            "lighting_profile_id": "fixed_default",
        },
        "variation": {
            "schema_version": "farpoint.variation.v3",
            "variation_id": "doll_box_pose_0000",
            "varied_axes": [
                "entities.pick_object.entity_type",
                "entities.placement_target.pose.position_m",
                "entities.placement_target.geometry.dimensions_m",
            ],
            "frozen_axes": ["lighting.profile"],
            "requested": copy.deepcopy(state),
            "resolved": copy.deepcopy(state),
            "split": "train",
        },
        "recording": {
            "fps": 30,
            "cameras": ["observation.images.front"],
            "frame_count": 1,
            "state_features": list(SIM_JOINT_NAMES),
            "action_features": list(SIM_JOINT_NAMES),
        },
        "outcome": {
            "success": True,
            "dataset_valid": True,
            "failure_category": None,
            "failure_reason": None,
        },
    }


def test_custom_doll_and_box_acceptance_region_are_open_ended():
    episode = _episode()
    target = next(
        entity
        for entity in episode["scene"]["entities"]
        if entity["role"] == "placement_target"
    )

    assert target["schema_version"] == ENTITY_SCHEMA_VERSION
    assert target["entity_type"] == "box"
    assert target["geometry"]["dimensions_m"] == [0.18, 0.16, 0.07]
    assert target["regions"][0]["relation"] == "inside"
    assert target["regions"][0]["geometry"]["dimensions_m"] == [
        0.15,
        0.13,
        0.055,
    ]
    assert target["regions"][0]["pose"]["position_m"] == [0.20, 0.10, 0.075]
    assert validate_contract(episode) == []
    assert validate_episode_semantics(episode) == []


def test_resolved_target_entity_must_match_the_spawned_scene():
    episode = _episode()
    episode["variation"]["resolved"]["entities"]["placement_target"]["pose"][
        "position_m"
    ][1] += 0.01

    assert (
        "variation.resolved.entities.placement_target does not match scene.entities"
        in validate_episode_semantics(episode)
    )


def test_target_physical_geometry_and_success_region_can_change_independently():
    target = placement_target_entity(_box_target(), entity_type="box", relation="inside")
    validate_scene_entity(target)
    assert target["geometry"]["dimensions_m"] != target["regions"][0][
        "geometry"
    ]["dimensions_m"]


def test_scene_entities_reject_duplicate_identity_and_invalid_dynamic_mass():
    state = bind_scene_entities(_object_state(), _box_target())
    entities = list(state["entities"].values())
    duplicate = copy.deepcopy(entities[1])
    duplicate["entity_id"] = entities[0]["entity_id"]
    with pytest.raises(ValueError, match="entity_id values must be unique"):
        validate_scene_entities([*entities, duplicate])

    invalid = copy.deepcopy(entities[0])
    invalid["physics"]["mass_kg"] = 0.0
    with pytest.raises(ValueError, match="mass_kg"):
        validate_scene_entity(invalid)
