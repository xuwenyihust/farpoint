from copy import deepcopy

from farpoint.contracts import task_definition, validate_contract, validate_episode_semantics
from farpoint.episode_metadata import validate_compatible_episode_metadata
from farpoint.scene_entities import legacy_object_entity, placement_target_entity

from v2_fixtures import episode_metadata_v2
from test_lerobot_exporter_v3 import _metadata as episode_metadata_v3


SHA = "a" * 64
JOINTS = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]


def _versioned(value, units=None):
    return {
        "requested": deepcopy(value), "resolved": deepcopy(value),
        "units": units or {}, "config_version": "1", "config_sha256": SHA,
    }


def _camera(camera_id):
    return {
        "camera_id": camera_id,
        "feature_key": f"observation.images.{camera_id}",
        "width": 640, "height": 480,
        "config_version": "1", "config_sha256": SHA,
        "calibration": {"model": "pinhole", "intrinsics": [600.0, 600.0, 320.0, 240.0]},
        "mount_transform": {"frame_id": "robot", "position_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "frame_timestamp_source": "control_tick",
    }


def episode_v4():
    obj = {
        "shape": "cube", "asset_id": "procedural_cube",
        "dimensions_m": [0.04, 0.04, 0.04], "position_m": [0.2, -0.05, 0.02],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0], "rgba": [0.9, 0.1, 0.1, 1.0],
        "mass_kg": 0.04, "static_friction": 0.8, "dynamic_friction": 0.6,
        "restitution": 0.0,
    }
    target_spec = {
        "target_id": "green-pad", "position_m": [0.16, 0.12, 0.005],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0], "dimensions_m": [0.12, 0.12, 0.01],
        "rgba": [0.1, 0.8, 0.2, 1.0], "body_type": "static", "material": {},
    }
    entities = [legacy_object_entity(obj), placement_target_entity(target_spec)]
    entities.append({
        "schema_version": "farpoint.scene-entity.v1", "entity_id": "table", "role": "support_surface",
        "entity_type": "table", "asset_id": "procedural_table",
        "pose": {"frame_id": "isaac_world", "position_m": [0.0, 0.0, -0.02], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "geometry": {"representation": "procedural", "shape": "cuboid", "dimensions_m": [0.6, 0.6, 0.04]},
        "physics": {"body_type": "static", "collision_enabled": True, "material": {"static_friction": 0.7, "dynamic_friction": 0.5, "restitution": 0.0, "friction_combine_mode": "average", "restitution_combine_mode": "max"}},
    })
    return {
        "schema_version": "farpoint.episode.v4",
        "identity": {"episode_id": "episode-1", "campaign_id": "campaign-1", "segment_id": "segment-0", "variation_seed": 11, "attempt_seed": 12, "task_id": "so101-pick-place", "split": "train"},
        "provenance": {"git_commit": "a" * 40, "campaign_sha256": SHA, "segment_sha256": SHA, "plan_sha256": SHA, "simulator_image_digest": "sha256:" + SHA, "oracle_profile_id": "default"},
        "task": {"task_id": "so101-pick-place", "instruction": "Pick up the object and place it on the target.", "success_criteria_id": "contact-lift-place-v1", "manipulated_entity_id": "pick_object", "target_entity_id": "placement_target", "acceptance_region_id": "placement_region"},
        "embodiment": {"robot": "so101", "joint_order": JOINTS},
        "scene": {
            "coordinate_frame": "isaac_world", "entities": entities,
            "object_archetype": _versioned({"archetype_id": "cube-v1", "semantic_type": "cube"}),
            "object_variant": _versioned({"variant_id": "red-40mm", "mass_kg": 0.04}, {"mass_kg": "kg"}),
            "feasible_region": _versioned({"region_id": "red-40mm-feasible-v1", "polygon_xy_m": [[0.1, -0.12], [0.27, -0.12], [0.27, 0.02], [0.1, 0.02]]}, {"polygon_xy_m": "m"}),
            "materials": {name: _versioned({"static_friction": 0.8, "dynamic_friction": 0.6, "restitution": 0.0, "friction_combine_mode": "average", "restitution_combine_mode": "max"}) for name in ("object", "table", "gripper")},
            "lighting_profile_id": "default-v1",
        },
        "variation": {
            "variation_id": "variation-11", "varied_axes": ["object_variant_id", "position_xy_m", "yaw_rad"],
            "frozen_axes": ["target.pose", "lighting.profile"],
            "requested": {"object_variant_id": "red-40mm", "position_xy_m": [0.2, -0.05], "yaw_rad": 0.1},
            "resolved": {"object_variant_id": "red-40mm", "position_xy_m": [0.2, -0.05], "yaw_rad": 0.1},
            "units": {"position_xy_m": "m", "yaw_rad": "rad"}, "config_version": "1", "config_sha256": SHA,
            "sampler": _versioned({"sampler_version": "farpoint.scrambled-sobol.v1", "sobol_index": 4}),
            "region_band": "middle", "yaw_stratum_id": "yaw00-18", "split": "train",
        },
        "recording": {
            "fps": 30, "control_hz": 120, "frame_count": 30,
            "state_features": JOINTS, "action_features": JOINTS,
            "cameras": [_camera("front"), _camera("wrist")],
            "synchronization": {"clock": "simulation", "same_control_tick": True, "timestamp_unit": "seconds", "maximum_skew_seconds": 0.0},
        },
        "outcome": {"success": True, "dataset_valid": True, "failure_category": None, "failure_reason": None},
    }


def test_episode_v4_requires_typed_dual_camera_and_entity_metadata():
    episode = episode_v4()
    assert validate_contract(episode) == []
    assert validate_episode_semantics(episode) == []
    assert task_definition(episode)["object_shape"] == "cube"


def test_episode_v4_rejects_camera_and_split_drift():
    episode = episode_v4()
    episode["recording"]["cameras"].pop()
    episode["variation"]["split"] = "validation"
    errors = validate_contract(episode) + validate_episode_semantics(episode)
    assert any("too short" in error for error in errors)
    assert "variation.split does not match identity.split" in errors
    assert "episode v4 requires exactly front and wrist cameras" in errors


def test_compatible_reader_accepts_v1_v2_v3_v4_without_rewriting():
    legacy = {"episode_id": "legacy", "task_name": "legacy", "episode_seed": 1}
    v2 = episode_metadata_v2()
    v3 = episode_metadata_v3(include_wrist=True, include_entities=True)
    v4 = episode_v4()
    assert validate_compatible_episode_metadata(legacy) == []
    assert validate_compatible_episode_metadata(v2) == []
    assert validate_compatible_episode_metadata(v3) == []
    assert validate_compatible_episode_metadata(v4) == []
