from copy import deepcopy

from farpoint.contracts import task_definition, validate_contract, validate_episode_semantics
from farpoint.demonstration import recovery_demonstration
from farpoint.episode_metadata import validate_compatible_episode_metadata
from farpoint.scene_entities import legacy_object_entity, placement_target_entity

from v2_fixtures import episode_metadata_v2
from test_lerobot_exporter_v3 import _metadata as episode_metadata_v3


SHA = "a" * 64
JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _versioned(value, units=None):
    return {
        "requested": deepcopy(value),
        "resolved": deepcopy(value),
        "units": units or {},
        "config_version": "1",
        "config_sha256": SHA,
    }


def _camera(camera_id):
    return {
        "camera_id": camera_id,
        "feature_key": f"observation.images.{camera_id}",
        "width": 640,
        "height": 480,
        "config_version": "1",
        "config_sha256": SHA,
        "calibration": {"model": "pinhole", "intrinsics": [600.0, 600.0, 320.0, 240.0]},
        "mount_transform": {
            "frame_id": "robot",
            "position_m": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "frame_timestamp_source": "control_tick",
        "video_artifact": {
            "path": f"videos/{camera_id}.mp4",
            "container": "mp4",
            "codec": "h264",
            "frame_count": 30,
            "width": 640,
            "height": 480,
            "fps": 30,
            "size_bytes": 100,
            "sha256": SHA,
            "decode_verified": True,
        },
    }


def episode_v4():
    object_material = {
        "static_friction": 0.8,
        "dynamic_friction": 0.6,
        "restitution": 0.0,
        "friction_combine_mode": "average",
        "restitution_combine_mode": "max",
    }
    table_material = {
        "static_friction": 0.7,
        "dynamic_friction": 0.5,
        "restitution": 0.0,
        "friction_combine_mode": "average",
        "restitution_combine_mode": "max",
    }
    gripper_material = {
        "static_friction": 1.0,
        "dynamic_friction": 0.8,
        "restitution": 0.0,
        "friction_combine_mode": "average",
        "restitution_combine_mode": "max",
    }
    obj = {
        "shape": "cube",
        "asset_id": "procedural_cube",
        "dimensions_m": [0.04, 0.04, 0.04],
        "position_m": [0.2, -0.05, 0.02],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "rgba": [0.9, 0.1, 0.1, 1.0],
        "mass_kg": 0.04,
        "static_friction": 0.8,
        "dynamic_friction": 0.6,
        "restitution": 0.0,
        "friction_combine_mode": "average",
        "restitution_combine_mode": "max",
    }
    target_spec = {
        "target_id": "green-pad",
        "position_m": [0.16, 0.12, 0.005],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "dimensions_m": [0.12, 0.12, 0.01],
        "rgba": [0.1, 0.8, 0.2, 1.0],
        "body_type": "static",
        "material": {},
    }
    entities = [legacy_object_entity(obj), placement_target_entity(target_spec)]
    entities.append(
        {
            "schema_version": "farpoint.scene-entity.v1",
            "entity_id": "table",
            "role": "support_surface",
            "entity_type": "table",
            "asset_id": "procedural_table",
            "pose": {
                "frame_id": "isaac_world",
                "position_m": [0.0, 0.0, -0.02],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "geometry": {
                "representation": "procedural",
                "shape": "cuboid",
                "dimensions_m": [0.6, 0.6, 0.04],
            },
            "physics": {
                "body_type": "static",
                "collision_enabled": True,
                "material": table_material,
            },
        }
    )
    return {
        "schema_version": "farpoint.episode.v4",
        "identity": {
            "episode_id": "episode-1",
            "campaign_id": "campaign-1",
            "segment_id": "segment-0",
            "variation_seed": 11,
            "attempt_seed": 12,
            "task_id": "so101-pick-place",
            "split": "train",
        },
        "provenance": {
            "git_commit": "a" * 40,
            "campaign_sha256": SHA,
            "segment_sha256": SHA,
            "plan_sha256": SHA,
            "simulator_image_digest": "sha256:" + SHA,
            "oracle_profile_id": "default",
        },
        "task": {
            "task_id": "so101-pick-place",
            "instruction": "Pick up the object and place it on the target.",
            "success_criteria_id": "contact-lift-place-v1",
            "manipulated_entity_id": "pick_object",
            "target_entity_id": "placement_target",
            "acceptance_region_id": "placement_region",
        },
        "embodiment": {"robot": "so101", "joint_order": JOINTS},
        "scene": {
            "coordinate_frame": "isaac_world",
            "entities": entities,
            "object_archetype": _versioned(
                {
                    "archetype_id": "cube-v1",
                    "semantic_type": "cube",
                    "geometry_representation": "procedural",
                    "anchor": "bottom_center",
                }
            ),
            "object_variant": _versioned(
                {
                    "variant_id": "red-40mm",
                    "archetype_id": "cube-v1",
                    "asset_id": "procedural_cube",
                    "dimensions_m": [0.04, 0.04, 0.04],
                    "rgba": [0.9, 0.1, 0.1, 1.0],
                    "mass_kg": 0.04,
                    "object_material": object_material,
                    "table_material": table_material,
                    "gripper_material": gripper_material,
                },
                {"dimensions_m": "m", "mass_kg": "kg"},
            ),
            "feasible_region": _versioned(
                {
                    "region_id": "red-40mm-feasible-v1",
                    "polygon_xy_m": [[0.1, -0.12], [0.27, -0.12], [0.27, 0.02], [0.1, 0.02]],
                },
                {"polygon_xy_m": "m"},
            ),
            "materials": {
                "object": _versioned(object_material),
                "table": _versioned(table_material),
                "gripper": _versioned(gripper_material),
            },
            "lighting_profile_id": "default-v1",
        },
        "variation": {
            "variation_id": "variation-11",
            "varied_axes": ["object_variant_id", "position_xy_m", "yaw_rad"],
            "frozen_axes": ["target.pose", "lighting.profile"],
            "requested": {
                "object_variant_id": "red-40mm",
                "position_xy_m": [0.2, -0.05],
                "yaw_rad": 0.1,
                "entities": {entity["entity_id"]: deepcopy(entity) for entity in entities},
            },
            "resolved": {
                "object_variant_id": "red-40mm",
                "position_xy_m": [0.2, -0.05],
                "yaw_rad": 0.1,
                "entities": {entity["entity_id"]: deepcopy(entity) for entity in entities},
            },
            "units": {"position_xy_m": "m", "yaw_rad": "rad"},
            "config_version": "1",
            "config_sha256": SHA,
            "sampler": _versioned(
                {"sampler_version": "farpoint.scrambled-sobol.v1", "sobol_index": 4}
            ),
            "region_band": "middle",
            "yaw_stratum_id": "yaw00-18",
            "split": "train",
        },
        "recording": {
            "fps": 30,
            "control_hz": 120,
            "frame_count": 30,
            "state_features": JOINTS,
            "action_features": JOINTS,
            "cameras": [_camera("front"), _camera("wrist")],
            "synchronization": {
                "clock": "simulation",
                "same_control_tick": True,
                "timestamp_unit": "seconds",
                "maximum_skew_seconds": 0.0,
            },
        },
        "outcome": {
            "success": True,
            "dataset_valid": True,
            "failure_category": None,
            "failure_reason": None,
        },
    }


def test_episode_v4_requires_typed_dual_camera_and_entity_metadata():
    episode = episode_v4()
    assert validate_contract(episode) == []
    assert validate_episode_semantics(episode) == []
    assert task_definition(episode)["object_shape"] == "cube"


def test_episode_v4_accepts_backward_compatible_missing_demonstration_section():
    episode = episode_v4()
    assert "demonstration" not in episode
    assert validate_contract(episode) == []
    assert validate_episode_semantics(episode) == []


def test_episode_v4_validates_explicit_recovery_handoff_stage_contract():
    episode = episode_v4()
    episode["demonstration"] = recovery_demonstration(
        oracle_profile_id="recovery-v1",
        source_policy={
            "policy_type": "act",
            "checkpoint_step": 20_000,
            "model_sha256": SHA,
            "training_run_id": "act-v010",
            "rollout_git_commit": "b" * 40,
        },
        trigger_id="approach-stall-v1",
        failure_class="approach_miss",
        control_step=119,
        handoff_stage="approach",
        trigger_reason="stage_progress_stall",
        trigger_evidence={"contact_forces_n": [0.0, 0.0]},
        source_rollout_id="rollout-1",
        source_scene_id="scene-1",
        state_snapshot={"joint_positions_rad": [0.0] * 6},
        recovery_strategy_id="regrasp-v1",
    )
    assert validate_contract(episode) == []
    assert validate_episode_semantics(episode) == []

    mismatched = deepcopy(episode)
    mismatched["demonstration"]["intervention"]["trigger"]["evidence"][
        "handoff_stage"
    ] = "grasp"
    assert "recovery trigger evidence handoff_stage does not match" in (
        validate_episode_semantics(mismatched)
    )


def test_episode_v4_validates_live_recovery_intervention_metadata():
    episode = episode_v4()
    episode["demonstration"] = {
        "schema_version": "farpoint.demonstration.v1",
        "type": "recovery",
        "controller": {"type": "oracle", "profile_id": "recovery-transport-v1"},
        "source_policy": {
            "policy_type": "act",
            "checkpoint_step": 20_000,
            "model_sha256": "b" * 64,
            "training_run_id": "act-v010-baseline-20k",
            "rollout_git_commit": "c" * 40,
        },
        "intervention": {
            "trigger": {
                "trigger_id": "target-progress-stall-v1",
                "failure_class": "transport_drift",
                "control_step": 684,
                "stage": "transport",
                "evidence": {"target_distance_progress_m": 0.0, "window_steps": 90},
            },
            "handoff": {
                "mode": "live_continuous_state",
                "source_rollout_id": "act-v010-recovery-source-001",
                "source_scene_id": "train-scene-001",
                "source_control_step": 684,
                "recovery_start_frame": 0,
                "physics_state_continuous": True,
                "reset_performed": False,
                "state_snapshot_sha256": "d" * 64,
            },
            "recovery_strategy_id": "stabilize-lift-preplace-v1",
        },
    }
    assert validate_contract(episode) == []
    assert validate_episode_semantics(episode) == []


def test_episode_v4_accepts_audited_physics_rate_intervention_command_trace():
    episode = episode_v4()
    episode["demonstration"] = {
        "schema_version": "farpoint.demonstration.v1",
        "type": "recovery",
        "controller": {"type": "oracle", "profile_id": "recovery-transport-v1"},
        "source_policy": {
            "policy_type": "act",
            "checkpoint_step": 20_000,
            "model_sha256": "b" * 64,
            "training_run_id": "act-v010-baseline-20k",
            "rollout_git_commit": "c" * 40,
        },
        "intervention": {
            "trigger": {
                "trigger_id": "target-progress-stall-v1",
                "failure_class": "transport_drift",
                "control_step": 684,
                "stage": "transport",
                "evidence": {"target_distance_progress_m": 0.0},
            },
            "handoff": {
                "mode": "live_continuous_state",
                "source_rollout_id": "act-v010-recovery-source-001",
                "source_scene_id": "train-scene-001",
                "source_control_step": 684,
                "recovery_start_frame": 0,
                "physics_state_continuous": True,
                "reset_performed": False,
                "state_snapshot_sha256": "d" * 64,
            },
            "command_trace": {
                "schema_version": "farpoint.command-trace.v1",
                "path": "oracle-commands.jsonl",
                "sha256": "e" * 64,
                "control_hz": 120,
                "sampling_stride": 1,
                "sample_count": 4,
                "first_control_step": 0,
                "last_control_step": 3,
                "joint_order": JOINTS,
                "unit": "radian",
                "action_semantics": ("actual_joint_position_target_sent_before_physics_step"),
            },
            "recovery_strategy_id": "stabilize-lift-preplace-v1",
        },
    }
    assert validate_contract(episode) == []
    assert validate_episode_semantics(episode) == []

    invalid = deepcopy(episode)
    invalid["demonstration"]["intervention"]["command_trace"]["control_hz"] = 60
    assert (
        "recovery command trace control_hz does not match recording.control_hz"
        in validate_episode_semantics(invalid)
    )


def test_episode_v4_accepts_recovery_validation_split_and_rejects_discontinuous_handoff():
    episode = episode_v4()
    episode["identity"]["split"] = "validation"
    episode["variation"]["split"] = "validation"
    episode["demonstration"] = {
        "schema_version": "farpoint.demonstration.v1",
        "type": "recovery",
        "controller": {"type": "oracle", "profile_id": "recovery-v1"},
        "source_policy": {
            "policy_type": "act",
            "checkpoint_step": 20_000,
            "model_sha256": "b" * 64,
            "training_run_id": "act-v010",
            "rollout_git_commit": "c" * 40,
        },
        "intervention": {
            "trigger": {
                "trigger_id": "stall-v1",
                "failure_class": "progress_stall",
                "control_step": 100,
                "stage": "transport",
                "evidence": {"window": 30},
            },
            "handoff": {
                "mode": "live_continuous_state",
                "source_rollout_id": "source",
                "source_scene_id": "scene",
                "source_control_step": 101,
                "recovery_start_frame": 0,
                "physics_state_continuous": True,
                "reset_performed": False,
                "state_snapshot_sha256": "d" * 64,
            },
            "recovery_strategy_id": "recover-v1",
        },
    }
    errors = validate_episode_semantics(episode)
    assert "recovery demonstrations must use the train or validation split" not in errors
    assert "recovery handoff source_control_step does not match trigger.control_step" in errors


def test_episode_v4_rejects_recovery_test_split():
    episode = episode_v4()
    episode["identity"]["split"] = "test"
    episode["variation"]["split"] = "test"
    episode["demonstration"] = {
        "schema_version": "farpoint.demonstration.v1",
        "type": "recovery",
        "controller": {"type": "oracle", "profile_id": "recovery-v1"},
        "source_policy": {
            "policy_type": "act",
            "checkpoint_step": 20_000,
            "model_sha256": "b" * 64,
            "training_run_id": "act-v010",
            "rollout_git_commit": "c" * 40,
        },
        "intervention": {
            "trigger": {
                "trigger_id": "stall-v1",
                "failure_class": "progress_stall",
                "control_step": 100,
                "stage": "transport",
                "evidence": {"window": 30},
            },
            "handoff": {
                "mode": "live_continuous_state",
                "source_rollout_id": "source",
                "source_scene_id": "scene",
                "source_control_step": 100,
                "recovery_start_frame": 0,
                "physics_state_continuous": True,
                "reset_performed": False,
                "state_snapshot_sha256": "d" * 64,
            },
            "recovery_strategy_id": "recover-v1",
        },
    }
    assert (
        "recovery demonstrations must use the train or validation split"
        in validate_episode_semantics(episode)
    )


def test_episode_v4_rejects_camera_and_split_drift():
    episode = episode_v4()
    episode["recording"]["cameras"].pop()
    episode["variation"]["split"] = "validation"
    errors = validate_contract(episode) + validate_episode_semantics(episode)
    assert any("too short" in error for error in errors)
    assert "variation.split does not match identity.split" in errors
    assert "episode v4 requires exactly front and wrist cameras" in errors


def test_episode_v4_rejects_resolved_variant_physics_drift():
    episode = episode_v4()
    episode["scene"]["object_variant"]["resolved"]["mass_kg"] = 0.03
    episode["scene"]["materials"]["gripper"]["resolved"]["static_friction"] = 0.9
    errors = validate_episode_semantics(episode)
    assert "scene.object_variant.resolved.mass_kg does not match the manipulated entity" in errors
    assert "scene.materials.gripper.resolved does not match the object variant" in errors


def test_episode_v4_requires_requested_and_resolved_entity_snapshots():
    episode = episode_v4()
    episode["variation"]["requested"].pop("entities")
    assert (
        "episode v4 variation must record requested/resolved entities"
        in validate_episode_semantics(episode)
    )


def test_compatible_reader_accepts_v1_v2_v3_v4_without_rewriting():
    legacy = {"episode_id": "legacy", "task_name": "legacy", "episode_seed": 1}
    v2 = episode_metadata_v2()
    v3 = episode_metadata_v3(include_wrist=True, include_entities=True)
    v4 = episode_v4()
    assert validate_compatible_episode_metadata(legacy) == []
    assert validate_compatible_episode_metadata(v2) == []
    assert validate_compatible_episode_metadata(v3) == []
    assert validate_compatible_episode_metadata(v4) == []
