from copy import deepcopy


SHA = "a" * 64
GIT_COMMIT = "b" * 40


def episode_metadata_v2(
    *,
    episode_id="episode-0000",
    trial_id="trial-0000",
    split="train",
    dataset_episode_index=0,
    shape="cube",
    instruction=None,
    position=None,
    frame_count=2,
):
    position = position or [0.5, 0.0, 0.05]
    instruction = instruction or f"Pick up the {shape} and place it in the target zone."
    values = {
        "object_position_m": position,
        "object_yaw_degrees": 0.0,
        "object_shape": shape,
        "object_dimensions_m": [0.05, 0.05, 0.05],
        "appearance_profile_id": "yellow_matte_v1",
        "camera_profile_id": "front_rgb_v1",
        "lighting_profile_id": "studio_v1",
    }
    return {
        "schema_version": "farpoint.episode.v2",
        "identity": {
            "episode_id": episode_id,
            "trial_id": trial_id,
            "task_id": f"pick_place_{shape}_v1",
            "split": split,
            "dataset_episode_index": dataset_episode_index,
        },
        "provenance": {
            "git_commit": GIT_COMMIT,
            "config_sha256": SHA,
            "simulator": "Isaac Sim",
            "physics_engine": "PhysX",
            "simulator_image": "nvcr.io/nvidia/isaac-sim:6.0.0",
            "simulator_image_digest": f"sha256:{SHA}",
            "robot_asset_id": "ur10e_robotiq_2f85_v1",
            "robot_asset_path": "/Assets/Robots/UR10eRobotiq.usd",
            "episode_seed": 7,
            "derived_seed": 7001,
        },
        "task": {
            "task_id": f"pick_place_{shape}_v1",
            "instruction": instruction,
            "object_shape": shape,
            "success_criteria_id": "contact_pick_place_v1",
        },
        "embodiment": {
            "robot": "ur10e",
            "gripper": "robotiq_2f85",
            "arm_dof": 6,
            "gripper_dof": 1,
            "controller": "rmpflow_lula_ik_v1",
            "control_mode": "articulation_drive",
            "grasp_mode": "contact_only",
        },
        "scene": {
            "coordinate_frame": "world",
            "object": {
                "shape": shape,
                "dimensions_m": [0.05, 0.05, 0.05],
                "mass_kg": 0.1,
                "material_id": "default_physics_v1",
                "initial_pose": {
                    "position_m": position,
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "yaw_degrees": 0.0,
                    "coordinate_frame": "world",
                },
            },
            "target": {
                "target_id": "target_zone_v1",
                "pose": {
                    "position_m": [0.65, 0.2, 0.05],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "yaw_degrees": 0.0,
                    "coordinate_frame": "world",
                },
            },
            "camera": {
                "profile_id": "front_rgb_v1",
                "calibration_id": "front_v1",
                "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 180.0},
                "extrinsics": {
                    "position_m": [1.5, -1.5, 1.2],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "yaw_degrees": 45.0,
                    "coordinate_frame": "world",
                },
            },
            "lighting_profile_id": "studio_v1",
            "appearance_profile_id": "yellow_matte_v1",
        },
        "variation": {
            "schema_version": "farpoint.variation.v2",
            "variation_id": "position_r00_c00_s00",
            "varied_axes": ["object_position_m"],
            "frozen_axes": [
                "object_shape", "object_dimensions_m", "object_yaw_degrees",
                "appearance_profile_id", "camera_profile_id", "lighting_profile_id",
            ],
            "cell_id": "r00_c00",
            "slot": 0,
            "requested": deepcopy(values),
            "resolved": deepcopy(values),
        },
        "recording": {
            "fps": 20,
            "cameras": ["observation.images.front"],
            "image_width": 640,
            "image_height": 360,
            "frame_count": frame_count,
        },
        "outcome": {
            "success": True,
            "dataset_valid": True,
            "failure_category": None,
            "failure_reason": None,
            "quality": {
                "final_xy_error_m": 0.01,
                "perception_error_m": 0.005,
                "bilateral_contact_frames": 30,
                "lift_height_m": 0.2,
                "settling_error": 0.01,
                "joint_smoothness_score": 0.95,
            },
        },
    }


def dataset_sidecar_v2(episodes):
    splits = {name: 0 for name in ("train", "validation", "test")}
    tasks = {}
    for episode in episodes:
        splits[episode["identity"]["split"]] += 1
        tasks[episode["task"]["task_id"]] = episode["task"]
    return {
        "schema_version": "farpoint.dataset.v2",
        "dataset_id": "farpoint-ur10e-robotiq-2f85",
        "format": "lerobot",
        "format_version": "v3",
        "demonstration_policy": "successful_only",
        "splits": splits,
        "tasks": [tasks[key] for key in sorted(tasks)],
        "robot": {"name": "ur10e", "gripper": "robotiq_2f85", "arm_dof": 6, "gripper_dof": 1},
        "simulation": {
            "simulator": "Isaac Sim",
            "image": "nvcr.io/nvidia/isaac-sim:6.0.0",
            "image_digest": f"sha256:{SHA}",
            "physics": "PhysX",
        },
        "recording": {
            "fps": 20,
            "cameras": ["observation.images.front"],
            "image_width": 640,
            "image_height": 360,
            "state_features": ["joint_0"],
            "action_features": ["joint_0"],
        },
        "contracts": {
            "episode": "farpoint.episode.v2",
            "variation": "farpoint.variation.v2",
            "benchmark": "farpoint.benchmark.v2",
        },
    }
