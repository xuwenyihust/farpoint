from farpoint.episode_metadata import normalize_episode_metadata


def test_legacy_metadata_is_explicitly_labeled_without_rewriting_provenance():
    normalized = normalize_episode_metadata(
        {
            "episode_id": "legacy-001",
            "episode_seed": 6,
            "task_name": "isaac_perception_contact_scene",
            "randomization": {"enabled": True, "pick_object_xy": [0.98, 0.26]},
        },
        {"success": True, "dataset_valid": True},
    )
    assert normalized["source_generation"] == "farpoint_legacy_randomized_v0"
    assert normalized["variation_id"] == "legacy_cube_randomized"
    assert normalized["object_position_bin"] == "legacy_randomized"
    assert normalized["randomization"]["pick_object_xy"] == [0.98, 0.26]


def test_profiled_metadata_preserves_v11_variation():
    normalized = normalize_episode_metadata(
        {
            "episode_id": "profiled-001",
            "episode_seed": 0,
            "task_name": "isaac_perception_contact_scene",
            "variation": {
                "variation_id": "cylinder_position_left",
                "object_type": "cylinder",
                "object_position_bin": "left",
                "grasp_profile": "cylinder_grip_v1",
            },
        },
        {"success": True, "dataset_valid": True},
    )
    assert normalized["source_generation"] == "farpoint_v1_1_profiled"
    assert normalized["variation_id"] == "cylinder_position_left"
    assert normalized["grasp_profile"] == "cylinder_grip_v1"
