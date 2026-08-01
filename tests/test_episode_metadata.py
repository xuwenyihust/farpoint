from copy import deepcopy

from farpoint.episode_metadata import normalize_episode_metadata, normalize_episode_metadata_v2

from v2_fixtures import episode_metadata_v2


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


def test_v2_normalizer_adds_export_identity_without_inventing_scene_values():
    expected = episode_metadata_v2()
    raw = deepcopy(expected)
    del raw["schema_version"]
    del raw["identity"]
    raw["episode_id"] = "episode-0000"
    raw["trial_id"] = "trial-0000"
    normalized = normalize_episode_metadata_v2(
        raw,
        {"success": True, "dataset_valid": True},
        split="train",
        dataset_episode_index=0,
    )
    assert normalized == expected


def test_v2_normalizer_refuses_missing_provenance():
    raw = episode_metadata_v2()
    raw.pop("provenance")
    try:
        normalize_episode_metadata_v2(raw, split="train", dataset_episode_index=0)
    except ValueError as error:
        assert "provenance" in str(error)
    else:
        raise AssertionError("missing provenance should fail")
