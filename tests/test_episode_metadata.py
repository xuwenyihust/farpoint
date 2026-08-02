from copy import deepcopy

from farpoint.episode_metadata import (
    normalize_episode_metadata,
    normalize_episode_metadata_v2,
    resolve_measured_object_pose,
    validate_simulator_metadata_v2,
)

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


def test_simulator_v2_metadata_is_validated_before_persistence():
    raw = episode_metadata_v2()
    raw["episode_id"] = raw["identity"]["episode_id"]
    raw["trial_id"] = raw["identity"]["trial_id"]
    raw["split"] = raw["identity"]["split"]
    validated = validate_simulator_metadata_v2(raw)
    assert validated["schema_version"] == "farpoint.episode.v2"
    assert validated["provenance"] == raw["provenance"]

    raw["recording"].pop("fps")
    try:
        validate_simulator_metadata_v2(raw)
    except ValueError as error:
        assert "recording" in str(error)
    else:
        raise AssertionError("invalid simulator recording metadata should fail")


def test_v2_contract_accepts_perception_failure_category():
    raw = episode_metadata_v2()
    raw["outcome"]["success"] = False
    raw["outcome"]["failure_category"] = "perception"
    raw["outcome"]["failure_reason"] = (
        "rgbd_pose_estimation_exceeded_tolerance"
    )
    raw["episode_id"] = raw["identity"]["episode_id"]
    raw["trial_id"] = raw["identity"]["trial_id"]
    raw["split"] = raw["identity"]["split"]

    validated = validate_simulator_metadata_v2(raw)

    assert validated["outcome"]["failure_category"] == "perception"


def test_measured_pose_updates_only_resolved_variation():
    raw = episode_metadata_v2()["variation"]
    requested = deepcopy(raw["requested"])
    measured = [0.87, 0.20, 0.4075]
    resolved = resolve_measured_object_pose(raw, measured)
    assert resolved["requested"] == requested
    assert resolved["resolved"]["object_position_m"] == measured
    assert raw["resolved"]["object_position_m"] != measured
