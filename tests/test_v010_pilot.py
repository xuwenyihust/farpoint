import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from farpoint.campaign import (
    create_segment,
    validate_campaign_semantics,
    validate_segment_semantics,
)
from farpoint.contracts import validate_contract, validate_episode_semantics
from farpoint.episode_v4 import build_so101_episode_v4
from farpoint.so101_collection import create_pilot_manifest
from farpoint.so101_pilot_report import (
    _required_object_region_errors,
    _v010_video_errors,
)
from farpoint.v010_pilot import (
    build_v010_integration_pilot_plan,
    initialize_v010_pilot_campaign,
    load_v010_pilot_config,
)
from farpoint.variation_engine import FeasibleRegion


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/variations/so101_v010_integration_pilot.json"
JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _plan():
    return build_v010_integration_pilot_plan(
        load_v010_pilot_config(CONFIG), pilot_id="so101-v010-pilot"
    )


def _camera(camera_id):
    return {
        "camera_id": camera_id,
        "feature_key": f"observation.images.{camera_id}",
        "width": 640,
        "height": 480,
        "config_version": "so101-front-wrist-v1",
        "config_sha256": "a" * 64,
        "calibration": {"model": "pinhole", "intrinsic_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        "mount_transform": {
            "parent_frame": "isaac_world",
            "position_m": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "frame_timestamp_source": "simulation_control_tick",
        "video_artifact": {
            "path": f"videos/{camera_id}.mp4",
            "container": "mp4",
            "codec": "h264",
            "frame_count": 30,
            "width": 640,
            "height": 480,
            "fps": 30,
            "size_bytes": 100,
            "sha256": "b" * 64,
            "decode_verified": True,
        },
    }


def test_v010_pilot_is_deterministic_continuous_and_exactly_stratified():
    first = _plan()
    second = _plan()
    assert first == second
    assert len(first["trials"]) == 12
    assert len({trial["seed"] for trial in first["trials"]}) == 12
    assert Counter(trial["split"] for trial in first["trials"]) == {
        "train": 10,
        "validation": 2,
    }
    assert validate_campaign_semantics(first["campaign_contract"]) == []
    assert first["campaign_contract"]["campaign_kind"] == "pilot"

    for object_id in ("red-40mm-40g", "blue-30mm-30g"):
        trials = [
            trial for trial in first["trials"] if trial["object_variant_id"] == object_id
        ]
        assert Counter(trial["region_band"] for trial in trials) == {
            "core": 2,
            "middle": 2,
            "outer": 2,
        }
        assert len({trial["yaw_stratum_id"] for trial in trials}) == 5
        for trial in trials:
            value = trial["feasible_region"]["resolved"]
            region = FeasibleRegion(
                region_id=value["region_id"],
                version=value["version"],
                frame_id=value["frame_id"],
                polygon_xy_m=tuple(tuple(point) for point in value["polygon_xy_m"]),
                max_clearance_m=value["max_clearance_m"],
                object_anchor=value["object_anchor"],
                footprint_xy_m=tuple(value["footprint_xy_m"]),
                generator_sha256=value["generator_sha256"],
                constraints_sha256=value["constraints_sha256"],
            )
            point = tuple(trial["resolved"]["position_m"][:2])
            assert region.contains(point)
            assert region.band(point) == trial["region_band"]


def test_v010_pilot_manifest_and_campaign_declarations_are_hash_bound(tmp_path):
    plan = _plan()
    manifest = create_pilot_manifest(
        plan, collection_id=plan["plan_id"], git_commit="a" * 40
    )
    assert manifest["required_successes"] == 10
    assert manifest["maximum_attempts"] == 12
    assert manifest["completion_policy"] == "all_planned_trials"
    bundle = initialize_v010_pilot_campaign(
        tmp_path, plan, git_commit="a" * 40
    )
    assert validate_campaign_semantics(bundle["campaign"]) == []
    assert validate_segment_semantics(bundle["segment"]) == []
    assert bundle["segment"]["plan_sha256"] == plan["plan_sha256"]
    with pytest.raises(FileExistsError):
        initialize_v010_pilot_campaign(tmp_path, plan, git_commit="a" * 40)


def test_v010_acceptance_requires_a_success_for_every_object_region_pair():
    plan = _plan()
    selected = [
        {"variation_id": trial["variation_id"]}
        for trial in plan["trials"]
        if not (
            trial["object_variant_id"] == "red-40mm-40g"
            and trial["region_band"] == "outer"
        )
    ]
    assert _required_object_region_errors(plan, selected) == [
        "required_object_region_failed:red-40mm-40g::outer"
    ]


def test_v010_episode_writer_emits_strict_v4_with_measured_pose():
    plan = _plan()
    campaign = plan["campaign_contract"]
    segment = create_segment(
        {
            "campaign_id": campaign["campaign_id"],
            "campaign_sha256": campaign["campaign_sha256"],
            "segment_id": "segment-000",
            "segment_index": 0,
            "git_commit": "a" * 40,
            "plan_sha256": plan["plan_sha256"],
            "parent_manifest_sha256": None,
            "oracle_profile_allowlist": [plan["oracle_profile_id"]],
            "execution_status": "RUNNING",
            "quality_status": "NOT_EVALUATED",
            "attempts": [],
        }
    )
    trial = plan["trials"][0]
    resolved_object = deepcopy(trial["resolved"])
    resolved_object["position_m"][0] += 1e-7
    metadata = build_so101_episode_v4(
        episode_id="episode-v010-000",
        campaign=campaign,
        segment=segment,
        plan=plan,
        trial=trial,
        attempt_seed=123,
        git_commit="a" * 40,
        simulator_image_digest="sha256:" + "b" * 64,
        resolved_object=resolved_object,
        target=plan["target"],
        table=plan["table"],
        camera_records=[_camera("front"), _camera("wrist")],
        embodiment={
            "robot": "so101",
            "joint_mapping": {"joint_order": JOINTS},
        },
        frame_count=30,
        control_hz=120,
        success=True,
        dataset_valid=True,
        failure_category=None,
        failure_reason=None,
        physics_audit={"mass": {"verified": True}},
    )
    assert metadata["schema_version"] == "farpoint.episode.v4"
    assert metadata["variation"]["resolved"]["position_xy_m"][0] == pytest.approx(
        resolved_object["position_m"][0]
    )
    assert set(metadata["variation"]["requested"]["entities"]) == {
        "pick_object",
        "placement_target",
        "table",
    }
    assert metadata["variation"]["resolved"]["entities"] == {
        entity["entity_id"]: entity for entity in metadata["scene"]["entities"]
    }
    assert validate_contract(metadata) == []
    assert validate_episode_semantics(metadata) == []
    assert json.loads(json.dumps(metadata)) == metadata


def test_v010_video_audit_rejects_missing_and_escaping_artifacts(tmp_path):
    metadata = {
        "recording": {
            "frame_count": 30,
            "cameras": [
                {"camera_id": "front", "video_artifact": {"path": "videos/front.mp4"}},
                {"camera_id": "wrist", "video_artifact": {"path": "../wrist.mp4"}},
            ],
        }
    }
    assert _v010_video_errors(tmp_path / "episode", metadata) == [
        "episode:front_video_missing",
        "episode:wrist_video_path_escape",
    ]
