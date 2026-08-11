import json

import pytest

from farpoint.campaign import (
    CampaignEventLog,
    attempt_seed,
    atomic_status_write,
    build_crossed_quotas,
    create_campaign,
    create_segment,
    validate_campaign_semantics,
    validate_event_sequence,
    validate_segment_semantics,
    variation_seed,
)


SHA = "a" * 64


def _campaign_spec():
    return {
        "campaign_id": "so101-v010",
        "lineage_id": "farpoint-so101-v010",
        "task_id": "so101-pick-place",
        "campaign_version": "0.1.0",
        "target": {"successful_episodes": 4, "splits": {"train": 3, "validation": 1}},
        "quotas": [
            {"object_variant_id": "red", "yaw_stratum_id": "yaw00", "region_band": "core", "split": "train", "count": 2},
            {"object_variant_id": "blue", "yaw_stratum_id": "yaw18", "region_band": "outer", "split": "train", "count": 1},
            {"object_variant_id": "blue", "yaw_stratum_id": "yaw18", "region_band": "outer", "split": "validation", "count": 1},
        ],
        "variation_contract": {"sampler": "farpoint.scrambled-sobol.v1"},
        "attempt_policy": {"maximum_attempts_per_variation": 3, "global_attempt_limit": None, "replacement_policy": "same_quota_new_variation_seed"},
        "watchdog_policy": {"profile": "so101-v010"},
        "rollout_holdout": {"scene_count": 20, "disjoint": True},
    }


def test_campaign_is_hash_bound_and_has_exact_quota_semantics():
    campaign = create_campaign(_campaign_spec())
    assert validate_campaign_semantics(campaign) == []
    campaign["target"]["successful_episodes"] = 5
    errors = validate_campaign_semantics(campaign)
    assert "campaign_sha256 does not match canonical campaign content" in errors
    assert "campaign quotas do not sum to the success target" in errors


def test_continuation_segment_requires_parent_manifest_hash():
    first = create_segment({
        "campaign_id": "so101-v010", "campaign_sha256": SHA,
        "segment_id": "segment-000", "segment_index": 0,
        "git_commit": "a" * 40, "plan_sha256": SHA,
        "parent_manifest_sha256": None, "oracle_profile_allowlist": ["default"],
        "execution_status": "RUNNING", "quality_status": "NOT_EVALUATED", "attempts": [],
    })
    assert validate_segment_semantics(first) == []
    with pytest.raises(ValueError, match="parent manifest"):
        create_segment({**first, "segment_id": "segment-001", "segment_index": 1, "segment_sha256": SHA})


def test_event_log_is_append_only_and_monotonic(tmp_path):
    path = tmp_path / "events.jsonl"
    log = CampaignEventLog(path, "so101-v010")
    assert log.append("campaign_created", {}, timestamp_unix=1.0)["sequence"] == 0
    assert log.append("heartbeat", {"alive": True}, timestamp_unix=2.0)["sequence"] == 1
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert validate_event_sequence(events) == []
    events[1]["sequence"] = 3
    assert validate_event_sequence(events) == ["event[1] sequence is not contiguous"]


def test_atomic_status_write_validates_independent_state_fields(tmp_path):
    path = tmp_path / "status.json"
    atomic_status_write(path, {"execution_status": "RUNNING", "quality_status": "NOT_EVALUATED", "heartbeat_unix": 4.0})
    assert json.loads(path.read_text())["execution_status"] == "RUNNING"
    with pytest.raises(ValueError, match="quality_status"):
        atomic_status_write(path, {"execution_status": "RUNNING", "quality_status": "DONE", "heartbeat_unix": 4.0})


def test_v010_crossed_quotas_are_exact_200_with_180_20_split():
    rows = build_crossed_quotas(
        object_variant_ids=("red-40mm-40g", "blue-30mm-30g"),
        yaw_stratum_ids=("yaw00", "yaw18", "yaw36", "yaw54", "yaw72"),
        population_per_yaw_region={"core": 5, "middle": 10, "outer": 5},
        validation_per_object_region={"core": 2, "middle": 5, "outer": 3},
        validation_per_object_yaw=2,
        seed=20260811,
    )
    assert sum(row["count"] for row in rows) == 200
    assert sum(row["count"] for row in rows if row["split"] == "train") == 180
    assert sum(row["count"] for row in rows if row["split"] == "validation") == 20
    assert {row["split"] for row in rows} == {"train", "validation"}
    for object_id in ("red-40mm-40g", "blue-30mm-30g"):
        selected = [row for row in rows if row["object_variant_id"] == object_id]
        assert sum(row["count"] for row in selected) == 100
        for yaw in ("yaw00", "yaw18", "yaw36", "yaw54", "yaw72"):
            yaw_rows = [row for row in selected if row["yaw_stratum_id"] == yaw]
            assert sum(row["count"] for row in yaw_rows) == 20
            assert sum(row["count"] for row in yaw_rows if row["split"] == "validation") == 2
        assert {
            region: sum(row["count"] for row in selected if row["split"] == "validation" and row["region_band"] == region)
            for region in ("core", "middle", "outer")
        } == {"core": 2, "middle": 5, "outer": 3}


def test_variation_and_attempt_seeds_are_stable_but_separate():
    kwargs = {
        "object_variant_id": "red", "yaw_stratum_id": "yaw00",
        "region_band": "core", "split": "train", "quota_ordinal": 0,
    }
    first = variation_seed(SHA, **kwargs)
    assert first == variation_seed(SHA, **kwargs)
    assert first != variation_seed(SHA, **kwargs, replacement_index=1)
    attempts = [attempt_seed(first, index) for index in range(3)]
    assert len(set(attempts)) == 3
    with pytest.raises(ValueError, match="0, 1, or 2"):
        attempt_seed(first, 3)
