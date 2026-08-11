import json

from farpoint.campaign import create_campaign
from farpoint.campaign_live import CampaignDashboardIndex, LiveCampaignPublisher


class Clock:
    def __init__(self, value=100.0):
        self.value = value

    def __call__(self):
        return self.value


def _campaign(campaign_id="so101-v010-live", *, campaign_kind=None):
    kind = {"campaign_kind": campaign_kind} if campaign_kind is not None else {}
    return create_campaign(
        {
            "campaign_id": campaign_id,
            "lineage_id": "farpoint-so101-v010",
            "task_id": "so101_cube_pick_place",
            "campaign_version": "0.1.0",
            **kind,
            "target": {"successful_episodes": 2, "splits": {"train": 1, "validation": 1}},
            "quotas": [
                {
                    "object_variant_id": "red-40mm-40g",
                    "yaw_stratum_id": "yaw00",
                    "region_band": "core",
                    "split": "train",
                    "count": 1,
                },
                {
                    "object_variant_id": "blue-30mm-30g",
                    "yaw_stratum_id": "yaw18",
                    "region_band": "outer",
                    "split": "validation",
                    "count": 1,
                },
            ],
            "variation_contract": {"sampler": "farpoint.scrambled-sobol.v1"},
            "attempt_policy": {
                "maximum_attempts_per_variation": 3,
                "global_attempt_limit": None,
                "replacement_policy": "same_quota_new_variation_seed",
            },
            "watchdog_policy": {"profile": "so101-v010"},
            "rollout_holdout": {"scene_count": 20, "disjoint": True},
        }
    )


def test_live_publisher_writes_atomic_status_preview_and_ordered_events(tmp_path):
    clock = Clock()
    (tmp_path / "campaign.json").write_text(json.dumps(_campaign()))
    publisher = LiveCampaignPublisher(
        tmp_path,
        "so101-v010-live",
        "segment-000",
        heartbeat_interval_seconds=3600,
        preview_interval_seconds=1,
        clock=clock,
    )
    publisher.start(payload={"target_successful_episodes": 2})
    publisher.update_status(startup_phase="environment_construction")
    publisher.attempt_started("attempt-000", "variation-000")
    assert publisher.publish_preview(b"\xff\xd8jpeg") is True
    assert publisher.publish_preview(b"\xff\xd8new") is False
    clock.value += 1
    assert publisher.publish_preview(b"\xff\xd8new") is True
    publisher.attempt_completed(
        attempt_id="attempt-000",
        variation_id="variation-000",
        success=True,
        dataset_valid=True,
        episode_id="episode-000",
        failure_reason=None,
    )
    publisher.finish(execution_status="FINISHED", quality_status="PASS")

    status = json.loads((tmp_path / "status.json").read_text())
    assert status["execution_status"] == "FINISHED"
    assert status["quality_status"] == "PASS"
    assert status["successful_episodes"] == 1
    assert status["startup_phase"] == "environment_construction"
    assert (tmp_path / "active-preview.jpg").read_bytes() == b"\xff\xd8new"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["sequence"] for event in events] == list(range(len(events)))
    assert [event["event_type"] for event in events] == [
        "segment_started",
        "attempt_started",
        "preview_updated",
        "preview_updated",
        "attempt_completed",
        "episode_completed",
        "segment_finished",
    ]


def test_campaign_dashboard_marks_stale_and_only_promotes_quality_pass(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    (live / "campaign.json").write_text(json.dumps(_campaign("live")))
    (live / "status.json").write_text(
        json.dumps(
            {
                "campaign_id": "live",
                "segment_id": "segment-000",
                "execution_status": "RUNNING",
                "quality_status": "NOT_EVALUATED",
                "heartbeat_unix": 10.0,
                "target_successful_episodes": 1,
                "successful_episodes": 1,
                "completed_attempts": 2,
            }
        )
    )
    passed = tmp_path / "passed"
    passed.mkdir()
    (passed / "campaign.json").write_text(
        json.dumps(_campaign("passed", campaign_kind="formal"))
    )
    (passed / "status.json").write_text(
        json.dumps(
            {
                "campaign_id": "passed",
                "segment_id": "segment-001",
                "execution_status": "FINISHED",
                "quality_status": "PASS",
                "heartbeat_unix": 90.0,
                "successful_episodes": 2,
                "completed_attempts": 3,
            }
        )
    )
    failed = tmp_path / "failed"
    failed.mkdir()
    (failed / "campaign.json").write_text(json.dumps(_campaign("failed")))
    (failed / "status.json").write_text(
        json.dumps(
            {
                "campaign_id": "failed",
                "segment_id": "segment-000",
                "execution_status": "FINISHED",
                "quality_status": "FAIL",
                "heartbeat_unix": 90.0,
            }
        )
    )
    integration = tmp_path / "integration"
    integration.mkdir()
    (integration / "campaign.json").write_text(
        json.dumps(_campaign("integration", campaign_kind="integration"))
    )
    (integration / "status.json").write_text(
        json.dumps(
            {
                "campaign_id": "integration",
                "segment_id": "segment-000",
                "execution_status": "FINISHED",
                "quality_status": "PASS",
                "heartbeat_unix": 90.0,
            }
        )
    )
    legacy_passed = tmp_path / "legacy-passed"
    legacy_passed.mkdir()
    (legacy_passed / "campaign.json").write_text(
        json.dumps(_campaign("legacy-passed"))
    )
    (legacy_passed / "status.json").write_text(
        json.dumps(
            {
                "campaign_id": "legacy-passed",
                "segment_id": "segment-000",
                "execution_status": "FINISHED",
                "quality_status": "PASS",
                "heartbeat_unix": 90.0,
            }
        )
    )

    index = CampaignDashboardIndex([tmp_path], stale_after_seconds=60)
    live_row = index.live_runs(now_unix=100)[0]
    assert live_row["execution_status"] == "STALE"
    assert live_row["target_successful_episodes"] == 1
    assert {row["campaign_id"] for row in index.collections(now_unix=100)} == {
        "live",
        "passed",
        "failed",
        "integration",
        "legacy-passed",
    }
    assert [row["campaign_id"] for row in index.benchmarks(now_unix=100)] == [
        "passed"
    ]
