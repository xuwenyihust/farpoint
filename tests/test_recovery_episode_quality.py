import json

import pytest

from farpoint.campaign import canonical_sha256, create_campaign, create_segment
from farpoint.campaign_recovery import (
    build_campaign_export_selection,
    evaluate_self_healing_campaign,
)
from farpoint.recovery_episode_quality import validate_recovery_episode_eligibility
from farpoint.so101_collection import create_manifest, next_attempt, record_attempt


ELIGIBILITY = {
    "schema_version": "farpoint.recovery-episode-eligibility.v1",
    "by_trigger_class": {
        "contact_without_lift": {
            "handoff_stage": "grasp",
            "trigger_evidence": {
                "ever_contact": True,
                "ever_lifted": False,
                "has_contact": True,
                "cube_lifted": False,
            },
            "handoff": {
                "physics_state_continuous": True,
                "reset_performed": False,
            },
        }
    },
}


def _metadata(*, stage="grasp", has_contact=True):
    return {
        "demonstration": {
            "type": "recovery",
            "intervention": {
                "trigger": {
                    "failure_class": "contact_without_lift",
                    "handoff_stage": stage,
                    "trigger_reason": "stage_progress_stall",
                    "evidence": {
                        "ever_contact": True,
                        "ever_lifted": False,
                        "has_contact": has_contact,
                        "cube_lifted": False,
                    },
                },
                "handoff": {
                    "physics_state_continuous": True,
                    "reset_performed": False,
                },
            },
        }
    }


def test_transport_eligibility_enforces_canonical_subclass_and_stages():
    eligibility = {
        "schema_version": "farpoint.recovery-episode-eligibility.v1",
        "by_trigger_class": {
            "transport_drift": {
                "handoff_stage": "transport",
                "trigger_fields": {
                    "failure_stage": "transport",
                    "last_completed_stage": "lift",
                },
                "allowed_failure_subclasses": [
                    "transport_drop",
                    "transport_stall",
                    "premature_release_outside_target",
                ],
                "trigger_evidence": {
                    "ever_lifted": True,
                    "ever_near_target": False,
                },
                "handoff": {
                    "physics_state_continuous": True,
                    "reset_performed": False,
                },
            }
        },
    }
    metadata = {
        "demonstration": {
            "type": "recovery",
            "intervention": {
                "trigger": {
                    "failure_class": "transport_drift",
                    "handoff_stage": "transport",
                    "failure_stage": "transport",
                    "last_completed_stage": "lift",
                    "failure_subclass": "transport_drop",
                    "evidence": {"ever_lifted": True, "ever_near_target": False},
                },
                "handoff": {
                    "physics_state_continuous": True,
                    "reset_performed": False,
                },
            },
        }
    }
    trial = {"recovery_trigger_class": "transport_drift"}
    assert validate_recovery_episode_eligibility(metadata, trial, eligibility) == []

    metadata["demonstration"]["intervention"]["trigger"][
        "failure_subclass"
    ] = "contact_without_lift"
    assert validate_recovery_episode_eligibility(metadata, trial, eligibility) == [
        "recovery failure_subclass is not allowed by campaign eligibility"
    ]


def _evidence(tmp_path, metadata):
    campaign = create_campaign(
        {
            "campaign_id": "grasp-recovery-quality",
            "campaign_kind": "formal",
            "lineage_id": "farpoint-so101-v012-grasp-recovery",
            "task_id": "so101_cube_pick_place",
            "campaign_version": "0.1.2-grasp-recovery.1",
            "target": {"successful_episodes": 1, "splits": {"train": 1}},
            "quotas": [
                {
                    "object_variant_id": "red-40mm-40g",
                    "yaw_stratum_id": "yaw00_18",
                    "region_band": "middle",
                    "split": "train",
                    "count": 1,
                }
            ],
            "variation_contract": {
                "kind": "live_policy_recovery",
                "episode_eligibility": ELIGIBILITY,
            },
            "attempt_policy": {
                "maximum_attempts_per_variation": 3,
                "global_attempt_limit": None,
                "replacement_policy": "same_quota_new_variation_seed",
            },
            "watchdog_policy": {"profile_id": "recovery-v1"},
            "rollout_holdout": {"policy": "excluded"},
        }
    )
    trial = {
        "trial_id": "trial-0",
        "variation_id": "variation-0",
        "seed": 123,
        "object_variant_id": "red-40mm-40g",
        "yaw_stratum_id": "yaw00_18",
        "region_band": "middle",
        "split": "train",
        "quota_ordinal": 0,
        "recovery_trigger_class": "contact_without_lift",
    }
    plan = {
        "task_id": "so101_cube_pick_place",
        "plan_id": "segment-000",
        "campaign_contract": campaign,
        "trials": [trial],
        "varied_axes": [],
        "frozen_axes": [],
        "collection": {
            "kind": "self_healing_campaign_segment",
            "required_successes": 1,
            "maximum_attempts": 3,
            "attempt_policy": {"maximum_attempts_per_variation": 3},
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan, omit=("plan_sha256",))
    manifest = create_manifest(plan, collection_id="segment-000", git_commit="abcdef1")
    attempt = next_attempt(manifest, plan)
    record_attempt(
        manifest,
        plan,
        attempt,
        episode_id="episode-0",
        success=True,
        dataset_valid=True,
    )
    episodes_root = tmp_path / "episodes"
    episode_root = episodes_root / "episode-0"
    episode_root.mkdir(parents=True)
    (episode_root / "metadata.json").write_text(json.dumps(metadata))
    segment = create_segment(
        {
            "campaign_id": campaign["campaign_id"],
            "campaign_sha256": campaign["campaign_sha256"],
            "segment_id": "segment-000",
            "segment_index": 0,
            "git_commit": "abcdef1",
            "plan_sha256": plan["plan_sha256"],
            "parent_manifest_sha256": None,
            "oracle_profile_allowlist": ["oracle-v1"],
            "execution_status": "FINISHED",
            "quality_status": "NOT_EVALUATED",
            "attempts": [],
        }
    )
    return campaign, [
        {
            "segment": segment,
            "plan": plan,
            "manifest": manifest,
            "episodes_root": str(episodes_root),
        }
    ]


def _policy():
    return {
        "schema_version": "farpoint.self-healing-policy.v1",
        "maximum_attempts_per_variation": 3,
        "distinct_structural_failure_limit": 12,
        "recent_window_attempts": 50,
        "minimum_recent_success_rate": 0.2,
        "no_success_timeout_seconds": 7200,
        "heartbeat_timeout_seconds": 60,
        "minimum_free_disk_bytes": 500 * 1024**3,
        "diagnostic_representatives_per_class": 3,
        "structural_failure_classes": ["runner_error", "phase_timeout"],
    }


def test_measured_grasp_recovery_eligibility_rejects_stage_and_contact_mismatch():
    trial = {"recovery_trigger_class": "contact_without_lift"}
    assert validate_recovery_episode_eligibility(
        _metadata(), trial, ELIGIBILITY
    ) == []
    errors = validate_recovery_episode_eligibility(
        _metadata(stage="lift", has_contact=False), trial, ELIGIBILITY
    )
    assert "recovery handoff_stage does not match campaign eligibility" in errors
    assert (
        "recovery trigger evidence has_contact does not match campaign eligibility"
        in errors
    )


def test_campaign_report_and_export_fail_closed_on_ineligible_recovery(tmp_path):
    campaign, evidence = _evidence(
        tmp_path, _metadata(stage="lift", has_contact=False)
    )
    report = evaluate_self_healing_campaign(
        campaign,
        evidence,
        _policy(),
        live_status={
            "execution_status": "FINISHED",
            "heartbeat_unix": 1000.0,
            "started_unix": 900.0,
        },
        free_disk_bytes=600 * 1024**3,
        now_unix=1001.0,
    )
    assert report["decision"] == "INVALID"
    assert report["progress"]["successful_quotas"] == 0
    assert any("recovery_episode_ineligible" in error for error in report["errors"])
    with pytest.raises(ValueError, match="selected recovery episode is ineligible"):
        build_campaign_export_selection(
            campaign, evidence, dataset_id="farpoint-so101-v012"
        )


def test_campaign_report_and_export_accept_exact_measured_grasp_recovery(tmp_path):
    campaign, evidence = _evidence(tmp_path, _metadata())
    report = evaluate_self_healing_campaign(
        campaign,
        evidence,
        _policy(),
        live_status={
            "execution_status": "FINISHED",
            "heartbeat_unix": 1000.0,
            "started_unix": 900.0,
        },
        free_disk_bytes=600 * 1024**3,
        now_unix=1001.0,
    )
    assert report["decision"] == "COMPLETE"
    assert report["progress"]["successful_quotas"] == 1
    assert report["recovery_episode_eligibility"] == {
        "validated_selected_episodes": 1,
        "failure_classes": {"contact_without_lift": 1},
        "handoff_stages": {"grasp": 1},
        "trigger_reasons": {"stage_progress_stall": 1},
        "current_contact_count": 1,
        "ever_lifted_count": 0,
    }
    selection = build_campaign_export_selection(
        campaign, evidence, dataset_id="farpoint-so101-v012"
    )
    assert len(selection["episodes"]) == 1
    assert selection["episodes"][0]["split"] == "train"
    assert selection["episodes"][0]["recovery_handoff"] == {
        "failure_class": "contact_without_lift",
        "handoff_stage": "grasp",
        "trigger_reason": "stage_progress_stall",
        "has_contact": True,
        "ever_lifted": False,
    }
