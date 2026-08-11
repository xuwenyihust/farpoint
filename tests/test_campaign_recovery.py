from copy import deepcopy

from farpoint.campaign import canonical_sha256, create_campaign, create_segment
from farpoint.campaign_recovery import (
    build_replacement_requests,
    create_continuation_segment,
    diagnostic_clusters,
    evaluate_self_healing_campaign,
    validate_replacement_plan,
    validate_self_healing_policy,
)
from farpoint.so101_collection import create_manifest, next_attempt, record_attempt


SHA = "a" * 64


def _policy(**overrides):
    value = {
        "schema_version": "farpoint.self-healing-policy.v1",
        "maximum_attempts_per_variation": 3,
        "distinct_structural_failure_limit": 12,
        "recent_window_attempts": 50,
        "minimum_recent_success_rate": 0.2,
        "no_success_timeout_seconds": 7200,
        "heartbeat_timeout_seconds": 60,
        "minimum_free_disk_bytes": 500 * 1024**3,
        "diagnostic_representatives_per_class": 3,
        "structural_failure_classes": [
            "runner_error",
            "bilateral_contact_lost",
            "phase_timeout",
        ],
    }
    value.update(overrides)
    return value


def _campaign(count=2):
    return create_campaign(
        {
            "campaign_id": "so101-v010-formal",
            "campaign_kind": "formal",
            "lineage_id": "farpoint-so101-v010",
            "task_id": "so101_cube_pick_place",
            "campaign_version": "0.1.0",
            "target": {"successful_episodes": count, "splits": {"train": count}},
            "quotas": [
                {
                    "object_variant_id": "red-40mm-40g",
                    "yaw_stratum_id": "yaw00_18",
                    "region_band": "core",
                    "split": "train",
                    "count": count,
                }
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


def _plan(campaign, count=2, *, replacement_index=0):
    trials = []
    for index in range(count):
        trials.append(
            {
                "trial_id": f"trial-{replacement_index}-{index}",
                "variation_id": f"variation-{replacement_index}-{index}",
                "seed": 1000 + replacement_index * 100 + index,
                "object_variant_id": "red-40mm-40g",
                "yaw_stratum_id": "yaw00_18",
                "region_band": "core",
                "split": "train",
                "quota_ordinal": index,
                "replacement_index": replacement_index,
            }
        )
    plan = {
        "task_id": "so101_cube_pick_place",
        "plan_id": f"segment-{replacement_index}",
        "campaign_contract": deepcopy(campaign),
        "trials": trials,
        "varied_axes": [],
        "frozen_axes": [],
        "collection": {
            "kind": "self_healing_campaign_segment",
            "required_successes": count,
            "maximum_attempts": count * 3,
            "attempt_policy": {"maximum_attempts_per_variation": 3},
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan, omit=("plan_sha256",))
    return plan


def _segment(campaign, plan):
    return create_segment(
        {
            "campaign_id": campaign["campaign_id"],
            "campaign_sha256": campaign["campaign_sha256"],
            "segment_id": "segment-000",
            "segment_index": 0,
            "git_commit": "abcdef1",
            "plan_sha256": plan["plan_sha256"],
            "parent_manifest_sha256": None,
            "oracle_profile_allowlist": ["profile-v1"],
            "execution_status": "RUNNING",
            "quality_status": "NOT_EVALUATED",
            "attempts": [],
        }
    )


def _fail_next(manifest, plan, reason="bilateral_contact_lost:static_hold"):
    attempt = next_attempt(manifest, plan)
    record_attempt(
        manifest,
        plan,
        attempt,
        episode_id=f"episode-{attempt['attempt_id']}",
        success=False,
        dataset_valid=True,
        failure_category="oracle",
        failure_reason=reason,
    )
    return attempt


def test_self_healing_manifest_hard_limits_each_variation_to_three_attempts():
    campaign = _campaign()
    plan = _plan(campaign)
    manifest = create_manifest(
        plan, collection_id="segment-000", git_commit="abcdef1"
    )

    while manifest["execution_status"] == "RUNNING":
        _fail_next(manifest, plan)

    assert len(manifest["attempts"]) == 6
    assert manifest["quality_status"] == "FAIL"
    assert next_attempt(manifest, plan) is None
    assert {row["attempt_index"] for row in manifest["attempts"]} == {0, 1, 2}


def test_segment_finishes_when_remaining_variation_is_deferred_before_global_budget():
    campaign = _campaign()
    plan = _plan(campaign)
    manifest = create_manifest(
        plan, collection_id="segment-000", git_commit="abcdef1"
    )
    _fail_next(manifest, plan)
    success = next_attempt(manifest, plan)
    record_attempt(
        manifest,
        plan,
        success,
        episode_id="episode-success",
        success=True,
        dataset_valid=True,
    )
    _fail_next(manifest, plan)
    _fail_next(manifest, plan)

    assert len(manifest["attempts"]) == 4
    assert manifest["execution_status"] == "FINISHED"
    assert manifest["quality_status"] == "FAIL"


def test_replacement_requests_preserve_quota_and_change_only_replacement_seed():
    campaign = _campaign()
    plan = _plan(campaign)
    manifest = create_manifest(
        plan, collection_id="segment-000", git_commit="abcdef1"
    )
    while manifest["execution_status"] == "RUNNING":
        _fail_next(manifest, plan)

    first = build_replacement_requests(campaign, plan, manifest)
    second = build_replacement_requests(campaign, plan, manifest)

    assert first == second
    assert len(first) == 2
    assert [row["replacement_index"] for row in first] == [1, 1]
    assert [row["quota"]["quota_ordinal"] for row in first] == [0, 1]
    assert {row["variation_seed"] for row in first}.isdisjoint(
        {trial["seed"] for trial in plan["trials"]}
    )
    continuation = _plan(campaign, replacement_index=1)
    for trial, request in zip(continuation["trials"], first, strict=True):
        trial["seed"] = request["variation_seed"]
    validate_replacement_plan(first, continuation)
    continuation["trials"][0]["region_band"] = "outer"
    try:
        validate_replacement_plan(first, continuation)
    except ValueError as error:
        assert "does not exactly realize" in str(error)
    else:
        raise AssertionError("replacement plan accepted a changed quota")


def test_distinct_structural_variations_pause_and_select_three_diagnostics():
    campaign = _campaign(12)
    plan = _plan(campaign, 12)
    manifest = create_manifest(
        plan, collection_id="segment-000", git_commit="abcdef1"
    )
    for index in range(12):
        attempt = next_attempt(manifest, plan)
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=f"episode-{index}",
            success=False,
            dataset_valid=True,
            failure_category="oracle",
            failure_reason="bilateral_contact_lost:bilateral_settle",
        )
    report = evaluate_self_healing_campaign(
        campaign,
        [{"segment": _segment(campaign, plan), "plan": plan, "manifest": manifest}],
        _policy(),
        live_status={"heartbeat_unix": 1000.0, "started_unix": 900.0},
        free_disk_bytes=600 * 1024**3,
        now_unix=1001.0,
    )

    assert report["decision"] == "PAUSE"
    assert report["next_action"] == "ORACLE_REPAIR"
    assert report["pause_reasons"] == [
        "distinct_structural_failures:bilateral_contact_lost:12"
    ]
    assert len(report["diagnostic_clusters"]["bilateral_contact_lost"]) == 3


def test_liveness_disk_and_integrity_fail_closed():
    campaign = _campaign()
    plan = _plan(campaign)
    manifest = create_manifest(
        plan, collection_id="segment-000", git_commit="abcdef1"
    )
    report = evaluate_self_healing_campaign(
        campaign,
        [{"segment": _segment(campaign, plan), "plan": plan, "manifest": manifest}],
        _policy(),
        live_status={"heartbeat_unix": 900.0, "started_unix": 900.0},
        free_disk_bytes=499 * 1024**3,
        integrity_errors=["wrist_video_checksum_mismatch"],
        now_unix=1000.0,
    )

    assert report["decision"] == "INVALID"
    assert report["next_action"] == "NONE"
    assert "wrist_video_checksum_mismatch" in report["errors"]
    assert "heartbeat_stale:100s" in report["pause_reasons"]
    assert any(reason.startswith("disk_below_minimum") for reason in report["pause_reasons"])


def test_continuation_binds_exact_parent_manifest_hash_and_next_index():
    campaign = _campaign()
    plan = _plan(campaign)
    manifest = create_manifest(
        plan, collection_id="segment-000", git_commit="abcdef1"
    )
    manifest["execution_status"] = "PAUSED"
    parent = _segment(campaign, plan)

    continuation = create_continuation_segment(
        campaign,
        parent,
        manifest,
        segment_id="segment-001",
        git_commit="abcdef2",
        plan_sha256="b" * 64,
        oracle_profile_allowlist=["profile-v1", "profile-v2"],
    )

    assert continuation["segment_index"] == 1
    assert continuation["parent_manifest_sha256"] == canonical_sha256(manifest)
    assert continuation["git_commit"] == "abcdef2"


def test_campaign_reuses_parent_successes_and_completes_from_continuation():
    campaign = _campaign()
    parent_plan = _plan(campaign)
    parent_manifest = create_manifest(
        parent_plan, collection_id="segment-000", git_commit="abcdef1"
    )
    _fail_next(parent_manifest, parent_plan)
    success = next_attempt(parent_manifest, parent_plan)
    record_attempt(
        parent_manifest,
        parent_plan,
        success,
        episode_id="episode-parent-success",
        success=True,
        dataset_valid=True,
    )
    _fail_next(parent_manifest, parent_plan)
    _fail_next(parent_manifest, parent_plan)
    parent_segment = _segment(campaign, parent_plan)
    requests = build_replacement_requests(campaign, parent_plan, parent_manifest)

    continuation_plan = _plan(campaign, count=1, replacement_index=1)
    continuation_plan["trials"][0]["seed"] = requests[0]["variation_seed"]
    continuation_plan["plan_sha256"] = canonical_sha256(
        continuation_plan, omit=("plan_sha256",)
    )
    continuation_segment = create_continuation_segment(
        campaign,
        parent_segment,
        parent_manifest,
        segment_id="segment-001",
        git_commit="abcdef2",
        plan_sha256=continuation_plan["plan_sha256"],
        oracle_profile_allowlist=["profile-v2"],
    )
    continuation_manifest = create_manifest(
        continuation_plan,
        collection_id="segment-001",
        git_commit="abcdef2",
    )
    replacement = next_attempt(continuation_manifest, continuation_plan)
    record_attempt(
        continuation_manifest,
        continuation_plan,
        replacement,
        episode_id="episode-continuation-success",
        success=True,
        dataset_valid=True,
    )

    report = evaluate_self_healing_campaign(
        campaign,
        [
            {
                "segment": parent_segment,
                "plan": parent_plan,
                "manifest": parent_manifest,
            },
            {
                "segment": continuation_segment,
                "plan": continuation_plan,
                "manifest": continuation_manifest,
            },
        ],
        _policy(),
        live_status={"heartbeat_unix": 1000.0, "started_unix": 900.0},
        free_disk_bytes=600 * 1024**3,
        now_unix=1001.0,
    )

    assert report["decision"] == "COMPLETE"
    assert report["progress"] == {
        "successful_quotas": 2,
        "target_successful_episodes": 2,
        "attempted_count": 5,
        "segment_count": 2,
    }


def test_diagnostic_clusters_use_distinct_variations():
    attempts = [
        {
            "variation_id": "a",
            "attempt_id": f"a-{index}",
            "attempt_seed": index,
            "success": False,
            "failure_category": "oracle",
            "failure_reason": "grasp_phase_timeout:slow_close",
        }
        for index in range(3)
    ]
    attempts.append(
        {
            "variation_id": "b",
            "attempt_id": "b-0",
            "attempt_seed": 10,
            "success": False,
            "failure_category": "oracle",
            "failure_reason": "grasp_phase_timeout:slow_close",
        }
    )

    clusters = diagnostic_clusters(attempts, representatives_per_class=3)
    assert [row["variation_id"] for row in clusters["phase_timeout"]] == ["a", "b"]


def test_policy_requires_exactly_three_attempts():
    invalid = _policy(maximum_attempts_per_variation=4)
    try:
        validate_self_healing_policy(invalid)
    except ValueError as error:
        assert "exactly three attempts" in str(error)
    else:
        raise AssertionError("policy unexpectedly accepted four attempts")
