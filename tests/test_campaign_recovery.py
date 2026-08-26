from copy import deepcopy

from farpoint.campaign import canonical_sha256, create_campaign, create_segment
from farpoint.campaign_recovery import (
    build_campaign_export_selection,
    build_continuation_requests,
    build_replacement_requests,
    create_campaign_quality_exclusions,
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


def _campaign(count=2, *, recovery=False):
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
            "variation_contract": (
                {"kind": "live_policy_recovery"}
                if recovery
                else {"sampler": "farpoint.scrambled-sobol.v1"}
            ),
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


def _success_next(manifest, plan, *, episode_id):
    attempt = next_attempt(manifest, plan)
    record_attempt(
        manifest,
        plan,
        attempt,
        episode_id=episode_id,
        success=True,
        dataset_valid=True,
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


def test_continuation_requests_preserve_partial_seed_and_replace_only_exhausted():
    campaign = _campaign()
    plan = _plan(campaign)
    manifest = create_manifest(
        plan, collection_id="segment-000", git_commit="abcdef1"
    )
    for _ in range(5):
        _fail_next(manifest, plan)
    manifest["execution_status"] = "ABORTED"
    manifest["quality_status"] = "NOT_EVALUATED"

    requests = build_continuation_requests(
        campaign,
        [{"segment": _segment(campaign, plan), "plan": plan, "manifest": manifest}],
    )

    by_quota = {row["quota"]["quota_ordinal"]: row for row in requests}
    exhausted = by_quota[0]
    partial = by_quota[1]
    assert exhausted["request_kind"] == "replacement"
    assert exhausted["replacement_index"] == 1
    assert exhausted["prior_attempt_count"] == 0
    assert exhausted["remaining_attempt_count"] == 3
    assert exhausted["variation_seed"] != plan["trials"][0]["seed"]
    assert partial["request_kind"] == "carryover"
    assert partial["replacement_index"] == 0
    assert partial["prior_attempt_count"] == 2
    assert partial["remaining_attempt_count"] == 1
    assert partial["variation_seed"] == plan["trials"][1]["seed"]


def test_continuation_trial_uses_only_remaining_cross_segment_attempt():
    campaign = _campaign(count=1)
    plan = _plan(campaign, count=1)
    trial = plan["trials"][0]
    trial["prior_attempt_count"] = 2
    plan["collection"]["maximum_attempts"] = 1
    plan["plan_sha256"] = canonical_sha256(plan, omit=("plan_sha256",))
    manifest = create_manifest(
        plan, collection_id="segment-001", git_commit="abcdef2"
    )

    attempt = _fail_next(manifest, plan)

    assert attempt["attempt_index"] == 2
    assert attempt["attempt_id"].endswith("__attempt02")
    assert next_attempt(manifest, plan) is None
    assert len(manifest["attempts"]) == 1


def test_aggregate_continuation_keeps_missing_quota_from_older_segment():
    campaign = _campaign()
    parent_plan = _plan(campaign)
    parent_manifest = create_manifest(
        parent_plan, collection_id="segment-000", git_commit="abcdef1"
    )
    for _ in range(5):
        _fail_next(parent_manifest, parent_plan)
    parent_manifest["execution_status"] = "ABORTED"
    parent_manifest["quality_status"] = "NOT_EVALUATED"
    parent_segment = _segment(campaign, parent_plan)
    initial_requests = build_continuation_requests(
        campaign,
        [
            {
                "segment": parent_segment,
                "plan": parent_plan,
                "manifest": parent_manifest,
            }
        ],
    )
    replacement_request = next(
        row for row in initial_requests if row["request_kind"] == "replacement"
    )
    continuation_plan = _plan(campaign, count=1, replacement_index=1)
    continuation_plan["trials"][0]["seed"] = replacement_request["variation_seed"]
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
        continuation_plan, collection_id="segment-001", git_commit="abcdef2"
    )
    attempt = next_attempt(continuation_manifest, continuation_plan)
    record_attempt(
        continuation_manifest,
        continuation_plan,
        attempt,
        episode_id="episode-replacement-success",
        success=True,
        dataset_valid=True,
    )

    remaining = build_continuation_requests(
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
    )

    assert len(remaining) == 1
    assert remaining[0]["quota"]["quota_ordinal"] == 1
    assert remaining[0]["request_kind"] == "carryover"
    assert remaining[0]["source_segment_id"] == "segment-000"
    assert remaining[0]["prior_attempt_count"] == 2
    assert remaining[0]["variation_seed"] == parent_plan["trials"][1]["seed"]


def _two_segment_gap_evidence(*, parent_hash_matches=True):
    campaign = _campaign()
    parent_plan = _plan(campaign)
    parent_manifest = create_manifest(
        parent_plan, collection_id="segment-000", git_commit="abcdef1"
    )
    for _ in range(5):
        _fail_next(parent_manifest, parent_plan)
    parent_manifest["execution_status"] = "ABORTED"
    parent_manifest["quality_status"] = "NOT_EVALUATED"
    parent_segment = _segment(campaign, parent_plan)
    request = next(
        row
        for row in build_continuation_requests(
            campaign,
            [{"segment": parent_segment, "plan": parent_plan, "manifest": parent_manifest}],
        )
        if row["request_kind"] == "replacement"
    )
    continuation_plan = _plan(campaign, count=1, replacement_index=1)
    continuation_plan["trials"][0]["seed"] = request["variation_seed"]
    continuation_plan["plan_sha256"] = canonical_sha256(
        continuation_plan, omit=("plan_sha256",)
    )
    continuation_segment = create_segment(
        {
            "campaign_id": campaign["campaign_id"],
            "campaign_sha256": campaign["campaign_sha256"],
            "segment_id": "segment-002",
            "segment_index": 2,
            "git_commit": "abcdef2",
            "plan_sha256": continuation_plan["plan_sha256"],
            "parent_manifest_sha256": (
                canonical_sha256(parent_manifest)
                if parent_hash_matches
                else "f" * 64
            ),
            "oracle_profile_allowlist": ["profile-v2"],
            "execution_status": "RUNNING",
            "quality_status": "NOT_EVALUATED",
            "attempts": [],
        }
    )
    continuation_manifest = create_manifest(
        continuation_plan, collection_id="segment-002", git_commit="abcdef2"
    )
    evidence = [
        {"segment": parent_segment, "plan": parent_plan, "manifest": parent_manifest},
        {
            "segment": continuation_segment,
            "plan": continuation_plan,
            "manifest": continuation_manifest,
        },
    ]
    return campaign, evidence


def test_continuation_allows_index_gaps_with_intact_parent_hash_chain():
    campaign, evidence = _two_segment_gap_evidence()

    assert len(build_continuation_requests(campaign, evidence)) == 2


def test_continuation_gap_still_requires_exact_parent_hash_chain():
    campaign, evidence = _two_segment_gap_evidence(parent_hash_matches=False)

    try:
        build_continuation_requests(campaign, evidence)
    except ValueError as error:
        assert "parent manifest hash mismatch" in str(error)
    else:
        raise AssertionError("continuation accepted a broken parent hash chain")


def test_campaign_evaluation_allows_index_gaps_with_intact_parent_hash_chain():
    campaign, evidence = _two_segment_gap_evidence()

    report = evaluate_self_healing_campaign(
        campaign,
        evidence,
        _policy(),
        live_status={"heartbeat_unix": 1000.0, "started_unix": 900.0},
        free_disk_bytes=600 * 1024**3,
        now_unix=1001.0,
    )

    assert report["decision"] != "INVALID"
    assert report["errors"] == []


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


def test_terminal_segment_ignores_stale_live_liveness_and_freezes_replacement():
    campaign = _campaign()
    plan = _plan(campaign)
    manifest = create_manifest(
        plan, collection_id="segment-000", git_commit="abcdef1"
    )
    success = _success_next(manifest, plan, episode_id="episode-success")
    for attempt in manifest["attempts"]:
        if attempt["attempt_id"] == success["attempt_id"]:
            attempt["finished_at"] = "1970-01-01T00:00:01Z"
    while manifest["execution_status"] == "RUNNING":
        _fail_next(manifest, plan)

    assert manifest["execution_status"] == "FINISHED"
    assert manifest["quality_status"] == "FAIL"
    report = evaluate_self_healing_campaign(
        campaign,
        [{"segment": _segment(campaign, plan), "plan": plan, "manifest": manifest}],
        _policy(),
        live_status={
            "execution_status": "FINISHED",
            "heartbeat_unix": 1.0,
            "started_unix": 1.0,
        },
        free_disk_bytes=600 * 1024**3,
        now_unix=10_000.0,
    )

    assert report["decision"] == "CONTINUE"
    assert report["next_action"] == "FREEZE_REPLACEMENT_SEGMENT"
    assert report["pause_reasons"] == []
    assert report["liveness"]["heartbeat_age_seconds"] == 9999.0
    assert report["liveness"]["no_success_age_seconds"] == 9999.0
    assert len(report["replacement_requests"]) == 1


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


def test_campaign_reuses_parent_successes_and_completes_from_continuation(tmp_path):
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
    selection = build_campaign_export_selection(
        campaign,
        [
            {
                "segment": parent_segment,
                "plan": parent_plan,
                "manifest": parent_manifest,
                "episodes_root": str(tmp_path / "parent-episodes"),
            },
            {
                "segment": continuation_segment,
                "plan": continuation_plan,
                "manifest": continuation_manifest,
                "episodes_root": str(tmp_path / "continuation-episodes"),
            },
        ],
        dataset_id="farpoint-so101",
    )
    assert selection["schema_version"] == "farpoint.export-selection.v1"
    assert len(selection["episodes"]) == 2
    assert {row["segment_id"] for row in selection["episodes"]} == {
        "segment-000",
        "segment-001",
    }
    assert all(row["split"] == "train" for row in selection["episodes"])


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


def test_quality_exclusion_reopens_selected_quota_without_mutating_parent(tmp_path):
    campaign = _campaign(count=1)
    parent_plan = _plan(campaign, count=1)
    parent_manifest = create_manifest(
        parent_plan, collection_id="segment-000", git_commit="abcdef1"
    )
    parent_attempt = _success_next(
        parent_manifest, parent_plan, episode_id="episode-parent-quality-fail"
    )
    parent_manifest["execution_status"] = "ABORTED"
    parent_manifest["quality_status"] = "NOT_EVALUATED"
    parent_segment = _segment(campaign, parent_plan)
    evidence = [
        {
            "segment": parent_segment,
            "plan": parent_plan,
            "manifest": parent_manifest,
            "episodes_root": str(tmp_path / "parent"),
        }
    ]
    parent_hash = canonical_sha256(parent_manifest)
    exclusions = create_campaign_quality_exclusions(
        campaign,
        evidence,
        [
            {
                "segment_id": "segment-000",
                "attempt_id": parent_attempt["attempt_id"],
                "reason_code": "recovery_action_slew_gate_failed",
                "evidence_sha256": "b" * 64,
            }
        ],
        exclusion_id="recovery-v011-slew-exclusions",
    )

    requests = build_continuation_requests(
        campaign, evidence, quality_exclusions=exclusions
    )

    assert canonical_sha256(parent_manifest) == parent_hash
    assert len(requests) == 1
    assert requests[0]["request_kind"] == "carryover"
    assert requests[0]["source_variation_id"] == parent_attempt["variation_id"]
    assert requests[0]["prior_attempt_count"] == 1
    assert requests[0]["remaining_attempt_count"] == 2


def test_legacy_recovery_source_ordinals_normalize_to_campaign_local_quotas():
    campaign = _campaign(count=2, recovery=True)
    plan = _plan(campaign, count=2)
    plan["schema_version"] = "farpoint.so101-recovery-plan.v1"
    plan["trials"][0]["quota_ordinal"] = 2
    plan["trials"][1]["quota_ordinal"] = 7
    plan["plan_sha256"] = canonical_sha256(plan, omit=("plan_sha256",))
    manifest = create_manifest(
        plan, collection_id="legacy-recovery-segment", git_commit="abcdef1"
    )
    segment = _segment(campaign, plan)

    requests = build_continuation_requests(
        campaign,
        [{"segment": segment, "plan": plan, "manifest": manifest}],
    )

    assert [request["quota"]["quota_ordinal"] for request in requests] == [0, 1]
    assert {request["source_variation_id"] for request in requests} == {
        "variation-0-0",
        "variation-0-1",
    }


def test_legacy_recovery_quotas_are_shared_by_evaluator_and_exporter(tmp_path):
    campaign = _campaign(count=2, recovery=True)
    plan = _plan(campaign, count=2)
    plan["schema_version"] = "farpoint.so101-recovery-plan.v1"
    plan["trials"][0]["quota_ordinal"] = 2
    plan["trials"][1]["quota_ordinal"] = 7
    plan["plan_sha256"] = canonical_sha256(plan, omit=("plan_sha256",))
    manifest = create_manifest(
        plan, collection_id="legacy-recovery-segment", git_commit="abcdef1"
    )
    _success_next(manifest, plan, episode_id="episode-legacy-0")
    _success_next(manifest, plan, episode_id="episode-legacy-1")
    evidence = {
        "segment": _segment(campaign, plan),
        "plan": plan,
        "manifest": manifest,
        "episodes_root": str(tmp_path / "episodes"),
    }

    report = evaluate_self_healing_campaign(
        campaign,
        [evidence],
        _policy(),
        live_status={"heartbeat_unix": 1000.0, "started_unix": 900.0},
        free_disk_bytes=600 * 1024**3,
        now_unix=1001.0,
    )
    selection = build_campaign_export_selection(
        campaign, [evidence], dataset_id="farpoint-so101"
    )

    assert report["decision"] == "COMPLETE"
    assert report["errors"] == []
    assert [row["quota"]["quota_ordinal"] for row in selection["episodes"]] == [
        0,
        1,
    ]


def test_quality_excluded_parent_is_replaced_by_continuation_selection(tmp_path):
    campaign = _campaign(count=1)
    parent_plan = _plan(campaign, count=1)
    parent_manifest = create_manifest(
        parent_plan, collection_id="segment-000", git_commit="abcdef1"
    )
    parent_attempt = _success_next(
        parent_manifest, parent_plan, episode_id="episode-parent-quality-fail"
    )
    parent_manifest["execution_status"] = "ABORTED"
    parent_manifest["quality_status"] = "NOT_EVALUATED"
    parent_segment = _segment(campaign, parent_plan)
    parent_evidence = {
        "segment": parent_segment,
        "plan": parent_plan,
        "manifest": parent_manifest,
        "episodes_root": str(tmp_path / "parent"),
    }
    exclusions = create_campaign_quality_exclusions(
        campaign,
        [parent_evidence],
        [
            {
                "segment_id": "segment-000",
                "attempt_id": parent_attempt["attempt_id"],
                "reason_code": "recovery_action_slew_gate_failed",
                "evidence_sha256": "b" * 64,
            }
        ],
        exclusion_id="recovery-v011-slew-exclusions",
    )
    request = build_continuation_requests(
        campaign, [parent_evidence], quality_exclusions=exclusions
    )[0]
    continuation_plan = _plan(campaign, count=1)
    trial = continuation_plan["trials"][0]
    trial["seed"] = request["variation_seed"]
    trial["prior_attempt_count"] = request["prior_attempt_count"]
    trial["continuation_provenance"] = {
        "request_kind": request["request_kind"]
    }
    continuation_plan["collection"]["maximum_attempts"] = 2
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
        continuation_plan, collection_id="segment-001", git_commit="abcdef2"
    )
    continuation_attempt = _success_next(
        continuation_manifest,
        continuation_plan,
        episode_id="episode-continuation-quality-pass",
    )
    continuation_evidence = {
        "segment": continuation_segment,
        "plan": continuation_plan,
        "manifest": continuation_manifest,
        "episodes_root": str(tmp_path / "continuation"),
    }
    all_evidence = [parent_evidence, continuation_evidence]

    selection = build_campaign_export_selection(
        campaign,
        all_evidence,
        dataset_id="farpoint-so101",
        quality_exclusions=exclusions,
    )
    report = evaluate_self_healing_campaign(
        campaign,
        all_evidence,
        _policy(),
        live_status={"heartbeat_unix": 1000.0, "started_unix": 900.0},
        free_disk_bytes=600 * 1024**3,
        quality_exclusions=exclusions,
        now_unix=1001.0,
    )

    assert [row["attempt_id"] for row in selection["episodes"]] == [
        continuation_attempt["attempt_id"]
    ]
    assert selection["quality_exclusions_sha256"] == exclusions[
        "quality_exclusions_sha256"
    ]
    assert report["decision"] == "COMPLETE"
    assert report["progress"]["successful_quotas"] == 1
    assert report["progress"]["quality_excluded_successes"] == 1


def test_quality_exclusions_fail_closed_on_tampered_manifest_binding():
    campaign = _campaign(count=1)
    plan = _plan(campaign, count=1)
    manifest = create_manifest(plan, collection_id="segment-000", git_commit="abcdef1")
    attempt = _success_next(manifest, plan, episode_id="episode-success")
    segment = _segment(campaign, plan)
    evidence = [{"segment": segment, "plan": plan, "manifest": manifest}]
    exclusions = create_campaign_quality_exclusions(
        campaign,
        evidence,
        [
            {
                "segment_id": "segment-000",
                "attempt_id": attempt["attempt_id"],
                "reason_code": "recovery_action_slew_gate_failed",
                "evidence_sha256": "b" * 64,
            }
        ],
        exclusion_id="recovery-v011-slew-exclusions",
    )
    exclusions["entries"][0]["manifest_sha256"] = "c" * 64
    exclusions["quality_exclusions_sha256"] = canonical_sha256(
        exclusions, omit=("quality_exclusions_sha256",)
    )

    try:
        build_continuation_requests(
            campaign, evidence, quality_exclusions=exclusions
        )
    except ValueError as error:
        assert "manifest hash mismatch" in str(error)
    else:
        raise AssertionError("tampered quality exclusion was accepted")
