"""Generic self-healing campaign evaluation and immutable replacement requests."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from farpoint.campaign import (
    canonical_sha256,
    create_segment,
    validate_campaign_semantics,
    validate_segment_semantics,
    variation_seed,
)
from farpoint.so101_collection import validate_manifest
from farpoint.so101_episode_analysis import classify_so101_failure


POLICY_SCHEMA = "farpoint.self-healing-policy.v1"
REPORT_SCHEMA = "farpoint.self-healing-campaign-report.v1"
DECISIONS = {"CONTINUE", "PAUSE", "COMPLETE", "INVALID"}
QUOTA_FIELDS = (
    "object_variant_id",
    "yaw_stratum_id",
    "region_band",
    "split",
    "quota_ordinal",
)


def validate_self_healing_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError(f"self-healing policy must use {POLICY_SCHEMA}")
    positive_integers = (
        "maximum_attempts_per_variation",
        "distinct_structural_failure_limit",
        "recent_window_attempts",
        "no_success_timeout_seconds",
        "heartbeat_timeout_seconds",
        "minimum_free_disk_bytes",
        "diagnostic_representatives_per_class",
    )
    for field in positive_integers:
        value = policy.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if policy["maximum_attempts_per_variation"] != 3:
        raise ValueError("self-healing policy must allow exactly three attempts")
    rate = policy.get("minimum_recent_success_rate")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool):
        raise ValueError("minimum_recent_success_rate must be numeric")
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError("minimum_recent_success_rate must be in [0, 1]")
    classes = policy.get("structural_failure_classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("structural_failure_classes must be a non-empty list")
    if any(not isinstance(value, str) or not value for value in classes):
        raise ValueError("structural_failure_classes must contain non-empty strings")
    if len(classes) != len(set(classes)):
        raise ValueError("structural_failure_classes must be unique")


def _quota_identity(trial: dict[str, Any]) -> tuple[Any, ...]:
    missing = [field for field in QUOTA_FIELDS if field not in trial]
    if missing:
        raise ValueError(f"trial is missing quota identity fields: {missing}")
    return tuple(trial[field] for field in QUOTA_FIELDS)


def _campaign_quota_identities(campaign: dict[str, Any]) -> set[tuple[Any, ...]]:
    identities = set()
    expected_count = 0
    for row in campaign.get("quotas") or []:
        count = int(row["count"])
        expected_count += count
        for quota_ordinal in range(count):
            identity = (
                row["object_variant_id"],
                row["yaw_stratum_id"],
                row["region_band"],
                row["split"],
                quota_ordinal,
            )
            if identity in identities:
                raise ValueError("campaign quota rows overlap")
            identities.add(identity)
    if len(identities) != expected_count:
        raise ValueError("campaign quota identities are inconsistent")
    return identities


def _trial_index(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {trial["variation_id"]: trial for trial in plan.get("trials") or []}
    if len(rows) != len(plan.get("trials") or []):
        raise ValueError("plan variation ids must be unique")
    return rows


def _parse_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _failure_class(attempt: dict[str, Any]) -> str:
    return classify_so101_failure(
        attempt.get("failure_reason"), attempt.get("failure_category")
    )


def diagnostic_clusters(
    attempts: Iterable[dict[str, Any]], *, representatives_per_class: int
) -> dict[str, list[dict[str, Any]]]:
    """Select stable, distinct variation representatives for each failure class."""
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for attempt in attempts:
        if attempt.get("success"):
            continue
        failure_class = _failure_class(attempt)
        variation_id = str(attempt.get("variation_id") or "")
        if not variation_id:
            continue
        grouped.setdefault(failure_class, {}).setdefault(variation_id, attempt)
    result = {}
    for failure_class, rows in sorted(grouped.items()):
        ordered = sorted(
            rows.values(),
            key=lambda row: (
                int(row.get("variation_seed", row.get("attempt_seed", 0))),
                str(row.get("variation_id")),
            ),
        )[:representatives_per_class]
        result[failure_class] = [
            {
                "variation_id": row.get("variation_id"),
                "variation_seed": row.get("variation_seed"),
                "attempt_seed": row.get("attempt_seed"),
                "attempt_id": row.get("attempt_id"),
                "episode_id": row.get("episode_id"),
                "failure_reason": row.get("failure_reason"),
            }
            for row in ordered
        ]
    return result


def build_replacement_requests(
    campaign: dict[str, Any], plan: dict[str, Any], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return deterministic same-quota seeds only for three-attempt failures."""
    if validate_campaign_semantics(campaign):
        raise ValueError("campaign contract is invalid")
    validate_manifest(manifest, plan)
    maximum = int(campaign["attempt_policy"]["maximum_attempts_per_variation"])
    if maximum != 3:
        raise ValueError("replacement generation requires a three-attempt campaign")
    trials = _trial_index(plan)
    allowed_quotas = _campaign_quota_identities(campaign)
    counts = Counter(row["variation_id"] for row in manifest.get("attempts") or [])
    requests = []
    for variation_id, count in sorted(counts.items()):
        if variation_id in (manifest.get("selected_variations") or {}) or count < maximum:
            continue
        trial = trials[variation_id]
        quota_identity = _quota_identity(trial)
        if quota_identity not in allowed_quotas:
            raise ValueError(f"deferred variation is outside campaign quotas: {variation_id}")
        quota = dict(zip(QUOTA_FIELDS, quota_identity))
        replacement_index = int(trial.get("replacement_index", 0)) + 1
        seed = variation_seed(
            campaign["campaign_sha256"],
            object_variant_id=str(quota["object_variant_id"]),
            yaw_stratum_id=str(quota["yaw_stratum_id"]),
            region_band=str(quota["region_band"]),
            split=str(quota["split"]),
            quota_ordinal=int(quota["quota_ordinal"]),
            replacement_index=replacement_index,
        )
        requests.append(
            {
                "deferred_variation_id": variation_id,
                "quota": quota,
                "replacement_index": replacement_index,
                "variation_seed": seed,
            }
        )
    return requests


def validate_replacement_plan(
    requests: Iterable[dict[str, Any]], continuation_plan: dict[str, Any]
) -> None:
    """Require a continuation plan to realize every request exactly once."""
    expected = {
        (
            tuple(request["quota"][field] for field in QUOTA_FIELDS),
            int(request["replacement_index"]),
            int(request["variation_seed"]),
        )
        for request in requests
    }
    observed = {
        (
            _quota_identity(trial),
            int(trial.get("replacement_index", 0)),
            int(trial["seed"]),
        )
        for trial in continuation_plan.get("trials") or []
    }
    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        raise ValueError(
            "continuation plan does not exactly realize replacement requests: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )


def create_continuation_segment(
    campaign: dict[str, Any],
    parent_segment: dict[str, Any],
    parent_manifest: dict[str, Any],
    *,
    segment_id: str,
    git_commit: str,
    plan_sha256: str,
    oracle_profile_allowlist: list[str],
) -> dict[str, Any]:
    """Freeze a continuation bound to the exact terminal parent manifest."""
    campaign_errors = validate_campaign_semantics(campaign)
    segment_errors = validate_segment_semantics(parent_segment)
    if campaign_errors or segment_errors:
        raise ValueError("campaign or parent segment is invalid")
    if parent_segment["campaign_sha256"] != campaign["campaign_sha256"]:
        raise ValueError("parent segment does not belong to campaign")
    if parent_manifest.get("execution_status") not in {"FINISHED", "ABORTED", "PAUSED"}:
        raise ValueError("continuation parent manifest must be terminal or paused")
    return create_segment(
        {
            "campaign_id": campaign["campaign_id"],
            "campaign_sha256": campaign["campaign_sha256"],
            "segment_id": segment_id,
            "segment_index": int(parent_segment["segment_index"]) + 1,
            "git_commit": git_commit,
            "plan_sha256": plan_sha256,
            "parent_manifest_sha256": canonical_sha256(parent_manifest),
            "oracle_profile_allowlist": list(oracle_profile_allowlist),
            "execution_status": "RUNNING",
            "quality_status": "NOT_EVALUATED",
            "attempts": [],
        }
    )


def evaluate_self_healing_campaign(
    campaign: dict[str, Any],
    segment_evidence: Iterable[dict[str, Any]],
    policy: dict[str, Any],
    *,
    live_status: dict[str, Any],
    free_disk_bytes: int,
    integrity_errors: Iterable[str] = (),
    now_unix: float | None = None,
) -> dict[str, Any]:
    """Evaluate aggregate quota progress without mutating immutable segments."""
    now = datetime.now(timezone.utc).timestamp() if now_unix is None else float(now_unix)
    errors = list(integrity_errors)
    reasons: list[str] = []
    try:
        validate_self_healing_policy(policy)
        campaign_errors = validate_campaign_semantics(campaign)
        if campaign_errors:
            raise ValueError("; ".join(campaign_errors))
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"invalid_campaign_or_policy:{error}")

    evidence = sorted(
        (deepcopy(row) for row in segment_evidence),
        key=lambda row: int((row.get("segment") or {}).get("segment_index", -1)),
    )
    all_attempts: list[dict[str, Any]] = []
    successful_quotas: dict[tuple[Any, ...], dict[str, Any]] = {}
    previous_manifest = None
    latest_plan = None
    latest_manifest = None
    try:
        expected_quotas = _campaign_quota_identities(campaign)
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"invalid_campaign_quotas:{error}")
        expected_quotas = set()
    for index, row in enumerate(evidence):
        segment = row.get("segment") or {}
        plan = row.get("plan") or {}
        manifest = row.get("manifest") or {}
        segment_errors = validate_segment_semantics(segment)
        if segment_errors:
            errors.extend(f"segment[{index}]:{error}" for error in segment_errors)
            continue
        if segment.get("campaign_sha256") != campaign.get("campaign_sha256"):
            errors.append(f"segment[{index}]:campaign_hash_mismatch")
        if int(segment.get("segment_index", -1)) != index:
            errors.append(f"segment[{index}]:non_contiguous_index")
        if index > 0 and segment.get("parent_manifest_sha256") != canonical_sha256(
            previous_manifest
        ):
            errors.append(f"segment[{index}]:parent_manifest_hash_mismatch")
        try:
            validate_manifest(manifest, plan)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"segment[{index}]:invalid_manifest:{error}")
            continue
        trials = _trial_index(plan)
        segment_quotas = []
        for trial in trials.values():
            try:
                quota = _quota_identity(trial)
            except ValueError as error:
                errors.append(f"segment[{index}]:{error}")
                continue
            if quota not in expected_quotas:
                errors.append(f"segment[{index}]:trial_outside_campaign_quota:{quota}")
            segment_quotas.append(quota)
        if len(segment_quotas) != len(set(segment_quotas)):
            errors.append(f"segment[{index}]:duplicate_quota_trials")
        for attempt in manifest.get("attempts") or []:
            enriched = deepcopy(attempt)
            trial = trials.get(attempt.get("variation_id")) or {}
            enriched.setdefault("variation_seed", trial.get("seed"))
            enriched["segment_id"] = segment.get("segment_id")
            all_attempts.append(enriched)
        for variation_id, attempt_id in (manifest.get("selected_variations") or {}).items():
            trial = trials[variation_id]
            quota = _quota_identity(trial)
            if quota not in expected_quotas:
                errors.append(f"segment[{index}]:success_outside_campaign_quota:{quota}")
                continue
            selected = {
                "quota": dict(zip(QUOTA_FIELDS, quota)),
                "variation_id": variation_id,
                "attempt_id": attempt_id,
                "segment_id": segment["segment_id"],
            }
            if quota in successful_quotas:
                errors.append(f"duplicate_success_for_quota:{quota}")
            else:
                successful_quotas[quota] = selected
        previous_manifest = manifest
        latest_plan = plan
        latest_manifest = manifest

    structural = set(policy.get("structural_failure_classes") or [])
    consecutive_class = None
    consecutive_variations: set[str] = set()
    for attempt in reversed(all_attempts):
        if attempt.get("success"):
            break
        failure_class = _failure_class(attempt)
        if consecutive_class is None:
            consecutive_class = failure_class
        if failure_class != consecutive_class:
            break
        consecutive_variations.add(str(attempt.get("variation_id")))
    if (
        consecutive_class in structural
        and len(consecutive_variations)
        >= int(policy.get("distinct_structural_failure_limit", 1))
    ):
        reasons.append(
            "distinct_structural_failures:"
            f"{consecutive_class}:{len(consecutive_variations)}"
        )

    recent_size = int(policy.get("recent_window_attempts", 1))
    recent = all_attempts[-recent_size:]
    recent_success_rate = (
        sum(bool(row.get("success")) for row in recent) / len(recent) if recent else None
    )
    if (
        len(recent) >= recent_size
        and recent_success_rate is not None
        and recent_success_rate < float(policy.get("minimum_recent_success_rate", 0.0))
    ):
        reasons.append(f"recent_success_rate:{recent_success_rate:.3f}")

    successful_times = [
        _parse_timestamp(row.get("finished_at"))
        for row in all_attempts
        if row.get("success")
    ]
    successful_times = [value for value in successful_times if value is not None]
    reference = max(successful_times) if successful_times else _parse_timestamp(
        live_status.get("started_unix") or live_status.get("started_at")
    )
    no_success_age = None if reference is None else max(0.0, now - reference)
    if no_success_age is not None and no_success_age > int(
        policy.get("no_success_timeout_seconds", 1)
    ):
        reasons.append(f"no_new_success:{no_success_age:.0f}s")

    heartbeat = _parse_timestamp(live_status.get("heartbeat_unix"))
    heartbeat_age = None if heartbeat is None else max(0.0, now - heartbeat)
    if heartbeat_age is None or heartbeat_age > int(
        policy.get("heartbeat_timeout_seconds", 1)
    ):
        reasons.append(
            "heartbeat_missing_or_stale"
            if heartbeat_age is None
            else f"heartbeat_stale:{heartbeat_age:.0f}s"
        )
    if int(free_disk_bytes) < int(policy.get("minimum_free_disk_bytes", 1)):
        reasons.append(f"disk_below_minimum:{int(free_disk_bytes)}")

    target = int((campaign.get("target") or {}).get("successful_episodes", 0))
    complete = set(successful_quotas) == expected_quotas and len(expected_quotas) == target
    replacements = []
    if latest_plan is not None and latest_manifest is not None:
        try:
            replacements = build_replacement_requests(
                campaign, latest_plan, latest_manifest
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"replacement_generation:{error}")

    if errors:
        decision = "INVALID"
        next_action = "NONE"
    elif complete:
        decision = "COMPLETE"
        next_action = "NONE"
    elif reasons:
        decision = "PAUSE"
        next_action = (
            "ORACLE_REPAIR"
            if any(
                reason.startswith("distinct_structural_failures")
                or reason.startswith("recent_success_rate")
                for reason in reasons
            )
            else "INSPECT"
        )
    else:
        decision = "CONTINUE"
        next_action = "FREEZE_REPLACEMENT_SEGMENT" if replacements else "COLLECT"

    report = {
        "schema_version": REPORT_SCHEMA,
        "campaign_id": campaign.get("campaign_id"),
        "campaign_sha256": campaign.get("campaign_sha256"),
        "decision": decision,
        "next_action": next_action,
        "errors": errors,
        "pause_reasons": reasons,
        "progress": {
            "successful_quotas": len(successful_quotas),
            "target_successful_episodes": target,
            "attempted_count": len(all_attempts),
            "segment_count": len(evidence),
        },
        "recent_window": {
            "attempt_count": len(recent),
            "success_rate": recent_success_rate,
            "consecutive_failure_class": consecutive_class,
            "consecutive_distinct_variations": len(consecutive_variations),
        },
        "liveness": {
            "heartbeat_age_seconds": heartbeat_age,
            "no_success_age_seconds": no_success_age,
            "free_disk_bytes": int(free_disk_bytes),
        },
        "replacement_requests": replacements,
        "diagnostic_clusters": diagnostic_clusters(
            all_attempts,
            representatives_per_class=int(
                policy.get("diagnostic_representatives_per_class", 3)
            ),
        ),
    }
    if decision not in DECISIONS:
        raise AssertionError("invalid self-healing decision")
    return report


def build_campaign_export_selection(
    campaign: dict[str, Any],
    segment_evidence: Iterable[dict[str, Any]],
    *,
    dataset_id: str,
) -> dict[str, Any]:
    """Compose one successful episode per exact quota across immutable segments."""
    campaign_errors = validate_campaign_semantics(campaign)
    if campaign_errors:
        raise ValueError("invalid campaign: " + "; ".join(campaign_errors))
    if not dataset_id:
        raise ValueError("dataset_id must be non-empty")
    expected_quotas = _campaign_quota_identities(campaign)
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    evidence = sorted(
        (deepcopy(row) for row in segment_evidence),
        key=lambda row: int((row.get("segment") or {}).get("segment_index", -1)),
    )
    previous_manifest = None
    for index, row in enumerate(evidence):
        segment = row.get("segment") or {}
        plan = row.get("plan") or {}
        manifest = row.get("manifest") or {}
        segment_errors = validate_segment_semantics(segment)
        if segment_errors:
            raise ValueError(f"invalid segment {index}: {'; '.join(segment_errors)}")
        if int(segment["segment_index"]) != index:
            raise ValueError("campaign segment indexes must be contiguous")
        if index > 0 and segment.get("parent_manifest_sha256") != canonical_sha256(
            previous_manifest
        ):
            raise ValueError("continuation parent manifest hash mismatch")
        validate_manifest(manifest, plan)
        trials = _trial_index(plan)
        attempts = {
            attempt["attempt_id"]: attempt for attempt in manifest.get("attempts") or []
        }
        episodes_root = str(row.get("episodes_root") or "")
        if not episodes_root:
            raise ValueError(f"segment {index} must define episodes_root")
        for variation_id, attempt_id in (manifest.get("selected_variations") or {}).items():
            trial = trials[variation_id]
            quota = _quota_identity(trial)
            if quota not in expected_quotas:
                raise ValueError(f"selected episode is outside campaign quota: {quota}")
            if quota in selected:
                raise ValueError(f"multiple selected episodes for campaign quota: {quota}")
            attempt = attempts[attempt_id]
            selected[quota] = {
                "episode_dir": str(Path(episodes_root) / attempt["episode_id"]),
                "trial_id": trial["trial_id"],
                "variation_id": variation_id,
                "split": trial["split"],
                "segment_id": segment["segment_id"],
                "segment_index": segment["segment_index"],
                "git_commit": segment["git_commit"],
                "attempt_id": attempt_id,
                "quota": dict(zip(QUOTA_FIELDS, quota)),
            }
        previous_manifest = manifest
    if set(selected) != expected_quotas:
        missing = expected_quotas - set(selected)
        raise ValueError(f"campaign selection is incomplete: missing {len(missing)} quotas")
    episodes = [selected[quota] for quota in sorted(selected)]
    split_counts = Counter(row["split"] for row in episodes)
    if dict(split_counts) != campaign["target"]["splits"]:
        raise ValueError("campaign selection split counts do not match target")
    return {
        "schema_version": "farpoint.export-selection.v1",
        "dataset_id": dataset_id,
        "collection_id": campaign["campaign_id"],
        "selection_policy": "one_success_per_exact_campaign_quota_across_segments",
        "episodes": episodes,
    }
