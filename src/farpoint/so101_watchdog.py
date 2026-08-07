"""Deterministic stop decisions for bounded SO-101 collections."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from farpoint.so101_collection import validate_manifest
from farpoint.so101_episode_analysis import classify_so101_failure


WATCHDOG_SCHEMA_VERSION = "farpoint.so101-watchdog-report.v1"
WATCHDOG_POLICY_SCHEMA_VERSION = "farpoint.so101-watchdog-policy.v1"
DECISIONS = {"CONTINUE", "STOP", "COMPLETE", "INVALID"}


def _sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _integer_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def validate_watchdog_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != WATCHDOG_POLICY_SCHEMA_VERSION:
        raise ValueError(
            f"watchdog policy must use {WATCHDOG_POLICY_SCHEMA_VERSION}"
        )
    positive_integer_fields = (
        "recent_window_attempts",
        "minimum_recent_attempts",
        "consecutive_failure_limit",
        "stale_run_state_seconds",
        "stale_manifest_seconds",
    )
    for field in positive_integer_fields:
        value = policy.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if policy["minimum_recent_attempts"] > policy["recent_window_attempts"]:
        raise ValueError(
            "minimum_recent_attempts cannot exceed recent_window_attempts"
        )
    fraction = policy.get("recent_failure_fraction")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
        raise ValueError("recent_failure_fraction must be numeric")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("recent_failure_fraction must be in (0, 1]")
    classes = policy.get("structural_failure_classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("structural_failure_classes must be a non-empty list")
    if any(not isinstance(value, str) or not value.strip() for value in classes):
        raise ValueError("structural_failure_classes must contain non-empty strings")
    if len(classes) != len(set(classes)):
        raise ValueError("structural_failure_classes must be unique")


def load_watchdog_policy(path: str | Path) -> dict[str, Any]:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_watchdog_policy(policy)
    return policy


def _live_run_states(
    episodes_root: Path | None,
    *,
    collection_id: str,
    now: datetime,
) -> tuple[list[dict[str, Any]], list[str]]:
    if episodes_root is None or not episodes_root.exists():
        return [], []
    live: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted(episodes_root.glob("*/run-state.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid_run_state:{path}:{type(error).__name__}")
            continue
        if state.get("execution_status") != "RUNNING":
            continue
        provenance = state.get("provenance") or {}
        if provenance.get("collection_id") != collection_id:
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        live.append(
            {
                "episode_id": (state.get("identity") or {}).get("episode_id")
                or path.parent.name,
                "path": str(path),
                "age_seconds": max(0.0, (now - modified).total_seconds()),
                "frame_count": _integer_or(
                    (state.get("recording") or {}).get("frame_count"), 0
                ),
            }
        )
    return live, errors


def evaluate_so101_collection(
    plan: dict[str, Any],
    manifest: dict[str, Any],
    policy: dict[str, Any],
    *,
    episodes_root: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen collection without mutating it or its artifacts."""
    generated_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    reasons: list[str] = []
    errors: list[str] = []
    try:
        validate_watchdog_policy(policy)
        validate_manifest(manifest, plan)
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"invalid_input:{type(error).__name__}:{error}")

    attempts_value = manifest.get("attempts") or []
    attempts = attempts_value if isinstance(attempts_value, list) else []
    selected_value = manifest.get("selected_variations") or {}
    selected_count = len(selected_value) if isinstance(selected_value, dict) else 0
    required_successes = _integer_or(manifest.get("required_successes"), 0)
    maximum_attempts = _integer_or(manifest.get("maximum_attempts"), 0)
    remaining_attempts = max(0, maximum_attempts - len(attempts))
    maximum_possible_successes = selected_count + remaining_attempts
    recent_window_size = max(1, _integer_or(policy.get("recent_window_attempts"), 1))
    recent_attempts = attempts[-recent_window_size:]
    recent_failures = [row for row in recent_attempts if not row.get("success")]
    recent_failure_classes = [
        classify_so101_failure(
            row.get("failure_reason"), row.get("failure_category")
        )
        for row in recent_failures
    ]
    recent_counts = Counter(recent_failure_classes)
    structural = set(policy.get("structural_failure_classes") or [])

    consecutive_class = None
    consecutive_count = 0
    for row in reversed(attempts):
        if row.get("success"):
            break
        failure_class = classify_so101_failure(
            row.get("failure_reason"), row.get("failure_category")
        )
        if consecutive_class is None:
            consecutive_class = failure_class
        if failure_class != consecutive_class:
            break
        consecutive_count += 1

    root = Path(episodes_root) if episodes_root is not None else None
    live, live_errors = _live_run_states(
        root,
        collection_id=str(manifest.get("collection_id") or ""),
        now=generated_at,
    )
    errors.extend(live_errors)
    if len(live) > 1:
        errors.append("multiple_live_attempts_for_collection")

    manifest_age_seconds = None
    try:
        manifest_age_seconds = max(
            0.0,
            (generated_at - _parse_time(str(manifest["updated_at"]))).total_seconds(),
        )
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"invalid_manifest_updated_at:{type(error).__name__}")

    status = manifest.get("execution_status")
    quality = manifest.get("quality_status")
    if errors:
        decision = "INVALID"
    elif status == "FINISHED":
        if quality == "PASS":
            decision = "COMPLETE"
            reasons.append("collection_quality_pass")
        else:
            decision = "STOP"
            reasons.append("collection_finished_without_quality_pass")
    elif status == "ABORTED":
        decision = "STOP"
        reasons.append(f"collection_aborted:{manifest.get('abort_reason') or 'unknown'}")
    elif status != "RUNNING":
        decision = "INVALID"
        errors.append(f"unsupported_execution_status:{status}")
    else:
        if maximum_possible_successes < required_successes:
            reasons.append("success_target_unreachable")
        if (
            consecutive_class in structural
            and consecutive_count >= int(policy["consecutive_failure_limit"])
        ):
            reasons.append(
                f"consecutive_structural_failure:{consecutive_class}:{consecutive_count}"
            )
        minimum_recent = int(policy["minimum_recent_attempts"])
        if len(recent_attempts) >= minimum_recent and recent_attempts:
            for failure_class, count in sorted(recent_counts.items()):
                fraction = count / len(recent_attempts)
                if (
                    failure_class in structural
                    and fraction >= float(policy["recent_failure_fraction"])
                ):
                    reasons.append(
                        f"recent_structural_failure:{failure_class}:{count}/{len(recent_attempts)}"
                    )
        if live and live[0]["age_seconds"] > int(policy["stale_run_state_seconds"]):
            reasons.append(
                f"stale_live_attempt:{live[0]['episode_id']}:{live[0]['age_seconds']:.0f}s"
            )
        if (
            not live
            and manifest_age_seconds is not None
            and manifest_age_seconds > int(policy["stale_manifest_seconds"])
        ):
            reasons.append(f"stale_collection:{manifest_age_seconds:.0f}s")
        decision = "STOP" if reasons else "CONTINUE"

    report = {
        "schema_version": WATCHDOG_SCHEMA_VERSION,
        "decision": decision,
        "generated_at": generated_at.isoformat(),
        "collection_id": manifest.get("collection_id"),
        "plan_id": plan.get("plan_id"),
        "plan_sha256": plan.get("plan_sha256"),
        "git_commit": manifest.get("git_commit"),
        "policy_sha256": _sha256(policy),
        "reasons": reasons,
        "errors": errors,
        "progress": {
            "attempted_count": len(attempts),
            "maximum_attempts": maximum_attempts,
            "remaining_attempts": remaining_attempts,
            "selected_successes": selected_count,
            "required_successes": required_successes,
            "maximum_possible_successes": maximum_possible_successes,
        },
        "recent_window": {
            "attempt_count": len(recent_attempts),
            "failure_count": len(recent_failures),
            "failure_class_counts": dict(sorted(recent_counts.items())),
            "consecutive_failure_class": consecutive_class,
            "consecutive_failure_count": consecutive_count,
        },
        "liveness": {
            "manifest_age_seconds": manifest_age_seconds,
            "live_attempts": live,
        },
    }
    if report["decision"] not in DECISIONS:
        raise AssertionError("invalid watchdog decision")
    return report
