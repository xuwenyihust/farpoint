"""Auditable evidence reports for SO-101 repeatability gates."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from farpoint.so101_collection import validate_manifest
from farpoint.so101_episode_analysis import (
    analyze_so101_episodes,
    classify_so101_failure,
)


def _evidence_errors(analysis: dict[str, Any], expected_episode_count: int) -> list[str]:
    errors: list[str] = []
    if analysis["episode_count"] != expected_episode_count:
        errors.append("episode_artifact_count_mismatch")
    if analysis["duplicate_observation_groups"]:
        errors.append("duplicate_observation_artifacts")
    for episode in analysis["episodes"]:
        name = Path(episode["episode_dir"]).name
        if episode["camera_frame_counts"] != {
            "front": episode["observation_count"]
        }:
            errors.append(f"{name}:not_front_only_complete")
        if episode["state_dimensions"] != [6]:
            errors.append(f"{name}:invalid_state_dimensions")
        if episode["action_dimensions"] != [6]:
            errors.append(f"{name}:invalid_action_dimensions")
        if not episode["timestamps_strictly_increasing"]:
            errors.append(f"{name}:invalid_timestamps")
        integrity = episode.get("camera_frame_integrity") or {}
        front = integrity.get("front") or {}
        if (
            front.get("referenced_frames") != episode["observation_count"]
            or front.get("existing_frames") != episode["observation_count"]
            or front.get("decodable_frames") != episode["observation_count"]
            or front.get("resolutions") != [[640, 480]]
            or front.get("modes") != ["RGB"]
            or front.get("unsafe_paths")
        ):
            errors.append(f"{name}:invalid_front_frame_artifacts")
    return errors


def build_so101_gate_report(
    plan: dict[str, Any],
    manifest: dict[str, Any],
    episodes_root: str | Path,
) -> dict[str, Any]:
    """Cross-check a gate manifest against raw, front-only episode evidence."""
    validate_manifest(manifest, plan)
    attempts = manifest.get("attempts") or []
    root = Path(episodes_root)
    episode_attempts = [row for row in attempts if row.get("episode_id")]
    missing_episode_ids = [
        row["episode_id"]
        for row in episode_attempts
        if not (root / row["episode_id"]).is_dir()
    ]
    existing_episode_dirs = [
        root / row["episode_id"]
        for row in episode_attempts
        if (root / row["episode_id"]).is_dir()
    ]
    analysis = (
        analyze_so101_episodes(existing_episode_dirs, verify_images=True)
        if existing_episode_dirs
        else {
            "episode_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "independent_observation_artifact_count": 0,
            "duplicate_observation_groups": [],
            "failure_reason_counts": {},
            "failure_class_counts": {},
            "episodes": [],
        }
    )
    evidence_errors = _evidence_errors(analysis, len(episode_attempts))
    evidence_errors.extend(f"missing_episode:{value}" for value in missing_episode_ids)

    episode_by_name = {
        Path(row["episode_dir"]).name: row for row in analysis["episodes"]
    }
    for attempt in episode_attempts:
        episode = episode_by_name.get(attempt["episode_id"])
        if episode is None:
            continue
        if episode["success"] != bool(attempt["success"]):
            evidence_errors.append(
                f"{attempt['episode_id']}:manifest_episode_success_mismatch"
            )
        if episode["dataset_valid"] != bool(attempt["dataset_valid"]):
            evidence_errors.append(
                f"{attempt['episode_id']}:manifest_episode_validity_mismatch"
            )

    success_count = sum(
        bool(row["success"] and row["dataset_valid"]) for row in attempts
    )
    attempted_count = len(attempts)
    maximum_attempts = int(manifest["maximum_attempts"])
    required_successes = int(manifest["required_successes"])
    execution_complete = (
        manifest["execution_status"] == "FINISHED"
        and attempted_count == maximum_attempts
    )
    threshold_met = success_count >= required_successes
    if evidence_errors:
        gate_status = "INVALID_EVIDENCE"
    elif not execution_complete:
        gate_status = "INCOMPLETE"
    elif threshold_met:
        gate_status = "PASS"
    else:
        gate_status = "FAIL"

    failure_reasons = Counter(
        row.get("failure_reason") or "unspecified_failure"
        for row in attempts
        if not row.get("success")
    )
    failure_classes = Counter(
        classify_so101_failure(
            row.get("failure_reason"), row.get("failure_category")
        )
        for row in attempts
        if not row.get("success")
    )
    gate = plan.get("gate") or {}
    return {
        "schema_version": "farpoint.so101-gate-report.v1",
        "gate_id": plan["plan_id"],
        "gate_kind": gate.get("kind"),
        "plan_sha256": plan["plan_sha256"],
        "collection_id": manifest["collection_id"],
        "git_commit": manifest["git_commit"],
        "gate_status": gate_status,
        "manifest_quality_status": manifest["quality_status"],
        "attempted_count": attempted_count,
        "maximum_attempts": maximum_attempts,
        "success_count": success_count,
        "required_successes": required_successes,
        "success_rate": success_count / maximum_attempts if maximum_attempts else 0.0,
        "attempt_seed_count": len({row["attempt_seed"] for row in attempts}),
        "variation_seed_count": len(
            {
                trial["seed"]
                for trial in plan["trials"]
                if trial["variation_id"]
                in {row["variation_id"] for row in attempts}
            }
        ),
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "failure_class_counts": dict(sorted(failure_classes.items())),
        "evidence_errors": sorted(set(evidence_errors)),
        "episode_evidence": analysis,
    }


def render_so101_gate_report_markdown(report: dict[str, Any]) -> str:
    """Render a concise human-reviewable gate result."""
    lines = [
        f"# SO-101 gate report: {report['gate_id']}",
        "",
        f"- Gate status: **{report['gate_status']}**",
        f"- Git commit: `{report['git_commit']}`",
        f"- Attempts: {report['attempted_count']}/{report['maximum_attempts']}",
        f"- Eligible successes: {report['success_count']}/{report['required_successes']}",
        f"- Success rate: {report['success_rate']:.1%}",
        f"- Independent episode artifacts: {report['episode_evidence']['independent_observation_artifact_count']}",
        "",
        "## Failure classes",
        "",
    ]
    if report["failure_class_counts"]:
        lines.extend(
            f"- {name}: {count}"
            for name, count in report["failure_class_counts"].items()
        )
    else:
        lines.append("No failed attempts recorded.")
    lines.extend(["", "## Evidence audit", ""])
    if report["evidence_errors"]:
        lines.extend(f"- {error}" for error in report["evidence_errors"])
    else:
        lines.append("Front-only episode artifacts, dimensions, timestamps, and hashes passed.")
    return "\n".join(lines) + "\n"
