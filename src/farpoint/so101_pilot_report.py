"""Auditable evidence report for the bounded SO-101 code-review pilot."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from farpoint.so101_collection import validate_manifest
from farpoint.so101_episode_analysis import analyze_so101_episodes, classify_so101_failure
from farpoint.so101_gate_report import so101_episode_evidence_errors


def build_so101_pilot_report(
    plan: dict[str, Any], manifest: dict[str, Any], episodes_root: str | Path
) -> dict[str, Any]:
    """Cross-check pilot completion, selection, physics, and front-only data."""
    validate_manifest(manifest, plan)
    attempts = manifest.get("attempts") or []
    root = Path(episodes_root)
    episode_attempts = [attempt for attempt in attempts if attempt.get("episode_id")]
    episode_dirs = [
        root / attempt["episode_id"]
        for attempt in episode_attempts
        if (root / attempt["episode_id"]).is_dir()
    ]
    analysis = analyze_so101_episodes(episode_dirs, verify_images=True)
    errors = so101_episode_evidence_errors(analysis, len(episode_attempts))
    for attempt in episode_attempts:
        if not (root / attempt["episode_id"]).is_dir():
            errors.append(f"missing_episode:{attempt['episode_id']}")
    by_name = {Path(item["episode_dir"]).name: item for item in analysis["episodes"]}
    for attempt in episode_attempts:
        episode = by_name.get(attempt["episode_id"])
        if episode is None:
            continue
        if episode["success"] != bool(attempt["success"]):
            errors.append(f"{attempt['episode_id']}:manifest_episode_success_mismatch")
        if episode["dataset_valid"] != bool(attempt["dataset_valid"]):
            errors.append(f"{attempt['episode_id']}:manifest_episode_validity_mismatch")
    selected = [attempt for attempt in attempts if attempt.get("selected_for_dataset")]
    selected_evidence = []
    for attempt in selected:
        episode = by_name.get(attempt.get("episode_id"))
        if episode is None:
            continue
        selected_evidence.append(episode)
        if not episode["success"] or not episode["dataset_valid"]:
            errors.append(f"{attempt['episode_id']}:selected_episode_not_eligible")
        if episode["terminal_phase"] != "retreat":
            errors.append(f"{attempt['episode_id']}:selected_episode_not_retreat")
        if episode["terminal_grasp_phase"] != "validated":
            errors.append(f"{attempt['episode_id']}:selected_grasp_not_validated")
        settle_frames = sum(
            phase["frame_count"]
            for phase in episode["phase_ranges"]
            if phase["phase"] == "settle"
        )
        if settle_frames < 15:
            errors.append(f"{attempt['episode_id']}:insufficient_settle_frames")
        proof = episode.get("proof_lift_tracking") or {}
        if float(proof.get("actual_max_m", 0.0)) < 0.005:
            errors.append(f"{attempt['episode_id']}:insufficient_proof_lift")

    attempt_seed_count = len({attempt["attempt_seed"] for attempt in attempts})
    attempted_ids = {attempt["variation_id"] for attempt in attempts}
    variation_seed_count = len(
        {trial["seed"] for trial in plan["trials"] if trial["variation_id"] in attempted_ids}
    )
    if attempt_seed_count != len(attempts):
        errors.append("attempt_seeds_not_unique")
    if variation_seed_count != len(attempts):
        errors.append("variation_seeds_not_unique")
    if len(selected) != int(manifest["required_successes"]):
        errors.append("selected_success_count_mismatch")
    completed = (
        manifest.get("execution_status") == "FINISHED"
        and manifest.get("quality_status") == "PASS"
        and len(selected) == int(manifest["required_successes"])
        and len(attempts) <= int(manifest["maximum_attempts"])
    )
    status = "INVALID_EVIDENCE" if errors else "PASS" if completed else "INCOMPLETE"
    failures = Counter(
        classify_so101_failure(attempt.get("failure_reason"), attempt.get("failure_category"))
        for attempt in attempts
        if not attempt.get("success")
    )
    return {
        "schema_version": "farpoint.so101-pilot-report.v1",
        "pilot_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "collection_id": manifest["collection_id"],
        "git_commit": manifest["git_commit"],
        "pilot_status": status,
        "attempted_count": len(attempts),
        "maximum_attempts": int(manifest["maximum_attempts"]),
        "success_count": len(selected),
        "required_successes": int(manifest["required_successes"]),
        "attempt_seed_count": attempt_seed_count,
        "variation_seed_count": variation_seed_count,
        "independent_episode_identity_count": len(
            {episode["metadata_sha256"] for episode in analysis["episodes"]}
        ),
        "failure_class_counts": dict(sorted(failures.items())),
        "evidence_errors": sorted(set(errors)),
        "minimum_selected_proof_lift_m": min(
            episode["proof_lift_tracking"]["actual_max_m"]
            for episode in selected_evidence
        )
        if selected_evidence
        else None,
        "minimum_selected_settle_frames": min(
            sum(
                phase["frame_count"]
                for phase in episode["phase_ranges"]
                if phase["phase"] == "settle"
            )
            for episode in selected_evidence
        )
        if selected_evidence
        else None,
        "episode_evidence": analysis,
    }


def render_so101_pilot_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# SO-101 pilot report: {report['pilot_id']}",
        "",
        f"- Pilot status: **{report['pilot_status']}**",
        f"- Git commit: `{report['git_commit']}`",
        f"- Attempts: {report['attempted_count']}/{report['maximum_attempts']}",
        f"- Eligible successes: {report['success_count']}/{report['required_successes']}",
        f"- Minimum proof lift: {report['minimum_selected_proof_lift_m']}",
        f"- Minimum settle frames: {report['minimum_selected_settle_frames']}",
        "",
        "## Evidence audit",
        "",
    ]
    if report["evidence_errors"]:
        lines.extend(f"- {error}" for error in report["evidence_errors"])
    else:
        lines.append("Selected physics and front-only episode artifacts passed.")
    return "\n".join(lines) + "\n"
