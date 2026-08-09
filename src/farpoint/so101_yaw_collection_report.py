"""Evidence report for formal SO-101 fixed-yaw, 30 mm collections."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from farpoint.so101_collection import validate_manifest
from farpoint.so101_episode_analysis import analyze_so101_episodes, classify_so101_failure
from farpoint.so101_gate_report import so101_episode_evidence_errors
from farpoint.so101_pilot_report import audit_yaw_mass_episodes
from farpoint.so101_yaw_collection import (
    COLLECTION_KIND,
    validate_yaw_collection_balance,
    yaw_collection_balance,
)


def build_so101_yaw_collection_report(
    plan: dict[str, Any], manifest: dict[str, Any], episodes_root: str | Path
) -> dict[str, Any]:
    validate_manifest(manifest, plan)
    profile = plan.get("collection") or {}
    if profile.get("kind") != COLLECTION_KIND:
        raise ValueError("plan is not a balanced yaw collection")
    attempts = manifest.get("attempts") or []
    selected_ids = set((manifest.get("selected_variations") or {}).values())
    selected = [row for row in attempts if row.get("attempt_id") in selected_ids]
    trials = {trial["variation_id"]: trial for trial in plan["trials"]}
    balance = yaw_collection_balance({"trials": [trials[row["variation_id"]] for row in selected]})
    errors = validate_yaw_collection_balance(
        balance, profile.get("balance_contract")
    )
    root = Path(episodes_root)
    episode_attempts = [row for row in attempts if row.get("episode_id")]
    episode_dirs = [root / row["episode_id"] for row in episode_attempts if (root / row["episode_id"]).is_dir()]
    analysis = analyze_so101_episodes(episode_dirs, verify_images=True)
    errors.extend(so101_episode_evidence_errors(analysis, len(episode_attempts)))
    by_name = {Path(item["episode_dir"]).name: item for item in analysis["episodes"]}
    for attempt in episode_attempts:
        episode = by_name.get(attempt["episode_id"])
        if episode is None:
            errors.append(f"missing_episode:{attempt['episode_id']}")
            continue
        if episode["success"] != bool(attempt["success"]):
            errors.append(f"{attempt['episode_id']}:manifest_episode_success_mismatch")
        if episode["dataset_valid"] != bool(attempt["dataset_valid"]):
            errors.append(f"{attempt['episode_id']}:manifest_episode_validity_mismatch")
    audits, audit_errors = audit_yaw_mass_episodes(plan, attempts, by_name, root, profile)
    errors.extend(audit_errors)
    for attempt in selected:
        episode = by_name.get(attempt.get("episode_id"))
        if episode is None:
            continue
        if episode.get("terminal_phase") != "retreat" or episode.get("terminal_grasp_phase") != "validated":
            errors.append(f"{attempt['episode_id']}:selected_episode_not_terminally_validated")
        proof = episode.get("proof_lift_tracking") or {}
        if float(proof.get("actual_max_m", 0.0)) < 0.005:
            errors.append(f"{attempt['episode_id']}:insufficient_proof_lift")
        settle = sum(row["frame_count"] for row in episode.get("phase_ranges", []) if row["phase"] == "settle")
        if settle < 15:
            errors.append(f"{attempt['episode_id']}:insufficient_settle_frames")
    complete = (
        manifest.get("execution_status") == "FINISHED"
        and manifest.get("quality_status") == "PASS"
        and len(selected) == int(profile["required_successes"])
        and len(attempts) <= int(profile["maximum_attempts"])
    )
    if not complete:
        errors.append("collection_not_complete_or_not_passed")
    status = "PASS" if not errors else ("INCOMPLETE" if manifest.get("execution_status") == "RUNNING" else "INVALID_EVIDENCE")
    failures = Counter(
        classify_so101_failure(row.get("failure_reason"), row.get("failure_category"))
        for row in attempts if not row.get("success")
    )
    return {
        "schema_version": "farpoint.so101-yaw-collection-report.v1",
        "collection_id": manifest["collection_id"],
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "git_commit": manifest["git_commit"],
        "status": status,
        "attempted_count": len(attempts),
        "maximum_attempts": int(profile["maximum_attempts"]),
        "yaw_degrees": float(profile["yaw_degrees"]),
        "success_count": len(selected),
        "required_successes": int(profile["required_successes"]),
        "complete_artifact_count": len(analysis["episodes"]),
        "yaw_mass_audit_count": len(audits),
        "yaw_mass_audits": audits,
        "balance": balance,
        "failure_class_counts": dict(sorted(failures.items())),
        "evidence_errors": sorted(set(errors)),
    }


def render_so101_yaw_collection_report_markdown(report: dict[str, Any]) -> str:
    balance = report["balance"]
    lines = [
        f"# SO-101 yaw={report['yaw_degrees']:g}°, 30 mm formal collection: "
        f"{report['collection_id']}",
        "",
        f"- Status: **{report['status']}**",
        f"- Git commit: `{report['git_commit']}`",
        f"- Successes: {report['success_count']}/{report['required_successes']}",
        f"- Attempts: {report['attempted_count']}/{report['maximum_attempts']}",
        f"- Complete artifacts: {report['complete_artifact_count']}/{report['attempted_count']}",
        f"- Yaw/mass audits: {report['yaw_mass_audit_count']}/{report['attempted_count']}", "",
        "## Frozen balance", "",
        f"- Splits: `{balance.get('splits')}`",
        f"- Masses: `{balance.get('masses_kg')}`",
        f"- Colors: `{balance.get('colors')}`",
        f"- Mass × color: `{balance.get('mass_color')}`",
        f"- Workspace cells: {len(balance.get('workspace_cells') or {})}/25", "",
        "## Evidence errors", "",
    ]
    lines.extend(["None."] if not report["evidence_errors"] else [f"- {error}" for error in report["evidence_errors"]])
    return "\n".join(lines) + "\n"
