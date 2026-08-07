"""Evidence report for a formal mirrored SO-101 mass collection."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from farpoint.so101_collection import validate_manifest
from farpoint.so101_episode_analysis import classify_so101_failure
from farpoint.so101_mass_collection import (
    COLLECTION_KIND,
    mirrored_balance,
    validate_mirrored_balance,
)
from farpoint.so101_mass_feasibility import audit_resolved_mass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_so101_mass_collection_report(
    plan: dict[str, Any], manifest: dict[str, Any], episodes_root: str | Path
) -> dict[str, Any]:
    validate_manifest(manifest, plan)
    profile = plan.get("collection") or {}
    if profile.get("kind") != COLLECTION_KIND:
        raise ValueError("plan is not a mirrored mass collection")
    trials = {trial["variation_id"]: trial for trial in plan["trials"]}
    attempts = manifest.get("attempts") or []
    selected_ids = set((manifest.get("selected_variations") or {}).values())
    selected_attempts = [row for row in attempts if row["attempt_id"] in selected_ids]
    selected_trials = [trials[row["variation_id"]] for row in selected_attempts]
    selected_plan = {"trials": selected_trials}
    balance = mirrored_balance(selected_plan)
    evidence_errors = validate_mirrored_balance(balance)
    root = Path(episodes_root)
    target_mass = float(profile["target_mass_kg"])
    tolerance = float(profile["actual_mass_tolerance_kg"])
    mass_audits = []
    artifact_count = 0
    for attempt in attempts:
        episode_id = attempt.get("episode_id")
        if not episode_id:
            evidence_errors.append(f"{attempt['attempt_id']}:missing_episode_id")
            continue
        episode = root / episode_id
        required = ("metadata.json", "metrics.json", "observations.jsonl", "run-state.json")
        missing = [name for name in required if not (episode / name).is_file()]
        if missing:
            evidence_errors.append(f"{episode_id}:missing_artifacts:{','.join(missing)}")
            continue
        rows = [line for line in (episode / "observations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            evidence_errors.append(f"{episode_id}:empty_observations")
            continue
        artifact_count += 1
        metadata = _read_json(episode / "metadata.json")
        metrics = _read_json(episode / "metrics.json")
        run_state = _read_json(episode / "run-state.json")
        if run_state.get("execution_status") != "FINISHED":
            evidence_errors.append(f"{episode_id}:run_state_not_finished")
        if bool(metrics.get("success")) != bool(attempt.get("success")):
            evidence_errors.append(f"{episode_id}:manifest_metrics_success_mismatch")
        metric_audit = (metrics.get("physics_audit") or {}).get("mass")
        metadata_audit = ((metadata.get("scene") or {}).get("object") or {}).get("mass_audit")
        if not isinstance(metric_audit, dict) or not isinstance(metadata_audit, dict):
            evidence_errors.append(f"{episode_id}:missing_mass_audit")
            continue
        keys = {
            "requested_mass_kg",
            "resolved_mass_kg",
            "physx_actual_mass_kg",
            "tolerance_kg",
            "verified",
        }
        if any(metric_audit.get(key) != metadata_audit.get(key) for key in keys):
            evidence_errors.append(f"{episode_id}:mass_audit_sidecar_mismatch")
        recomputed = audit_resolved_mass(
            requested_mass_kg=float(metric_audit.get("requested_mass_kg", math.inf)),
            resolved_mass_kg=float(metric_audit.get("resolved_mass_kg", math.inf)),
            physx_actual_mass_kg=float(metric_audit.get("physx_actual_mass_kg", math.inf)),
            tolerance_kg=tolerance,
        )
        mass_audits.append({"episode_id": episode_id, **recomputed})
        if (
            not recomputed["verified"]
            or not bool(metric_audit.get("verified"))
            or abs(float(metric_audit.get("requested_mass_kg", math.inf)) - target_mass) > tolerance
            or abs(float(metric_audit.get("resolved_mass_kg", math.inf)) - target_mass) > tolerance
            or float(metric_audit.get("tolerance_kg", math.inf)) != tolerance
        ):
            evidence_errors.append(f"{episode_id}:mass_audit_failed")
    execution_complete = (
        manifest.get("execution_status") == "FINISHED"
        and manifest.get("quality_status") == "PASS"
        and len(selected_attempts) == int(profile["required_successes"])
        and len(attempts) <= int(profile["maximum_attempts"])
    )
    if not execution_complete:
        evidence_errors.append("collection_not_complete_or_not_passed")
    status = "PASS" if not evidence_errors else (
        "INCOMPLETE" if manifest.get("execution_status") == "RUNNING" else "INVALID_EVIDENCE"
    )
    failure_counts = Counter(
        classify_so101_failure(row.get("failure_reason"), row.get("failure_category"))
        for row in attempts
        if not row.get("success")
    )
    return {
        "schema_version": "farpoint.so101-mass-collection-report.v1",
        "collection_id": manifest["collection_id"],
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "git_commit": manifest["git_commit"],
        "status": status,
        "target_mass_kg": target_mass,
        "attempted_count": len(attempts),
        "maximum_attempts": int(profile["maximum_attempts"]),
        "success_count": len(selected_attempts),
        "required_successes": int(profile["required_successes"]),
        "complete_artifact_count": artifact_count,
        "mass_audit_count": len(mass_audits),
        "mass_audits": mass_audits,
        "balance": balance,
        "failure_class_counts": dict(sorted(failure_counts.items())),
        "evidence_errors": sorted(set(evidence_errors)),
    }


def render_so101_mass_collection_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# SO-101 0.03 kg formal collection: {report['collection_id']}",
        "",
        f"- Status: **{report['status']}**",
        f"- Git commit: `{report['git_commit']}`",
        f"- Successes: {report['success_count']}/{report['required_successes']}",
        f"- Attempts: {report['attempted_count']}/{report['maximum_attempts']}",
        f"- Complete artifacts: {report['complete_artifact_count']}/{report['attempted_count']}",
        f"- PhysX mass audits: {report['mass_audit_count']}/{report['attempted_count']}",
        "",
        "## Mirrored balance",
        "",
        f"- Splits: `{report['balance'].get('splits')}`",
        f"- Sizes: `{report['balance'].get('sizes')}`",
        f"- Colors: `{report['balance'].get('colors')}`",
        f"- Workspace cells: {len(report['balance'].get('workspace_cells') or {})}/25",
        "",
        "## Evidence errors",
        "",
    ]
    lines.extend(
        ["None."] if not report["evidence_errors"] else [f"- {error}" for error in report["evidence_errors"]]
    )
    return "\n".join(lines) + "\n"
