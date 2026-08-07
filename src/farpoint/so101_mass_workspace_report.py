"""Candidate-only workspace evidence for an SO-101 cube mass."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from farpoint.so101_gate_report import build_so101_gate_report
from farpoint.so101_mass_feasibility import audit_resolved_mass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_so101_mass_workspace_report(
    plan: dict[str, Any],
    manifest: dict[str, Any],
    episodes_root: str | Path,
) -> dict[str, Any]:
    """Verify candidate success coverage and actual PhysX mass."""
    gate = plan.get("gate") or {}
    if gate.get("kind") != "cube_mass_workspace_pilot":
        raise ValueError("plan is not a cube mass workspace pilot")
    base = build_so101_gate_report(plan, manifest, episodes_root)
    root = Path(episodes_root)
    trials = {trial["variation_id"]: trial for trial in plan["trials"]}
    evidence_errors = list(base["evidence_errors"])
    mass_audits = []
    successful_positions = []
    tolerance = float(gate["actual_mass_tolerance_kg"])
    candidate_mass = float(gate["candidate_mass_kg"])
    for attempt in manifest.get("attempts") or []:
        trial = trials[attempt["variation_id"]]
        episode_id = attempt.get("episode_id")
        if attempt.get("success") and attempt.get("dataset_valid"):
            successful_positions.append(trial["resolved"]["position_m"][:2])
        if not episode_id:
            continue
        episode_root = root / episode_id
        metadata_path = episode_root / "metadata.json"
        metrics_path = episode_root / "metrics.json"
        if not metadata_path.is_file() or not metrics_path.is_file():
            continue
        metadata = _read_json(metadata_path)
        metrics = _read_json(metrics_path)
        metadata_audit = (
            metadata.get("scene", {}).get("object", {}).get("mass_audit")
        )
        metric_audit = metrics.get("physics_audit", {}).get("mass")
        if not isinstance(metadata_audit, dict) or not isinstance(metric_audit, dict):
            evidence_errors.append(f"{episode_id}:missing_mass_audit")
            continue
        keys = {
            "requested_mass_kg",
            "resolved_mass_kg",
            "physx_actual_mass_kg",
            "tolerance_kg",
            "verified",
        }
        if any(metadata_audit.get(key) != metric_audit.get(key) for key in keys):
            evidence_errors.append(f"{episode_id}:mass_audit_sidecar_mismatch")
        recomputed = audit_resolved_mass(
            requested_mass_kg=float(trial["requested"]["mass_kg"]),
            resolved_mass_kg=float(trial["resolved"]["mass_kg"]),
            physx_actual_mass_kg=float(metric_audit["physx_actual_mass_kg"]),
            tolerance_kg=tolerance,
        )
        mass_audits.append({"episode_id": episode_id, **recomputed})
        recorded_matches_plan = (
            abs(float(metric_audit.get("requested_mass_kg", math.inf)) - candidate_mass)
            <= tolerance
            and abs(
                float(metric_audit.get("resolved_mass_kg", math.inf))
                - candidate_mass
            )
            <= tolerance
            and float(metric_audit.get("tolerance_kg", math.inf)) == tolerance
        )
        if (
            not recomputed["verified"]
            or not bool(metric_audit.get("verified"))
            or not recorded_matches_plan
        ):
            evidence_errors.append(f"{episode_id}:mass_audit_failed")

    if evidence_errors:
        pilot_status = "INVALID_EVIDENCE"
    elif base["gate_status"] == "PASS":
        pilot_status = "PASS"
    else:
        pilot_status = base["gate_status"]
    recommendation = (
        "EXPAND_MASS_AXIS"
        if pilot_status == "PASS"
        else "DO_NOT_EXPAND_MASS_AXIS"
    )
    return {
        "schema_version": "farpoint.so101-mass-workspace-report.v1",
        "pilot_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "collection_id": manifest["collection_id"],
        "git_commit": manifest["git_commit"],
        "pilot_status": pilot_status,
        "recommendation": recommendation,
        "candidate_mass_kg": candidate_mass,
        "attempted_count": base["attempted_count"],
        "maximum_attempts": base["maximum_attempts"],
        "success_count": base["success_count"],
        "required_successes": base["required_successes"],
        "successful_positions_xy_m": successful_positions,
        "mass_audits": mass_audits,
        "historical_baseline": gate["historical_baseline"],
        "historical_comparison_is_contemporaneous": False,
        "failure_class_counts": base["failure_class_counts"],
        "evidence_errors": sorted(set(evidence_errors)),
        "episode_evidence": base["episode_evidence"],
    }


def render_so101_mass_workspace_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# SO-101 mass workspace pilot: {report['pilot_id']}",
        "",
        f"- Pilot status: **{report['pilot_status']}**",
        f"- Recommendation: **{report['recommendation']}**",
        f"- Candidate mass: {report['candidate_mass_kg']:.3f} kg",
        f"- Successes: {report['success_count']}/{report['attempted_count']} "
        f"(required {report['required_successes']})",
        f"- Git commit: `{report['git_commit']}`",
        "",
        "## Historical reference policy",
        "",
        "The v0.0.0 episodes prove that these positions were solvable at 0.04 kg. "
        "They are not a contemporaneous control and are not used for a causal "
        "trajectory comparison.",
        "",
        "## PhysX mass audit",
        "",
    ]
    if report["evidence_errors"]:
        lines.extend(f"- {error}" for error in report["evidence_errors"])
    else:
        lines.append("Every candidate episode passed requested/resolved/actual mass audit.")
    return "\n".join(lines) + "\n"
