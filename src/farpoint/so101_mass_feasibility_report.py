"""Evidence and paired-behavior reports for SO-101 cube-mass feasibility."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from farpoint.so101_collection import validate_manifest
from farpoint.so101_episode_analysis import (
    analyze_so101_episodes,
    classify_so101_failure,
)
from farpoint.so101_gate_report import so101_episode_evidence_errors
from farpoint.so101_mass_feasibility import audit_resolved_mass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _relative_delta(baseline: float, candidate: float) -> float:
    return (candidate - baseline) / max(abs(baseline), 1e-12)


def _behavior_features(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    actions = [row["action_joint_positions"] for row in rows]
    action_path = sum(
        math.sqrt(
            sum((float(right) - float(left)) ** 2 for left, right in zip(a, b))
        )
        for a, b in zip(actions, actions[1:])
    )
    lift_forces = []
    for row in rows:
        if row.get("phase") != "lift":
            continue
        forces = row.get("contact_forces_newtons") or {}
        lift_forces.append(
            min(
                float(forces.get("left_finger", 0.0)),
                float(forces.get("right_finger", 0.0)),
            )
        )
    return {
        "action_path_length_rad": action_path,
        "frame_count": len(rows),
        "mean_lift_bilateral_force_n": (
            statistics.fmean(lift_forces) if lift_forces else 0.0
        ),
    }


def build_so101_mass_feasibility_report(
    plan: dict[str, Any],
    manifest: dict[str, Any],
    episodes_root: str | Path,
) -> dict[str, Any]:
    """Verify actual masses, per-mass yield, and matched-pair behavior."""
    validate_manifest(manifest, plan)
    gate = plan.get("gate") or {}
    if gate.get("kind") != "cube_mass_feasibility":
        raise ValueError("plan is not a cube-mass feasibility profile")
    trials = {trial["variation_id"]: trial for trial in plan["trials"]}
    attempts = manifest.get("attempts") or []
    root = Path(episodes_root)
    episode_attempts = [row for row in attempts if row.get("episode_id")]
    missing = [
        row["episode_id"]
        for row in episode_attempts
        if not (root / row["episode_id"]).is_dir()
    ]
    existing = [
        root / row["episode_id"]
        for row in episode_attempts
        if all(
            (root / row["episode_id"] / name).is_file()
            for name in ("metadata.json", "metrics.json", "observations.jsonl")
        )
    ]
    incomplete = [
        row["episode_id"]
        for row in episode_attempts
        if (root / row["episode_id"]).is_dir()
        and (root / row["episode_id"]) not in existing
    ]
    analysis = (
        analyze_so101_episodes(existing, verify_images=True)
        if existing
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
    evidence_errors = so101_episode_evidence_errors(
        analysis, len(episode_attempts), allow_duplicate_observations=True
    )
    evidence_errors.extend(f"missing_episode:{episode_id}" for episode_id in missing)
    evidence_errors.extend(
        f"incomplete_episode:{episode_id}" for episode_id in incomplete
    )
    analyzed = {Path(row["episode_dir"]).name: row for row in analysis["episodes"]}

    success_by_role: Counter[str] = Counter()
    attempts_by_role: Counter[str] = Counter()
    pair_records: dict[str, dict[str, Any]] = defaultdict(dict)
    episode_features: dict[str, dict[str, Any]] = {}
    mass_audits: list[dict[str, Any]] = []
    for attempt in attempts:
        trial = trials[attempt["variation_id"]]
        role = trial["mass_role"]
        attempts_by_role[role] += 1
        if attempt.get("success") and attempt.get("dataset_valid"):
            success_by_role[role] += 1
        pair_records[trial["mass_pair_id"]][role] = {
            "variation_id": trial["variation_id"],
            "environment_seed": trial["environment_seed"],
            "success": bool(attempt.get("success") and attempt.get("dataset_valid")),
            "episode_id": attempt.get("episode_id"),
        }
        episode_id = attempt.get("episode_id")
        if not episode_id or episode_id not in analyzed:
            continue
        episode = analyzed[episode_id]
        if episode["success"] != bool(attempt.get("success")):
            evidence_errors.append(f"{episode_id}:manifest_episode_success_mismatch")
        episode_root = root / episode_id
        metadata = _read_json(episode_root / "metadata.json")
        metrics = _read_json(episode_root / "metrics.json")
        metric_audit = metrics.get("physics_audit", {}).get("mass")
        metadata_audit = (
            metadata.get("scene", {}).get("object", {}).get("mass_audit")
        )
        if not isinstance(metric_audit, dict) or not isinstance(metadata_audit, dict):
            evidence_errors.append(f"{episode_id}:missing_mass_audit")
            continue
        audit_keys = {
            "requested_mass_kg",
            "resolved_mass_kg",
            "physx_actual_mass_kg",
            "tolerance_kg",
            "verified",
        }
        if any(metric_audit.get(key) != metadata_audit.get(key) for key in audit_keys):
            evidence_errors.append(f"{episode_id}:mass_audit_sidecar_mismatch")
        recorded = metric_audit
        recomputed = audit_resolved_mass(
            requested_mass_kg=float(trial["requested"]["mass_kg"]),
            resolved_mass_kg=float(trial["resolved"]["mass_kg"]),
            physx_actual_mass_kg=float(recorded["physx_actual_mass_kg"]),
            tolerance_kg=float(gate["actual_mass_tolerance_kg"]),
        )
        audit_row = {"episode_id": episode_id, "mass_role": role, **recomputed}
        mass_audits.append(audit_row)
        recorded_matches_plan = (
            abs(
                float(recorded.get("requested_mass_kg", math.inf))
                - float(trial["requested"]["mass_kg"])
            )
            <= float(gate["actual_mass_tolerance_kg"])
            and abs(
                float(recorded.get("resolved_mass_kg", math.inf))
                - float(trial["resolved"]["mass_kg"])
            )
            <= float(gate["actual_mass_tolerance_kg"])
            and float(recorded.get("tolerance_kg", math.inf))
            == float(gate["actual_mass_tolerance_kg"])
        )
        if (
            not recomputed["verified"]
            or not bool(recorded.get("verified"))
            or not recorded_matches_plan
        ):
            evidence_errors.append(f"{episode_id}:mass_audit_failed")
        rows = _read_rows(episode_root / "observations.jsonl")
        features = _behavior_features(rows)
        episode_features[episode_id] = features
        pair_records[trial["mass_pair_id"]][role]["features"] = features

    paired_deltas = []
    for pair_id, pair in sorted(pair_records.items()):
        if set(pair) != {"baseline", "candidate"}:
            evidence_errors.append(f"{pair_id}:incomplete_mass_pair")
            continue
        if pair["baseline"]["environment_seed"] != pair["candidate"]["environment_seed"]:
            evidence_errors.append(f"{pair_id}:environment_seed_mismatch")
        if not (
            pair["baseline"]["success"]
            and pair["candidate"]["success"]
            and "features" in pair["baseline"]
            and "features" in pair["candidate"]
        ):
            continue
        baseline = pair["baseline"]["features"]
        candidate = pair["candidate"]["features"]
        paired_deltas.append(
            {
                "pair_id": pair_id,
                "action_path_relative": _relative_delta(
                    baseline["action_path_length_rad"],
                    candidate["action_path_length_rad"],
                ),
                "mean_lift_bilateral_force_relative": _relative_delta(
                    baseline["mean_lift_bilateral_force_n"],
                    candidate["mean_lift_bilateral_force_n"],
                ),
                "frame_count_absolute": int(candidate["frame_count"])
                - int(baseline["frame_count"]),
            }
        )

    thresholds = gate["behavior_change_thresholds"]
    minimum_pairs = int(gate["minimum_successful_pairs_for_behavior"])
    median_deltas = {}
    if paired_deltas:
        for key in thresholds:
            median_deltas[key] = statistics.median(
                abs(float(row[key])) for row in paired_deltas
            )
    behavior_signal_detected = len(paired_deltas) >= minimum_pairs and any(
        median_deltas.get(key, 0.0) >= float(threshold)
        for key, threshold in thresholds.items()
    )
    execution_complete = (
        manifest.get("execution_status") == "FINISHED"
        and len(attempts) == int(gate["maximum_attempts"])
    )
    minimum_successes = int(gate["minimum_successes_per_mass"])
    per_mass_threshold_met = all(
        success_by_role[role] >= minimum_successes
        for role in ("baseline", "candidate")
    )
    if evidence_errors:
        feasibility_status = "INVALID_EVIDENCE"
    elif not execution_complete:
        feasibility_status = "INCOMPLETE"
    elif not per_mass_threshold_met:
        feasibility_status = "FAIL"
    else:
        feasibility_status = "PASS"
    if feasibility_status != "PASS":
        recommendation = "DO_NOT_EXPAND_FEASIBILITY_FAILED"
    elif len(paired_deltas) < minimum_pairs:
        recommendation = "INCONCLUSIVE_INSUFFICIENT_SUCCESSFUL_PAIRS"
    elif behavior_signal_detected:
        recommendation = "EXPAND_PHYSICS_ROBUSTNESS_PILOT"
    else:
        recommendation = "DO_NOT_EXPAND_NO_MEASURABLE_BEHAVIOR_SIGNAL"
    failures = Counter(
        classify_so101_failure(row.get("failure_reason"), row.get("failure_category"))
        for row in attempts
        if not row.get("success")
    )
    return {
        "schema_version": "farpoint.so101-mass-feasibility-report.v1",
        "profile_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "collection_id": manifest["collection_id"],
        "git_commit": manifest["git_commit"],
        "feasibility_status": feasibility_status,
        "recommendation": recommendation,
        "attempted_count": len(attempts),
        "maximum_attempts": int(gate["maximum_attempts"]),
        "success_by_role": dict(success_by_role),
        "attempts_by_role": dict(attempts_by_role),
        "minimum_successes_per_mass": minimum_successes,
        "mass_kg_by_role": {
            "baseline": float(gate["baseline_mass_kg"]),
            "candidate": float(gate["candidate_mass_kg"]),
        },
        "mass_audits": mass_audits,
        "successful_pair_count": len(paired_deltas),
        "minimum_successful_pairs_for_behavior": minimum_pairs,
        "paired_behavior_deltas": paired_deltas,
        "median_absolute_behavior_deltas": median_deltas,
        "behavior_change_thresholds": thresholds,
        "behavior_signal_detected": behavior_signal_detected,
        "episode_behavior_features": episode_features,
        "failure_class_counts": dict(sorted(failures.items())),
        "evidence_errors": sorted(set(evidence_errors)),
        "episode_evidence": analysis,
    }


def render_so101_mass_feasibility_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# SO-101 cube-mass feasibility: {report['profile_id']}",
        "",
        f"- Feasibility: **{report['feasibility_status']}**",
        f"- Recommendation: **{report['recommendation']}**",
        f"- Git commit: `{report['git_commit']}`",
        f"- Attempts: {report['attempted_count']}/{report['maximum_attempts']}",
        f"- Successful matched pairs: {report['successful_pair_count']}",
        "",
        "| Role | Mass (kg) | Eligible successes | Attempts |",
        "|---|---:|---:|---:|",
    ]
    for role in ("baseline", "candidate"):
        lines.append(
            f"| {role} | {report['mass_kg_by_role'][role]:.3f} | "
            f"{report['success_by_role'].get(role, 0)} | "
            f"{report['attempts_by_role'].get(role, 0)} |"
        )
    lines.extend(["", "## PhysX mass audit", ""])
    if report["evidence_errors"]:
        lines.extend(f"- {error}" for error in report["evidence_errors"])
    else:
        lines.append("All recorded requested/resolved/PhysX masses agree within tolerance.")
    lines.extend(["", "## Paired behavior signal", ""])
    if report["median_absolute_behavior_deltas"]:
        for name, value in report["median_absolute_behavior_deltas"].items():
            lines.append(
                f"- {name}: median absolute delta `{value:.6g}` "
                f"(threshold `{report['behavior_change_thresholds'][name]}`)"
            )
    else:
        lines.append("No complete successful pair was available for comparison.")
    return "\n".join(lines) + "\n"
