"""Auditable evidence report for the bounded SO-101 code-review pilot."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from farpoint.so101_collection import validate_manifest
from farpoint.so101_episode_analysis import analyze_so101_episodes, classify_so101_failure
from farpoint.so101_gate_report import so101_episode_evidence_errors


def _pilot_status(
    execution_status: str,
    quality_status: str,
    selected_count: int,
    required_successes: int,
    evidence_errors: list[str],
    acceptance_errors: list[str],
) -> str:
    """Keep evidence integrity separate from an ordinary pilot gate failure."""
    if evidence_errors:
        return "INVALID_EVIDENCE"
    if execution_status != "FINISHED":
        return "INCOMPLETE"
    if acceptance_errors:
        return "FAIL"
    if quality_status == "PASS" and selected_count >= required_successes:
        return "PASS"
    return "FAIL"


def _expectation_errors(
    expectations: dict[str, dict[str, Any]], attempts: list[dict[str, Any]]
) -> list[str]:
    by_trial_id = {attempt.get("trial_id"): attempt for attempt in attempts}
    errors = []
    for trial_id, expectation in expectations.items():
        attempt = by_trial_id.get(trial_id)
        if attempt is None:
            errors.append(f"{trial_id}:missing_expected_attempt")
            continue
        expected_success = bool(expectation["success"])
        if bool(attempt.get("success")) != expected_success:
            errors.append(f"{trial_id}:expected_success_{str(expected_success).lower()}")
            continue
        expected_reason = expectation.get("failure_reason")
        if not expected_success and attempt.get("failure_reason") != expected_reason:
            errors.append(f"{trial_id}:unexpected_failure_reason")
    return errors


def _required_success_cell_errors(
    plan: dict[str, Any], selected: list[dict[str, Any]]
) -> list[str]:
    trials = {trial["variation_id"]: trial for trial in plan.get("trials") or []}
    successful_cells = {
        trials[attempt["variation_id"]]["cell_id"]
        for attempt in selected
        if attempt.get("variation_id") in trials
    }
    return [
        f"required_success_cell_failed:{cell_id}"
        for cell_id in (plan.get("pilot") or {}).get("required_success_cells") or []
        if cell_id not in successful_cells
    ]


def _quaternion_error_degrees(actual: list[float], expected: list[float]) -> float:
    if len(actual) != 4 or len(expected) != 4:
        return math.inf
    actual_norm = math.sqrt(sum(float(value) ** 2 for value in actual))
    expected_norm = math.sqrt(sum(float(value) ** 2 for value in expected))
    if actual_norm <= 0.0 or expected_norm <= 0.0:
        return math.inf
    dot = sum(float(a) * float(b) for a, b in zip(actual, expected))
    normalized_dot = min(1.0, max(-1.0, dot / (actual_norm * expected_norm)))
    return math.degrees(2.0 * math.acos(abs(normalized_dot)))


def _quaternions_equivalent(
    actual: list[float], expected: list[float], *, tolerance_degrees: float
) -> bool:
    return _quaternion_error_degrees(actual, expected) <= tolerance_degrees


def audit_yaw_mass_episodes(
    plan: dict[str, Any],
    attempts: list[dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
    episodes_root: Path,
    profile: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    profile = profile or plan.get("pilot") or {}
    if "yaw_degrees" not in profile:
        return [], []
    by_trial_id = {trial["trial_id"]: trial for trial in plan["trials"]}
    audits = []
    errors = []
    for attempt in attempts:
        episode_id = attempt.get("episode_id")
        episode = by_name.get(episode_id)
        trial = by_trial_id.get(attempt.get("trial_id"))
        if episode is None or trial is None or not episode_id:
            continue
        root = episodes_root / episode_id
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        expected_orientation = [float(value) for value in trial["resolved"]["orientation_xyzw"]]
        orientation_tolerance_degrees = float(
            profile.get("actual_orientation_tolerance_degrees", 2.0)
        )
        initial_orientation = [float(value) for value in episode["initial_object_pose_xyzw"][3:]]
        initial_orientation_error_degrees = _quaternion_error_degrees(
            initial_orientation, expected_orientation
        )
        variation = metadata.get("variation") or {}
        recorded_orientations = []
        for role in ("requested", "resolved"):
            payload = variation.get(role) or {}
            recorded_orientations.append(payload.get("orientation_xyzw") or [])
            recorded_orientations.append(
                (((payload.get("entities") or {}).get("pick_object") or {}).get("pose") or {}).get(
                    "orientation_xyzw"
                )
                or []
            )
        scene_orientation = (
            ((metadata.get("scene") or {}).get("object") or {}).get("initial_pose") or {}
        ).get("orientation_xyzw") or []
        orientation_verified = _quaternions_equivalent(
            initial_orientation,
            expected_orientation,
            tolerance_degrees=orientation_tolerance_degrees,
        ) and all(
            _quaternions_equivalent(
                value,
                expected_orientation,
                tolerance_degrees=orientation_tolerance_degrees,
            )
            for value in [*recorded_orientations, scene_orientation]
        )
        if not orientation_verified:
            errors.append(f"{episode_id}:yaw_orientation_audit_failed")

        metric_mass = (metrics.get("physics_audit") or {}).get("mass")
        metadata_mass = ((metadata.get("scene") or {}).get("object") or {}).get("mass_audit")
        expected_mass = float(trial["resolved"]["mass_kg"])
        mass_verified = (
            isinstance(metric_mass, dict)
            and metric_mass == metadata_mass
            and metric_mass.get("verified") is True
            and abs(float(metric_mass.get("requested_mass_kg", math.inf)) - expected_mass) <= 1e-6
            and abs(float(metric_mass.get("resolved_mass_kg", math.inf)) - expected_mass) <= 1e-6
            and abs(float(metric_mass.get("physx_actual_mass_kg", math.inf)) - expected_mass)
            <= 1e-6
        )
        if not mass_verified:
            errors.append(f"{episode_id}:mass_audit_failed")
        audits.append(
            {
                "episode_id": episode_id,
                "trial_id": attempt["trial_id"],
                "expected_yaw_degrees": float(profile["yaw_degrees"]),
                "expected_orientation_xyzw": expected_orientation,
                "initial_orientation_xyzw": initial_orientation,
                "initial_orientation_error_degrees": initial_orientation_error_degrees,
                "orientation_tolerance_degrees": orientation_tolerance_degrees,
                "orientation_verified": orientation_verified,
                "expected_mass_kg": expected_mass,
                "physx_actual_mass_kg": (
                    metric_mass.get("physx_actual_mass_kg")
                    if isinstance(metric_mass, dict)
                    else None
                ),
                "mass_verified": mass_verified,
            }
        )
    if len(audits) != len([attempt for attempt in attempts if attempt.get("episode_id")]):
        errors.append("yaw_audit_count_mismatch")
    return audits, errors


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
    yaw_audits, yaw_errors = audit_yaw_mass_episodes(plan, attempts, by_name, root)
    errors.extend(yaw_errors)
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
            phase["frame_count"] for phase in episode["phase_ranges"] if phase["phase"] == "settle"
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
    acceptance_errors = []
    if (plan.get("pilot") or {}).get("kind") == "targeted_yaw_pilot":
        if len(selected) < int(manifest["required_successes"]):
            acceptance_errors.append("selected_success_count_below_threshold")
        acceptance_errors.extend(_required_success_cell_errors(plan, selected))
    elif len(selected) != int(manifest["required_successes"]):
        acceptance_errors.append("selected_success_count_mismatch")
    if len(attempts) > int(manifest["maximum_attempts"]):
        acceptance_errors.append("attempt_budget_exceeded")
    acceptance_errors.extend(
        _expectation_errors((plan.get("pilot") or {}).get("expectations") or {}, attempts)
    )
    status = _pilot_status(
        str(manifest.get("execution_status")),
        str(manifest.get("quality_status")),
        len(selected),
        int(manifest["required_successes"]),
        errors,
        acceptance_errors,
    )
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
        "yaw_audit_count": len(yaw_audits),
        "yaw_audits": yaw_audits,
        "evidence_errors": sorted(set(errors)),
        "acceptance_errors": sorted(set(acceptance_errors)),
        "minimum_selected_proof_lift_m": min(
            episode["proof_lift_tracking"]["actual_max_m"] for episode in selected_evidence
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
    if report.get("acceptance_errors"):
        lines.extend(
            ["", "## Acceptance gate", ""] + [f"- {error}" for error in report["acceptance_errors"]]
        )
    return "\n".join(lines) + "\n"
