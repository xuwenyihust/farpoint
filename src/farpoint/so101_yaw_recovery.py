"""Recovery and completion evidence for fixed-yaw SO-101 collections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from farpoint.so101_collection_recovery import (
    build_completion_artifacts,
    build_missing_variation_recovery_plan,
    selected_rows,
    sha256_json,
    validate_recovery_bindings,
)
from farpoint.so101_episode_analysis import analyze_so101_episodes
from farpoint.so101_gate_report import so101_episode_evidence_errors
from farpoint.so101_pilot_report import audit_yaw_mass_episodes
from farpoint.so101_yaw_collection import (
    COLLECTION_KIND,
    validate_yaw_collection_balance,
    yaw_collection_balance,
)


def build_yaw_recovery_plan(
    reference_plan: dict[str, Any],
    source_collections: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    recovery_id: str,
    maximum_attempts: int,
) -> dict[str, Any]:
    if (reference_plan.get("collection") or {}).get("kind") != COLLECTION_KIND:
        raise ValueError("reference plan is not a balanced yaw collection")
    return build_missing_variation_recovery_plan(
        reference_plan,
        source_collections,
        recovery_id=recovery_id,
        maximum_attempts=maximum_attempts,
    )


def _audit_selected_source(
    reference_plan: dict[str, Any],
    manifest: dict[str, Any],
    episodes_root: Path,
) -> tuple[list[str], int, set[str], set[str]]:
    rows = list(selected_rows(manifest).values())
    episode_dirs = [episodes_root / row["episode_id"] for row in rows]
    missing = [path.name for path in episode_dirs if not path.is_dir()]
    existing = [path for path in episode_dirs if path.is_dir()]
    if existing:
        analysis = analyze_so101_episodes(existing, verify_images=True)
    else:
        analysis = {
            "episode_count": 0,
            "duplicate_observation_groups": [],
            "episodes": [],
        }
    errors = so101_episode_evidence_errors(analysis, len(rows))
    errors.extend(f"missing_episode:{episode_id}" for episode_id in missing)
    by_name = {Path(item["episode_dir"]).name: item for item in analysis["episodes"]}
    for row in rows:
        episode = by_name.get(row.get("episode_id"))
        if episode is None:
            continue
        prefix = row["episode_id"]
        if not row.get("success") or not row.get("dataset_valid"):
            errors.append(f"{prefix}:selected_manifest_row_not_eligible")
        if not episode.get("success") or not episode.get("dataset_valid"):
            errors.append(f"{prefix}:selected_episode_not_eligible")
        if episode.get("terminal_phase") != "retreat":
            errors.append(f"{prefix}:selected_episode_not_retreat")
        if episode.get("terminal_grasp_phase") != "validated":
            errors.append(f"{prefix}:selected_grasp_not_validated")
        proof = episode.get("proof_lift_tracking") or {}
        if float(proof.get("actual_max_m", 0.0)) < 0.005:
            errors.append(f"{prefix}:insufficient_proof_lift")
        settle = sum(
            phase["frame_count"]
            for phase in episode.get("phase_ranges") or []
            if phase["phase"] == "settle"
        )
        if settle < 15:
            errors.append(f"{prefix}:insufficient_settle_frames")
    profile = reference_plan.get("collection") or {}
    audits, audit_errors = audit_yaw_mass_episodes(
        reference_plan, rows, by_name, episodes_root, profile
    )
    errors.extend(audit_errors)
    return (
        errors,
        len(audits),
        {episode["metadata_sha256"] for episode in analysis["episodes"]},
        {episode["observations_sha256"] for episode in analysis["episodes"]},
    )


def build_yaw_completion_report(
    reference_plan: dict[str, Any],
    historical_sources: list[tuple[dict[str, Any], dict[str, Any], str | Path]],
    recovery_plan: dict[str, Any],
    recovery_manifest: dict[str, Any],
    *,
    recovery_episodes_root: str | Path,
) -> dict[str, Any]:
    """Prove exact frozen balance across historical successes and recovery."""
    profile = reference_plan.get("collection") or {}
    if profile.get("kind") != COLLECTION_KIND:
        raise ValueError("reference plan is not a balanced yaw collection")
    errors, normalized = validate_recovery_bindings(
        reference_plan, historical_sources, recovery_plan, recovery_manifest
    )
    sources = [
        *normalized,
        (recovery_plan, recovery_manifest, Path(recovery_episodes_root)),
    ]
    observed: set[str] = set()
    source_summaries: list[dict[str, Any]] = []
    yaw_mass_audit_count = 0
    metadata_identities: set[str] = set()
    observation_identities: set[str] = set()
    for _plan, manifest, root in sources:
        rows = selected_rows(manifest)
        observed.update(rows)
        source_errors, audit_count, metadata_hashes, observation_hashes = _audit_selected_source(
            reference_plan, manifest, root
        )
        errors.extend(source_errors)
        yaw_mass_audit_count += audit_count
        if metadata_identities & metadata_hashes:
            errors.append("duplicate_episode_identity_across_sources")
        if observation_identities & observation_hashes:
            errors.append("duplicate_observation_artifacts_across_sources")
        metadata_identities.update(metadata_hashes)
        observation_identities.update(observation_hashes)
        source_summaries.append(
            {
                "collection_id": manifest["collection_id"],
                "execution_status": manifest["execution_status"],
                "quality_status": manifest["quality_status"],
                "attempted_count": len(manifest.get("attempts") or []),
                "selected_successes": len(rows),
                "manifest_sha256": sha256_json(manifest),
            }
        )
    trials = {trial["variation_id"]: trial for trial in reference_plan["trials"]}
    for _plan, manifest, _root in sources:
        for variation_id, row in selected_rows(manifest).items():
            if variation_id in trials and row.get("split") != trials[variation_id].get("split"):
                errors.append(f"{variation_id}:split_mismatch")
    balance = yaw_collection_balance(
        {
            "trials": [
                trial for trial in reference_plan["trials"] if trial["variation_id"] in observed
            ]
        }
    )
    errors.extend(validate_yaw_collection_balance(balance, profile.get("balance_contract")))
    errors = sorted(set(errors))
    return {
        "schema_version": "farpoint.so101-yaw-completion-report.v1",
        "status": "PASS" if not errors else "INVALID_EVIDENCE",
        "reference_plan_id": reference_plan["plan_id"],
        "reference_plan_sha256": reference_plan["plan_sha256"],
        "yaw_degrees": float(profile["yaw_degrees"]),
        "selected_successes": len(observed),
        "required_successes": len(reference_plan["trials"]),
        "yaw_mass_audit_count": yaw_mass_audit_count,
        "balance": balance,
        "sources": source_summaries,
        "evidence_errors": errors,
    }


def build_yaw_completion_selection(
    reference_plan: dict[str, Any],
    historical_sources: list[tuple[dict[str, Any], dict[str, Any], str | Path]],
    recovery_plan: dict[str, Any],
    recovery_manifest: dict[str, Any],
    *,
    recovery_episodes_root: str | Path,
    collection_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report = build_yaw_completion_report(
        reference_plan,
        historical_sources,
        recovery_plan,
        recovery_manifest,
        recovery_episodes_root=recovery_episodes_root,
    )
    if report["status"] != "PASS":
        raise ValueError(
            "yaw completion evidence did not pass: " + "; ".join(report["evidence_errors"])
        )
    sources = [
        *historical_sources,
        (recovery_plan, recovery_manifest, recovery_episodes_root),
    ]
    manifest, selection = build_completion_artifacts(
        reference_plan,
        sources,
        collection_id=collection_id,
        git_commit=recovery_manifest["git_commit"],
        balance=report["balance"],
    )
    return manifest, selection, report
