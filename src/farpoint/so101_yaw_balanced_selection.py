"""Frozen balanced30 selection contract for the aborted SO-101 yaw collection."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from farpoint.balanced_selection import (
    load_selection_policy,
    select_balanced,
    selection_stats as policy_selection_stats,
    validate_balance as validate_policy_balance,
)
from farpoint.so101_episode_analysis import analyze_so101_episodes
from farpoint.so101_gate_report import so101_episode_evidence_errors
from farpoint.so101_pilot_report import audit_yaw_mass_episodes


SCHEMA_VERSION = "farpoint.collection-selection.v1"
VALIDATION_SCHEMA_VERSION = "farpoint.collection-selection-validation.v2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = PROJECT_ROOT / "configs" / "selections" / "so101_yaw0_balanced30.json"
POLICY = load_selection_policy(POLICY_PATH)
POLICY_ID = POLICY["policy_id"]
TARGET_COUNT = int(POLICY["target_count"])
SELECTION_SEED = int(POLICY["seed"])
SPLIT_TARGET = dict(POLICY["split_targets"])


def _constraint_targets(key: str) -> dict[str, int]:
    return next(
        dict(row["targets"])
        for row in POLICY["constraints"]
        if row["kind"] == "counts" and row["key"] == key
    )


ROW_TARGET = _constraint_targets("workspace_rows")
COLUMN_TARGET = _constraint_targets("workspace_columns")
MISSING_CELLS = set(POLICY["coverage_summary"]["universe"]) - set(
    next(
        row["required"]
        for row in POLICY["constraints"]
        if row["kind"] == "coverage" and row["key"] == "workspace_cells"
    )
)
MASS_COLOR_TARGET = {
    f"mass_{key}": value for key, value in _constraint_targets("mass_color").items()
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_aborted_source(
    manifest: dict[str, Any], plan: dict[str, Any], abort_record: dict[str, Any]
) -> list[str]:
    """Require an immutable, owner-aborted yaw collection lineage."""
    errors: list[str] = []
    if manifest.get("execution_status") != "ABORTED":
        errors.append("source_execution_status_not_aborted")
    if manifest.get("quality_status") != "NOT_EVALUATED":
        errors.append("source_quality_status_not_not_evaluated")
    if manifest.get("collection_id") != abort_record.get("collection_id"):
        errors.append("abort_record_collection_id_mismatch")
    if abort_record.get("execution_status") != "ABORTED":
        errors.append("abort_record_status_not_aborted")
    if abort_record.get("reason") != manifest.get("abort_reason"):
        errors.append("abort_reason_mismatch")
    if manifest.get("plan_id") != plan.get("plan_id"):
        errors.append("source_plan_id_mismatch")
    if manifest.get("plan_sha256") != plan.get("plan_sha256"):
        errors.append("source_plan_sha256_mismatch")
    profile = plan.get("collection") or {}
    if profile.get("kind") != "balanced_yaw_success_collection":
        errors.append("source_profile_not_balanced_yaw_collection")
    if float(profile.get("yaw_degrees", math.nan)) != 0.0:
        errors.append("source_yaw_not_zero")
    if float(profile.get("cube_size_m", math.nan)) != 0.03:
        errors.append("source_cube_size_not_30mm")
    completed = sum(
        bool(row.get("finished_at")) and row.get("failure_reason") != "aborted"
        for row in manifest.get("attempts") or []
    )
    if int(abort_record.get("completed_attempt_count", -1)) != completed:
        errors.append("abort_record_completed_attempt_count_mismatch")
    eligible = sum(
        bool(row.get("success") and row.get("dataset_valid"))
        for row in manifest.get("attempts") or []
    )
    selected_variations = manifest.get("selected_variations") or {}
    eligible_by_attempt = {
        row["attempt_id"]: row
        for row in manifest.get("attempts") or []
        if row.get("success") and row.get("dataset_valid")
    }
    if eligible != len(selected_variations):
        errors.append("eligible_attempt_count_selected_variation_count_mismatch")
    if any(
        attempt_id not in eligible_by_attempt
        or eligible_by_attempt[attempt_id].get("variation_id") != variation_id
        for variation_id, attempt_id in selected_variations.items()
    ):
        errors.append("selected_variation_mapping_not_eligible")
    if int(abort_record.get("selected_variation_count", -1)) != len(selected_variations):
        errors.append("abort_record_selected_variation_count_mismatch")
    return errors


def _legacy_stats(stats: dict[str, Any]) -> dict[str, Any]:
    converted = copy.deepcopy(stats)
    converted["mass_color"] = {
        f"mass_{key}": value for key, value in stats.get("mass_color", {}).items()
    }
    return converted


def _policy_stats(stats: dict[str, Any]) -> dict[str, Any]:
    converted = copy.deepcopy(stats)
    converted["mass_color"] = {
        key.removeprefix("mass_"): value for key, value in stats.get("mass_color", {}).items()
    }
    return converted


def selection_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = []
    for row in rows:
        item = copy.deepcopy(row)
        item.update(
            {
                "workspace_cells": row["cell_id"],
                "workspace_rows": row["cell_id"].split("_")[0],
                "workspace_columns": row["cell_id"].split("_")[1],
                "sizes": row["size_label"],
                "colors": row["color_label"],
                "masses_kg": row["mass_label"].removeprefix("mass_"),
                "yaw_degrees": f"{float(row['yaw_degrees']):.1f}",
            }
        )
        normalized.append(item)
    return _legacy_stats(policy_selection_stats(normalized, POLICY))


def validate_balance(stats: dict[str, Any]) -> list[str]:
    return validate_policy_balance(_policy_stats(stats), POLICY)


def select_balanced30(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    *,
    seed: int = SELECTION_SEED,
    iterations: int = 100_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select through the generic policy while preserving the historical API."""
    selected, stats = select_balanced(manifest, plan, POLICY, seed=seed, iterations=iterations)
    legacy_rows = []
    for row in selected:
        item = copy.deepcopy(row)
        item["cell_id"] = item.pop("workspace_cells")
        item.pop("workspace_rows")
        item.pop("workspace_columns")
        item["size_label"] = item.pop("sizes")
        item["color_label"] = item.pop("colors")
        item["mass_label"] = f"mass_{item.pop('masses_kg')}"
        item["yaw_degrees"] = float(item["yaw_degrees"])
        legacy_rows.append(item)
    return legacy_rows, _legacy_stats(stats)


def build_artifacts(
    source_manifest: dict[str, Any],
    plan: dict[str, Any],
    abort_record: dict[str, Any],
    selected: list[dict[str, Any]],
    stats: dict[str, Any],
    *,
    collection_id: str,
    dataset_id: str,
    episodes_root: str | Path,
    git_commit: str,
    source_manifest_file_sha256: str,
    abort_record_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timestamp = _now()
    attempts = []
    episodes = []
    internal_labels = {
        "cell_id",
        "size_label",
        "color_label",
        "mass_label",
        "yaw_degrees",
    }
    for row in selected:
        attempt = {
            key: copy.deepcopy(value) for key, value in row.items() if key not in internal_labels
        }
        attempt["selected_for_dataset"] = True
        attempts.append(attempt)
        episodes.append(
            {
                "episode_dir": str(Path(episodes_root).resolve() / row["episode_id"]),
                "trial_id": row["trial_id"],
                "variation_id": row["variation_id"],
                "split": row["split"],
                "source_collection_id": source_manifest["collection_id"],
                "source_attempt_id": row["attempt_id"],
            }
        )
    lineage = {
        "collection_id": source_manifest["collection_id"],
        "execution_status": "ABORTED",
        "quality_status": "NOT_EVALUATED",
        "manifest_sha256": source_manifest_file_sha256,
        "manifest_canonical_sha256": canonical_sha256(source_manifest),
        "abort_record_sha256": abort_record_file_sha256,
        "abort_reason": abort_record["reason"],
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "collection_id": collection_id,
        "task_id": source_manifest["task_id"],
        "git_commit": git_commit,
        "execution_status": "FINISHED",
        "quality_status": "PASS",
        "release_status": "CANDIDATE",
        "required_successes": TARGET_COUNT,
        "maximum_attempts": TARGET_COUNT,
        "attempts": attempts,
        "selected_variations": {row["variation_id"]: row["attempt_id"] for row in selected},
        "selection_policy": POLICY_ID,
        "balance": copy.deepcopy(stats),
        "source_collection": lineage,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    selection = {
        "schema_version": "farpoint.export-selection.v1",
        "dataset_id": dataset_id,
        "collection_id": collection_id,
        "selection_policy": POLICY_ID,
        "source_collection": lineage,
        "episodes": episodes,
    }
    return manifest, selection


def validate_episode_evidence(
    selected: list[dict[str, Any]],
    plan: dict[str, Any],
    episodes_root: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    root = Path(episodes_root).resolve()
    missing = [row["episode_id"] for row in selected if not (root / row["episode_id"]).is_dir()]
    episode_dirs = [
        root / row["episode_id"] for row in selected if row["episode_id"] not in missing
    ]
    analysis = analyze_so101_episodes(episode_dirs, verify_images=True)
    errors = so101_episode_evidence_errors(analysis, TARGET_COUNT)
    errors.extend(f"missing_episode:{episode_id}" for episode_id in missing)
    by_name = {Path(row["episode_dir"]).name: row for row in analysis["episodes"]}
    for attempt in selected:
        episode = by_name.get(attempt["episode_id"])
        if episode is None:
            continue
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
        proof_lift = float((episode.get("proof_lift_tracking") or {}).get("actual_max_m", 0.0))
        if proof_lift < 0.005:
            errors.append(f"{attempt['episode_id']}:insufficient_proof_lift")
    audits, audit_errors = audit_yaw_mass_episodes(
        plan, selected, by_name, root, profile=plan.get("collection") or {}
    )
    errors.extend(audit_errors)
    return analysis, audits, sorted(set(errors))


def build_validation_report(
    *,
    collection_id: str,
    source_manifest: dict[str, Any],
    plan: dict[str, Any],
    abort_record: dict[str, Any],
    selected: list[dict[str, Any]],
    stats: dict[str, Any],
    episodes_root: str | Path,
    source_manifest_path: str | Path,
    abort_record_path: str | Path,
    source_manifest_file_sha256: str | None = None,
    abort_record_file_sha256: str | None = None,
) -> dict[str, Any]:
    errors = validate_aborted_source(source_manifest, plan, abort_record)
    errors.extend(validate_balance(stats))
    source_attempts = {row["attempt_id"]: row for row in source_manifest.get("attempts") or []}
    if len({row["attempt_id"] for row in selected}) != TARGET_COUNT:
        errors.append("selected_attempt_ids_not_unique")
    if len({row["variation_id"] for row in selected}) != TARGET_COUNT:
        errors.append("selected_variation_ids_not_unique")
    for row in selected:
        source = source_attempts.get(row["attempt_id"])
        if source is None:
            errors.append(f"{row['attempt_id']}:not_in_source_manifest")
        elif not (source.get("success") and source.get("dataset_valid")):
            errors.append(f"{row['attempt_id']}:source_attempt_not_eligible")
        elif source.get("variation_id") != row.get("variation_id"):
            errors.append(f"{row['attempt_id']}:source_identity_mismatch")
    analysis, audits, evidence_errors = validate_episode_evidence(selected, plan, episodes_root)
    errors.extend(evidence_errors)
    current_manifest_sha256 = file_sha256(source_manifest_path)
    current_abort_sha256 = file_sha256(abort_record_path)
    if (
        source_manifest_file_sha256 is not None
        and current_manifest_sha256 != source_manifest_file_sha256
    ):
        errors.append("source_manifest_changed_during_validation")
    if abort_record_file_sha256 is not None and current_abort_sha256 != abort_record_file_sha256:
        errors.append("abort_record_changed_during_validation")
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "collection_id": collection_id,
        "selection_policy": POLICY_ID,
        "valid": not errors,
        "source_collection": {
            "collection_id": source_manifest["collection_id"],
            "manifest_sha256": current_manifest_sha256,
            "manifest_canonical_sha256": canonical_sha256(source_manifest),
            "abort_record_sha256": current_abort_sha256,
            "plan_id": plan["plan_id"],
            "plan_sha256": plan["plan_sha256"],
        },
        "selected_attempt_ids": [row["attempt_id"] for row in selected],
        "selected_trial_ids": [row["trial_id"] for row in selected],
        "balance": copy.deepcopy(stats),
        "yaw_mass_audits": audits,
        "episode_evidence": analysis,
        "errors": sorted(set(errors)),
        "validated_at": _now(),
    }


def render_validation_markdown(report: dict[str, Any]) -> str:
    balance = report["balance"]
    lines = [
        f"# SO-101 balanced30 validation: {report['collection_id']}",
        "",
        f"- Status: **{'PASS' if report['valid'] else 'FAIL'}**",
        f"- Selection policy: `{report['selection_policy']}`",
        f"- Source: `{report['source_collection']['collection_id']}` (ABORTED, immutable)",
        f"- Episodes: {balance['total']}",
        f"- Splits: {balance['splits']}",
        f"- Workspace: {balance['covered_cell_count']}/25 cells; missing {balance['missing_cells']}",
        f"- Masses: {balance['masses_kg']}",
        f"- Colors: {balance['colors']}",
        "",
        "## Validation errors",
        "",
    ]
    lines.extend(f"- {error}" for error in report["errors"])
    if not report["errors"]:
        lines.append("- None.")
    return "\n".join(lines) + "\n"
