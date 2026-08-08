"""Frozen balanced30 selection contract for the aborted SO-101 yaw collection."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from farpoint.so101_episode_analysis import analyze_so101_episodes
from farpoint.so101_gate_report import so101_episode_evidence_errors
from farpoint.so101_pilot_report import audit_yaw_mass_episodes


SCHEMA_VERSION = "farpoint.collection-selection.v1"
VALIDATION_SCHEMA_VERSION = "farpoint.collection-selection-validation.v2"
POLICY_ID = "so101_yaw0_30mm_balanced30_v1"
TARGET_COUNT = 30
SELECTION_SEED = 202608081159369
SPLIT_TARGET = {"train": 24, "validation": 1, "test": 5}
ROW_TARGET = {f"r{row:02d}": 6 for row in range(5)}
COLUMN_TARGET = {"c00": 5, "c01": 6, "c02": 7, "c03": 6, "c04": 6}
MISSING_CELLS = {"r04_c00", "r04_c01"}
MASS_COLOR_TARGET = {
    "mass_0.03__color_0": 8,
    "mass_0.03__color_1": 7,
    "mass_0.04__color_0": 7,
    "mass_0.04__color_1": 8,
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
    if int(abort_record.get("selected_variation_count", -1)) != len(
        selected_variations
    ):
        errors.append("abort_record_selected_variation_count_mismatch")
    return errors


def _candidate_rows(
    manifest: dict[str, Any], plan: dict[str, Any]
) -> list[dict[str, Any]]:
    trials = {trial["variation_id"]: trial for trial in plan.get("trials") or []}
    selected_variations = manifest.get("selected_variations") or {}
    rows = []
    for attempt in manifest.get("attempts") or []:
        if not (attempt.get("success") and attempt.get("dataset_valid")):
            continue
        if selected_variations.get(attempt.get("variation_id")) != attempt.get(
            "attempt_id"
        ):
            continue
        trial = trials.get(attempt.get("variation_id"))
        if trial is None:
            raise ValueError(
                f"eligible attempt has no planned variation: {attempt.get('attempt_id')}"
            )
        material = trial.get("seed_material") or {}
        mass = float(trial["resolved"]["mass_kg"])
        rows.append(
            {
                **copy.deepcopy(attempt),
                "cell_id": str(trial["cell_id"]),
                "size_label": f"size_{int(material['size_index'])}",
                "color_label": f"color_{int(material['color_index'])}",
                "mass_label": f"mass_{mass:.2f}",
                "yaw_degrees": float(trial["object_yaw_degrees"]),
            }
        )
    return sorted(rows, key=lambda row: row["trial_id"])


def selection_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = Counter(row["cell_id"] for row in rows)
    all_cells = {
        f"r{row:02d}_c{column:02d}"
        for row in range(5)
        for column in range(5)
    }
    return {
        "total": len(rows),
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "workspace_cells": dict(sorted(cells.items())),
        "workspace_rows": dict(
            sorted(Counter(row["cell_id"].split("_")[0] for row in rows).items())
        ),
        "workspace_columns": dict(
            sorted(Counter(row["cell_id"].split("_")[1] for row in rows).items())
        ),
        "sizes": dict(sorted(Counter(row["size_label"] for row in rows).items())),
        "colors": dict(sorted(Counter(row["color_label"] for row in rows).items())),
        "masses_kg": dict(
            sorted(
                Counter(row["mass_label"].removeprefix("mass_") for row in rows).items()
            )
        ),
        "mass_color": dict(
            sorted(
                Counter(
                    f"{row['mass_label']}__{row['color_label']}" for row in rows
                ).items()
            )
        ),
        "yaw_degrees": dict(
            sorted(Counter(f"{row['yaw_degrees']:.1f}" for row in rows).items())
        ),
        "covered_cell_count": len(cells),
        "missing_cells": sorted(all_cells - set(cells)),
    }


def validate_balance(stats: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "total": TARGET_COUNT,
        "splits": SPLIT_TARGET,
        "workspace_rows": ROW_TARGET,
        "workspace_columns": COLUMN_TARGET,
        "sizes": {"size_0": 30},
        "colors": {"color_0": 15, "color_1": 15},
        "masses_kg": {"0.03": 15, "0.04": 15},
        "mass_color": MASS_COLOR_TARGET,
        "yaw_degrees": {"0.0": 30},
        "covered_cell_count": 23,
        "missing_cells": sorted(MISSING_CELLS),
    }
    for key, value in expected.items():
        if stats.get(key) != value:
            errors.append(f"{key}_mismatch:{stats.get(key)!r}")
    cells = stats.get("workspace_cells") or {}
    if any(int(count) not in {1, 2} for count in cells.values()):
        errors.append(f"workspace_cell_multiplicity_invalid:{cells!r}")
    return errors


def _score(rows: list[dict[str, Any]]) -> int:
    stats = selection_stats(rows)
    penalties = 0
    penalties += 1_000_000 * len(
        set(stats["missing_cells"]).symmetric_difference(MISSING_CELLS)
    )
    penalties += 100_000 * sum(
        abs(stats["mass_color"].get(key, 0) - value)
        for key, value in MASS_COLOR_TARGET.items()
    )
    penalties += 10_000 * sum(
        abs(stats["workspace_rows"].get(key, 0) - value)
        for key, value in ROW_TARGET.items()
    )
    penalties += 1_000 * sum(
        abs(stats["workspace_columns"].get(key, 0) - value)
        for key, value in COLUMN_TARGET.items()
    )
    return penalties


def select_balanced30(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    *,
    seed: int = SELECTION_SEED,
    iterations: int = 100_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the frozen 30-episode contract without changing planned splits."""
    candidates = _candidate_rows(manifest, plan)
    by_split = {
        split: [row for row in candidates if row["split"] == split]
        for split in SPLIT_TARGET
    }
    for split, target in SPLIT_TARGET.items():
        if len(by_split[split]) < target:
            raise ValueError(
                f"split {split} has {len(by_split[split])} eligible episodes; {target} required"
            )
    rng = random.Random(seed)
    selected = []
    for split, target in SPLIT_TARGET.items():
        selected.extend(rng.sample(by_split[split], target))
    selected_ids = {row["attempt_id"] for row in selected}
    current_score = _score(selected)
    best = list(selected)
    best_score = current_score
    train_indexes = [
        index for index, row in enumerate(selected) if row["split"] == "train"
    ]
    for step in range(iterations):
        index = train_indexes[rng.randrange(len(train_indexes))]
        outgoing = selected[index]
        available = [
            row
            for row in by_split["train"]
            if row["attempt_id"] not in selected_ids
        ]
        incoming = available[rng.randrange(len(available))]
        selected[index] = incoming
        proposal_score = _score(selected)
        temperature = max(1.0, 50_000.0 * (1.0 - (step % 10_000) / 10_000))
        accept = proposal_score <= current_score or rng.random() < math.exp(
            min(0.0, (current_score - proposal_score) / temperature)
        )
        if accept:
            selected_ids.remove(outgoing["attempt_id"])
            selected_ids.add(incoming["attempt_id"])
            current_score = proposal_score
            if proposal_score < best_score:
                best = list(selected)
                best_score = proposal_score
                if best_score == 0:
                    break
        else:
            selected[index] = outgoing
    best.sort(key=lambda row: (tuple(SPLIT_TARGET).index(row["split"]), row["trial_id"]))
    stats = selection_stats(best)
    errors = validate_balance(stats)
    if errors:
        raise ValueError("balanced30 search did not satisfy contract: " + "; ".join(errors))
    stats["selection_seed"] = seed
    stats["selection_score"] = best_score
    return best, stats


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
            key: copy.deepcopy(value)
            for key, value in row.items()
            if key not in internal_labels
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
        "selected_variations": {
            row["variation_id"]: row["attempt_id"] for row in selected
        },
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
    missing = [
        row["episode_id"]
        for row in selected
        if not (root / row["episode_id"]).is_dir()
    ]
    episode_dirs = [
        root / row["episode_id"]
        for row in selected
        if row["episode_id"] not in missing
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
            phase["frame_count"]
            for phase in episode["phase_ranges"]
            if phase["phase"] == "settle"
        )
        if settle_frames < 15:
            errors.append(f"{attempt['episode_id']}:insufficient_settle_frames")
        proof_lift = float(
            (episode.get("proof_lift_tracking") or {}).get("actual_max_m", 0.0)
        )
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
    source_attempts = {
        row["attempt_id"]: row for row in source_manifest.get("attempts") or []
    }
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
    analysis, audits, evidence_errors = validate_episode_evidence(
        selected, plan, episodes_root
    )
    errors.extend(evidence_errors)
    current_manifest_sha256 = file_sha256(source_manifest_path)
    current_abort_sha256 = file_sha256(abort_record_path)
    if (
        source_manifest_file_sha256 is not None
        and current_manifest_sha256 != source_manifest_file_sha256
    ):
        errors.append("source_manifest_changed_during_validation")
    if (
        abort_record_file_sha256 is not None
        and current_abort_sha256 != abort_record_file_sha256
    ):
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
