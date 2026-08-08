"""Auditable continuation and completion evidence for SO-101 mass collection."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from farpoint.so101_collection import validate_manifest
from farpoint.so101_mass_collection import (
    COLLECTION_KIND,
    mirrored_balance,
    validate_mirrored_balance,
)
from farpoint.so101_mass_feasibility import audit_resolved_mass


CONTINUATION_POLICY = "missing_variation_continuation_v1"
COMPLETION_POLICY = "aborted_parent_plus_continuation_v1"


def _sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_rows(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    attempts = {
        row["attempt_id"]: row for row in manifest.get("attempts") or []
    }
    return {
        variation_id: attempts[attempt_id]
        for variation_id, attempt_id in (
            manifest.get("selected_variations") or {}
        ).items()
    }


def build_mass_continuation_plan(
    parent_plan: dict[str, Any],
    parent_manifest: dict[str, Any],
    *,
    continuation_id: str,
) -> dict[str, Any]:
    """Freeze only uncovered variations from an aborted mass collection."""
    validate_manifest(parent_manifest, parent_plan)
    profile = parent_plan.get("collection") or {}
    if profile.get("kind") != COLLECTION_KIND:
        raise ValueError("parent plan is not a mirrored mass collection")
    if parent_manifest.get("execution_status") != "ABORTED":
        raise ValueError("parent collection must be ABORTED")
    if parent_manifest.get("quality_status") != "NOT_EVALUATED":
        raise ValueError("parent collection quality must be NOT_EVALUATED")
    selected = set(parent_manifest.get("selected_variations") or {})
    trials = [
        copy.deepcopy(trial)
        for trial in parent_plan["trials"]
        if trial["variation_id"] not in selected
    ]
    if not trials:
        raise ValueError("parent collection has no uncovered variations")
    remaining_attempts = int(parent_manifest["maximum_attempts"]) - len(
        parent_manifest.get("attempts") or []
    )
    if remaining_attempts < len(trials):
        raise ValueError("parent attempt budget cannot cover missing variations")
    plan = copy.deepcopy(parent_plan)
    plan["plan_id"] = str(continuation_id)
    plan["config_revision"] = (
        f"{parent_plan.get('config_revision', 'unknown')}:continuation"
    )
    plan["trials"] = trials
    continuation_balance = mirrored_balance(plan)
    plan["collection"] = {
        **copy.deepcopy(profile),
        "required_successes": len(trials),
        "maximum_attempts": remaining_attempts,
        "selection_policy": CONTINUATION_POLICY,
        "continuation_balance": continuation_balance,
        "parent_collection": {
            "collection_id": parent_manifest["collection_id"],
            "execution_status": parent_manifest["execution_status"],
            "manifest_sha256": _sha256(parent_manifest),
            "plan_id": parent_plan["plan_id"],
            "plan_sha256": parent_plan["plan_sha256"],
            "attempted_count": len(parent_manifest.get("attempts") or []),
            "selected_successes": len(selected),
        },
    }
    plan["collection"].pop("balance", None)
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = _sha256(plan)
    return plan


def _audit_episode(
    row: dict[str, Any],
    root: Path,
    *,
    target_mass_kg: float,
    tolerance_kg: float,
) -> list[str]:
    episode_id = row.get("episode_id")
    prefix = str(episode_id or row.get("attempt_id") or "unknown")
    if not episode_id:
        return [f"{prefix}:missing_episode_id"]
    episode = root / episode_id
    required = (
        "metadata.json",
        "metrics.json",
        "observations.jsonl",
        "run-state.json",
    )
    missing = [name for name in required if not (episode / name).is_file()]
    if missing:
        return [f"{prefix}:missing_artifacts:{','.join(missing)}"]
    errors: list[str] = []
    observations = [
        line
        for line in (episode / "observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if not observations:
        errors.append(f"{prefix}:empty_observations")
    metadata = _read_json(episode / "metadata.json")
    metrics = _read_json(episode / "metrics.json")
    run_state = _read_json(episode / "run-state.json")
    if run_state.get("execution_status") != "FINISHED":
        errors.append(f"{prefix}:run_state_not_finished")
    if not metrics.get("success") or not metrics.get("dataset_valid"):
        errors.append(f"{prefix}:selected_metrics_not_eligible")
    metric_audit = (metrics.get("physics_audit") or {}).get("mass")
    metadata_audit = (
        ((metadata.get("scene") or {}).get("object") or {}).get("mass_audit")
    )
    if not isinstance(metric_audit, dict) or not isinstance(
        metadata_audit, dict
    ):
        errors.append(f"{prefix}:missing_mass_audit")
        return errors
    audit_keys = {
        "requested_mass_kg",
        "resolved_mass_kg",
        "physx_actual_mass_kg",
        "tolerance_kg",
        "verified",
    }
    if any(
        metric_audit.get(key) != metadata_audit.get(key)
        for key in audit_keys
    ):
        errors.append(f"{prefix}:mass_audit_sidecar_mismatch")
    try:
        recomputed = audit_resolved_mass(
            requested_mass_kg=float(metric_audit["requested_mass_kg"]),
            resolved_mass_kg=float(metric_audit["resolved_mass_kg"]),
            physx_actual_mass_kg=float(
                metric_audit["physx_actual_mass_kg"]
            ),
            tolerance_kg=tolerance_kg,
        )
    except (KeyError, TypeError, ValueError):
        errors.append(f"{prefix}:invalid_mass_audit")
        return errors
    if (
        not recomputed["verified"]
        or not metric_audit.get("verified")
        or abs(float(metric_audit["requested_mass_kg"]) - target_mass_kg)
        > tolerance_kg
        or abs(float(metric_audit["resolved_mass_kg"]) - target_mass_kg)
        > tolerance_kg
        or float(metric_audit.get("tolerance_kg", -1.0)) != tolerance_kg
    ):
        errors.append(f"{prefix}:mass_audit_failed")
    return errors


def build_mass_completion_report(
    parent_plan: dict[str, Any],
    parent_manifest: dict[str, Any],
    continuation_plan: dict[str, Any],
    continuation_manifest: dict[str, Any],
    *,
    parent_episodes_root: str | Path,
    continuation_episodes_root: str | Path,
) -> dict[str, Any]:
    """Validate exact full coverage across the immutable parent and continuation."""
    evidence_errors: list[str] = []
    validate_manifest(parent_manifest, parent_plan)
    validate_manifest(continuation_manifest, continuation_plan)
    parent_profile = parent_plan.get("collection") or {}
    continuation_profile = continuation_plan.get("collection") or {}
    parent_binding = continuation_profile.get("parent_collection") or {}
    if parent_manifest.get("execution_status") != "ABORTED":
        evidence_errors.append("parent_collection_not_aborted")
    if continuation_manifest.get("execution_status") != "FINISHED":
        evidence_errors.append("continuation_not_finished")
    if continuation_manifest.get("quality_status") != "PASS":
        evidence_errors.append("continuation_quality_not_pass")
    if parent_binding.get("collection_id") != parent_manifest.get(
        "collection_id"
    ):
        evidence_errors.append("continuation_parent_collection_mismatch")
    if parent_binding.get("manifest_sha256") != _sha256(parent_manifest):
        evidence_errors.append("continuation_parent_manifest_hash_mismatch")
    if parent_binding.get("plan_sha256") != parent_plan.get("plan_sha256"):
        evidence_errors.append("continuation_parent_plan_hash_mismatch")
    parent_rows = _selected_rows(parent_manifest)
    continuation_rows = _selected_rows(continuation_manifest)
    overlap = set(parent_rows) & set(continuation_rows)
    if overlap:
        evidence_errors.append(
            "duplicate_selected_variations:" + ",".join(sorted(overlap))
        )
    expected = {trial["variation_id"] for trial in parent_plan["trials"]}
    observed = set(parent_rows) | set(continuation_rows)
    missing = expected - observed
    extra = observed - expected
    if missing:
        evidence_errors.append(
            "missing_selected_variations:" + ",".join(sorted(missing))
        )
    if extra:
        evidence_errors.append(
            "unknown_selected_variations:" + ",".join(sorted(extra))
        )
    trial_by_id = {
        trial["variation_id"]: trial for trial in parent_plan["trials"]
    }
    for variation_id, row in {**parent_rows, **continuation_rows}.items():
        trial = trial_by_id.get(variation_id)
        if trial is not None and row.get("split") != trial.get("split"):
            evidence_errors.append(f"{variation_id}:split_mismatch")
    target_mass = float(parent_profile["target_mass_kg"])
    tolerance = float(parent_profile["actual_mass_tolerance_kg"])
    for rows, root in (
        (parent_rows, Path(parent_episodes_root)),
        (continuation_rows, Path(continuation_episodes_root)),
    ):
        for row in rows.values():
            evidence_errors.extend(
                _audit_episode(
                    row,
                    root,
                    target_mass_kg=target_mass,
                    tolerance_kg=tolerance,
                )
            )
    selected_plan = {
        "trials": [
            trial
            for trial in parent_plan["trials"]
            if trial["variation_id"] in observed
        ]
    }
    balance = mirrored_balance(selected_plan)
    evidence_errors.extend(validate_mirrored_balance(balance))
    return {
        "schema_version": "farpoint.so101-mass-completion-report.v1",
        "status": "PASS" if not evidence_errors else "INVALID_EVIDENCE",
        "parent_collection_id": parent_manifest["collection_id"],
        "continuation_collection_id": continuation_manifest["collection_id"],
        "target_mass_kg": target_mass,
        "selected_successes": len(observed),
        "required_successes": len(expected),
        "parent_attempted_count": len(parent_manifest.get("attempts") or []),
        "continuation_attempted_count": len(
            continuation_manifest.get("attempts") or []
        ),
        "balance": balance,
        "evidence_errors": sorted(set(evidence_errors)),
    }


def build_mass_completion_selection(
    parent_plan: dict[str, Any],
    parent_manifest: dict[str, Any],
    continuation_plan: dict[str, Any],
    continuation_manifest: dict[str, Any],
    *,
    parent_episodes_root: str | Path,
    continuation_episodes_root: str | Path,
    collection_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build a PASS selection only after combined evidence reaches 50/50."""
    report = build_mass_completion_report(
        parent_plan,
        parent_manifest,
        continuation_plan,
        continuation_manifest,
        parent_episodes_root=parent_episodes_root,
        continuation_episodes_root=continuation_episodes_root,
    )
    if report["status"] != "PASS":
        raise ValueError(
            "mass completion evidence did not pass: "
            + "; ".join(report["evidence_errors"])
        )
    parent_rows = _selected_rows(parent_manifest)
    continuation_rows = _selected_rows(continuation_manifest)
    attempts: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    selected_variations: dict[str, str] = {}
    sources = (
        (
            parent_manifest,
            parent_rows,
            Path(parent_episodes_root),
        ),
        (
            continuation_manifest,
            continuation_rows,
            Path(continuation_episodes_root),
        ),
    )
    for source_manifest, rows, root in sources:
        source_id = source_manifest["collection_id"]
        for variation_id, source_row in rows.items():
            row = copy.deepcopy(source_row)
            source_attempt_id = row["attempt_id"]
            row["attempt_id"] = f"{source_id}__{source_attempt_id}"
            row["source_attempt_id"] = source_attempt_id
            row["source_collection_id"] = source_id
            row["selected_for_dataset"] = True
            attempts.append(row)
            selected_variations[variation_id] = row["attempt_id"]
            episodes.append(
                {
                    "episode_dir": str(root / row["episode_id"]),
                    "trial_id": row["trial_id"],
                    "variation_id": variation_id,
                    "split": row["split"],
                }
            )
    order = {
        trial["variation_id"]: index
        for index, trial in enumerate(parent_plan["trials"])
    }
    attempts.sort(key=lambda row: order[row["variation_id"]])
    episodes.sort(key=lambda row: order[row["variation_id"]])
    timestamp = _now()
    manifest = {
        "schema_version": "farpoint.collection-selection.v1",
        "collection_id": collection_id,
        "task_id": parent_manifest["task_id"],
        "git_commit": continuation_manifest["git_commit"],
        "execution_status": "FINISHED",
        "quality_status": "PASS",
        "release_status": "CANDIDATE",
        "required_successes": 50,
        "maximum_attempts": 50,
        "created_at": timestamp,
        "updated_at": timestamp,
        "attempts": attempts,
        "selected_variations": selected_variations,
        "selection_policy": COMPLETION_POLICY,
        "balance": report["balance"],
        "source_collections": [
            {
                "collection_id": source_manifest["collection_id"],
                "execution_status": source_manifest["execution_status"],
                "quality_status": source_manifest["quality_status"],
                "manifest_sha256": _sha256(source_manifest),
                "episode_root": str(root),
            }
            for source_manifest, _rows, root in sources
        ],
    }
    selection = {
        "schema_version": "farpoint.export-selection.v1",
        "dataset_id": (parent_plan.get("collection") or {}).get(
            "dataset_id", "farpoint_so101"
        ),
        "collection_id": collection_id,
        "selection_policy": COMPLETION_POLICY,
        "episodes": episodes,
    }
    return manifest, selection, report
