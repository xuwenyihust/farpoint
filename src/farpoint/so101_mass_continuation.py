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
RECOVERY_POLICY = "multi_source_missing_variation_recovery_v1"
MULTI_SOURCE_COMPLETION_POLICY = "multi_source_mass_completion_v1"


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


def _source_binding(
    plan: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    return {
        "collection_id": manifest["collection_id"],
        "execution_status": manifest["execution_status"],
        "quality_status": manifest["quality_status"],
        "manifest_sha256": _sha256(manifest),
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "attempted_count": len(manifest.get("attempts") or []),
        "selected_successes": len(manifest.get("selected_variations") or {}),
    }


def _is_reusable_terminal_source(manifest: dict[str, Any]) -> bool:
    state = (
        manifest.get("execution_status"),
        manifest.get("quality_status"),
    )
    return state in {
        ("ABORTED", "NOT_EVALUATED"),
        ("FINISHED", "PASS"),
    }


def build_mass_recovery_plan(
    reference_plan: dict[str, Any],
    source_collections: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    recovery_id: str,
    maximum_attempts: int = 150,
) -> dict[str, Any]:
    """Freeze variations missing across multiple immutable source collections."""
    if not recovery_id:
        raise ValueError("recovery_id must be non-empty")
    if not source_collections:
        raise ValueError("recovery requires at least one source collection")
    attempt_budget = int(maximum_attempts)
    reference_profile = reference_plan.get("collection") or {}
    if reference_profile.get("kind") != COLLECTION_KIND:
        raise ValueError("reference plan is not a mirrored mass collection")
    expected_trials = {
        trial["variation_id"]: trial for trial in reference_plan["trials"]
    }
    selected: set[str] = set()
    bindings: list[dict[str, Any]] = []
    for source_plan, source_manifest in source_collections:
        validate_manifest(source_manifest, source_plan)
        if not _is_reusable_terminal_source(source_manifest):
            raise ValueError("recovery source must be terminal reusable evidence")
        source_profile = source_plan.get("collection") or {}
        if source_profile.get("kind") != COLLECTION_KIND:
            raise ValueError("recovery source is not a mirrored mass collection")
        source_rows = _selected_rows(source_manifest)
        overlap = selected & set(source_rows)
        if overlap:
            raise ValueError(
                "recovery sources overlap selected variations: "
                + ",".join(sorted(overlap))
            )
        unknown = set(source_rows) - set(expected_trials)
        if unknown:
            raise ValueError(
                "recovery source has unknown selected variations: "
                + ",".join(sorted(unknown))
            )
        for variation_id, row in source_rows.items():
            if row.get("split") != expected_trials[variation_id].get("split"):
                raise ValueError(f"recovery source split mismatch: {variation_id}")
        selected.update(source_rows)
        bindings.append(_source_binding(source_plan, source_manifest))
    trials = [
        copy.deepcopy(trial)
        for trial in reference_plan["trials"]
        if trial["variation_id"] not in selected
    ]
    if not trials:
        raise ValueError("source collections already cover every variation")
    if attempt_budget < len(trials):
        raise ValueError("recovery attempt budget cannot cover missing variations")
    plan = copy.deepcopy(reference_plan)
    plan["plan_id"] = str(recovery_id)
    plan["config_revision"] = (
        f"{reference_plan.get('config_revision', 'unknown')}:multi-source-recovery"
    )
    plan["trials"] = trials
    plan["collection"] = {
        **copy.deepcopy(reference_profile),
        "required_successes": len(trials),
        "maximum_attempts": attempt_budget,
        "selection_policy": RECOVERY_POLICY,
        "recovery_balance": mirrored_balance(plan),
        "source_collections": bindings,
    }
    plan["collection"].pop("balance", None)
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = _sha256(plan)
    return plan


def build_mass_multi_source_completion_report(
    reference_plan: dict[str, Any],
    historical_sources: list[
        tuple[dict[str, Any], dict[str, Any], str | Path]
    ],
    recovery_plan: dict[str, Any],
    recovery_manifest: dict[str, Any],
    *,
    recovery_episodes_root: str | Path,
) -> dict[str, Any]:
    """Validate exact balanced coverage across historical and recovery sources."""
    evidence_errors: list[str] = []
    validate_manifest(recovery_manifest, recovery_plan)
    recovery_profile = recovery_plan.get("collection") or {}
    bindings = recovery_profile.get("source_collections") or []
    if len(bindings) != len(historical_sources):
        evidence_errors.append("recovery_source_binding_count_mismatch")
    expected_trials = {
        trial["variation_id"]: trial for trial in reference_plan["trials"]
    }
    historical_observed: set[str] = set()
    sources: list[
        tuple[dict[str, Any], dict[str, Any], Path]
    ] = []
    for index, (plan, manifest, episodes_root) in enumerate(historical_sources):
        validate_manifest(manifest, plan)
        sources.append((plan, manifest, Path(episodes_root)))
        if not _is_reusable_terminal_source(manifest):
            evidence_errors.append(
                f"source_not_terminal_reusable:{manifest.get('collection_id')}"
            )
        rows = _selected_rows(manifest)
        overlap = historical_observed & set(rows)
        if overlap:
            evidence_errors.append(
                "duplicate_selected_variations:" + ",".join(sorted(overlap))
            )
        historical_observed.update(rows)
        if index < len(bindings):
            expected_binding = _source_binding(plan, manifest)
            if bindings[index] != expected_binding:
                evidence_errors.append(
                    f"recovery_source_binding_mismatch:{manifest.get('collection_id')}"
                )
    if recovery_manifest.get("execution_status") != "FINISHED":
        evidence_errors.append("recovery_not_finished")
    if recovery_manifest.get("quality_status") != "PASS":
        evidence_errors.append("recovery_quality_not_pass")
    frozen_recovery = {
        trial["variation_id"] for trial in recovery_plan["trials"]
    }
    expected_recovery = set(expected_trials) - historical_observed
    if frozen_recovery != expected_recovery:
        evidence_errors.append("recovery_trial_set_mismatch")
    sources.append(
        (recovery_plan, recovery_manifest, Path(recovery_episodes_root))
    )
    selected_by_source: list[dict[str, dict[str, Any]]] = []
    observed: set[str] = set()
    duplicates: set[str] = set()
    for _plan, manifest, _root in sources:
        rows = _selected_rows(manifest)
        duplicates.update(observed & set(rows))
        observed.update(rows)
        selected_by_source.append(rows)
    if duplicates:
        evidence_errors.append(
            "duplicate_selected_variations:" + ",".join(sorted(duplicates))
        )
    missing = set(expected_trials) - observed
    extra = observed - set(expected_trials)
    if missing:
        evidence_errors.append(
            "missing_selected_variations:" + ",".join(sorted(missing))
        )
    if extra:
        evidence_errors.append(
            "unknown_selected_variations:" + ",".join(sorted(extra))
        )
    reference_profile = reference_plan.get("collection") or {}
    target_mass = float(reference_profile["target_mass_kg"])
    tolerance = float(reference_profile["actual_mass_tolerance_kg"])
    source_summaries: list[dict[str, Any]] = []
    for (_plan, manifest, root), rows in zip(sources, selected_by_source):
        for variation_id, row in rows.items():
            trial = expected_trials.get(variation_id)
            if trial is not None and row.get("split") != trial.get("split"):
                evidence_errors.append(f"{variation_id}:split_mismatch")
            evidence_errors.extend(
                _audit_episode(
                    row,
                    root,
                    target_mass_kg=target_mass,
                    tolerance_kg=tolerance,
                )
            )
        source_summaries.append(
            {
                "collection_id": manifest["collection_id"],
                "execution_status": manifest["execution_status"],
                "quality_status": manifest["quality_status"],
                "attempted_count": len(manifest.get("attempts") or []),
                "selected_successes": len(rows),
                "manifest_sha256": _sha256(manifest),
            }
        )
    selected_plan = {
        "trials": [
            trial
            for trial in reference_plan["trials"]
            if trial["variation_id"] in observed
        ]
    }
    balance = mirrored_balance(selected_plan)
    evidence_errors.extend(validate_mirrored_balance(balance))
    return {
        "schema_version": "farpoint.so101-mass-completion-report.v2",
        "status": "PASS" if not evidence_errors else "INVALID_EVIDENCE",
        "target_mass_kg": target_mass,
        "selected_successes": len(observed),
        "required_successes": len(expected_trials),
        "balance": balance,
        "sources": source_summaries,
        "evidence_errors": sorted(set(evidence_errors)),
    }


def build_mass_multi_source_completion_selection(
    reference_plan: dict[str, Any],
    historical_sources: list[
        tuple[dict[str, Any], dict[str, Any], str | Path]
    ],
    recovery_plan: dict[str, Any],
    recovery_manifest: dict[str, Any],
    *,
    recovery_episodes_root: str | Path,
    collection_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build a candidate selection from any number of immutable sources."""
    report = build_mass_multi_source_completion_report(
        reference_plan,
        historical_sources,
        recovery_plan,
        recovery_manifest,
        recovery_episodes_root=recovery_episodes_root,
    )
    if report["status"] != "PASS":
        raise ValueError(
            "mass multi-source completion evidence did not pass: "
            + "; ".join(report["evidence_errors"])
        )
    sources = [
        *historical_sources,
        (recovery_plan, recovery_manifest, recovery_episodes_root),
    ]
    attempts: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    selected_variations: dict[str, str] = {}
    source_records: list[dict[str, Any]] = []
    for _plan, source_manifest, episodes_root in sources:
        rows = _selected_rows(source_manifest)
        source_id = source_manifest["collection_id"]
        root = Path(episodes_root)
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
        source_records.append(
            {
                "collection_id": source_id,
                "execution_status": source_manifest["execution_status"],
                "quality_status": source_manifest["quality_status"],
                "manifest_sha256": _sha256(source_manifest),
                "episode_root": str(root),
            }
        )
    order = {
        trial["variation_id"]: index
        for index, trial in enumerate(reference_plan["trials"])
    }
    attempts.sort(key=lambda row: order[row["variation_id"]])
    episodes.sort(key=lambda row: order[row["variation_id"]])
    timestamp = _now()
    manifest = {
        "schema_version": "farpoint.collection-selection.v1",
        "collection_id": collection_id,
        "task_id": recovery_manifest["task_id"],
        "git_commit": recovery_manifest["git_commit"],
        "execution_status": "FINISHED",
        "quality_status": "PASS",
        "release_status": "CANDIDATE",
        "required_successes": len(reference_plan["trials"]),
        "maximum_attempts": len(reference_plan["trials"]),
        "created_at": timestamp,
        "updated_at": timestamp,
        "attempts": attempts,
        "selected_variations": selected_variations,
        "selection_policy": MULTI_SOURCE_COMPLETION_POLICY,
        "balance": report["balance"],
        "source_collections": source_records,
    }
    selection = {
        "schema_version": "farpoint.export-selection.v1",
        "dataset_id": (reference_plan.get("collection") or {}).get(
            "dataset_id", "farpoint_so101"
        ),
        "collection_id": collection_id,
        "selection_policy": MULTI_SOURCE_COMPLETION_POLICY,
        "episodes": episodes,
    }
    return manifest, selection, report
