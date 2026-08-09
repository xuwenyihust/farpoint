"""Generic immutable-source recovery contracts for SO-101 collections."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from farpoint.so101_collection import validate_manifest


RECOVERY_POLICY = "missing_variation_recovery_v1"
COMPLETION_POLICY = "immutable_sources_plus_recovery_v1"


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def selected_rows(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    attempts = {row["attempt_id"]: row for row in manifest.get("attempts") or []}
    return {
        variation_id: attempts[attempt_id]
        for variation_id, attempt_id in (manifest.get("selected_variations") or {}).items()
    }


def source_binding(plan: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "collection_id": manifest["collection_id"],
        "execution_status": manifest["execution_status"],
        "quality_status": manifest["quality_status"],
        "manifest_sha256": sha256_json(manifest),
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "attempted_count": len(manifest.get("attempts") or []),
        "selected_successes": len(manifest.get("selected_variations") or {}),
    }


def is_reusable_terminal_source(manifest: dict[str, Any]) -> bool:
    return (manifest.get("execution_status"), manifest.get("quality_status")) in {
        ("ABORTED", "NOT_EVALUATED"),
        ("FINISHED", "PASS"),
    }


def build_missing_variation_recovery_plan(
    reference_plan: dict[str, Any],
    source_collections: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    recovery_id: str,
    maximum_attempts: int,
) -> dict[str, Any]:
    """Freeze exactly the variations not selected by immutable source evidence."""
    if not recovery_id:
        raise ValueError("recovery_id must be non-empty")
    if not source_collections:
        raise ValueError("recovery requires at least one source collection")
    expected = {trial["variation_id"]: trial for trial in reference_plan["trials"]}
    selected: set[str] = set()
    bindings: list[dict[str, Any]] = []
    reference_kind = (reference_plan.get("collection") or {}).get("kind")
    for source_plan, source_manifest in source_collections:
        validate_manifest(source_manifest, source_plan)
        if not is_reusable_terminal_source(source_manifest):
            raise ValueError("recovery source must be terminal reusable evidence")
        if (source_plan.get("collection") or {}).get("kind") != reference_kind:
            raise ValueError("recovery source collection kind mismatch")
        rows = selected_rows(source_manifest)
        overlap = selected & set(rows)
        if overlap:
            raise ValueError(
                "recovery sources overlap selected variations: " + ",".join(sorted(overlap))
            )
        unknown = set(rows) - set(expected)
        if unknown:
            raise ValueError(
                "recovery source has unknown selected variations: " + ",".join(sorted(unknown))
            )
        for variation_id, row in rows.items():
            if row.get("split") != expected[variation_id].get("split"):
                raise ValueError(f"recovery source split mismatch: {variation_id}")
        selected.update(rows)
        bindings.append(source_binding(source_plan, source_manifest))
    trials = [
        copy.deepcopy(trial)
        for trial in reference_plan["trials"]
        if trial["variation_id"] not in selected
    ]
    if not trials:
        raise ValueError("source collections already cover every variation")
    budget = int(maximum_attempts)
    if budget < len(trials):
        raise ValueError("recovery attempt budget cannot cover missing variations")
    plan = copy.deepcopy(reference_plan)
    plan["plan_id"] = recovery_id
    plan["config_revision"] = f"{reference_plan.get('config_revision', 'unknown')}:recovery"
    plan["trials"] = trials
    profile = copy.deepcopy(reference_plan.get("collection") or {})
    profile.update(
        {
            "required_successes": len(trials),
            "maximum_attempts": budget,
            "selection_policy": RECOVERY_POLICY,
            "reference_collection": {
                "plan_id": reference_plan["plan_id"],
                "plan_sha256": reference_plan["plan_sha256"],
                "required_successes": len(reference_plan["trials"]),
            },
            "source_collections": bindings,
        }
    )
    profile.pop("balance", None)
    profile.pop("balance_contract", None)
    plan["collection"] = profile
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = sha256_json(plan)
    return plan


def validate_recovery_bindings(
    reference_plan: dict[str, Any],
    historical_sources: list[tuple[dict[str, Any], dict[str, Any], str | Path]],
    recovery_plan: dict[str, Any],
    recovery_manifest: dict[str, Any],
) -> tuple[list[str], list[tuple[dict[str, Any], dict[str, Any], Path]]]:
    """Validate source hashes and exact variation partitioning."""
    errors: list[str] = []
    validate_manifest(recovery_manifest, recovery_plan)
    if recovery_manifest.get("execution_status") != "FINISHED":
        errors.append("recovery_not_finished")
    if recovery_manifest.get("quality_status") != "PASS":
        errors.append("recovery_quality_not_pass")
    profile = recovery_plan.get("collection") or {}
    reference = profile.get("reference_collection") or {}
    if reference.get("plan_sha256") != reference_plan.get("plan_sha256"):
        errors.append("recovery_reference_plan_hash_mismatch")
    bindings = profile.get("source_collections") or []
    if len(bindings) != len(historical_sources):
        errors.append("recovery_source_binding_count_mismatch")
    expected = {trial["variation_id"]: trial for trial in reference_plan["trials"]}
    observed: set[str] = set()
    normalized: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    for index, (plan, manifest, root) in enumerate(historical_sources):
        validate_manifest(manifest, plan)
        normalized.append((plan, manifest, Path(root)))
        if not is_reusable_terminal_source(manifest):
            errors.append(f"source_not_terminal_reusable:{manifest.get('collection_id')}")
        rows = selected_rows(manifest)
        overlap = observed & set(rows)
        if overlap:
            errors.append("duplicate_selected_variations:" + ",".join(sorted(overlap)))
        observed.update(rows)
        if index >= len(bindings) or bindings[index] != source_binding(plan, manifest):
            errors.append(f"recovery_source_binding_mismatch:{manifest.get('collection_id')}")
    frozen = {trial["variation_id"] for trial in recovery_plan["trials"]}
    if frozen != set(expected) - observed:
        errors.append("recovery_trial_set_mismatch")
    recovery_rows = selected_rows(recovery_manifest)
    overlap = observed & set(recovery_rows)
    if overlap:
        errors.append("duplicate_selected_variations:" + ",".join(sorted(overlap)))
    final = observed | set(recovery_rows)
    missing = set(expected) - final
    extra = final - set(expected)
    if missing:
        errors.append("missing_selected_variations:" + ",".join(sorted(missing)))
    if extra:
        errors.append("unknown_selected_variations:" + ",".join(sorted(extra)))
    return sorted(set(errors)), normalized


def build_completion_artifacts(
    reference_plan: dict[str, Any],
    sources: list[tuple[dict[str, Any], dict[str, Any], str | Path]],
    *,
    collection_id: str,
    git_commit: str,
    balance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compose an ordered candidate manifest and export selection."""
    attempts: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    selected_variations: dict[str, str] = {}
    source_records: list[dict[str, Any]] = []
    for _plan, manifest, episodes_root in sources:
        rows = selected_rows(manifest)
        source_id = manifest["collection_id"]
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
                "execution_status": manifest["execution_status"],
                "quality_status": manifest["quality_status"],
                "manifest_sha256": sha256_json(manifest),
                "episode_root": str(root),
            }
        )
    order = {trial["variation_id"]: i for i, trial in enumerate(reference_plan["trials"])}
    attempts.sort(key=lambda row: order[row["variation_id"]])
    episodes.sort(key=lambda row: order[row["variation_id"]])
    now = datetime.now(timezone.utc).isoformat()
    required = len(reference_plan["trials"])
    manifest = {
        "schema_version": "farpoint.collection-selection.v1",
        "collection_id": collection_id,
        "task_id": sources[-1][1]["task_id"],
        "git_commit": git_commit,
        "execution_status": "FINISHED",
        "quality_status": "PASS",
        "release_status": "CANDIDATE",
        "required_successes": required,
        "maximum_attempts": required,
        "created_at": now,
        "updated_at": now,
        "attempts": attempts,
        "selected_variations": selected_variations,
        "selection_policy": COMPLETION_POLICY,
        "balance": balance,
        "source_collections": source_records,
    }
    selection = {
        "schema_version": "farpoint.export-selection.v1",
        "dataset_id": (reference_plan.get("collection") or {}).get("dataset_id", "farpoint_so101"),
        "collection_id": collection_id,
        "selection_policy": COMPLETION_POLICY,
        "episodes": episodes,
    }
    return manifest, selection
