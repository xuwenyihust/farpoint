"""Resumable, coverage-preserving collection state for SO-101 demonstrations."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "farpoint.collection.v2"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_manifest(
    plan: dict[str, Any],
    *,
    collection_id: str,
    git_commit: str,
    maximum_attempts: int = 150,
) -> dict[str, Any]:
    trials = plan.get("trials") or []
    if len(trials) != 100:
        raise ValueError("SO-101 collection requires exactly 100 planned variations")
    if maximum_attempts < len(trials):
        raise ValueError("maximum_attempts cannot be less than the planned variation count")
    return {
        "schema_version": SCHEMA_VERSION,
        "collection_id": collection_id,
        "task_id": plan["task_id"],
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "git_commit": git_commit,
        "required_successes": 100,
        "maximum_attempts": maximum_attempts,
        "attempts": [],
        "selected_variations": {},
        "execution_status": "RUNNING",
        "quality_status": "NOT_EVALUATED",
        "release_status": "PILOT",
        "created_at": _now(),
        "updated_at": _now(),
    }


def load_manifest(path: str | Path, plan: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"collection manifest must use {SCHEMA_VERSION}")
    if manifest.get("plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("collection manifest does not match the variation plan")
    validate_manifest(manifest, plan)
    return manifest


def validate_manifest(manifest: dict[str, Any], plan: dict[str, Any]) -> None:
    known = {trial["variation_id"] for trial in plan["trials"]}
    selected = manifest.get("selected_variations") or {}
    unknown = set(selected) - known
    if unknown:
        raise ValueError(f"collection selected unknown variations: {sorted(unknown)}")
    attempts = manifest.get("attempts") or []
    if len(attempts) > int(manifest["maximum_attempts"]):
        raise ValueError("collection exceeds its frozen maximum attempt budget")
    attempt_ids = [row.get("attempt_id") for row in attempts]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("collection attempt ids must be unique")
    for variation_id, attempt_id in selected.items():
        matches = [row for row in attempts if row.get("attempt_id") == attempt_id]
        if len(matches) != 1 or matches[0].get("variation_id") != variation_id:
            raise ValueError(f"invalid selected attempt for variation {variation_id}")
        if not matches[0].get("success") or not matches[0].get("dataset_valid"):
            raise ValueError(f"selected attempt is not eligible: {attempt_id}")


def next_attempt(manifest: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any] | None:
    """Return the next uncovered variation, preserving the original split."""
    validate_manifest(manifest, plan)
    if len(manifest["selected_variations"]) == int(manifest["required_successes"]):
        return None
    if len(manifest["attempts"]) >= int(manifest["maximum_attempts"]):
        return None
    counts = Counter(row["variation_id"] for row in manifest["attempts"])
    selected = manifest["selected_variations"]
    candidates = [trial for trial in plan["trials"] if trial["variation_id"] not in selected]
    candidates.sort(key=lambda trial: (counts[trial["variation_id"]], trial["trial_id"]))
    trial = copy.deepcopy(candidates[0])
    attempt_index = counts[trial["variation_id"]]
    attempt_id = f"{trial['trial_id']}__attempt{attempt_index:02d}"
    retry_material = {
        "plan_sha256": plan["plan_sha256"],
        "trial_id": trial["trial_id"],
        "attempt_index": attempt_index,
    }
    retry_seed = int.from_bytes(
        hashlib.sha256(
            json.dumps(retry_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()[:8],
        "big",
    )
    trial.update(
        {
            "attempt_id": attempt_id,
            "attempt_index": attempt_index,
            "attempt_seed": retry_seed,
        }
    )
    return trial


def record_attempt(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    attempt: dict[str, Any],
    *,
    episode_id: str | None,
    success: bool,
    dataset_valid: bool,
    failure_category: str | None = None,
    failure_reason: str | None = None,
) -> None:
    expected = next_attempt(manifest, plan)
    if expected is None or expected["attempt_id"] != attempt.get("attempt_id"):
        raise ValueError("attempt is not the next frozen collection attempt")
    row = {
        "attempt_id": attempt["attempt_id"],
        "trial_id": attempt["trial_id"],
        "variation_id": attempt["variation_id"],
        "split": attempt["split"],
        "attempt_index": attempt["attempt_index"],
        "attempt_seed": attempt["attempt_seed"],
        "episode_id": episode_id,
        "success": bool(success),
        "dataset_valid": bool(dataset_valid),
        "failure_category": failure_category,
        "failure_reason": failure_reason,
        "selected_for_dataset": bool(success and dataset_valid),
        "finished_at": _now(),
    }
    manifest["attempts"].append(row)
    if row["selected_for_dataset"]:
        manifest["selected_variations"][attempt["variation_id"]] = attempt["attempt_id"]
    complete = len(manifest["selected_variations"]) == int(manifest["required_successes"])
    exhausted = len(manifest["attempts"]) >= int(manifest["maximum_attempts"])
    if complete or exhausted:
        manifest["execution_status"] = "FINISHED"
        manifest["quality_status"] = "PASS" if complete else "FAIL"
    manifest["updated_at"] = _now()
    validate_manifest(manifest, plan)


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def build_export_selection(manifest: dict[str, Any], episodes_root: str = "outputs/episodes") -> dict:
    attempts = {row["attempt_id"]: row for row in manifest["attempts"]}
    selected = []
    for variation_id, attempt_id in sorted(manifest["selected_variations"].items()):
        row = attempts[attempt_id]
        selected.append(
            {
                "episode_dir": f"{episodes_root}/{row['episode_id']}",
                "trial_id": row["trial_id"],
                "variation_id": variation_id,
                "split": row["split"],
            }
        )
    return {
        "schema_version": "farpoint.export-selection.v1",
        "dataset_id": "farpoint-so101-cube-pick-place",
        "collection_id": manifest["collection_id"],
        "selection_policy": "one_success_per_stratified_variation",
        "episodes": selected,
    }
