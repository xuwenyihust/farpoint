"""Compatibility adapter for the configuration-driven SO-101 balanced50 policy."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from farpoint.balanced_selection import (
    load_selection_policy,
    select_balanced,
    validate_balance as validate_policy_balance,
)


SCHEMA_VERSION = "farpoint.collection-selection.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = PROJECT_ROOT / "configs" / "selections" / "so101_balanced50.json"
POLICY = load_selection_policy(POLICY_PATH)
POLICY_ID = POLICY["policy_id"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def select_balanced_attempts(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    *,
    target_count: int = 50,
    seed: int = 101,
    iterations: int = 250_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the historical API while executing the checked-in generic policy."""
    if target_count != POLICY["target_count"]:
        raise ValueError("balanced50 compatibility API requires target_count=50")
    return select_balanced(
        manifest,
        plan,
        POLICY,
        seed=seed,
        iterations=iterations,
    )


def validate_balance(stats: dict[str, Any], *, target_count: int = 50) -> list[str]:
    if target_count != POLICY["target_count"]:
        return [f"selected {stats.get('total')} episodes instead of {target_count}"]
    errors = validate_policy_balance(stats, POLICY)
    return [
        "selection does not cover all 25 workspace cells"
        if error.startswith("workspace_cells_coverage_mismatch")
        else error
        for error in errors
    ]


def build_artifacts(
    source_manifest: dict[str, Any],
    plan: dict[str, Any],
    selected: list[dict[str, Any]],
    stats: dict[str, Any],
    *,
    collection_id: str,
    dataset_id: str,
    episodes_root: str | Path,
    git_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timestamp = _now()
    source_digest = _sha256(source_manifest)
    internal_labels = set(POLICY["labels"])
    attempts = []
    episodes = []
    for row in selected:
        attempt = {
            key: copy.deepcopy(value) for key, value in row.items() if key not in internal_labels
        }
        attempt["selected_for_dataset"] = True
        attempts.append(attempt)
        episodes.append(
            {
                "episode_dir": str(Path(episodes_root) / row["episode_id"]),
                "trial_id": row["trial_id"],
                "variation_id": row["variation_id"],
                "split": row["split"],
            }
        )
    selected_variations = {row["variation_id"]: row["attempt_id"] for row in selected}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "collection_id": collection_id,
        "task_id": source_manifest["task_id"],
        "git_commit": git_commit,
        "execution_status": "FINISHED",
        "quality_status": "PASS",
        "release_status": "CANDIDATE",
        "required_successes": len(selected),
        "maximum_attempts": len(selected),
        "attempts": attempts,
        "selected_variations": selected_variations,
        "selection_policy": POLICY_ID,
        "balance": copy.deepcopy(stats),
        "source_collection": {
            "collection_id": source_manifest["collection_id"],
            "manifest_sha256": source_digest,
            "plan_id": plan["plan_id"],
            "plan_sha256": plan["plan_sha256"],
        },
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    selection = {
        "schema_version": "farpoint.export-selection.v1",
        "dataset_id": dataset_id,
        "collection_id": collection_id,
        "selection_policy": POLICY_ID,
        "source_collection_id": source_manifest["collection_id"],
        "episodes": episodes,
    }
    return manifest, selection
