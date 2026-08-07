"""Build the 0.04/0.03 kg balanced SO-101 dataset candidate selection."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from farpoint.so101_collection import build_export_selection, validate_manifest


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_mass_dataset_candidate(
    baseline: dict[str, Any],
    candidate_manifest: dict[str, Any],
    candidate_plan: dict[str, Any],
    *,
    collection_id: str,
    baseline_episodes_root: str | Path,
    candidate_episodes_root: str | Path,
    dataset_id: str = "farpoint_so101",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if baseline.get("schema_version") != "farpoint.collection-selection.v1":
        raise ValueError("baseline must be a balanced collection selection")
    if baseline.get("quality_status") != "PASS" or len(baseline.get("attempts") or []) != 50:
        raise ValueError("baseline balanced50 evidence is not accepted")
    validate_manifest(candidate_manifest, candidate_plan)
    if candidate_manifest.get("quality_status") != "PASS":
        raise ValueError("candidate collection has not passed")
    profile = candidate_plan.get("collection") or {}
    if baseline.get("collection_id") != profile.get("source_selection_id"):
        raise ValueError("candidate plan does not reference this baseline selection")
    candidate_selection = build_export_selection(
        candidate_manifest, str(candidate_episodes_root)
    )
    baseline_attempts = baseline["attempts"]
    candidate_attempts_by_id = {
        row["attempt_id"]: row for row in candidate_manifest["attempts"]
    }
    candidate_trials = {
        trial["variation_id"]: trial for trial in candidate_plan["trials"]
    }
    source_to_candidate = {}
    for variation_id, attempt_id in candidate_manifest["selected_variations"].items():
        source_to_candidate[candidate_trials[variation_id]["source_trial_id"]] = (
            candidate_attempts_by_id[attempt_id], candidate_trials[variation_id]
        )
    baseline_ids = {row["trial_id"] for row in baseline_attempts}
    if baseline_ids != set(source_to_candidate):
        raise ValueError("candidate successes do not exactly mirror balanced50 trial identities")
    for row in baseline_attempts:
        candidate_row, candidate_trial = source_to_candidate[row["trial_id"]]
        if row["split"] != candidate_row["split"] or row["split"] != candidate_trial["split"]:
            raise ValueError(f"split mismatch for mirrored trial {row['trial_id']}")
    attempts = [copy.deepcopy(row) for row in baseline_attempts]
    attempts.extend(
        copy.deepcopy(candidate_attempts_by_id[attempt_id])
        for attempt_id in candidate_manifest["selected_variations"].values()
    )
    selected_variations = {
        row["variation_id"]: row["attempt_id"] for row in attempts
    }
    splits = Counter(row["split"] for row in attempts)
    if dict(splits) != {"train": 80, "validation": 10, "test": 10}:
        raise ValueError(f"combined split balance is invalid: {dict(splits)}")
    manifest = {
        "schema_version": "farpoint.collection-selection.v1",
        "collection_id": collection_id,
        "task_id": candidate_manifest["task_id"],
        "git_commit": candidate_manifest["git_commit"],
        "execution_status": "FINISHED",
        "quality_status": "PASS",
        "release_status": "CANDIDATE",
        "required_successes": 100,
        "maximum_attempts": 100,
        "attempts": attempts,
        "selected_variations": selected_variations,
        "selection_policy": "so101_balanced_mirrored_mass_v1",
        "balance": {
            "total": 100,
            "splits": {key: splits[key] for key in ("train", "validation", "test")},
            "mass_kg": {"0.03": 50, "0.04": 50},
            "mirrored_trial_pairs": 50,
            "workspace_cells_per_mass": 25,
            "sizes_per_mass": {"0.03": {"0.03": 25, "0.04": 25}, "0.04": {"0.03": 25, "0.04": 25}},
            "colors_per_mass": {"0.03": {"red": 25, "blue": 25}, "0.04": {"red": 25, "blue": 25}},
        },
        "source_collections": [
            {
                "collection_id": baseline["collection_id"],
                "manifest_sha256": _sha256(baseline),
                "mass_kg": 0.04,
            },
            {
                "collection_id": candidate_manifest["collection_id"],
                "manifest_sha256": _sha256(candidate_manifest),
                "plan_sha256": candidate_plan["plan_sha256"],
                "mass_kg": 0.03,
            },
        ],
    }
    baseline_entries = [
        {
            "episode_dir": str(Path(baseline_episodes_root) / row["episode_id"]),
            "trial_id": row["trial_id"],
            "variation_id": row["variation_id"],
            "split": row["split"],
        }
        for row in baseline_attempts
    ]
    selection = {
        "schema_version": "farpoint.export-selection.v1",
        "dataset_id": dataset_id,
        "collection_id": collection_id,
        "selection_policy": "so101_balanced_mirrored_mass_v1",
        "episodes": baseline_entries + candidate_selection["episodes"],
    }
    return manifest, selection
