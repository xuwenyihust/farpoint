"""Deterministic balanced subset selection for successful SO-101 episodes."""

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


SCHEMA_VERSION = "farpoint.collection-selection.v1"
POLICY_ID = "so101_balanced_stratified_subset_v1"
SPLIT_ORDER = ("train", "validation", "test")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _labels(trial: dict[str, Any]) -> tuple[str, str, str]:
    material = trial.get("seed_material") or {}
    cell = trial.get("cell_id")
    if not cell:
        row = material.get("row")
        column = material.get("column")
        if row is None or column is None:
            raise ValueError(f"trial {trial.get('trial_id')} has no workspace cell")
        cell = f"r{int(row):02d}_c{int(column):02d}"
    size = material.get("size_index")
    color = material.get("color_index")
    if size is None or color is None:
        raise ValueError(f"trial {trial.get('trial_id')} has no size/color indexes")
    return str(cell), f"size_{int(size)}", f"color_{int(color)}"


def _quota(total: int, counts: Counter[str]) -> dict[str, int]:
    population = sum(counts.values())
    exact = {key: total * value / population for key, value in counts.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remainder = total - sum(quotas.values())
    for key in sorted(counts, key=lambda name: (-(exact[name] - quotas[name]), name)):
        if remainder == 0:
            break
        quotas[key] += 1
        remainder -= 1
    return quotas


def _selection_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    splits = Counter(row["split"] for row in rows)
    cells = Counter(row["cell_id"] for row in rows)
    sizes = Counter(row["size_label"] for row in rows)
    colors = Counter(row["color_label"] for row in rows)
    joints = Counter(f"{row['size_label']}__{row['color_label']}" for row in rows)
    workspace_rows = Counter(cell.split("_")[0] for cell in cells for _ in range(cells[cell]))
    workspace_columns = Counter(cell.split("_")[1] for cell in cells for _ in range(cells[cell]))
    return {
        "total": len(rows),
        "splits": dict(sorted(splits.items())),
        "workspace_cells": dict(sorted(cells.items())),
        "workspace_rows": dict(sorted(workspace_rows.items())),
        "workspace_columns": dict(sorted(workspace_columns.items())),
        "sizes": dict(sorted(sizes.items())),
        "colors": dict(sorted(colors.items())),
        "size_color": dict(sorted(joints.items())),
    }


def _score(rows: list[dict[str, Any]], all_cells: set[str]) -> int:
    stats = _selection_stats(rows)
    total = len(rows)

    def even_penalty(values: dict[str, int], categories: int) -> int:
        expected = total / categories
        return round(sum(abs(value - expected) for value in values.values()) * 2)

    primary = even_penalty(stats["sizes"], 2) + even_penalty(stats["colors"], 2)
    joint_values = list(stats["size_color"].values())
    joint_range = max(joint_values) - min(joint_values) if joint_values else total
    missing_cells = len(all_cells.difference(stats["workspace_cells"]))
    row_column = even_penalty(stats["workspace_rows"], 5) + even_penalty(
        stats["workspace_columns"], 5
    )
    expected_cell = total / len(all_cells)
    cell_penalty = round(
        sum(abs(stats["workspace_cells"].get(cell, 0) - expected_cell) for cell in all_cells)
        * 2
    )
    return (
        primary * 100_000
        + max(0, joint_range - 1) * 20_000
        + missing_cells * 10_000
        + row_column * 100
        + cell_penalty
    )


def select_balanced_attempts(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    *,
    target_count: int = 50,
    seed: int = 101,
    iterations: int = 250_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose a reproducible subset while preserving proportional split quotas."""
    trials = {trial["variation_id"]: trial for trial in plan.get("trials") or []}
    candidates = []
    for attempt in manifest.get("attempts") or []:
        if not (attempt.get("success") and attempt.get("dataset_valid")):
            continue
        trial = trials.get(attempt.get("variation_id"))
        if trial is None:
            raise ValueError(f"successful attempt has no planned variation: {attempt.get('attempt_id')}")
        cell, size, color = _labels(trial)
        candidates.append(
            {
                **copy.deepcopy(attempt),
                "cell_id": cell,
                "size_label": size,
                "color_label": color,
            }
        )
    candidates.sort(key=lambda row: row["trial_id"])
    if len(candidates) < target_count:
        raise ValueError(f"only {len(candidates)} eligible successes for target {target_count}")
    plan_splits = Counter(trial["split"] for trial in plan.get("trials") or [])
    quotas = _quota(target_count, plan_splits)
    by_split = {
        split: [row for row in candidates if row["split"] == split] for split in SPLIT_ORDER
    }
    for split, required in quotas.items():
        if len(by_split.get(split, ())) < required:
            raise ValueError(
                f"split {split} has {len(by_split.get(split, ()))} eligible successes; {required} required"
            )
    all_cells = {str(trial.get("cell_id") or _labels(trial)[0]) for trial in plan["trials"]}
    rng = random.Random(seed)
    selected = []
    for split in SPLIT_ORDER:
        selected.extend(rng.sample(by_split[split], quotas.get(split, 0)))
    best = list(selected)
    best_score = _score(best, all_cells)
    current_score = best_score
    selected_ids = {row["attempt_id"] for row in selected}
    for step in range(iterations):
        split = SPLIT_ORDER[rng.randrange(len(SPLIT_ORDER))]
        chosen = [row for row in selected if row["split"] == split]
        available = [row for row in by_split[split] if row["attempt_id"] not in selected_ids]
        if not chosen or not available:
            continue
        outgoing = chosen[rng.randrange(len(chosen))]
        incoming = available[rng.randrange(len(available))]
        index = selected.index(outgoing)
        selected[index] = incoming
        proposal_score = _score(selected, all_cells)
        # A single swap may temporarily disturb an exactly balanced marginal
        # while improving its paired size/color group on the next swap.  The
        # restart cycle must therefore cross the 20k joint-balance barrier.
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
        else:
            selected[index] = outgoing
    # Finish with deterministic steepest descent.  Annealing finds the basin;
    # this pass removes residual one-row or one-column imbalances without
    # allowing a later hot restart to move away from the best candidate.
    improved = True
    while improved:
        improved = False
        best_swap = None
        best_swap_score = best_score
        best_ids = {row["attempt_id"] for row in best}
        for index, outgoing in enumerate(best):
            for incoming in by_split[outgoing["split"]]:
                if incoming["attempt_id"] in best_ids:
                    continue
                proposal = list(best)
                proposal[index] = incoming
                proposal_score = _score(proposal, all_cells)
                swap_key = (outgoing["attempt_id"], incoming["attempt_id"])
                if proposal_score < best_swap_score or (
                    proposal_score == best_swap_score
                    and best_swap is not None
                    and swap_key < best_swap[0]
                ):
                    best_swap_score = proposal_score
                    best_swap = (swap_key, index, incoming)
        if best_swap is not None and best_swap_score < best_score:
            _, index, incoming = best_swap
            best[index] = incoming
            best_score = best_swap_score
            improved = True
    best.sort(key=lambda row: (SPLIT_ORDER.index(row["split"]), row["trial_id"]))
    stats = _selection_stats(best)
    stats["split_quotas"] = {key: quotas[key] for key in SPLIT_ORDER}
    stats["score"] = best_score
    return best, stats


def validate_balance(stats: dict[str, Any], *, target_count: int = 50) -> list[str]:
    errors = []
    expected_splits = {"train": 40, "validation": 5, "test": 5}
    if target_count == 50 and stats.get("splits") != expected_splits:
        errors.append(f"split counts are not {expected_splits}: {stats.get('splits')}")
    if stats.get("total") != target_count:
        errors.append(f"selected {stats.get('total')} episodes instead of {target_count}")
    for axis in ("sizes", "colors"):
        values = list((stats.get(axis) or {}).values())
        if len(values) != 2 or max(values) - min(values) != 0:
            errors.append(f"{axis} are not evenly balanced: {stats.get(axis)}")
    joint = list((stats.get("size_color") or {}).values())
    if len(joint) != 4 or max(joint) - min(joint) > 1:
        errors.append(f"size/color combinations differ by more than one: {stats.get('size_color')}")
    if len(stats.get("workspace_cells") or {}) != 25:
        errors.append("selection does not cover all 25 workspace cells")
    for axis in ("workspace_rows", "workspace_columns"):
        values = list((stats.get(axis) or {}).values())
        if len(values) != 5 or len(set(values)) != 1:
            errors.append(f"{axis} are not evenly balanced: {stats.get(axis)}")
    return errors


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
    attempts = []
    episodes = []
    for row in selected:
        attempt = {key: copy.deepcopy(value) for key, value in row.items() if key not in {"size_label", "color_label"}}
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
