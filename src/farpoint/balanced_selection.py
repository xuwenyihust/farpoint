"""Configuration-driven balanced selection for successful collection attempts."""

from __future__ import annotations

import copy
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


POLICY_SCHEMA_VERSION = "farpoint.balanced-selection-policy.v1"


def load_selection_policy(path: str | Path) -> dict[str, Any]:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_selection_policy(policy)
    return policy


def validate_selection_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError(f"selection policy must use {POLICY_SCHEMA_VERSION}")
    if not isinstance(policy.get("policy_id"), str) or not policy["policy_id"]:
        raise ValueError("selection policy requires policy_id")
    target_count = policy.get("target_count")
    if not isinstance(target_count, int) or isinstance(target_count, bool) or target_count <= 0:
        raise ValueError("selection target_count must be a positive integer")
    split_targets = policy.get("split_targets")
    if (
        not isinstance(split_targets, dict)
        or not split_targets
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for name, value in split_targets.items()
        )
        or sum(split_targets.values()) != target_count
    ):
        raise ValueError("split_targets must sum to target_count")
    labels = policy.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError("selection policy requires labels")
    for name, definition in labels.items():
        if not name or not isinstance(definition, dict) or not definition.get("path"):
            raise ValueError("every selection label requires a path")
        if definition.get("transform") not in {None, "cell_row", "cell_column"}:
            raise ValueError(f"unsupported label transform: {definition.get('transform')}")
    known_keys = set(labels) | set(policy.get("joints") or {})
    for name, axes in (policy.get("joints") or {}).items():
        if (
            not name
            or not isinstance(axes, list)
            or not axes
            or any(axis not in labels for axis in axes)
        ):
            raise ValueError("selection joints must reference one or more labels")
    for constraint in policy.get("constraints") or []:
        if constraint.get("kind") not in {"counts", "balanced", "coverage", "range"}:
            raise ValueError(f"unsupported selection constraint: {constraint.get('kind')}")
        if constraint.get("key") not in known_keys:
            raise ValueError(
                f"selection constraint references unknown key: {constraint.get('key')}"
            )
    for objective in policy.get("objectives") or []:
        if objective.get("key") not in known_keys:
            raise ValueError(f"selection objective references unknown key: {objective.get('key')}")


def _path_value(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"selection path is missing: {path}")
        value = value[part]
    return value


def _label_value(trial: dict[str, Any], definition: dict[str, Any]) -> str:
    value = _path_value(trial, str(definition["path"]))
    transform = definition.get("transform")
    if transform == "cell_row":
        value = str(value).split("_")[0]
    elif transform == "cell_column":
        value = str(value).split("_")[1]
    elif transform is not None:
        raise ValueError(f"unsupported label transform: {transform}")
    template = definition.get("template")
    if template:
        return str(template).format(value=value)
    number_format = definition.get("number_format")
    if number_format:
        return format(float(value), str(number_format))
    return str(value)


def candidate_rows(
    manifest: dict[str, Any], plan: dict[str, Any], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    validate_selection_policy(policy)
    trials = {trial["variation_id"]: trial for trial in plan.get("trials") or []}
    selected_variations = manifest.get("selected_variations") or {}
    require_selected = bool(policy.get("require_selected_variation", False))
    rows = []
    for attempt in manifest.get("attempts") or []:
        if not (attempt.get("success") and attempt.get("dataset_valid")):
            continue
        if require_selected and selected_variations.get(attempt.get("variation_id")) != attempt.get(
            "attempt_id"
        ):
            continue
        trial = trials.get(attempt.get("variation_id"))
        if trial is None:
            raise ValueError(
                f"eligible attempt has no planned variation: {attempt.get('attempt_id')}"
            )
        row = copy.deepcopy(attempt)
        for output, definition in policy["labels"].items():
            row[output] = _label_value(trial, definition)
        rows.append(row)
    return sorted(rows, key=lambda row: row["trial_id"])


def selection_stats(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "total": len(rows),
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
    }
    for output in policy["labels"]:
        stats[output] = dict(sorted(Counter(row[output] for row in rows).items()))
    for output, axes in (policy.get("joints") or {}).items():
        stats[output] = dict(
            sorted(Counter("__".join(row[axis] for axis in axes) for row in rows).items())
        )
    summary = policy.get("coverage_summary") or {}
    if summary:
        observed = set(stats.get(summary["axis"], {}))
        universe = set(summary.get("universe") or [])
        stats["covered_cell_count"] = len(observed)
        stats["missing_cells"] = sorted(universe - observed)
    return stats


def _constraint_error(stats: dict[str, Any], constraint: dict[str, Any]) -> str | None:
    key = str(constraint["key"])
    values = stats.get(key) or {}
    kind = constraint["kind"]
    if kind == "counts":
        expected = constraint["targets"]
        if values != expected:
            return f"{key}_mismatch:{values!r}"
    elif kind == "balanced":
        categories = list(constraint["categories"])
        counts = [int(values.get(category, 0)) for category in categories]
        if set(values) != set(categories) or max(counts) - min(counts) > int(
            constraint.get("max_difference", 0)
        ):
            return f"{key}_not_balanced:{values!r}"
    elif kind == "coverage":
        required = set(constraint.get("required") or [])
        excluded = set(constraint.get("excluded") or [])
        observed = set(values)
        if not required.issubset(observed) or observed.intersection(excluded):
            return f"{key}_coverage_mismatch:{sorted(observed)!r}"
    elif kind == "range":
        minimum = int(constraint.get("minimum", 0))
        maximum = int(constraint.get("maximum", 2**63 - 1))
        if any(not minimum <= int(value) <= maximum for value in values.values()):
            return f"{key}_range_mismatch:{values!r}"
    return None


def validate_balance(stats: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    errors = []
    if stats.get("total") != int(policy["target_count"]):
        errors.append(f"total_mismatch:{stats.get('total')!r}")
    if stats.get("splits") != policy["split_targets"]:
        errors.append(f"splits_mismatch:{stats.get('splits')!r}")
    for constraint in policy.get("constraints") or []:
        error = _constraint_error(stats, constraint)
        if error:
            errors.append(error)
    return errors


def _score(stats: dict[str, Any], policy: dict[str, Any]) -> int:
    score = 0.0
    for objective in policy.get("objectives") or []:
        values = stats.get(objective["key"]) or {}
        weight = float(objective.get("weight", 1))
        if "targets" in objective:
            targets = objective["targets"]
            score += weight * sum(
                abs(int(values.get(key, 0)) - int(target)) for key, target in targets.items()
            )
            score += weight * sum(int(value) for key, value in values.items() if key not in targets)
        elif "required" in objective:
            required = set(objective["required"])
            score += weight * len(required.symmetric_difference(values))
        elif "categories" in objective:
            categories = list(objective["categories"])
            expected = sum(values.values()) / len(categories)
            score += weight * sum(abs(int(values.get(key, 0)) - expected) for key in categories)
    return round(score)


def select_balanced(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    policy: dict[str, Any],
    *,
    seed: int | None = None,
    iterations: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic constrained subset described entirely by policy."""
    candidates = candidate_rows(manifest, plan, policy)
    target_count = int(policy["target_count"])
    if len(candidates) < target_count:
        raise ValueError(f"only {len(candidates)} eligible successes for target {target_count}")
    split_targets = {str(key): int(value) for key, value in policy["split_targets"].items()}
    by_split = {
        split: [row for row in candidates if row["split"] == split] for split in split_targets
    }
    for split, target in split_targets.items():
        if len(by_split[split]) < target:
            raise ValueError(
                f"split {split} has {len(by_split[split])} eligible episodes; {target} required"
            )
    rng = random.Random(int(policy.get("seed", 0) if seed is None else seed))
    selected = []
    for split, target in split_targets.items():
        selected.extend(rng.sample(by_split[split], target))
    selected_ids = {row["attempt_id"] for row in selected}
    current_score = _score(selection_stats(selected, policy), policy)
    best = list(selected)
    best_score = current_score
    mutable_splits = list(policy.get("mutable_splits") or split_targets)
    iteration_count = int(policy.get("iterations", 100_000) if iterations is None else iterations)
    for step in range(iteration_count):
        available_splits = [
            split
            for split in mutable_splits
            if any(row["attempt_id"] not in selected_ids for row in by_split[split])
        ]
        if not available_splits:
            break
        split = available_splits[rng.randrange(len(available_splits))]
        chosen_indexes = [index for index, row in enumerate(selected) if row["split"] == split]
        index = chosen_indexes[rng.randrange(len(chosen_indexes))]
        outgoing = selected[index]
        available = [row for row in by_split[split] if row["attempt_id"] not in selected_ids]
        incoming = available[rng.randrange(len(available))]
        selected[index] = incoming
        proposal_score = _score(selection_stats(selected, policy), policy)
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
    order = {name: index for index, name in enumerate(split_targets)}
    best.sort(key=lambda row: (order[row["split"]], row["trial_id"]))
    stats = selection_stats(best, policy)
    errors = validate_balance(stats, policy)
    if errors:
        raise ValueError("balanced selection did not satisfy policy: " + "; ".join(errors))
    stats["selection_seed"] = int(policy.get("seed", 0) if seed is None else seed)
    stats["selection_score"] = best_score
    return best, stats
