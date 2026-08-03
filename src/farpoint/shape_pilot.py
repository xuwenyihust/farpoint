"""Frozen sentinel selection and acceptance for shape-position pilots."""

from __future__ import annotations

from collections import Counter
from typing import Any


SENTINEL_CELLS = ("r00_c00", "r00_c04", "r02_c02", "r04_c00", "r04_c04")


def pilot_trials(plan: dict[str, Any]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in plan["trials"]
        if row["cell_id"] in SENTINEL_CELLS and row["slot"] < 3
    ]
    if len(selected) != 15:
        raise ValueError("shape pilot requires three frozen candidates in five sentinel cells")
    return sorted(selected, key=lambda row: (row["slot"], row["row"], row["column"]))


def scheduled_pilot_trials(
    plan: dict[str, Any], attempts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    completed = {row["trial_id"] for row in attempts}
    covered = {
        row["cell_id"]
        for row in attempts
        if row.get("success") is True and row.get("dataset_valid") is True
    }
    return [
        row
        for row in pilot_trials(plan)
        if row["trial_id"] not in completed and row["cell_id"] not in covered
    ]


def pilot_acceptance(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        row for row in attempts if row.get("success") is True and row.get("dataset_valid") is True
    ]
    per_cell = dict(sorted(Counter(row["cell_id"] for row in successful).items()))
    accepted = set(per_cell) == set(SENTINEL_CELLS) and set(per_cell.values()) == {1}
    return {
        "accepted": accepted,
        "required_cells": list(SENTINEL_CELLS),
        "observed_cells": sorted(per_cell),
        "successful_episodes": len(successful),
        "task_attempts": len(attempts),
    }


def impossible_pilot_cell(plan: dict[str, Any], attempts: list[dict[str, Any]]) -> str | None:
    covered = set(pilot_acceptance(attempts)["observed_cells"])
    remaining = Counter(row["cell_id"] for row in scheduled_pilot_trials(plan, attempts))
    for cell in SENTINEL_CELLS:
        if cell not in covered and remaining[cell] == 0:
            return cell
    return None
