"""Selection and acceptance helpers for the v0.0.1 yaw-aware pilot."""

from __future__ import annotations

from typing import Any


PILOT_CELLS = {(0, 0), (2, 2), (4, 4)}
PILOT_YAWS = {0.0, 15.0, 30.0, 45.0}


def pilot_trials(plan: dict[str, Any]) -> list[dict[str, Any]]:
    selected = [
        trial for trial in plan["trials"]
        if (trial["row"], trial["column"]) in PILOT_CELLS
        and float(trial["object_yaw_degrees"]) in PILOT_YAWS
    ]
    if len(selected) != 12:
        raise ValueError(f"yaw pilot must contain 12 trials, found {len(selected)}")
    return sorted(selected, key=lambda row: (row["object_yaw_degrees"], row["row"], row["column"]))


def yaw_audit_accepted(metrics: dict[str, Any], *, max_error_degrees: float = 10.0) -> bool:
    audit = metrics.get("yaw_aware") or {}
    return (
        audit.get("control_source") == "rgbd_cube_yaw"
        and audit.get("alignment_stable") is True
        and float(audit.get("audit_error_degrees", float("inf"))) <= max_error_degrees
    )
