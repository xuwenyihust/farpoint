"""Dependency-light contracts and safety metrics for closed-loop policy rollout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from farpoint.contracts import validate_contract
from farpoint.so101 import LEROBOT_MAX, LEROBOT_MIN


def json_default(value: Any) -> Any:
    """Convert NumPy scalar values emitted by simulation into JSON scalars."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def load_rollout_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_contract(payload)
    if errors:
        raise ValueError("invalid policy rollout contract:\n" + "\n".join(errors))
    scene_ids = [scene["scene_id"] for scene in payload["scenes"]]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("policy rollout scene_id values must be unique")
    acceptance = payload["acceptance"]
    if acceptance["required_completed_episodes"] != len(scene_ids):
        raise ValueError("required_completed_episodes must equal the frozen scene count")
    if acceptance["minimum_task_successes"] > len(scene_ids):
        raise ValueError("minimum_task_successes exceeds the frozen scene count")
    if payload["control"]["physics_hz"] % payload["control"]["policy_hz"] != 0:
        raise ValueError("physics_hz must be divisible by policy_hz")
    if payload["task"]["evaluation_class"].startswith("independent_holdout"):
        source = payload.get("holdout_source")
        if source is None:
            raise ValueError("independent holdout rollout requires holdout_source")
        if source["evaluated_scene_count"] != len(scene_ids):
            raise ValueError("evaluated holdout scene count does not match scenes")
    return payload


def constrain_policy_action(
    raw_action: Any,
    current_position: Any,
    *,
    max_delta: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(raw_action, dtype=np.float64)
    current = np.asarray(current_position, dtype=np.float64)
    if raw.shape != (6,) or current.shape != (6,):
        raise ValueError("SO-101 rollout actions and positions must have shape (6,)")
    if not np.all(np.isfinite(raw)):
        raise ValueError("SO-101 rollout action contains non-finite values")
    if not np.all(np.isfinite(current)):
        raise ValueError("SO-101 current position contains non-finite values")
    hard_mask = (raw < LEROBOT_MIN) | (raw > LEROBOT_MAX)
    hard_clipped = np.clip(raw, LEROBOT_MIN, LEROBOT_MAX)
    delta = np.clip(hard_clipped - current, -max_delta, max_delta)
    applied = current + delta
    diagnostics = {
        "hard_range_violation_count": int(np.count_nonzero(hard_mask)),
        "delta_limited_count": int(np.count_nonzero(np.abs(hard_clipped - current) > max_delta)),
        "maximum_raw_abs": float(np.max(np.abs(raw))),
        "maximum_applied_delta": float(np.max(np.abs(delta))),
    }
    return applied.astype(np.float32), diagnostics


def evaluate_rollout_acceptance(
    spec: dict[str, Any], episode_results: list[dict[str, Any]]
) -> dict[str, Any]:
    acceptance = spec["acceptance"]
    expected_ids = [scene["scene_id"] for scene in spec["scenes"]]
    actual_ids = [result["scene_id"] for result in episode_results]
    errors = []
    if actual_ids != expected_ids:
        errors.append("episode results do not match the frozen scene order")
    completed = sum(result.get("execution_status") == "FINISHED" for result in episode_results)
    successes = sum(bool(result.get("task_success")) for result in episode_results)
    nonfinite = sum(int(result.get("nonfinite_action_count", 0)) for result in episode_results)
    range_violations = sum(
        int(result.get("hard_range_violation_count", 0)) for result in episode_results
    )
    delta_limited = sum(int(result.get("delta_limited_count", 0)) for result in episode_results)
    if completed != acceptance["required_completed_episodes"]:
        errors.append("completed episode count does not meet the smoke contract")
    if successes < acceptance["minimum_task_successes"]:
        errors.append("task success count is below the frozen minimum")
    if nonfinite > acceptance["maximum_nonfinite_actions"]:
        errors.append("non-finite action count exceeds the frozen maximum")
    if range_violations > acceptance["maximum_hard_range_violations"]:
        errors.append("hard-range action violations exceed the frozen maximum")
    stage_names = (
        "ever_cube_contact",
        "ever_bilateral_contact",
        "ever_lifted",
        "ever_entered_target",
    )
    stage_progress = {
        name: sum(
            bool((result.get("stage_evidence") or {}).get(name)) for result in episode_results
        )
        for name in stage_names
    }
    terminal_reasons: dict[str, int] = {}
    for result in episode_results:
        reason = str(result.get("terminal_reason", "unknown"))
        terminal_reasons[reason] = terminal_reasons.get(reason, 0) + 1
    return {
        "status": "PASS" if not errors else "FAIL",
        "acceptance_errors": errors,
        "completed_episodes": completed,
        "task_successes": successes,
        "task_success_rate": successes / len(expected_ids),
        "nonfinite_action_count": nonfinite,
        "hard_range_violation_count": range_violations,
        "delta_limited_count": delta_limited,
        "stage_progress": stage_progress,
        "terminal_reason_counts": dict(sorted(terminal_reasons.items())),
    }
