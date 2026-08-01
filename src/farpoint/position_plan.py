"""Deterministic two-dimensional position plans for Farpoint simulation trials."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "farpoint.variation.v2"
PLANNER_REVISION = "cube-position-grid-v1"
SPLITS = ("train", "validation", "test")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def planner_sha256() -> str:
    """Identify the exact planner implementation used to create a manifest."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_position_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_position_config(config)
    return config


def validate_position_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("position plan config must be an object")
    required = {
        "schema_version",
        "plan_id",
        "task_id",
        "config_revision",
        "grid",
        "frozen_factors",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"position plan config is missing fields: {sorted(missing)}")
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    grid = config["grid"]
    if not isinstance(grid, dict):
        raise ValueError("grid must be an object")
    for axis in ("x_bounds_m", "y_bounds_m"):
        bounds = grid.get(axis)
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not all(isinstance(value, (int, float)) for value in bounds)
            or float(bounds[0]) >= float(bounds[1])
        ):
            raise ValueError(f"grid.{axis} must contain increasing numeric bounds")
    if int(grid.get("rows", 0)) != 5 or int(grid.get("columns", 0)) != 5:
        raise ValueError("the v1.3 cube baseline requires a 5x5 grid")
    if int(grid.get("primary_slots_per_cell", 0)) != 3:
        raise ValueError("the v1.3 cube baseline requires 3 primary slots per cell")
    if int(grid.get("reserve_candidates_per_slot", -1)) != 2:
        raise ValueError("each primary slot must predeclare 2 reserve candidates")
    interior = grid.get("interior_fraction")
    if (
        not isinstance(interior, list)
        or len(interior) != 2
        or not 0.0 < float(interior[0]) < float(interior[1]) < 1.0
    ):
        raise ValueError("grid.interior_fraction must be two fractions inside (0, 1)")
    frozen = config["frozen_factors"]
    if not isinstance(frozen, dict) or frozen.get("object_shape") != "cube":
        raise ValueError("frozen_factors.object_shape must be cube")
    if float(frozen.get("object_yaw_degrees", float("nan"))) != 0.0:
        raise ValueError("frozen_factors.object_yaw_degrees must be 0")


def _split_for(row: int, column: int, slot: int) -> str:
    if slot in (0, 1):
        return "train"
    return "validation" if (row + column) % 2 == 0 else "test"


def _seed_material(plan_id: str, row: int, column: int, slot: int, reserve_index: int) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan_id,
        "cell_row": row,
        "cell_column": column,
        "slot": slot,
        "reserve_index": reserve_index,
    }


def _candidate(
    config: dict[str, Any], row: int, column: int, slot: int, reserve_index: int
) -> dict[str, Any]:
    grid = config["grid"]
    x_min, x_max = (float(value) for value in grid["x_bounds_m"])
    y_min, y_max = (float(value) for value in grid["y_bounds_m"])
    x_width = (x_max - x_min) / int(grid["columns"])
    y_width = (y_max - y_min) / int(grid["rows"])
    low, high = (float(value) for value in grid["interior_fraction"])
    cell_x_min = x_min + column * x_width
    cell_y_min = y_min + row * y_width
    material = _seed_material(config["plan_id"], row, column, slot, reserve_index)
    seed = int.from_bytes(hashlib.sha256(_canonical_json(material)).digest()[:8], "big") & (
        (1 << 63) - 1
    )
    rng = random.Random(seed)
    position = [
        round(cell_x_min + x_width * rng.uniform(low, high), 9),
        round(cell_y_min + y_width * rng.uniform(low, high), 9),
    ]
    suffix = f"r{row:02d}_c{column:02d}_s{slot:02d}"
    if reserve_index:
        suffix += f"_reserve{reserve_index}"
    return {
        "candidate_id": suffix,
        "reserve_index": reserve_index,
        "seed": seed,
        "seed_material": material,
        "object_position_xy_m": position,
    }


def generate_position_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Generate the immutable 75-primary v1.3 cube position manifest."""
    validate_position_config(config)
    grid = config["grid"]
    trials = []
    all_positions: set[tuple[float, float]] = set()
    for row in range(int(grid["rows"])):
        for column in range(int(grid["columns"])):
            for slot in range(int(grid["primary_slots_per_cell"])):
                primary = _candidate(config, row, column, slot, 0)
                reserves = [
                    _candidate(config, row, column, slot, reserve_index)
                    for reserve_index in range(1, int(grid["reserve_candidates_per_slot"]) + 1)
                ]
                for candidate in (primary, *reserves):
                    key = tuple(candidate["object_position_xy_m"])
                    if key in all_positions:
                        raise ValueError(f"position collision in generated plan: {key}")
                    all_positions.add(key)
                trial_id = f"primary_r{row:02d}_c{column:02d}_s{slot:02d}"
                trials.append({
                    "trial_id": trial_id,
                    "variation_id": f"position_r{row:02d}_c{column:02d}_s{slot:02d}",
                    "cell_id": f"r{row:02d}_c{column:02d}",
                    "row": row,
                    "column": column,
                    "slot": slot,
                    "split": _split_for(row, column, slot),
                    **{key: primary[key] for key in ("seed", "seed_material", "object_position_xy_m")},
                    "reserve_candidates": reserves,
                })

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "plan_id": config["plan_id"],
        "task_id": config["task_id"],
        "config_revision": str(config["config_revision"]),
        "planner_revision": PLANNER_REVISION,
        "planner_sha256": planner_sha256(),
        "config_sha256": _sha256(config),
        "dimensions": [
            {"name": "object_initial_position_x_m", "kind": "continuous_grid", "values": grid["x_bounds_m"], "unit": "m"},
            {"name": "object_initial_position_y_m", "kind": "continuous_grid", "values": grid["y_bounds_m"], "unit": "m"},
        ],
        "varied_axes": ["object_initial_position_x_m", "object_initial_position_y_m"],
        "grid": copy.deepcopy(grid),
        "frozen_factors": copy.deepcopy(config["frozen_factors"]),
        "trials": trials,
    }
    manifest["plan_sha256"] = _sha256(manifest)
    validate_position_plan(manifest)
    return manifest


def validate_position_plan(manifest: dict[str, Any]) -> None:
    """Validate plan integrity and all Goal 2 distribution invariants."""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"plan schema_version must be {SCHEMA_VERSION!r}")
    supplied_hash = manifest.get("plan_sha256")
    unhashed = {key: value for key, value in manifest.items() if key != "plan_sha256"}
    if supplied_hash != _sha256(unhashed):
        raise ValueError("plan_sha256 does not match the manifest contents")
    if manifest.get("varied_axes") != [
        "object_initial_position_x_m",
        "object_initial_position_y_m",
    ]:
        raise ValueError("only object initial X/Y may be varied")
    trials = manifest.get("trials")
    if not isinstance(trials, list) or len(trials) != 75:
        raise ValueError("position plan must contain exactly 75 primary trials")
    ids = {trial.get("trial_id") for trial in trials}
    positions = {tuple(trial.get("object_position_xy_m", ())) for trial in trials}
    if len(ids) != 75 or len(positions) != 75:
        raise ValueError("primary trial ids and positions must be unique")
    split_counts = {split: 0 for split in SPLITS}
    cell_counts: dict[str, int] = {}
    grid = manifest["grid"]
    x_min, x_max = grid["x_bounds_m"]
    y_min, y_max = grid["y_bounds_m"]
    x_width = (x_max - x_min) / grid["columns"]
    y_width = (y_max - y_min) / grid["rows"]
    low, high = grid["interior_fraction"]
    for trial in trials:
        row, column, slot = trial["row"], trial["column"], trial["slot"]
        expected_cell = f"r{row:02d}_c{column:02d}"
        if trial["cell_id"] != expected_cell or trial["split"] != _split_for(row, column, slot):
            raise ValueError(f"invalid cell or split assignment for {trial['trial_id']}")
        x, y = trial["object_position_xy_m"]
        expected_x = (x_min + column * x_width + low * x_width, x_min + column * x_width + high * x_width)
        expected_y = (y_min + row * y_width + low * y_width, y_min + row * y_width + high * y_width)
        if not expected_x[0] <= x <= expected_x[1] or not expected_y[0] <= y <= expected_y[1]:
            raise ValueError(f"position lies outside the interior of {expected_cell}")
        if len(trial.get("reserve_candidates", [])) != 2:
            raise ValueError(f"{trial['trial_id']} must declare two reserve candidates")
        split_counts[trial["split"]] += 1
        cell_counts[expected_cell] = cell_counts.get(expected_cell, 0) + 1
    if split_counts != {"train": 50, "validation": 13, "test": 12}:
        raise ValueError(f"invalid split counts: {split_counts}")
    if len(cell_counts) != 25 or set(cell_counts.values()) != {3}:
        raise ValueError("all 25 cells must contain exactly 3 primary slots")


def load_position_plan(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_position_plan(manifest)
    return manifest


def resolve_position_trial(
    manifest: dict[str, Any], trial_id: str, *, reserve_index: int = 0
) -> dict[str, Any]:
    """Resolve a primary or predeclared reserve into episode variation metadata."""
    validate_position_plan(manifest)
    trial = next((item for item in manifest["trials"] if item["trial_id"] == trial_id), None)
    if trial is None:
        raise ValueError(f"unknown trial_id: {trial_id}")
    if reserve_index == 0:
        candidate = trial
    else:
        candidate = next(
            (item for item in trial["reserve_candidates"] if item["reserve_index"] == reserve_index),
            None,
        )
        if candidate is None:
            raise ValueError(f"reserve_index must be 0, 1, or 2 for {trial_id}")
    frozen = manifest["frozen_factors"]
    z = float(frozen["object_initial_z_m"])
    values = {
        "object_position_m": [*candidate["object_position_xy_m"], z],
        "object_yaw_degrees": float(frozen["object_yaw_degrees"]),
        "object_shape": frozen["object_shape"],
        "object_dimensions_m": copy.deepcopy(frozen["object_dimensions_m"]),
        "appearance_profile_id": frozen["appearance_profile_id"],
        "camera_profile_id": frozen["camera_profile_id"],
        "lighting_profile_id": frozen["lighting_profile_id"],
    }
    variation_id = trial["variation_id"]
    if reserve_index:
        variation_id += f"_reserve{reserve_index}"
    return {
        "trial_id": trial_id,
        "split": trial["split"],
        "seed": candidate["seed"],
        "reserve_index": reserve_index,
        "plan_id": manifest["plan_id"],
        "plan_sha256": manifest["plan_sha256"],
        "variation": {
            "schema_version": SCHEMA_VERSION,
            "variation_id": variation_id,
            "varied_axes": copy.deepcopy(manifest["varied_axes"]),
            "frozen_axes": sorted(frozen),
            "cell_id": trial["cell_id"],
            "slot": trial["slot"],
            "requested": copy.deepcopy(values),
            "resolved": copy.deepcopy(values),
        },
    }


def apply_position_trial(
    task: dict[str, Any], manifest: dict[str, Any], trial_id: str, *, reserve_index: int = 0
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one manifest trial to a task while freezing every non-position factor."""
    if manifest["task_id"] != task.get("name"):
        raise ValueError(
            f"position plan task_id {manifest['task_id']!r} does not match task {task.get('name')!r}"
        )
    resolved = resolve_position_trial(manifest, trial_id, reserve_index=reserve_index)
    configured = copy.deepcopy(task)
    configured["scene"]["pick_object"]["position"] = list(
        resolved["variation"]["resolved"]["object_position_m"]
    )
    configured["scene"]["target_zone"]["position"] = list(
        manifest["frozen_factors"]["target_position_m"]
    )
    configured["randomization"]["enabled"] = False
    return configured, resolved
