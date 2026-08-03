"""Deterministic shape-aware tabletop position plans."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "farpoint.position-plan.v1"
VARIATION_SCHEMA_VERSION = "farpoint.variation.v2"
PLANNER_REVISION = "shape-position-grid-v1"
SUPPORTED_SHAPES = {"cube", "cylinder"}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def planner_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_shape_position_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_shape_position_config(config)
    return config


def validate_shape_position_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "plan_id",
        "task_id",
        "task_template_id",
        "language_instruction",
        "config_revision",
        "grid",
        "frozen_factors",
    }
    if not isinstance(config, dict) or required - config.keys():
        raise ValueError("shape position config is missing required fields")
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    grid = config["grid"]
    if int(grid.get("rows", 0)) != 5 or int(grid.get("columns", 0)) != 5:
        raise ValueError("shape position collection requires a 5x5 grid")
    if int(grid.get("candidates_per_cell", 0)) != 6:
        raise ValueError("shape position collection requires six candidates per cell")
    for axis in ("x_bounds_m", "y_bounds_m"):
        bounds = grid.get(axis)
        if not isinstance(bounds, list) or len(bounds) != 2 or bounds[0] >= bounds[1]:
            raise ValueError(f"grid.{axis} must contain increasing bounds")
    interior = grid.get("interior_fraction")
    if not isinstance(interior, list) or len(interior) != 2 or not 0 < interior[0] < interior[1] < 1:
        raise ValueError("grid.interior_fraction must lie inside (0, 1)")
    frozen = config["frozen_factors"]
    if frozen.get("object_shape") not in SUPPORTED_SHAPES:
        raise ValueError("unsupported frozen object shape")
    dimensions = frozen.get("object_dimensions_m")
    if not isinstance(dimensions, list) or len(dimensions) != 3 or min(dimensions) <= 0:
        raise ValueError("object_dimensions_m must contain three positive values")
    if frozen["object_shape"] == "cylinder":
        radius = float(frozen.get("object_radius_m", 0))
        height = float(frozen.get("object_height_m", 0))
        if radius <= 0 or height <= 0:
            raise ValueError("cylinder radius and height must be positive")
        if dimensions != [2 * radius, 2 * radius, height]:
            raise ValueError("cylinder dimensions must equal diameter, diameter, height")
    words = set(config["language_instruction"].lower().replace("-", " ").split())
    if frozen["object_shape"] not in words:
        raise ValueError("language instruction must name the object shape")


def _split_for(row: int, column: int) -> str:
    validation = {(0, 0), (0, 4), (4, 0), (4, 4)}
    test = {(0, 2), (2, 0), (2, 2), (2, 4)}
    if (row, column) in validation:
        return "validation"
    if (row, column) in test:
        return "test"
    return "train"


def _candidate(config: dict[str, Any], row: int, column: int, slot: int) -> dict[str, Any]:
    grid = config["grid"]
    material = {
        "schema_version": SCHEMA_VERSION,
        "plan_id": config["plan_id"],
        "cell_row": row,
        "cell_column": column,
        "slot": slot,
    }
    seed = int.from_bytes(hashlib.sha256(_canonical_json(material)).digest()[:8], "big") & (
        (1 << 63) - 1
    )
    rng = random.Random(seed)
    x_min, x_max = map(float, grid["x_bounds_m"])
    y_min, y_max = map(float, grid["y_bounds_m"])
    x_width = (x_max - x_min) / grid["columns"]
    y_width = (y_max - y_min) / grid["rows"]
    low, high = map(float, grid["interior_fraction"])
    position = [
        round(x_min + column * x_width + x_width * rng.uniform(low, high), 9),
        round(y_min + row * y_width + y_width * rng.uniform(low, high), 9),
    ]
    shape = config["frozen_factors"]["object_shape"]
    return {
        "trial_id": f"{shape}_r{row:02d}_c{column:02d}_s{slot:02d}",
        "variation_id": f"{shape}_position_r{row:02d}_c{column:02d}_s{slot:02d}",
        "cell_id": f"r{row:02d}_c{column:02d}",
        "row": row,
        "column": column,
        "slot": slot,
        "split": _split_for(row, column),
        "seed": seed,
        "seed_material": material,
        "object_position_xy_m": position,
    }


def generate_shape_position_plan(config: dict[str, Any]) -> dict[str, Any]:
    validate_shape_position_config(config)
    grid = config["grid"]
    trials = [
        _candidate(config, row, column, slot)
        for slot in range(grid["candidates_per_cell"])
        for row in range(grid["rows"])
        for column in range(grid["columns"])
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "plan_id": config["plan_id"],
        "task_id": config["task_id"],
        "task_template_id": config["task_template_id"],
        "language_instruction": config["language_instruction"],
        "config_revision": str(config["config_revision"]),
        "planner_revision": PLANNER_REVISION,
        "planner_sha256": planner_sha256(),
        "config_sha256": _sha256(config),
        "varied_axes": ["object_initial_position_x_m", "object_initial_position_y_m"],
        "grid": copy.deepcopy(grid),
        "frozen_factors": copy.deepcopy(config["frozen_factors"]),
        "trials": trials,
    }
    manifest["plan_sha256"] = _sha256(manifest)
    validate_shape_position_plan(manifest)
    return manifest


def validate_shape_position_plan(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"plan schema_version must be {SCHEMA_VERSION}")
    supplied = manifest.get("plan_sha256")
    if supplied != _sha256({key: value for key, value in manifest.items() if key != "plan_sha256"}):
        raise ValueError("plan_sha256 does not match the manifest contents")
    trials = manifest.get("trials") or []
    if len(trials) != 150:
        raise ValueError("shape position plan must contain 150 candidates")
    if len({row["trial_id"] for row in trials}) != 150:
        raise ValueError("trial ids must be unique")
    if len({row["seed"] for row in trials}) != 150:
        raise ValueError("candidate seeds must be unique")
    if len({tuple(row["object_position_xy_m"]) for row in trials}) != 150:
        raise ValueError("candidate positions must be unique")
    counts = {split: sum(row["split"] == split for row in trials) for split in ("train", "validation", "test")}
    if counts != {"train": 102, "validation": 24, "test": 24}:
        raise ValueError(f"invalid candidate split counts: {counts}")
    per_cell = {cell: sum(row["cell_id"] == cell for row in trials) for cell in {row["cell_id"] for row in trials}}
    if len(per_cell) != 25 or set(per_cell.values()) != {6}:
        raise ValueError("all 25 cells must contain six candidates")


def load_shape_position_plan(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_shape_position_plan(manifest)
    return manifest


def resolve_shape_position_trial(manifest: dict[str, Any], trial_id: str) -> dict[str, Any]:
    validate_shape_position_plan(manifest)
    trial = next((row for row in manifest["trials"] if row["trial_id"] == trial_id), None)
    if trial is None:
        raise ValueError(f"unknown trial_id: {trial_id}")
    frozen = manifest["frozen_factors"]
    values = {
        "object_position_m": [*trial["object_position_xy_m"], float(frozen["object_initial_z_m"])],
        "object_yaw_degrees": float(frozen["object_yaw_degrees"]),
        "object_shape": frozen["object_shape"],
        "object_dimensions_m": copy.deepcopy(frozen["object_dimensions_m"]),
        "appearance_profile_id": frozen["appearance_profile_id"],
        "camera_profile_id": frozen["camera_profile_id"],
        "lighting_profile_id": frozen["lighting_profile_id"],
    }
    return {
        "trial_id": trial_id,
        "split": trial["split"],
        "seed": trial["seed"],
        "reserve_index": 0,
        "plan_id": manifest["plan_id"],
        "plan_sha256": manifest["plan_sha256"],
        "variation": {
            "schema_version": VARIATION_SCHEMA_VERSION,
            "variation_id": trial["variation_id"],
            "varied_axes": copy.deepcopy(manifest["varied_axes"]),
            "frozen_axes": sorted(frozen),
            "cell_id": trial["cell_id"],
            "slot": trial["slot"],
            "requested": copy.deepcopy(values),
            "resolved": copy.deepcopy(values),
        },
    }


def apply_shape_position_trial(task: dict[str, Any], manifest: dict[str, Any], trial_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if task.get("name") != manifest["task_template_id"]:
        raise ValueError("shape position task template does not match the loaded task")
    resolved = resolve_shape_position_trial(manifest, trial_id)
    configured = copy.deepcopy(task)
    configured["name"] = manifest["task_id"]
    configured["language_instruction"] = manifest["language_instruction"]
    configured["scene"]["pick_object"]["position"] = list(resolved["variation"]["resolved"]["object_position_m"])
    configured["scene"]["pick_object"]["scale"] = list(manifest["frozen_factors"]["object_dimensions_m"])
    configured["scene"]["pick_object"]["cylinder_radius_scale"] = 0.5
    configured["scene"]["target_zone"]["position"] = list(manifest["frozen_factors"]["target_position_m"])
    for key, value in (manifest["frozen_factors"].get("pickup_overrides") or {}).items():
        configured["pickup"][key] = copy.deepcopy(value)
    configured["randomization"]["enabled"] = False
    return configured, resolved
