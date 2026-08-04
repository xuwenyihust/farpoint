"""Deterministic 5x5 cube yaw plans for the v0.0.1 collection."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "farpoint.yaw-variation.v1"
PLANNER_REVISION = "cube-yaw-grid-v1"
SPLITS = ("train", "validation", "test")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def planner_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def yaw_quaternion_xyzw(yaw_degrees: float) -> list[float]:
    half = math.radians(float(yaw_degrees)) / 2.0
    return [0.0, 0.0, round(math.sin(half), 12), round(math.cos(half), 12)]


def canonical_cube_yaw_degrees(yaw_degrees: float) -> float:
    """Return a cube's upright orientation in its [0, 90) symmetry class."""
    return round(float(yaw_degrees) % 90.0, 9)


def load_yaw_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_yaw_config(config)
    return config


def validate_yaw_config(config: dict[str, Any]) -> None:
    required = {"schema_version", "plan_id", "task_id", "config_revision", "grid", "yaw_degrees", "object_spec", "frozen_factors"}
    if not isinstance(config, dict) or required - config.keys():
        raise ValueError("yaw plan config is missing required fields")
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    grid = config["grid"]
    if int(grid.get("rows", 0)) != 5 or int(grid.get("columns", 0)) != 5:
        raise ValueError("yaw plan requires a 5x5 grid")
    if int(grid.get("reserve_candidates_per_condition", -1)) != 2:
        raise ValueError("yaw plan requires two reserve candidates")
    for name in ("x_bounds_m", "y_bounds_m"):
        values = grid.get(name)
        if not isinstance(values, list) or len(values) != 2 or float(values[0]) >= float(values[1]):
            raise ValueError(f"grid.{name} must contain increasing bounds")
    yaws = config["yaw_degrees"]
    if yaws != [0.0, 15.0, 30.0, 45.0]:
        raise ValueError("v0.0.1 yaw plan requires [0, 15, 30, 45]")
    spec = config["object_spec"]
    if spec.get("object_shape") != "cube" or spec.get("object_variant_id") != "cube_55mm_red_v1":
        raise ValueError("v0.0.1 requires cube_55mm_red_v1")
    if spec.get("object_dimensions_m") != [0.055, 0.055, 0.055]:
        raise ValueError("v0.0.1 requires 55mm cube dimensions")


def _split(row: int, column: int) -> str:
    # 17/4/4 per yaw: first 17 cells train; stable interleaving for holds.
    ordinal = row * 5 + column
    return "train" if ordinal < 17 else ("validation" if ordinal < 21 else "test")


def _candidate(config: dict[str, Any], row: int, column: int, yaw: float, reserve_index: int) -> dict[str, Any]:
    grid = config["grid"]
    material = {"schema_version": SCHEMA_VERSION, "plan_id": config["plan_id"], "row": row, "column": column, "yaw_degrees": yaw, "reserve_index": reserve_index}
    seed = int.from_bytes(hashlib.sha256(_canonical(material)).digest()[:8], "big") & ((1 << 63) - 1)
    rng = random.Random(seed)
    x0, x1 = map(float, grid["x_bounds_m"])
    y0, y1 = map(float, grid["y_bounds_m"])
    width_x, width_y = (x1 - x0) / 5, (y1 - y0) / 5
    low, high = (float(value) for value in grid.get("interior_fraction", [0.1, 0.9]))
    return {
        "reserve_index": reserve_index,
        "seed": seed,
        "seed_material": material,
        "object_position_xy_m": [round(x0 + column * width_x + width_x * rng.uniform(low, high), 9), round(y0 + row * width_y + width_y * rng.uniform(low, high), 9)],
    }


def generate_yaw_plan(config: dict[str, Any]) -> dict[str, Any]:
    validate_yaw_config(config)
    trials = []
    for yaw in config["yaw_degrees"]:
        for row in range(5):
            for column in range(5):
                primary = _candidate(config, row, column, yaw, 0)
                reserves = [_candidate(config, row, column, yaw, index) for index in (1, 2)]
                trials.append({
                    "trial_id": f"yaw{int(yaw):02d}_r{row:02d}_c{column:02d}",
                    "variation_id": f"cube_55mm_yaw{int(yaw):02d}_r{row:02d}_c{column:02d}",
                    "cell_id": f"r{row:02d}_c{column:02d}", "row": row, "column": column,
                    "split": _split(row, column), "object_yaw_degrees": yaw,
                    **primary, "reserve_candidates": reserves,
                })
    manifest = {"schema_version": SCHEMA_VERSION, "plan_id": config["plan_id"], "task_id": config["task_id"], "config_revision": str(config["config_revision"]), "planner_revision": PLANNER_REVISION, "planner_sha256": planner_sha256(), "config_sha256": _sha(config), "varied_axes": ["object_initial_position_x_m", "object_initial_position_y_m", "object_initial_yaw_degrees"], "grid": copy.deepcopy(config["grid"]), "yaw_degrees": copy.deepcopy(config["yaw_degrees"]), "object_spec": copy.deepcopy(config["object_spec"]), "frozen_factors": copy.deepcopy(config["frozen_factors"]), "trials": trials}
    manifest["plan_sha256"] = _sha(manifest)
    validate_yaw_plan(manifest)
    return manifest


def validate_yaw_plan(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION or len(manifest.get("trials", [])) != 100:
        raise ValueError("yaw plan must contain 100 trials")
    supplied = manifest.get("plan_sha256")
    if supplied != _sha({key: value for key, value in manifest.items() if key != "plan_sha256"}):
        raise ValueError("plan_sha256 does not match the manifest contents")
    counts = {split: sum(trial["split"] == split for trial in manifest["trials"]) for split in SPLITS}
    if counts != {"train": 68, "validation": 16, "test": 16}:
        raise ValueError(f"invalid split counts: {counts}")
    ids = {trial["trial_id"] for trial in manifest["trials"]}
    if len(ids) != 100:
        raise ValueError("yaw trial ids must be unique")


def load_yaw_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_yaw_plan(plan)
    return plan


def resolve_yaw_trial(plan: dict[str, Any], trial_id: str, *, reserve_index: int = 0) -> dict[str, Any]:
    validate_yaw_plan(plan)
    trial = next((row for row in plan["trials"] if row["trial_id"] == trial_id), None)
    if trial is None:
        raise ValueError(f"unknown trial_id: {trial_id}")
    candidate = trial if reserve_index == 0 else next((row for row in trial["reserve_candidates"] if row["reserve_index"] == reserve_index), None)
    if candidate is None:
        raise ValueError("reserve_index must be 0, 1, or 2")
    spec = plan["object_spec"]
    values = {"object_position_m": [*candidate["object_position_xy_m"], float(plan["frozen_factors"]["object_initial_z_m"])], "object_yaw_degrees": float(trial["object_yaw_degrees"]), "object_orientation_xyzw": yaw_quaternion_xyzw(trial["object_yaw_degrees"]), "object_shape": spec["object_shape"], "object_dimensions_m": copy.deepcopy(spec["object_dimensions_m"]), "object_variant_id": spec["object_variant_id"], "grasp_profile_id": spec["grasp_profile_id"], "perception_profile_id": spec["perception_profile_id"], "appearance_profile_id": spec["appearance_profile_id"], "camera_profile_id": plan["frozen_factors"]["camera_profile_id"], "lighting_profile_id": plan["frozen_factors"]["lighting_profile_id"]}
    return {"trial_id": trial_id, "split": trial["split"], "seed": candidate["seed"], "reserve_index": reserve_index, "plan_id": plan["plan_id"], "plan_sha256": plan["plan_sha256"], "variation": {"schema_version": "farpoint.variation.v2", "variation_id": trial["variation_id"] + (f"_reserve{reserve_index}" if reserve_index else ""), "varied_axes": copy.deepcopy(plan["varied_axes"]), "frozen_axes": sorted({*plan["frozen_factors"], *spec} - {"object_dimensions_m"}), "cell_id": trial["cell_id"], "slot": 0, "split": trial["split"], "requested": copy.deepcopy(values), "resolved": copy.deepcopy(values)}}


def apply_yaw_trial(task: dict[str, Any], plan: dict[str, Any], trial_id: str, *, reserve_index: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    if task.get("name") != plan.get("task_id"):
        raise ValueError("yaw plan task does not match task")
    resolved = resolve_yaw_trial(plan, trial_id, reserve_index=reserve_index)
    configured = copy.deepcopy(task)
    obj = configured["scene"]["pick_object"]
    obj["position"] = list(resolved["variation"]["resolved"]["object_position_m"])
    obj["rotation_degrees"] = [0.0, 0.0, resolved["variation"]["resolved"]["object_yaw_degrees"]]
    configured["scene"]["target_zone"]["position"] = list(plan["frozen_factors"]["target_position_m"])
    configured["randomization"]["enabled"] = False
    return configured, resolved
