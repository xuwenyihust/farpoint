"""Frozen repeatability-gate plans for SO-101 simulation evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from farpoint.object_variation import generate_variation_plan


def _sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_fixed_cube_gate_plan(
    variation_config: dict[str, Any],
    *,
    gate_id: str,
    edge_m: float,
    position_xy_m: tuple[float, float],
    repetitions: int = 20,
) -> dict[str, Any]:
    """Build independent seeded trials with identical resolved scene factors."""
    if not gate_id:
        raise ValueError("gate_id must be non-empty")
    configured_sizes = [float(value) for value in variation_config["object"]["edge_sizes_m"]]
    if float(edge_m) not in configured_sizes:
        raise ValueError("edge_m must be one of the configured cube sizes")
    if len(position_xy_m) != 2 or not all(
        isinstance(value, (int, float)) for value in position_xy_m
    ):
        raise ValueError("position_xy_m must contain two numbers")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    base_plan = generate_variation_plan(variation_config)
    template = next(
        trial
        for trial in base_plan["trials"]
        if trial["resolved"]["dimensions_m"][0] == float(edge_m)
        and trial["resolved"]["rgba"][0] > trial["resolved"]["rgba"][2]
    )
    trials = []
    for repetition in range(repetitions):
        trial = copy.deepcopy(template)
        trial_id = f"{gate_id}_rep{repetition:02d}"
        seed_material = {
            "gate_id": gate_id,
            "repetition": repetition,
            "edge_m": float(edge_m),
            "position_xy_m": [float(value) for value in position_xy_m],
        }
        seed = int.from_bytes(
            hashlib.sha256(
                json.dumps(
                    seed_material, sort_keys=True, separators=(",", ":")
                ).encode()
            ).digest()[:4],
            "big",
        )
        position = [
            float(position_xy_m[0]),
            float(position_xy_m[1]),
            float(variation_config["workspace"]["table_z_m"]) + edge_m / 2,
        ]
        trial.update(
            {
                "trial_id": trial_id,
                "variation_id": trial_id,
                "cell_id": "fixed_gate",
                "split": "train",
                "seed": seed,
                "seed_material": seed_material,
            }
        )
        for key in ("requested", "resolved"):
            trial[key]["dimensions_m"] = [edge_m, edge_m, edge_m]
            trial[key]["position_m"] = position
        trials.append(trial)

    plan = {
        **base_plan,
        "plan_id": gate_id,
        "config_revision": f"gate:{variation_config['config_revision']}",
        "trials": trials,
        "gate": {
            "kind": "fixed_cube_repeatability",
            "required_successes": repetitions,
            "maximum_attempts": repetitions,
            "edge_m": edge_m,
            "position_xy_m": list(position_xy_m),
            "repetitions": repetitions,
        },
    }
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = _sha256(plan)
    return plan


def build_cube_workspace_matrix_plan(
    variation_config: dict[str, Any],
    *,
    gate_id: str,
    positions_xy_m: list[tuple[float, float]],
    minimum_success_rate: float = 0.90,
) -> dict[str, Any]:
    """Freeze one trial for each configured cube size at five workspace points."""
    if not gate_id:
        raise ValueError("gate_id must be non-empty")
    if len(positions_xy_m) != 5:
        raise ValueError("workspace matrix requires exactly five positions")
    if len(set(positions_xy_m)) != 5:
        raise ValueError("workspace matrix positions must be unique")
    if not 0.0 < minimum_success_rate <= 1.0:
        raise ValueError("minimum_success_rate must be in (0, 1]")
    x_bounds = variation_config["workspace"]["x_bounds_m"]
    y_bounds = variation_config["workspace"]["y_bounds_m"]
    edge_sizes = [float(value) for value in variation_config["object"]["edge_sizes_m"]]
    if len(edge_sizes) != 2:
        raise ValueError("workspace matrix requires exactly two configured cube sizes")
    target_position = variation_config["target"]["position_m"]
    target_dimensions = variation_config["target"]["dimensions_m"]
    # A cube at arbitrary yaw fits inside a horizontal circle with radius
    # edge/sqrt(2).  Reject any center whose conservative footprint intersects
    # the fixed target pad, so the workspace gate never starts with the object
    # on or under the placement target.
    cube_half_extent = max(edge_sizes) / math.sqrt(2.0)
    target_x_min = float(target_position[0]) - float(target_dimensions[0]) / 2.0
    target_x_max = float(target_position[0]) + float(target_dimensions[0]) / 2.0
    target_y_min = float(target_position[1]) - float(target_dimensions[1]) / 2.0
    target_y_max = float(target_position[1]) + float(target_dimensions[1]) / 2.0
    for position in positions_xy_m:
        if len(position) != 2:
            raise ValueError("each workspace position must contain x and y")
        if not x_bounds[0] <= position[0] <= x_bounds[1]:
            raise ValueError("workspace matrix x position is outside configured bounds")
        if not y_bounds[0] <= position[1] <= y_bounds[1]:
            raise ValueError("workspace matrix y position is outside configured bounds")
        overlaps_target = (
            float(position[0]) + cube_half_extent > target_x_min
            and float(position[0]) - cube_half_extent < target_x_max
            and float(position[1]) + cube_half_extent > target_y_min
            and float(position[1]) - cube_half_extent < target_y_max
        )
        if overlaps_target:
            raise ValueError("workspace matrix cube footprint overlaps the target pad")

    base_plan = generate_variation_plan(variation_config)
    templates = {
        edge_m: next(
            trial
            for trial in base_plan["trials"]
            if trial["resolved"]["dimensions_m"][0] == edge_m
            and trial["resolved"]["rgba"][0] > trial["resolved"]["rgba"][2]
        )
        for edge_m in edge_sizes
    }
    trials = []
    table_z = float(variation_config["workspace"]["table_z_m"])
    for edge_m in edge_sizes:
        for position_index, position_xy in enumerate(positions_xy_m):
            trial = copy.deepcopy(templates[edge_m])
            size_mm = int(round(edge_m * 1000))
            trial_id = f"{gate_id}_{size_mm}mm_pos{position_index:02d}"
            seed_material = {
                "gate_id": gate_id,
                "edge_m": edge_m,
                "position_index": position_index,
                "position_xy_m": [float(value) for value in position_xy],
            }
            seed = int.from_bytes(
                hashlib.sha256(
                    json.dumps(
                        seed_material, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).digest()[:4],
                "big",
            )
            position = [
                float(position_xy[0]),
                float(position_xy[1]),
                table_z + edge_m / 2,
            ]
            trial.update(
                {
                    "trial_id": trial_id,
                    "variation_id": trial_id,
                    "cell_id": f"{size_mm}mm_pos{position_index:02d}",
                    "split": "train",
                    "seed": seed,
                    "seed_material": seed_material,
                }
            )
            for key in ("requested", "resolved"):
                trial[key]["dimensions_m"] = [edge_m, edge_m, edge_m]
                trial[key]["position_m"] = position
            trials.append(trial)

    required_successes = math.ceil(minimum_success_rate * len(trials))
    plan = {
        **base_plan,
        "plan_id": gate_id,
        "config_revision": f"gate:{variation_config['config_revision']}",
        "trials": trials,
        "gate": {
            "kind": "cube_workspace_matrix",
            "required_successes": required_successes,
            "maximum_attempts": len(trials),
            "minimum_success_rate": minimum_success_rate,
            "edge_sizes_m": edge_sizes,
            "positions_xy_m": [list(position) for position in positions_xy_m],
            "trials_per_cell": 1,
        },
    }
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = _sha256(plan)
    return plan
