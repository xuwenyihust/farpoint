"""Paired, bounded feasibility plans for SO-101 cube mass variation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from farpoint.object_variation import generate_variation_plan
from farpoint.scene_entities import bind_scene_entities


def _sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _seed(value: Any) -> int:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def audit_resolved_mass(
    *,
    requested_mass_kg: float,
    resolved_mass_kg: float,
    physx_actual_mass_kg: float,
    tolerance_kg: float = 1e-6,
) -> dict[str, Any]:
    """Build a fail-closed audit proving the configured mass reached PhysX."""
    values = {
        "requested_mass_kg": float(requested_mass_kg),
        "resolved_mass_kg": float(resolved_mass_kg),
        "physx_actual_mass_kg": float(physx_actual_mass_kg),
        "tolerance_kg": float(tolerance_kg),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("mass audit values must be finite")
    if min(
        values["requested_mass_kg"],
        values["resolved_mass_kg"],
        values["physx_actual_mass_kg"],
    ) <= 0.0:
        raise ValueError("mass audit masses must be positive")
    if values["tolerance_kg"] < 0.0:
        raise ValueError("mass audit tolerance must be non-negative")
    requested_resolved_error = abs(
        values["requested_mass_kg"] - values["resolved_mass_kg"]
    )
    resolved_physx_error = abs(
        values["resolved_mass_kg"] - values["physx_actual_mass_kg"]
    )
    verified = (
        requested_resolved_error <= values["tolerance_kg"]
        and resolved_physx_error <= values["tolerance_kg"]
    )
    return {
        **values,
        "requested_resolved_absolute_error_kg": requested_resolved_error,
        "resolved_physx_absolute_error_kg": resolved_physx_error,
        "verified": verified,
    }


def build_cube_mass_feasibility_plan(
    variation_config: dict[str, Any],
    *,
    profile_id: str,
    baseline_mass_kg: float = 0.04,
    candidate_mass_kg: float = 0.03,
    edge_m: float = 0.03,
    position_xy_m: tuple[float, float] = (0.20, -0.095),
    repetitions_per_mass: int = 5,
    minimum_successes_per_mass: int = 4,
) -> dict[str, Any]:
    """Build matched environment-seed pairs for baseline/candidate masses."""
    if not profile_id:
        raise ValueError("profile_id must be non-empty")
    masses = (float(baseline_mass_kg), float(candidate_mass_kg))
    if any(mass <= 0.0 for mass in masses):
        raise ValueError("cube masses must be positive")
    if masses[0] == masses[1]:
        raise ValueError("baseline and candidate masses must differ")
    if repetitions_per_mass <= 0:
        raise ValueError("repetitions_per_mass must be positive")
    if not 0 < minimum_successes_per_mass <= repetitions_per_mass:
        raise ValueError(
            "minimum_successes_per_mass must be within the repetition budget"
        )
    configured_sizes = [
        float(value) for value in variation_config["object"]["edge_sizes_m"]
    ]
    if float(edge_m) not in configured_sizes:
        raise ValueError("edge_m must be one of the configured cube sizes")
    if len(position_xy_m) != 2:
        raise ValueError("position_xy_m must contain x and y")
    x_bounds = variation_config["workspace"]["x_bounds_m"]
    y_bounds = variation_config["workspace"]["y_bounds_m"]
    if not x_bounds[0] <= position_xy_m[0] <= x_bounds[1]:
        raise ValueError("position x is outside the configured workspace")
    if not y_bounds[0] <= position_xy_m[1] <= y_bounds[1]:
        raise ValueError("position y is outside the configured workspace")

    base_plan = generate_variation_plan(variation_config)
    template = next(
        trial
        for trial in base_plan["trials"]
        if trial["resolved"]["dimensions_m"][0] == float(edge_m)
        and trial["resolved"]["rgba"][0] > trial["resolved"]["rgba"][2]
    )
    table_z = float(variation_config["workspace"]["table_z_m"])
    position = [
        float(position_xy_m[0]),
        float(position_xy_m[1]),
        table_z + float(edge_m) / 2.0,
    ]
    trials = []
    for repetition in range(repetitions_per_mass):
        environment_seed = _seed(
            {"profile_id": profile_id, "pair_index": repetition}
        )
        pair_id = f"pair{repetition:02d}"
        for role, mass in zip(("baseline", "candidate"), masses):
            trial = copy.deepcopy(template)
            grams = int(round(mass * 1000))
            trial_id = f"{profile_id}_{pair_id}_{role}_{grams}g"
            seed_material = {
                "profile_id": profile_id,
                "pair_index": repetition,
                "mass_role": role,
                "mass_kg": mass,
            }
            trial.update(
                {
                    "trial_id": trial_id,
                    "variation_id": trial_id,
                    "cell_id": pair_id,
                    "split": "train",
                    "seed": _seed(seed_material),
                    "seed_material": seed_material,
                    "environment_seed": environment_seed,
                    "mass_pair_id": pair_id,
                    "mass_role": role,
                    "mass_audit_tolerance_kg": 1e-6,
                }
            )
            for key in ("requested", "resolved"):
                trial[key]["dimensions_m"] = [edge_m, edge_m, edge_m]
                trial[key]["position_m"] = position
                trial[key]["mass_kg"] = mass
                trial[key] = bind_scene_entities(
                    trial[key], variation_config["target"]
                )
            trials.append(trial)

    plan = {
        **base_plan,
        "plan_id": profile_id,
        "config_revision": f"mass-feasibility:{variation_config['config_revision']}",
        "varied_axes": ["entities.pick_object.physics.mass_kg"],
        "frozen_axes": [
            "entities.pick_object.entity_type",
            "entities.pick_object.pose",
            "entities.pick_object.geometry",
            "entities.pick_object.appearance",
            "entities.pick_object.physics.body_type",
            "entities.pick_object.physics.collision_enabled",
            "entities.pick_object.physics.material",
            "entities.placement_target.pose",
            "entities.placement_target.geometry",
            "lighting.profile",
        ],
        "trials": trials,
        "gate": {
            "kind": "cube_mass_feasibility",
            "required_successes": minimum_successes_per_mass * 2,
            "maximum_attempts": len(trials),
            "baseline_mass_kg": masses[0],
            "candidate_mass_kg": masses[1],
            "edge_m": float(edge_m),
            "position_xy_m": [float(value) for value in position_xy_m],
            "repetitions_per_mass": repetitions_per_mass,
            "minimum_successes_per_mass": minimum_successes_per_mass,
            "paired_environment_seeds": True,
            "actual_mass_tolerance_kg": 1e-6,
            "minimum_successful_pairs_for_behavior": 3,
            "behavior_change_thresholds": {
                "action_path_relative": 0.005,
                "mean_lift_bilateral_force_relative": 0.05,
                "frame_count_absolute": 1,
            },
        },
    }
    plan.pop("pilot", None)
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = _sha256(plan)
    return plan
