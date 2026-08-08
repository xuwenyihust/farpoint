"""Frozen balanced SO-101 yaw collection plans."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from farpoint.object_variation import generate_variation_plan


CONFIG_SCHEMA_VERSION = "farpoint.so101-yaw-collection-config.v1"
COLLECTION_KIND = "balanced_yaw_success_collection"


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _seed(value: Any) -> int:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _yaw_quaternion_xyzw(yaw_degrees: float) -> list[float]:
    half = math.radians(yaw_degrees) / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def load_yaw_collection_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_yaw_collection_config(config)
    return config


def validate_yaw_collection_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"yaw collection config must use {CONFIG_SCHEMA_VERSION}")
    required_keys = (
        "profile_id", "source_plan_sha256", "cube_size_m", "yaw_degrees",
        "mass_kg", "required_successes", "maximum_attempts", "selection_policy",
    )
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError("yaw collection config is missing " + ", ".join(missing))
    if float(config["cube_size_m"]) != 0.03:
        raise ValueError("first yaw collection tranche must use 30 mm cubes")
    yaw = float(config["yaw_degrees"])
    if not math.isfinite(yaw) or yaw != 0.0:
        raise ValueError("first yaw collection tranche must use yaw=0 degrees")
    masses = [float(value) for value in config["mass_kg"]]
    if masses != [0.03, 0.04]:
        raise ValueError("yaw collection masses must be [0.03, 0.04]")
    if int(config["required_successes"]) != 50:
        raise ValueError("yaw collection must require 50 successes")
    if int(config["maximum_attempts"]) != 150:
        raise ValueError("yaw collection must freeze a 150-attempt ceiling")
    if config["selection_policy"] != "workspace_mass_color_checkerboard_v1":
        raise ValueError("unsupported yaw collection selection policy")


def yaw_collection_balance(plan: dict[str, Any]) -> dict[str, Any]:
    trials = plan.get("trials") or []
    mass_color = Counter(
        f"mass_{float(trial['resolved']['mass_kg']):.2f}__"
        f"color_{int(trial['seed_material']['color_index'])}"
        for trial in trials
    )
    return {
        "total": len(trials),
        "splits": dict(sorted(Counter(trial["split"] for trial in trials).items())),
        "workspace_cells": dict(sorted(Counter(trial["cell_id"] for trial in trials).items())),
        "workspace_rows": dict(sorted(Counter(trial["cell_id"].split("_")[0] for trial in trials).items())),
        "workspace_columns": dict(sorted(Counter(trial["cell_id"].split("_")[1] for trial in trials).items())),
        "sizes": dict(sorted(Counter(f"size_{int(trial['seed_material']['size_index'])}" for trial in trials).items())),
        "colors": dict(sorted(Counter(f"color_{int(trial['seed_material']['color_index'])}" for trial in trials).items())),
        "masses_kg": dict(sorted(Counter(f"{float(trial['resolved']['mass_kg']):.2f}" for trial in trials).items())),
        "mass_color": dict(sorted(mass_color.items())),
        "yaw_degrees": dict(sorted(Counter(f"{float(trial['object_yaw_degrees']):.1f}" for trial in trials).items())),
    }


def validate_yaw_collection_balance(balance: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if balance.get("total") != 50:
        errors.append("yaw collection must contain exactly 50 variations")
    if balance.get("splits") != {"test": 5, "train": 40, "validation": 5}:
        errors.append("yaw collection splits must be train=40, validation=5, test=5")
    if balance.get("sizes") != {"size_0": 50}:
        errors.append("yaw collection must contain only 30 mm cubes")
    for key in ("colors", "masses_kg"):
        if sorted((balance.get(key) or {}).values()) != [25, 25]:
            errors.append(f"yaw collection {key} counts must be 25/25")
    if sorted((balance.get("mass_color") or {}).values()) != [12, 12, 13, 13]:
        errors.append("yaw collection mass/color counts must be 12/12/13/13")
    cells = balance.get("workspace_cells") or {}
    if len(cells) != 25 or set(cells.values()) != {2}:
        errors.append("yaw collection must contain two variations in every workspace cell")
    for key in ("workspace_rows", "workspace_columns"):
        if sorted((balance.get(key) or {}).values()) != [10] * 5:
            errors.append(f"yaw collection {key} counts must all equal 10")
    if balance.get("yaw_degrees") != {"0.0": 50}:
        errors.append("yaw collection must contain only yaw=0 degrees")
    return errors


def build_yaw_collection_plan(
    variation_config: dict[str, Any], collection_config: dict[str, Any]
) -> dict[str, Any]:
    validate_yaw_collection_config(collection_config)
    base = generate_variation_plan(variation_config)
    if base["plan_sha256"] != collection_config["source_plan_sha256"]:
        raise ValueError("base variation plan does not match frozen source_plan_sha256")
    yaw = float(collection_config["yaw_degrees"])
    orientation = _yaw_quaternion_xyzw(yaw)
    trials = []
    for source in base["trials"]:
        if int(source["seed_material"]["size_index"]) != 0:
            continue
        trial = copy.deepcopy(source)
        row = int(trial["cell_id"][1:3])
        column = int(trial["cell_id"][5:7])
        color = int(trial["seed_material"]["color_index"])
        mass = 0.03 if color == ((row + column) % 2) else 0.04
        grams = int(round(mass * 1000))
        trial_id = f"{source['trial_id']}_yaw00000_m{grams:03d}g"
        material = copy.deepcopy(trial["seed_material"])
        material.update({
            "yaw_collection_profile_id": collection_config["profile_id"],
            "source_trial_id": source["trial_id"],
            "yaw_degrees": yaw,
            "mass_kg": mass,
        })
        trial.update({
            "trial_id": trial_id,
            "variation_id": trial_id,
            "source_trial_id": source["trial_id"],
            "object_yaw_degrees": yaw,
            "seed_material": material,
            "seed": _seed(material),
            "mass_audit_tolerance_kg": float(collection_config.get("actual_mass_tolerance_kg", 1e-6)),
        })
        # The base 30 mm slice is 40/6/4. Freeze one source identity as test
        # before collection so the formal tranche is 40/5/5 without any
        # outcome-dependent relabeling.
        if source["trial_id"] == "cube_r04_c02_s0_k1":
            if trial["split"] != "validation":
                raise ValueError("frozen yaw split override source is not validation")
            trial["split"] = "test"
            trial["split_source"] = "yaw_formal_40_5_5_override_v1"
        for role in ("requested", "resolved"):
            trial[role]["mass_kg"] = mass
            trial[role]["orientation_xyzw"] = orientation
            obj = trial[role]["entities"]["pick_object"]
            obj["physics"]["mass_kg"] = mass
            obj["pose"]["orientation_xyzw"] = orientation
        trials.append(trial)
    plan = {
        **base,
        "plan_id": str(collection_config["profile_id"]),
        "config_revision": f"yaw-formal:{variation_config['config_revision']}:{collection_config.get('profile_revision', '1')}",
        "varied_axes": [*base["varied_axes"], "entities.pick_object.pose.orientation_xyzw", "entities.pick_object.physics.mass_kg"],
        "dimensions": [
            *base["dimensions"],
            {"name": "object_yaw_degrees", "kind": "categorical", "values": [yaw]},
            {"name": "object_mass_kg", "kind": "categorical", "values": [0.03, 0.04]},
        ],
        "trials": trials,
        "collection": {
            "kind": COLLECTION_KIND,
            "required_successes": 50,
            "maximum_attempts": 150,
            "cube_size_m": 0.03,
            "yaw_degrees": yaw,
            "orientation_xyzw": orientation,
            "actual_orientation_tolerance_degrees": float(collection_config.get("actual_orientation_tolerance_degrees", 2.0)),
            "mass_kg": [0.03, 0.04],
            "actual_mass_tolerance_kg": float(collection_config.get("actual_mass_tolerance_kg", 1e-6)),
            "source_plan_id": base["plan_id"],
            "source_plan_sha256": base["plan_sha256"],
            "selection_policy": collection_config["selection_policy"],
            "dataset_id": str(collection_config.get("dataset_id", "farpoint_so101")),
        },
    }
    balance = yaw_collection_balance(plan)
    errors = validate_yaw_collection_balance(balance)
    if errors:
        raise ValueError("invalid yaw collection balance: " + "; ".join(errors))
    plan["collection"]["balance"] = balance
    plan.pop("pilot", None)
    plan.pop("gate", None)
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = _sha256(plan)
    return plan
