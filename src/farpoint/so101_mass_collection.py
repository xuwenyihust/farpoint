"""Frozen mirrored-mass collection plans for the SO-101 dataset."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from farpoint.object_variation import generate_variation_plan


CONFIG_SCHEMA_VERSION = "farpoint.so101-mass-collection-config.v1"
COLLECTION_KIND = "mirrored_mass_success_collection"


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _seed(value: Any) -> int:
    return int.from_bytes(hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).digest()[:8], "big")


def load_mass_collection_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_mass_collection_config(config)
    return config


def validate_mass_collection_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"mass collection config must use {CONFIG_SCHEMA_VERSION}")
    for key in (
        "profile_id",
        "source_plan_sha256",
        "source_selection_id",
        "source_selection_manifest_sha256",
        "source_trial_ids",
        "target_mass_kg",
        "required_successes",
        "maximum_attempts",
    ):
        if key not in config:
            raise ValueError(f"mass collection config is missing {key}")
    trial_ids = config["source_trial_ids"]
    if not isinstance(trial_ids, list) or not trial_ids:
        raise ValueError("source_trial_ids must be a non-empty list")
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("source_trial_ids must be unique")
    required = int(config["required_successes"])
    maximum = int(config["maximum_attempts"])
    if required != len(trial_ids):
        raise ValueError("required_successes must equal the mirrored trial count")
    if maximum < required:
        raise ValueError("maximum_attempts cannot be smaller than required_successes")
    if float(config["target_mass_kg"]) <= 0.0:
        raise ValueError("target_mass_kg must be positive")
    tolerance = float(config.get("actual_mass_tolerance_kg", 1e-6))
    if tolerance < 0.0:
        raise ValueError("actual_mass_tolerance_kg must be non-negative")


def _set_mass(payload: dict[str, Any], mass_kg: float) -> None:
    payload["mass_kg"] = mass_kg
    payload["entities"]["pick_object"]["physics"]["mass_kg"] = mass_kg


def mirrored_balance(plan: dict[str, Any]) -> dict[str, Any]:
    trials = plan.get("trials") or []
    splits = Counter(trial["split"] for trial in trials)
    cells = Counter(trial["cell_id"] for trial in trials)
    sizes = Counter(
        f"size_{int(trial['seed_material']['size_index'])}" for trial in trials
    )
    colors = Counter(
        f"color_{int(trial['seed_material']['color_index'])}" for trial in trials
    )
    joints = Counter(
        f"size_{int(trial['seed_material']['size_index'])}__"
        f"color_{int(trial['seed_material']['color_index'])}"
        for trial in trials
    )
    rows = Counter(trial["cell_id"].split("_")[0] for trial in trials)
    columns = Counter(trial["cell_id"].split("_")[1] for trial in trials)
    return {
        "total": len(trials),
        "splits": dict(sorted(splits.items())),
        "workspace_cells": dict(sorted(cells.items())),
        "workspace_rows": dict(sorted(rows.items())),
        "workspace_columns": dict(sorted(columns.items())),
        "sizes": dict(sorted(sizes.items())),
        "colors": dict(sorted(colors.items())),
        "size_color": dict(sorted(joints.items())),
    }


def validate_mirrored_balance(balance: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if balance.get("total") != 50:
        errors.append("mirrored collection must contain exactly 50 variations")
    if balance.get("splits") != {"test": 5, "train": 40, "validation": 5}:
        errors.append("mirrored split counts must be train=40, validation=5, test=5")
    for key in ("sizes", "colors"):
        if sorted((balance.get(key) or {}).values()) != [25, 25]:
            errors.append(f"mirrored {key} counts must be 25/25")
    joint = sorted((balance.get("size_color") or {}).values())
    if joint != [12, 12, 13, 13]:
        errors.append("mirrored size/color joint counts must be 12/12/13/13")
    if len(balance.get("workspace_cells") or {}) != 25:
        errors.append("mirrored collection must cover all 25 workspace cells")
    for key in ("workspace_rows", "workspace_columns"):
        if sorted((balance.get(key) or {}).values()) != [10, 10, 10, 10, 10]:
            errors.append(f"mirrored {key} counts must all equal 10")
    return errors


def build_mirrored_mass_collection_plan(
    variation_config: dict[str, Any], collection_config: dict[str, Any]
) -> dict[str, Any]:
    """Mirror the exact balanced50 identities while changing only cube mass."""
    validate_mass_collection_config(collection_config)
    base_plan = generate_variation_plan(variation_config)
    if base_plan["plan_sha256"] != collection_config["source_plan_sha256"]:
        raise ValueError("base variation plan does not match frozen source_plan_sha256")
    by_id = {trial["trial_id"]: trial for trial in base_plan["trials"]}
    missing = [name for name in collection_config["source_trial_ids"] if name not in by_id]
    if missing:
        raise ValueError(f"source trial ids are missing from the base plan: {missing}")
    target_mass = float(collection_config["target_mass_kg"])
    grams = int(round(target_mass * 1000))
    trials = []
    for source_id in collection_config["source_trial_ids"]:
        trial = copy.deepcopy(by_id[source_id])
        trial_id = f"{source_id}_m{grams:03d}g"
        material = copy.deepcopy(trial["seed_material"])
        material.update(
            {
                "mass_collection_profile_id": collection_config["profile_id"],
                "source_trial_id": source_id,
                "mass_kg": target_mass,
            }
        )
        trial.update(
            {
                "trial_id": trial_id,
                "variation_id": trial_id,
                "seed": _seed(material),
                "seed_material": material,
                "source_trial_id": source_id,
                "mass_audit_tolerance_kg": float(
                    collection_config.get("actual_mass_tolerance_kg", 1e-6)
                ),
            }
        )
        for key in ("requested", "resolved"):
            _set_mass(trial[key], target_mass)
        trials.append(trial)
    plan = {
        **base_plan,
        "plan_id": str(collection_config["profile_id"]),
        "config_revision": (
            f"mass-mirror:{variation_config['config_revision']}:"
            f"{collection_config.get('profile_revision', '1')}"
        ),
        "varied_axes": [
            "entities.pick_object.pose.position_m.x",
            "entities.pick_object.pose.position_m.y",
            "entities.pick_object.geometry.dimensions_m",
            "entities.pick_object.appearance.rgba",
            "entities.pick_object.physics.mass_kg",
        ],
        "frozen_axes": [
            "entities.pick_object.entity_type",
            "entities.pick_object.physics.body_type",
            "entities.pick_object.physics.collision_enabled",
            "entities.pick_object.physics.material",
            "entities.placement_target.pose",
            "entities.placement_target.geometry",
            "lighting.profile",
        ],
        "dimensions": [
            *base_plan["dimensions"],
            {"name": "object_mass_kg", "kind": "categorical", "values": [target_mass]},
        ],
        "trials": trials,
        "collection": {
            "kind": COLLECTION_KIND,
            "required_successes": int(collection_config["required_successes"]),
            "maximum_attempts": int(collection_config["maximum_attempts"]),
            "target_mass_kg": target_mass,
            "actual_mass_tolerance_kg": float(
                collection_config.get("actual_mass_tolerance_kg", 1e-6)
            ),
            "source_plan_id": base_plan["plan_id"],
            "source_plan_sha256": base_plan["plan_sha256"],
            "source_selection_id": collection_config["source_selection_id"],
            "source_selection_manifest_sha256": collection_config[
                "source_selection_manifest_sha256"
            ],
            "selection_policy": "exact_trial_identity_mirror_v1",
            "dataset_id": str(collection_config.get("dataset_id", "farpoint_so101")),
        },
    }
    balance = mirrored_balance(plan)
    errors = validate_mirrored_balance(balance)
    if errors:
        raise ValueError("invalid mirrored balance: " + "; ".join(errors))
    plan["collection"]["balance"] = balance
    plan.pop("gate", None)
    plan.pop("pilot", None)
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = _sha256(plan)
    return plan
