"""Frozen, stratified trial ordering for the SO-101 code-review pilot."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from typing import Any

from farpoint.object_variation import generate_variation_plan


DEFAULT_PRIMARY_TRIAL_IDS = (
    "cube_r00_c00_s0_k0",
    "cube_r00_c04_s1_k1",
    "cube_r01_c01_s1_k0",
    "cube_r01_c03_s0_k1",
    "cube_r02_c02_s0_k0",
    "cube_r03_c00_s1_k1",
    "cube_r03_c04_s0_k1",
    "cube_r04_c01_s0_k0",
    "cube_r04_c03_s1_k0",
    "cube_r02_c04_s1_k1",
)

DEFAULT_FALLBACK_TRIAL_IDS = (
    "cube_r00_c02_s1_k1",
    "cube_r01_c04_s0_k0",
    "cube_r02_c00_s1_k0",
    "cube_r03_c02_s0_k1",
    "cube_r04_c00_s1_k1",
)


def _sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _seed(value: Any) -> int:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _set_mass(payload: dict[str, Any], mass_kg: float) -> None:
    payload["mass_kg"] = mass_kg
    payload["entities"]["pick_object"]["physics"]["mass_kg"] = mass_kg


def _yaw_quaternion_xyzw(yaw_degrees: float) -> list[float]:
    half_angle = math.radians(float(yaw_degrees)) / 2.0
    return [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]


def _set_orientation(payload: dict[str, Any], orientation_xyzw: list[float]) -> None:
    payload["orientation_xyzw"] = copy.deepcopy(orientation_xyzw)
    payload["entities"]["pick_object"]["pose"]["orientation_xyzw"] = copy.deepcopy(
        orientation_xyzw
    )


def build_so101_pilot_plan(
    variation_config: dict[str, Any],
    *,
    pilot_id: str,
    primary_trial_ids: tuple[str, ...] = DEFAULT_PRIMARY_TRIAL_IDS,
    fallback_trial_ids: tuple[str, ...] = DEFAULT_FALLBACK_TRIAL_IDS,
) -> dict[str, Any]:
    """Reorder all 100 variations around a frozen 10-of-15 pilot budget."""
    if not pilot_id:
        raise ValueError("pilot_id must be non-empty")
    if len(primary_trial_ids) != 10 or len(fallback_trial_ids) != 5:
        raise ValueError("pilot requires 10 primary and 5 fallback trial ids")
    ordered_ids = tuple(primary_trial_ids) + tuple(fallback_trial_ids)
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("pilot trial ids must be unique")

    plan = generate_variation_plan(variation_config)
    by_id = {trial["trial_id"]: trial for trial in plan["trials"]}
    missing = sorted(set(ordered_ids) - set(by_id))
    if missing:
        raise ValueError("unknown pilot trial ids: " + ", ".join(missing))
    remaining = [trial for trial in plan["trials"] if trial["trial_id"] not in ordered_ids]
    plan.update(
        {
            "plan_id": pilot_id,
            "config_revision": f"pilot:{variation_config['config_revision']}",
            "trials": [by_id[trial_id] for trial_id in ordered_ids] + remaining,
            "pilot": {
                "kind": "stratified_success_pilot",
                "required_successes": len(primary_trial_ids),
                "maximum_attempts": len(ordered_ids),
                "primary_trial_ids": list(primary_trial_ids),
                "fallback_trial_ids": list(fallback_trial_ids),
            },
        }
    )
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = _sha256(plan)
    return plan


def build_targeted_mass_diagnostic_pilot_plan(
    variation_config: dict[str, Any],
    *,
    pilot_id: str,
    source_trial_ids: tuple[str, ...],
    target_mass_kg: float,
    required_successes: int,
    expectations: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a bounded pilot for named variations at one audited cube mass."""
    if not pilot_id:
        raise ValueError("pilot_id must be non-empty")
    if not source_trial_ids or len(set(source_trial_ids)) != len(source_trial_ids):
        raise ValueError("targeted pilot trial ids must be non-empty and unique")
    if target_mass_kg <= 0.0:
        raise ValueError("target mass must be positive")
    if not 0 < required_successes <= len(source_trial_ids):
        raise ValueError("required successes must fit within the targeted pilot")

    plan = generate_variation_plan(variation_config)
    available_ids = {trial["trial_id"] for trial in plan["trials"]}
    missing = sorted(set(source_trial_ids) - available_ids)
    if missing:
        raise ValueError("unknown targeted pilot trial ids: " + ", ".join(missing))
    expectations = copy.deepcopy(expectations or {})
    if set(expectations) != set(source_trial_ids):
        raise ValueError("targeted pilot expectations must cover every source trial")
    for trial_id, expectation in expectations.items():
        if not isinstance(expectation.get("success"), bool):
            raise ValueError(f"targeted pilot expectation requires success: {trial_id}")
        if not expectation["success"] and not expectation.get("failure_reason"):
            raise ValueError(
                f"failed targeted pilot expectation requires failure_reason: {trial_id}"
            )

    grams = int(round(target_mass_kg * 1000.0))
    transformed = []
    for source in plan["trials"]:
        trial = copy.deepcopy(source)
        source_id = source["trial_id"]
        trial_id = f"{source_id}_m{grams:03d}g"
        seed_material = copy.deepcopy(source["seed_material"])
        seed_material.update(
            {
                "targeted_pilot_id": pilot_id,
                "source_trial_id": source_id,
                "mass_kg": target_mass_kg,
            }
        )
        trial.update(
            {
                "trial_id": trial_id,
                "variation_id": trial_id,
                "seed": _seed(seed_material),
                "seed_material": seed_material,
                "source_trial_id": source_id,
                "mass_audit_tolerance_kg": 1e-6,
            }
        )
        for key in ("requested", "resolved"):
            _set_mass(trial[key], target_mass_kg)
        transformed.append(trial)

    by_source_id = {trial["source_trial_id"]: trial for trial in transformed}
    selected = [by_source_id[trial_id] for trial_id in source_trial_ids]
    selected_ids = {trial["trial_id"] for trial in selected}
    expectations_by_trial_id = {
        by_source_id[source_id]["trial_id"]: expectations[source_id]
        for source_id in source_trial_ids
    }
    remaining = [trial for trial in transformed if trial["trial_id"] not in selected_ids]
    plan.update(
        {
            "plan_id": pilot_id,
            "config_revision": (
                f"targeted-mass-pilot:{variation_config['config_revision']}"
            ),
            "varied_axes": [
                *plan["varied_axes"],
                "entities.pick_object.physics.mass_kg",
            ],
            "dimensions": [
                *plan["dimensions"],
                {
                    "name": "object_mass_kg",
                    "kind": "categorical",
                    "values": [target_mass_kg],
                },
            ],
            "trials": selected + remaining,
            "pilot": {
                "kind": "targeted_mass_diagnostic_pilot",
                "required_successes": required_successes,
                "maximum_attempts": len(selected),
                "trial_ids": [trial["trial_id"] for trial in selected],
                "source_trial_ids": list(source_trial_ids),
                "target_mass_kg": target_mass_kg,
                "actual_mass_tolerance_kg": 1e-6,
                "expectations": expectations_by_trial_id,
            },
        }
    )
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = _sha256(plan)
    return plan


def build_so101_yaw_pilot_plan(
    variation_config: dict[str, Any],
    *,
    pilot_id: str,
    yaw_degrees: float,
    trial_profiles: list[dict[str, Any]],
    required_successes: int = 10,
) -> dict[str, Any]:
    """Build a frozen 12-attempt SO-101 yaw pilot across both proven masses."""
    if not pilot_id:
        raise ValueError("pilot_id must be non-empty")
    if not math.isfinite(yaw_degrees) or not 0.0 <= yaw_degrees < 90.0:
        raise ValueError("cube yaw must be finite and in [0, 90) degrees")
    if len(trial_profiles) != 12:
        raise ValueError("SO-101 yaw pilot requires exactly 12 trial profiles")
    if required_successes != 10:
        raise ValueError("SO-101 yaw pilot requires exactly 10 successes")
    source_ids = [str(profile.get("source_trial_id", "")) for profile in trial_profiles]
    if any(not value for value in source_ids) or len(set(source_ids)) != len(source_ids):
        raise ValueError("yaw pilot source trial ids must be non-empty and unique")
    masses = [float(profile.get("mass_kg", 0.0)) for profile in trial_profiles]
    if set(masses) != {0.03, 0.04} or masses.count(0.03) != 6:
        raise ValueError("yaw pilot requires six trials at each of 0.03 kg and 0.04 kg")

    plan = generate_variation_plan(variation_config)
    by_id = {trial["trial_id"]: trial for trial in plan["trials"]}
    missing = sorted(set(source_ids) - set(by_id))
    if missing:
        raise ValueError("unknown yaw pilot trial ids: " + ", ".join(missing))

    orientation = _yaw_quaternion_xyzw(yaw_degrees)
    transformed = []
    for profile, mass_kg in zip(trial_profiles, masses):
        source_id = str(profile["source_trial_id"])
        trial = copy.deepcopy(by_id[source_id])
        grams = int(round(mass_kg * 1000.0))
        yaw_millidegrees = int(round(yaw_degrees * 1000.0))
        trial_id = f"{source_id}_yaw{yaw_millidegrees:05d}_m{grams:03d}g"
        seed_material = copy.deepcopy(trial["seed_material"])
        seed_material.update(
            {
                "yaw_pilot_id": pilot_id,
                "source_trial_id": source_id,
                "yaw_degrees": float(yaw_degrees),
                "mass_kg": mass_kg,
            }
        )
        trial.update(
            {
                "trial_id": trial_id,
                "variation_id": trial_id,
                "seed": _seed(seed_material),
                "seed_material": seed_material,
                "source_trial_id": source_id,
                "object_yaw_degrees": float(yaw_degrees),
                "mass_audit_tolerance_kg": 1e-6,
            }
        )
        for key in ("requested", "resolved"):
            _set_mass(trial[key], mass_kg)
            _set_orientation(trial[key], orientation)
        transformed.append(trial)

    transformed_ids = {trial["trial_id"] for trial in transformed}
    coverage = {
        "splits": dict(sorted(Counter(trial["split"] for trial in transformed).items())),
        "workspace_cells": sorted(trial["cell_id"] for trial in transformed),
        "sizes": dict(
            sorted(
                Counter(
                    f"size_{trial['seed_material']['size_index']}"
                    for trial in transformed
                ).items()
            )
        ),
        "colors": dict(
            sorted(
                Counter(
                    f"color_{trial['seed_material']['color_index']}"
                    for trial in transformed
                ).items()
            )
        ),
        "size_color": dict(
            sorted(
                Counter(
                    f"size_{trial['seed_material']['size_index']}__"
                    f"color_{trial['seed_material']['color_index']}"
                    for trial in transformed
                ).items()
            )
        ),
    }
    if coverage["splits"] != {"test": 2, "train": 8, "validation": 2}:
        raise ValueError("yaw pilot split coverage must be train=8, validation=2, test=2")
    if len(set(coverage["workspace_cells"])) != 12:
        raise ValueError("yaw pilot must cover 12 distinct workspace cells")
    if sorted(coverage["sizes"].values()) != [6, 6]:
        raise ValueError("yaw pilot cube sizes must be balanced 6/6")
    if sorted(coverage["colors"].values()) != [6, 6]:
        raise ValueError("yaw pilot cube colors must be balanced 6/6")
    if sorted(coverage["size_color"].values()) != [3, 3, 3, 3]:
        raise ValueError("yaw pilot size/color combinations must be balanced 3 each")
    remaining = [
        trial for trial in plan["trials"] if trial["trial_id"] not in set(source_ids)
    ]
    plan.update(
        {
            "plan_id": pilot_id,
            "config_revision": f"yaw-pilot:{variation_config['config_revision']}",
            "varied_axes": [
                *plan["varied_axes"],
                "entities.pick_object.pose.orientation_xyzw",
                "entities.pick_object.physics.mass_kg",
            ],
            "dimensions": [
                *plan["dimensions"],
                {
                    "name": "object_yaw_degrees",
                    "kind": "categorical",
                    "values": [float(yaw_degrees)],
                },
                {
                    "name": "object_mass_kg",
                    "kind": "categorical",
                    "values": [0.03, 0.04],
                },
            ],
            "trials": transformed + remaining,
            "pilot": {
                "kind": "targeted_yaw_pilot",
                "required_successes": required_successes,
                "maximum_attempts": len(transformed),
                "trial_ids": [trial["trial_id"] for trial in transformed],
                "source_trial_ids": source_ids,
                "yaw_degrees": float(yaw_degrees),
                "orientation_xyzw": orientation,
                "actual_orientation_tolerance_degrees": 1.0,
                "mass_kg_counts": {"0.03": 6, "0.04": 6},
                "actual_mass_tolerance_kg": 1e-6,
                "selection_policy": "balanced_representative_yaw_pilot_v1",
                "coverage": coverage,
            },
        }
    )
    if transformed_ids & {trial["trial_id"] for trial in remaining}:
        raise ValueError("yaw pilot trial ids collide with base variations")
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = _sha256(plan)
    return plan
