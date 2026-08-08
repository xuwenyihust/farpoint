"""Frozen, stratified trial ordering for the SO-101 code-review pilot."""

from __future__ import annotations

import copy
import hashlib
import json
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
