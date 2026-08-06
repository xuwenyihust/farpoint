"""Frozen, stratified trial ordering for the SO-101 code-review pilot."""

from __future__ import annotations

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
