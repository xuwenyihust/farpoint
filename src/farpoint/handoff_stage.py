"""Versioned, controller-independent recovery handoff stages."""

from __future__ import annotations

from typing import Any

import numpy as np


HANDOFF_STAGE_SCHEMA_VERSION = "1"
HANDOFF_STAGES = (
    "approach",
    "grasp",
    "lift",
    "transport",
    "place",
    "release",
    "settle",
)


def derive_handoff_stage(evidence: dict[str, Any]) -> str:
    """Classify one live handoff from measured task milestones.

    The ordering is intentionally late-stage first.  Historical progress is
    retained after a drop, while current target/release state distinguishes
    place, release, and settle without relying on an ACT-internal phase label.
    """
    forces = np.asarray(evidence.get("contact_forces_n", (0.0, 0.0)), dtype=np.float64)
    if forces.shape != (2,) or not np.all(np.isfinite(forces)):
        raise ValueError("handoff contact_forces_n must contain two finite values")

    cube_in_target = bool(evidence.get("cube_in_target", False))
    gripper_released = bool(evidence.get("gripper_released", False))
    if cube_in_target and gripper_released:
        return "settle"
    if cube_in_target:
        return "release"
    if bool(evidence.get("ever_near_target", False)):
        return "place"
    if bool(evidence.get("ever_lifted", evidence.get("cube_lifted", False))):
        return "transport"

    threshold = float(evidence.get("contact_force_threshold_n", 0.1))
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("handoff contact force threshold must be finite and non-negative")
    secure_grasp = bool(evidence.get("secure_grasp", np.min(forces) >= threshold))
    if secure_grasp and not gripper_released:
        return "lift"
    if bool(evidence.get("ever_contact", np.max(forces) >= threshold)):
        return "grasp"
    return "approach"


def normalize_handoff_stage(
    trigger: dict[str, Any],
    *,
    observed_milestones: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Return an explicit stage while retaining a legacy source label.

    Published v0.1.1 records only contain the broad label ``pre_lift``.  A
    caller auditing their pre-handoff trace can supply measured milestones;
    new records already carry the versioned stage and are returned unchanged.
    """
    explicit = trigger.get("handoff_stage")
    if explicit is not None:
        if trigger.get("handoff_stage_schema_version") != HANDOFF_STAGE_SCHEMA_VERSION:
            raise ValueError("unsupported handoff stage schema version")
        if explicit not in HANDOFF_STAGES:
            raise ValueError(f"unsupported handoff stage: {explicit}")
        return {
            "handoff_stage_schema_version": HANDOFF_STAGE_SCHEMA_VERSION,
            "handoff_stage": str(explicit),
            **(
                {"source_stage_label": str(trigger["stage"])}
                if trigger.get("stage") and trigger.get("stage") != explicit
                else {}
            ),
        }

    evidence = dict(trigger.get("evidence") or trigger)
    if observed_milestones:
        evidence.update(observed_milestones)
    stage = derive_handoff_stage(evidence)
    result = {
        "handoff_stage_schema_version": HANDOFF_STAGE_SCHEMA_VERSION,
        "handoff_stage": stage,
    }
    if trigger.get("stage"):
        result["source_stage_label"] = str(trigger["stage"])
    return result
