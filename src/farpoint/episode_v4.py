"""Build strict simulator-authored SO-101 episode metadata v4."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from farpoint.campaign import validate_campaign_semantics, validate_segment_semantics
from farpoint.contracts import validate_contract, validate_episode_semantics
from farpoint.demonstration import nominal_demonstration
from farpoint.scene_entities import (
    bind_scene_entities,
    support_surface_entity,
)
from farpoint.v010_pilot import PILOT_KIND, versioned_config


FORMAL_SEGMENT_KIND = "self_healing_campaign_segment"


def is_v010_episode_plan(plan: dict[str, Any]) -> bool:
    """Return whether a plan must emit the v0.1.0 episode v4 contract."""
    return (plan.get("pilot") or {}).get("kind") == PILOT_KIND or (
        plan.get("collection") or {}
    ).get("kind") == FORMAL_SEGMENT_KIND


def _variant_resolved_state(
    trial: dict[str, Any],
    resolved_object: dict[str, Any],
) -> dict[str, Any]:
    requested = deepcopy(trial["object_variant"]["requested"])
    material = {
        key: resolved_object[key]
        for key in (
            "static_friction",
            "dynamic_friction",
            "restitution",
            "friction_combine_mode",
            "restitution_combine_mode",
        )
    }
    requested.update(
        {
            "asset_id": resolved_object["asset_id"],
            "dimensions_m": deepcopy(resolved_object["dimensions_m"]),
            "rgba": deepcopy(resolved_object["rgba"]),
            "mass_kg": float(resolved_object["mass_kg"]),
            "object_material": material,
        }
    )
    return requested


def _canonical_cube_yaw_degrees(orientation_xyzw: list[float]) -> float:
    x, y, z, w = (float(value) for value in orientation_xyzw)
    yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return yaw % 90.0


def build_so101_episode_v4(
    *,
    episode_id: str,
    campaign: dict[str, Any],
    segment: dict[str, Any],
    plan: dict[str, Any],
    trial: dict[str, Any],
    attempt_seed: int,
    git_commit: str,
    simulator_image_digest: str,
    resolved_object: dict[str, Any],
    target: dict[str, Any],
    table: dict[str, Any],
    camera_records: list[dict[str, Any]],
    embodiment: dict[str, Any],
    frame_count: int,
    control_hz: float,
    success: bool,
    dataset_valid: bool,
    failure_category: str | None,
    failure_reason: str | None,
    physics_audit: dict[str, Any],
    demonstration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind immutable campaign/segment provenance to one measured episode."""
    if not is_v010_episode_plan(plan):
        raise ValueError("episode v4 writer requires a v0.1.0 plan")
    campaign_errors = validate_campaign_semantics(campaign)
    segment_errors = validate_segment_semantics(segment)
    if campaign_errors or segment_errors:
        raise ValueError(
            "invalid episode v4 campaign context: " + "; ".join([*campaign_errors, *segment_errors])
        )
    if campaign["campaign_id"] != segment["campaign_id"]:
        raise ValueError("campaign and segment ids do not match")
    if campaign["campaign_sha256"] != segment["campaign_sha256"]:
        raise ValueError("campaign and segment hashes do not match")
    if campaign["campaign_sha256"] != plan.get("campaign_sha256"):
        raise ValueError("plan and campaign hashes do not match")
    if segment["plan_sha256"] != plan.get("plan_sha256"):
        raise ValueError("segment and plan hashes do not match")
    if segment["git_commit"] != git_commit:
        raise ValueError("segment and runtime git commits do not match")
    if plan["oracle_profile_id"] not in segment["oracle_profile_allowlist"]:
        raise ValueError("Oracle profile is outside the segment allowlist")
    if not simulator_image_digest.startswith("sha256:"):
        raise ValueError("simulator image digest must be a sha256 identifier")
    if frame_count <= 0:
        raise ValueError("episode v4 frame_count must be positive")
    if {record.get("camera_id") for record in camera_records} != {"front", "wrist"}:
        raise ValueError("episode v4 requires exactly front and wrist camera records")
    if any(
        (record.get("video_artifact") or {}).get("frame_count") != frame_count
        or (record.get("video_artifact") or {}).get("decode_verified") is not True
        for record in camera_records
    ):
        raise ValueError("episode v4 requires frame-aligned decoded video artifacts")

    resolved_state = deepcopy(resolved_object)
    resolved_variation_entities = bind_scene_entities(resolved_state, target)
    object_entity = resolved_variation_entities["entities"]["pick_object"]
    target_entity = resolved_variation_entities["entities"]["placement_target"]
    table_material = deepcopy(plan["materials"]["table"])
    table_entity = support_surface_entity(table, material=table_material)
    requested_entities = deepcopy(trial["requested"]["entities"])
    requested_entities[table_entity["entity_id"]] = deepcopy(table_entity)
    resolved_entities = deepcopy(resolved_variation_entities["entities"])
    resolved_entities[table_entity["entity_id"]] = deepcopy(table_entity)
    variant_resolved = _variant_resolved_state(trial, resolved_state)
    variant = deepcopy(trial["object_variant"])
    variant["resolved"] = variant_resolved

    materials = {
        role: versioned_config(
            deepcopy(plan["materials"][role]),
            config_version=plan["config_version"],
        )
        for role in ("object", "table", "gripper")
    }
    variation_resolved = deepcopy(trial["variation_resolved"])
    variation_resolved["position_xy_m"] = [
        float(value) for value in resolved_state["position_m"][:2]
    ]
    variation_resolved["yaw_degrees"] = _canonical_cube_yaw_degrees(
        resolved_state["orientation_xyzw"]
    )
    variation_requested = deepcopy(trial["variation_requested"])
    variation_requested["entities"] = requested_entities
    variation_resolved["entities"] = resolved_entities
    demonstration_record = (
        deepcopy(demonstration)
        if demonstration is not None
        else nominal_demonstration(oracle_profile_id=plan["oracle_profile_id"])
    )
    metadata = {
        "schema_version": "farpoint.episode.v4",
        "identity": {
            "episode_id": episode_id,
            "campaign_id": campaign["campaign_id"],
            "segment_id": segment["segment_id"],
            "variation_seed": int(trial["seed"]),
            "attempt_seed": int(attempt_seed),
            "task_id": plan["task_id"],
            "split": trial["split"],
        },
        "provenance": {
            "git_commit": git_commit,
            "campaign_sha256": campaign["campaign_sha256"],
            "segment_sha256": segment["segment_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "simulator_image_digest": simulator_image_digest,
            "oracle_profile_id": plan["oracle_profile_id"],
            "simulator": "Isaac Sim 6.0.0",
            "physics_engine": "PhysX",
        },
        "task": {
            "task_id": plan["task_id"],
            "instruction": "Pick up the cube and place it on the green target pad.",
            "success_criteria_id": "contact_pick_place_footprint_v2",
            "manipulated_entity_id": "pick_object",
            "target_entity_id": "placement_target",
            "acceptance_region_id": "placement_region",
        },
        "embodiment": deepcopy(embodiment),
        "demonstration": demonstration_record,
        "scene": {
            "coordinate_frame": "isaac_world",
            "entities": [object_entity, target_entity, table_entity],
            "object_archetype": deepcopy(trial["object_archetype"]),
            "object_variant": variant,
            "feasible_region": deepcopy(trial["feasible_region"]),
            "materials": materials,
            "lighting_profile_id": plan["lighting_profile_id"],
        },
        "variation": {
            "variation_id": trial["variation_id"],
            "varied_axes": deepcopy(plan["varied_axes"]),
            "frozen_axes": deepcopy(plan["frozen_axes"]),
            "requested": variation_requested,
            "resolved": variation_resolved,
            "units": deepcopy(trial["variation_units"]),
            "config_version": plan["config_version"],
            "config_sha256": plan["config_sha256"],
            "sampler": deepcopy(trial["sampler"]),
            "region_band": trial["region_band"],
            "yaw_stratum_id": trial["yaw_stratum_id"],
            "split": trial["split"],
        },
        "recording": {
            "fps": 30,
            "control_hz": float(control_hz),
            "frame_count": int(frame_count),
            "state_features": deepcopy(embodiment["joint_mapping"]["joint_order"]),
            "action_features": deepcopy(embodiment["joint_mapping"]["joint_order"]),
            "cameras": deepcopy(camera_records),
            "synchronization": {
                "clock": "simulation_control_tick",
                "same_control_tick": True,
                "timestamp_unit": "seconds",
                "maximum_skew_seconds": 0.0,
            },
        },
        "outcome": {
            "success": bool(success),
            "dataset_valid": bool(dataset_valid),
            "failure_category": failure_category,
            "failure_reason": failure_reason,
            "physics_audit": deepcopy(physics_audit),
        },
    }
    errors = validate_contract(metadata) + validate_episode_semantics(metadata)
    if errors:
        raise ValueError("invalid SO-101 episode v4 metadata: " + "; ".join(errors))
    return metadata
