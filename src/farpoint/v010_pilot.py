"""Continuous, object-aware SO-101 v0.1.0 integration pilot plans."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from farpoint.campaign import (
    canonical_sha256,
    create_campaign,
    create_segment,
    variation_seed,
)
from farpoint.object_catalog import MaterialSpec, ObjectArchetype, ObjectVariant
from farpoint.scene_entities import bind_scene_entities
from farpoint.variation_engine import FeasibleRegion, ScrambledSobolSampler


CONFIG_SCHEMA = "farpoint.so101-v010-integration-pilot-config.v1"
PLAN_SCHEMA = "farpoint.so101-v010-integration-pilot-plan.v1"
PILOT_KIND = "v010_integration_pilot"


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _material(value: dict[str, Any]) -> MaterialSpec:
    return MaterialSpec(
        static_friction=float(value["static_friction"]),
        dynamic_friction=float(value["dynamic_friction"]),
        restitution=float(value["restitution"]),
        friction_combine_mode=str(value["friction_combine_mode"]),
        restitution_combine_mode=str(value["restitution_combine_mode"]),
    )


def _archetype(config: dict[str, Any]) -> ObjectArchetype:
    value = config["object_archetype"]
    return ObjectArchetype(
        archetype_id=str(value["archetype_id"]),
        version=str(value["version"]),
        semantic_type=str(value["semantic_type"]),
        geometry_representation=str(value["geometry_representation"]),
        anchor=str(value["anchor"]),
        default_orientation_xyzw=tuple(value["default_orientation_xyzw"]),
    )


def _variants(config: dict[str, Any]) -> dict[str, ObjectVariant]:
    materials = config["materials"]
    values = {}
    for row in config["object_variants"]:
        variant = ObjectVariant(
            variant_id=str(row["variant_id"]),
            version=str(row["version"]),
            archetype_id=str(row["archetype_id"]),
            asset_id=str(row["asset_id"]),
            dimensions_m=tuple(float(value) for value in row["dimensions_m"]),
            rgba=tuple(float(value) for value in row["rgba"]),
            mass_kg=float(row["mass_kg"]),
            object_material=_material(materials["object"]),
            table_material=_material(materials["table"]),
            gripper_material=_material(materials["gripper"]),
        )
        variant.validate()
        if variant.variant_id in values:
            raise ValueError(f"duplicate object variant: {variant.variant_id}")
        values[variant.variant_id] = variant
    return values


def _region(config: dict[str, Any], row: dict[str, Any]) -> FeasibleRegion:
    identity = {
        "config_version": config["config_version"],
        "region_id": row["region_id"],
        "evidence_basis": row["evidence_basis"],
        "formal_eligible": row["formal_eligible"],
    }
    constraints = {
        "polygon_xy_m": row["polygon_xy_m"],
        "max_clearance_m": row["max_clearance_m"],
        "object_variant_id": row["object_variant_id"],
        "footprint_xy_m": row["footprint_xy_m"],
    }
    return FeasibleRegion(
        region_id=str(row["region_id"]),
        version=str(row["version"]),
        frame_id=str(row["frame_id"]),
        polygon_xy_m=tuple(tuple(float(value) for value in point) for point in row["polygon_xy_m"]),
        max_clearance_m=float(row["max_clearance_m"]),
        object_anchor=str(row["object_anchor"]),
        footprint_xy_m=tuple(float(value) for value in row["footprint_xy_m"]),
        generator_sha256=_sha256(identity),
        constraints_sha256=_sha256(constraints),
    )


def _region_record(config: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    region = _region(config, row)
    return {
        "region_id": region.region_id,
        "version": region.version,
        "frame_id": region.frame_id,
        "polygon_xy_m": [list(point) for point in region.polygon_xy_m],
        "max_clearance_m": region.max_clearance_m,
        "object_anchor": region.object_anchor,
        "footprint_xy_m": list(region.footprint_xy_m),
        "generator_sha256": region.generator_sha256,
        "constraints_sha256": region.constraints_sha256,
        "formal_eligible": bool(row["formal_eligible"]),
        "evidence_basis": str(row["evidence_basis"]),
    }


def _archetype_state(value: ObjectArchetype) -> dict[str, Any]:
    return {
        "archetype_id": value.archetype_id,
        "semantic_type": value.semantic_type,
        "geometry_representation": value.geometry_representation,
        "anchor": value.anchor,
    }


def _variant_state(value: ObjectVariant) -> dict[str, Any]:
    return {
        "variant_id": value.variant_id,
        "archetype_id": value.archetype_id,
        "asset_id": value.asset_id,
        "dimensions_m": list(value.dimensions_m),
        "rgba": list(value.rgba),
        "mass_kg": value.mass_kg,
        "object_material": value.object_material.to_dict(),
        "table_material": value.table_material.to_dict(),
        "gripper_material": value.gripper_material.to_dict(),
    }


def versioned_config(
    requested: dict[str, Any],
    *,
    resolved: dict[str, Any] | None = None,
    units: dict[str, str] | None = None,
    config_version: str,
) -> dict[str, Any]:
    """Build a requested/resolved value bound to one canonical config hash."""
    payload = {
        "requested": deepcopy(requested),
        "resolved": deepcopy(requested if resolved is None else resolved),
        "units": deepcopy(units or {}),
        "config_version": str(config_version),
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def validate_v010_pilot_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"v0.1.0 pilot config must use {CONFIG_SCHEMA}")
    if not config.get("config_version") or not config.get("task_id"):
        raise ValueError("v0.1.0 pilot config identity must be non-empty")
    archetype = _archetype(config)
    archetype.validate()
    variants = _variants(config)
    if len(variants) != 2:
        raise ValueError("v0.1.0 pilot requires exactly two object variants")
    if {value.mass_kg for value in variants.values()} != {0.03, 0.04}:
        raise ValueError("v0.1.0 pilot object masses must be 0.03 and 0.04 kg")
    if {value.dimensions_m for value in variants.values()} != {
        (0.03, 0.03, 0.03),
        (0.04, 0.04, 0.04),
    }:
        raise ValueError("v0.1.0 pilot object dimensions must be 30 and 40 mm cubes")

    regions = config.get("feasible_regions") or []
    if {row.get("object_variant_id") for row in regions} != set(variants):
        raise ValueError("every object variant requires one feasible region")
    for row in regions:
        region = _region(config, row)
        variant = variants[row["object_variant_id"]]
        if region.footprint_xy_m != variant.dimensions_m[:2]:
            raise ValueError("feasible-region footprint must match its object variant")
        if row.get("formal_eligible") is not False:
            raise ValueError("integration pilot regions must not claim formal eligibility")

    yaws = config.get("yaw_strata") or []
    yaw_ids = [row.get("yaw_stratum_id") for row in yaws]
    if len(yaws) != 5 or len(set(yaw_ids)) != 5:
        raise ValueError("v0.1.0 pilot requires five unique yaw strata")
    for row in yaws:
        low = float(row["minimum_degrees"])
        high = float(row["maximum_degrees"])
        value = float(row["pilot_degrees"])
        if not 0.0 <= low < value < high <= 90.0:
            raise ValueError("yaw pilot value must lie inside its [0, 90) stratum")

    profiles = config.get("pilot_profiles") or []
    if len(profiles) != 12:
        raise ValueError("v0.1.0 integration pilot requires exactly 12 profiles")
    if any(profile.get("split") not in {"train", "validation"} for profile in profiles):
        raise ValueError("v0.1.0 pilot splits must be train or validation")
    if Counter(profile["split"] for profile in profiles) != Counter(
        {"train": 10, "validation": 2}
    ):
        raise ValueError("v0.1.0 pilot split coverage must be train=10, validation=2")
    for variant_id in variants:
        selected = [row for row in profiles if row.get("object_variant_id") == variant_id]
        if len(selected) != 6:
            raise ValueError("each object variant requires six pilot profiles")
        if Counter(row.get("region_band") for row in selected) != Counter(
            {"core": 2, "middle": 2, "outer": 2}
        ):
            raise ValueError("each object variant requires two profiles per region band")
        if {row.get("yaw_stratum_id") for row in selected} != set(yaw_ids):
            raise ValueError("each object variant must cover all five yaw strata")
    if any(row.get("yaw_stratum_id") not in set(yaw_ids) for row in profiles):
        raise ValueError("pilot profile references an unknown yaw stratum")
    if any(int(row.get("sample_ordinal", -1)) < 0 for row in profiles):
        raise ValueError("pilot sample ordinals must be non-negative")
    if len(
        {
            (
                row["object_variant_id"],
                row["region_band"],
                row["sample_ordinal"],
            )
            for row in profiles
        }
    ) != len(profiles):
        raise ValueError("pilot Sobol sample identities must be unique")

    acceptance = config.get("acceptance") or {}
    if acceptance != {
        "required_successes": 10,
        "maximum_attempts": 12,
        "minimum_successes_per_object_region": 1,
    }:
        raise ValueError("v0.1.0 pilot must freeze the 10-of-12 object-region gate")
    if not config.get("oracle_profile_id") or not config.get("lighting_profile_id"):
        raise ValueError("v0.1.0 pilot requires Oracle and lighting profile ids")


def load_v010_pilot_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_v010_pilot_config(config)
    return config


def _campaign_for_config(
    config: dict[str, Any], *, campaign_id: str
) -> dict[str, Any]:
    profiles = config["pilot_profiles"]
    quota_counts = Counter(
        (
            row["object_variant_id"],
            row["yaw_stratum_id"],
            row["region_band"],
            row["split"],
        )
        for row in profiles
    )
    quotas = [
        {
            "object_variant_id": object_id,
            "yaw_stratum_id": yaw_id,
            "region_band": region,
            "split": split,
            "count": count,
        }
        for (object_id, yaw_id, region, split), count in sorted(quota_counts.items())
    ]
    split_counts = dict(sorted(Counter(row["split"] for row in profiles).items()))
    config_hash = _sha256(config)
    return create_campaign(
        {
            "campaign_id": campaign_id,
            "lineage_id": "farpoint-so101-v010",
            "task_id": config["task_id"],
            "campaign_version": config["config_version"],
            "campaign_kind": "pilot",
            "target": {
                "successful_episodes": len(profiles),
                "splits": split_counts,
            },
            "quotas": quotas,
            "variation_contract": {
                "config_schema": CONFIG_SCHEMA,
                "config_sha256": config_hash,
                "sampler_version": config["sampler"]["sampler_version"],
                "formal_eligible": False,
            },
            "attempt_policy": {
                "maximum_attempts_per_variation": 3,
                "global_attempt_limit": None,
                "replacement_policy": "same_quota_new_variation_seed",
            },
            "watchdog_policy": {"profile": "so101_watchdog_p0"},
            "rollout_holdout": {"scene_count": 20, "disjoint": True},
        }
    )


def build_v010_integration_pilot_plan(
    config: dict[str, Any], *, pilot_id: str
) -> dict[str, Any]:
    """Build 12 continuous Sobol variations with exact object/region/yaw coverage."""
    validate_v010_pilot_config(config)
    if not pilot_id:
        raise ValueError("v0.1.0 pilot_id must be non-empty")
    campaign = _campaign_for_config(config, campaign_id=pilot_id)
    variants = _variants(config)
    archetype = _archetype(config)
    region_rows = {
        row["object_variant_id"]: row for row in config["feasible_regions"]
    }
    yaw_rows = {
        row["yaw_stratum_id"]: row for row in config["yaw_strata"]
    }
    sampler = ScrambledSobolSampler(
        max_candidates_per_sample=int(config["sampler"]["maximum_candidates_per_sample"])
    )
    archetype_versioned = versioned_config(
        _archetype_state(archetype),
        config_version=archetype.version,
    )
    trials = []
    for index, profile in enumerate(config["pilot_profiles"]):
        object_id = profile["object_variant_id"]
        variant = variants[object_id]
        region_row = region_rows[object_id]
        region = _region(config, region_row)
        seed = variation_seed(
            campaign["campaign_sha256"],
            object_variant_id=object_id,
            yaw_stratum_id=profile["yaw_stratum_id"],
            region_band=profile["region_band"],
            split=profile["split"],
            quota_ordinal=int(profile["sample_ordinal"]),
        )
        sampled = sampler.sample(
            region,
            ordinal=int(profile["sample_ordinal"]),
            seed=seed,
            band=profile["region_band"],
        )
        yaw = yaw_rows[profile["yaw_stratum_id"]]
        yaw_degrees = float(yaw["pilot_degrees"])
        half_angle = math.radians(yaw_degrees) / 2.0
        orientation = [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]
        position = [
            *sampled["position_xy_m"],
            0.032 + variant.dimensions_m[2] / 2.0,
        ]
        object_state = {
            "shape": archetype.semantic_type,
            "asset_id": variant.asset_id,
            "dimensions_m": list(variant.dimensions_m),
            "position_m": position,
            "orientation_xyzw": orientation,
            "rgba": list(variant.rgba),
            "mass_kg": variant.mass_kg,
            **variant.object_material.to_dict(),
        }
        requested = bind_scene_entities(object_state, config["target"])
        region_record = _region_record(config, region_row)
        variant_versioned = versioned_config(
            _variant_state(variant),
            units={"dimensions_m": "m", "mass_kg": "kg"},
            config_version=variant.version,
        )
        sampler_requested = {
            "sampler_version": config["sampler"]["sampler_version"],
            "base_seed": int(config["sampler"]["seed"]),
            "scramble_seed": seed,
            "sample_ordinal": int(profile["sample_ordinal"]),
            "requested_region_band": profile["region_band"],
        }
        sampler_versioned = versioned_config(
            sampler_requested,
            resolved={**sampler_requested, **sampled},
            units={"position_xy_m": "m", "normalized_clearance": "ratio"},
            config_version=config["config_version"],
        )
        compact_requested = {
            "object_variant_id": object_id,
            "position_xy_m": sampled["position_xy_m"],
            "yaw_degrees": yaw_degrees,
            "region_band": profile["region_band"],
            "yaw_stratum_id": profile["yaw_stratum_id"],
        }
        token = re.sub(r"[^a-z0-9]+", "_", object_id.lower()).strip("_")
        trial_id = (
            f"{token}_{profile['region_band']}_{profile['yaw_stratum_id']}"
            f"_q{int(profile['sample_ordinal']):02d}"
        )
        trials.append(
            {
                "trial_id": trial_id,
                "variation_id": trial_id,
                "split": profile["split"],
                "seed": seed,
                "seed_material": {
                    "campaign_sha256": campaign["campaign_sha256"],
                    "profile_index": index,
                    **deepcopy(profile),
                },
                "requested": requested,
                "resolved": deepcopy(requested),
                "object_variant_id": object_id,
                "object_archetype": deepcopy(archetype_versioned),
                "object_variant": variant_versioned,
                "feasible_region": versioned_config(
                    region_record,
                    units={
                        "polygon_xy_m": "m",
                        "max_clearance_m": "m",
                        "footprint_xy_m": "m",
                    },
                    config_version=region.version,
                ),
                "sampler": sampler_versioned,
                "region_band": profile["region_band"],
                "yaw_stratum_id": profile["yaw_stratum_id"],
                "object_yaw_degrees": yaw_degrees,
                "variation_requested": compact_requested,
                "variation_resolved": deepcopy(compact_requested),
                "variation_units": {
                    "position_xy_m": "m",
                    "yaw_degrees": "degree",
                    "region_band": "category",
                    "yaw_stratum_id": "category",
                    "object_variant_id": "category",
                },
                "mass_audit_tolerance_kg": 1e-6,
                "projection_cell_id": None,
            }
        )

    required_pairs = sorted(
        {
            f"{trial['object_variant_id']}::{trial['region_band']}"
            for trial in trials
        }
    )
    plan = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": pilot_id,
        "task_id": config["task_id"],
        "config_version": config["config_version"],
        "config_sha256": _sha256(config),
        "campaign_sha256": campaign["campaign_sha256"],
        "campaign_contract": campaign,
        "oracle_profile_id": config["oracle_profile_id"],
        "lighting_profile_id": config["lighting_profile_id"],
        "varied_axes": [
            "object_variant_id",
            "entities.pick_object.pose.position_m.x",
            "entities.pick_object.pose.position_m.y",
            "entities.pick_object.pose.orientation_xyzw",
        ],
        "frozen_axes": [
            "entities.placement_target",
            "entities.support_surface",
            "lighting.profile",
            "camera.profile",
            "success_criteria",
        ],
        "target": deepcopy(config["target"]),
        "table": deepcopy(config["table"]),
        "materials": deepcopy(config["materials"]),
        "trials": trials,
        "pilot": {
            "kind": PILOT_KIND,
            "required_successes": 10,
            "maximum_attempts": 12,
            "trial_ids": [trial["trial_id"] for trial in trials],
            "required_object_region_pairs": required_pairs,
            "minimum_successes_per_object_region": 1,
            "required_yaw_strata": [
                row["yaw_stratum_id"] for row in config["yaw_strata"]
            ],
            "coverage": {
                "objects": dict(sorted(Counter(trial["object_variant_id"] for trial in trials).items())),
                "regions": dict(sorted(Counter(trial["region_band"] for trial in trials).items())),
                "yaw_strata": dict(sorted(Counter(trial["yaw_stratum_id"] for trial in trials).items())),
                "splits": dict(sorted(Counter(trial["split"] for trial in trials).items())),
            },
        },
    }
    plan["plan_sha256"] = _sha256(plan)
    return plan


def initialize_v010_pilot_campaign(
    root: str | Path,
    plan: dict[str, Any],
    *,
    git_commit: str,
    segment_id: str = "segment-000",
) -> dict[str, Any]:
    """Write immutable campaign and first-segment declarations beside a workflow."""
    if (plan.get("pilot") or {}).get("kind") != PILOT_KIND:
        raise ValueError("plan is not a v0.1.0 integration pilot")
    destination = Path(root)
    campaign = deepcopy(plan["campaign_contract"])
    if campaign["campaign_sha256"] != plan["campaign_sha256"]:
        raise ValueError("plan campaign hash is inconsistent")
    segment = create_segment(
        {
            "campaign_id": campaign["campaign_id"],
            "campaign_sha256": campaign["campaign_sha256"],
            "segment_id": segment_id,
            "segment_index": 0,
            "git_commit": git_commit,
            "plan_sha256": plan["plan_sha256"],
            "parent_manifest_sha256": None,
            "oracle_profile_allowlist": [plan["oracle_profile_id"]],
            "execution_status": "RUNNING",
            "quality_status": "NOT_EVALUATED",
            "attempts": [],
        }
    )
    campaign_path = destination / "campaign.json"
    segment_path = destination / "segments" / segment_id / "segment.json"
    if campaign_path.exists() or segment_path.exists():
        raise FileExistsError("v0.1.0 campaign declaration already exists")
    campaign_path.parent.mkdir(parents=True, exist_ok=True)
    segment_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_path.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    segment_path.write_text(
        json.dumps(segment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"campaign": campaign, "segment": segment}
