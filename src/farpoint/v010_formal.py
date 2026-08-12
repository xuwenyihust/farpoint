"""Frozen 200-scene SO-101 v0.1.0 campaign and rollout holdout planning."""

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
    build_crossed_quotas,
    create_campaign,
    create_segment,
    validate_campaign_semantics,
    variation_seed,
)
from farpoint.scene_entities import bind_scene_entities
from farpoint.variation_engine import ScrambledSobolSampler
from farpoint.v010_pilot import (
    _archetype,
    _archetype_state,
    _region,
    _region_record,
    _variant_state,
    _variants,
    validate_v010_pilot_config,
    versioned_config,
)


CONFIG_SCHEMA = "farpoint.so101-v010-formal-config.v1"
PLAN_SCHEMA = "farpoint.so101-v010-formal-plan.v1"
FORMAL_KIND = "self_healing_campaign_segment"


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit_fraction(seed: int, domain: str) -> float:
    material = f"farpoint-v010-{domain}:{seed}".encode()
    integer = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return (integer + 0.5) / (1 << 64)


def validate_v010_formal_config(
    config: dict[str, Any], base_config: dict[str, Any]
) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"v0.1.0 formal config must use {CONFIG_SCHEMA}")
    validate_v010_pilot_config(base_config)
    if _sha256(base_config) != (config.get("base_variation_config") or {}).get(
        "config_sha256"
    ):
        raise ValueError("formal config base variation hash does not match")
    if config.get("target") != {
        "successful_episodes": 200,
        "splits": {"train": 180, "validation": 20},
    }:
        raise ValueError("formal target must be exactly 200 with train=180 validation=20")
    if config.get("population_per_object_yaw_region") != {
        "core": 5,
        "middle": 10,
        "outer": 5,
    }:
        raise ValueError("formal region population must be core/middle/outer 5/10/5")
    if config.get("validation_per_object_region") != {
        "core": 2,
        "middle": 5,
        "outer": 3,
    }:
        raise ValueError("formal validation regions must be core/middle/outer 2/5/3")
    if int(config.get("validation_per_object_yaw", -1)) != 2:
        raise ValueError("formal validation requires two scenes per object/yaw")
    if config.get("attempt_policy") != {
        "maximum_attempts_per_variation": 3,
        "global_attempt_limit": None,
        "replacement_policy": "same_quota_new_variation_seed",
    }:
        raise ValueError("formal attempt policy must freeze three attempts and no global limit")
    holdout = config.get("rollout_holdout") or {}
    if holdout.get("scene_count") != 20 or holdout.get("disjoint") is not True:
        raise ValueError("formal rollout holdout must contain 20 disjoint scenes")
    authorization = config.get("pilot_authorization") or {}
    hashes = ("plan_sha256", "manifest_sha256", "report_sha256")
    if any(
        not isinstance(authorization.get(field), str)
        or len(authorization[field]) != 64
        for field in hashes
    ):
        raise ValueError("pilot authorization must bind plan, manifest, and report hashes")
    if authorization.get("required_cameras") != ["front", "wrist"]:
        raise ValueError("formal authorization requires front and wrist cameras")
    for field in ("runtime_watchdog", "self_healing_policy"):
        binding = config.get(field) or {}
        if not binding.get("path") or not isinstance(binding.get("sha256"), str) or len(
            binding["sha256"]
        ) != 64:
            raise ValueError(f"{field} must bind a path and SHA256")


def load_v010_formal_config(
    path: str | Path, *, project_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root)
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    base_path = root / config["base_variation_config"]["path"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    validate_v010_formal_config(config, base)
    for field in ("runtime_watchdog", "self_healing_policy"):
        binding = config[field]
        if _file_sha256(root / binding["path"]) != binding["sha256"]:
            raise ValueError(f"{field} SHA256 does not match")
    return config, base


def validate_pilot_authorization(
    config: dict[str, Any], *, report_path: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    """Fail closed unless the exact immutable pilot evidence passed its gate."""
    expected = config["pilot_authorization"]
    if _file_sha256(report_path) != expected["report_sha256"]:
        raise ValueError("pilot report SHA256 does not match formal authorization")
    if _file_sha256(manifest_path) != expected["manifest_sha256"]:
        raise ValueError("pilot manifest SHA256 does not match formal authorization")
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if report.get("pilot_status") != "PASS":
        raise ValueError("pilot report did not pass")
    if report.get("collection_id") != expected["collection_id"]:
        raise ValueError("pilot report collection identity mismatch")
    if report.get("git_commit") != expected["git_commit"]:
        raise ValueError("pilot report git commit mismatch")
    if report.get("plan_sha256") != expected["plan_sha256"]:
        raise ValueError("pilot report plan hash mismatch")
    if report.get("required_cameras") != expected["required_cameras"]:
        raise ValueError("pilot report camera gate mismatch")
    if int(report.get("success_count", 0)) < int(expected["required_successes"]):
        raise ValueError("pilot report success gate was not met")
    if int(report.get("independent_episode_identity_count", 0)) != int(
        expected["required_attempted_identities"]
    ):
        raise ValueError("pilot report identity count mismatch")
    if report.get("acceptance_errors") or report.get("evidence_errors"):
        raise ValueError("pilot report contains validation errors")
    if manifest.get("execution_status") != "FINISHED" or manifest.get(
        "quality_status"
    ) != "PASS":
        raise ValueError("pilot manifest is not FINISHED/PASS")
    if manifest.get("git_commit") != expected["git_commit"] or manifest.get(
        "plan_sha256"
    ) != expected["plan_sha256"]:
        raise ValueError("pilot manifest provenance mismatch")
    if len(manifest.get("attempts") or []) != int(
        expected["required_attempted_identities"]
    ):
        raise ValueError("pilot manifest attempt count mismatch")
    return {
        "collection_id": expected["collection_id"],
        "git_commit": expected["git_commit"],
        "plan_sha256": expected["plan_sha256"],
        "manifest_sha256": expected["manifest_sha256"],
        "report_sha256": expected["report_sha256"],
        "pilot_status": report["pilot_status"],
        "success_count": int(report["success_count"]),
        "attempted_identity_count": int(report["independent_episode_identity_count"]),
        "required_cameras": deepcopy(report["required_cameras"]),
    }


def _formalized_base(
    config: dict[str, Any], base_config: dict[str, Any], authorization: dict[str, Any]
) -> dict[str, Any]:
    base = deepcopy(base_config)
    base["config_version"] = config["config_version"]
    base["sampler"] = deepcopy(config["sampler"])
    for row in base["feasible_regions"]:
        row["formal_eligible"] = True
        row["evidence_basis"] = (
            f"integration pilot {authorization['collection_id']} PASS; "
            f"report_sha256={authorization['report_sha256']}"
        )
    return base


def _orientation(yaw_degrees: float) -> list[float]:
    half_angle = math.radians(yaw_degrees) / 2.0
    return [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]


def _trial_record(
    *,
    config: dict[str, Any],
    base: dict[str, Any],
    campaign: dict[str, Any],
    object_id: str,
    yaw_row: dict[str, Any],
    region_band: str,
    split: str,
    quota_ordinal: int,
    replacement_index: int,
    seed: int,
    sample_ordinal: int,
    trial_prefix: str,
) -> dict[str, Any]:
    variants = _variants(base)
    archetype = _archetype(base)
    region_row = next(
        row for row in base["feasible_regions"] if row["object_variant_id"] == object_id
    )
    region = _region(base, region_row)
    sampler = ScrambledSobolSampler(
        max_candidates_per_sample=int(config["sampler"]["maximum_candidates_per_sample"])
    )
    sampled = sampler.sample(
        region, ordinal=sample_ordinal, seed=seed, band=region_band
    )
    low = float(yaw_row["minimum_degrees"])
    high = float(yaw_row["maximum_degrees"])
    yaw_degrees = low + (high - low) * _unit_fraction(seed, "yaw")
    variant = variants[object_id]
    object_state = {
        "shape": archetype.semantic_type,
        "asset_id": variant.asset_id,
        "dimensions_m": list(variant.dimensions_m),
        "position_m": [
            *sampled["position_xy_m"],
            0.032 + variant.dimensions_m[2] / 2.0,
        ],
        "orientation_xyzw": _orientation(yaw_degrees),
        "rgba": list(variant.rgba),
        "mass_kg": variant.mass_kg,
        **variant.object_material.to_dict(),
    }
    requested = bind_scene_entities(object_state, base["target"])
    region_record = _region_record(base, region_row)
    sampler_requested = {
        "sampler_version": config["sampler"]["sampler_version"],
        "base_seed": int(config["sampler"]["seed"]),
        "scramble_seed": seed,
        "sample_ordinal": sample_ordinal,
        "requested_region_band": region_band,
    }
    compact = {
        "object_variant_id": object_id,
        "position_xy_m": sampled["position_xy_m"],
        "yaw_degrees": yaw_degrees,
        "region_band": region_band,
        "yaw_stratum_id": yaw_row["yaw_stratum_id"],
    }
    token = re.sub(r"[^a-z0-9]+", "_", object_id.lower()).strip("_")
    trial_id = (
        f"{trial_prefix}_{token}_{region_band}_{yaw_row['yaw_stratum_id']}_"
        f"{split}_q{quota_ordinal:02d}_r{replacement_index:02d}"
    )
    return {
        "trial_id": trial_id,
        "variation_id": trial_id,
        "split": split,
        "seed": seed,
        "seed_material": {
            "campaign_sha256": campaign["campaign_sha256"],
            "object_variant_id": object_id,
            "yaw_stratum_id": yaw_row["yaw_stratum_id"],
            "region_band": region_band,
            "split": split,
            "quota_ordinal": quota_ordinal,
            "replacement_index": replacement_index,
        },
        "quota_ordinal": quota_ordinal,
        "replacement_index": replacement_index,
        "requested": requested,
        "resolved": deepcopy(requested),
        "object_variant_id": object_id,
        "object_archetype": versioned_config(
            _archetype_state(archetype), config_version=archetype.version
        ),
        "object_variant": versioned_config(
            _variant_state(variant),
            units={"dimensions_m": "m", "mass_kg": "kg"},
            config_version=variant.version,
        ),
        "feasible_region": versioned_config(
            region_record,
            units={
                "polygon_xy_m": "m",
                "max_clearance_m": "m",
                "footprint_xy_m": "m",
            },
            config_version=region.version,
        ),
        "sampler": versioned_config(
            sampler_requested,
            resolved={**sampler_requested, **sampled},
            units={"position_xy_m": "m", "normalized_clearance": "ratio"},
            config_version=config["config_version"],
        ),
        "region_band": region_band,
        "yaw_stratum_id": yaw_row["yaw_stratum_id"],
        "object_yaw_degrees": yaw_degrees,
        "variation_requested": compact,
        "variation_resolved": deepcopy(compact),
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


def _campaign_contract(
    config: dict[str, Any], base: dict[str, Any], *, campaign_id: str
) -> dict[str, Any]:
    objects = [row["variant_id"] for row in base["object_variants"]]
    yaws = [row["yaw_stratum_id"] for row in base["yaw_strata"]]
    quotas = build_crossed_quotas(
        object_variant_ids=objects,
        yaw_stratum_ids=yaws,
        population_per_yaw_region=config["population_per_object_yaw_region"],
        validation_per_object_region=config["validation_per_object_region"],
        validation_per_object_yaw=int(config["validation_per_object_yaw"]),
        seed=int(config["sampler"]["seed"]),
    )
    return create_campaign(
        {
            "campaign_id": campaign_id,
            "lineage_id": "farpoint-so101-v010",
            "task_id": base["task_id"],
            "campaign_version": config["config_version"],
            "campaign_kind": "formal",
            "target": deepcopy(config["target"]),
            "quotas": quotas,
            "variation_contract": {
                "config_schema": CONFIG_SCHEMA,
                "config_sha256": _sha256(config),
                "base_config_sha256": config["base_variation_config"]["config_sha256"],
                "sampler_version": config["sampler"]["sampler_version"],
                "formal_eligible": True,
                "pilot_report_sha256": config["pilot_authorization"]["report_sha256"],
            },
            "attempt_policy": deepcopy(config["attempt_policy"]),
            "watchdog_policy": deepcopy(config["runtime_watchdog"]),
            "rollout_holdout": deepcopy(config["rollout_holdout"]),
        }
    )


def build_v010_formal_plan(
    config: dict[str, Any],
    base_config: dict[str, Any],
    authorization: dict[str, Any],
    *,
    campaign_id: str,
) -> dict[str, Any]:
    """Freeze exact quotas, continuous scenes, and disjoint rollout holdouts."""
    validate_v010_formal_config(config, base_config)
    base = _formalized_base(config, base_config, authorization)
    campaign = _campaign_contract(config, base, campaign_id=campaign_id)
    yaw_rows = {row["yaw_stratum_id"]: row for row in base["yaw_strata"]}
    trials = []
    for quota in campaign["quotas"]:
        for ordinal in range(int(quota["count"])):
            seed = variation_seed(
                campaign["campaign_sha256"],
                object_variant_id=quota["object_variant_id"],
                yaw_stratum_id=quota["yaw_stratum_id"],
                region_band=quota["region_band"],
                split=quota["split"],
                quota_ordinal=ordinal,
                replacement_index=0,
            )
            trials.append(
                _trial_record(
                    config=config,
                    base=base,
                    campaign=campaign,
                    object_id=quota["object_variant_id"],
                    yaw_row=yaw_rows[quota["yaw_stratum_id"]],
                    region_band=quota["region_band"],
                    split=quota["split"],
                    quota_ordinal=ordinal,
                    replacement_index=0,
                    seed=seed,
                    sample_ordinal=ordinal,
                    trial_prefix="formal",
                )
            )

    holdout_scenes = []
    holdout_regions = (
        "core",
        "middle",
        "outer",
        "middle",
        "outer",
        "core",
        "middle",
        "outer",
        "core",
        "middle",
    )
    training_seeds = {trial["seed"] for trial in trials}
    for object_index, object_id in enumerate(
        row["variant_id"] for row in base["object_variants"]
    ):
        for local_index in range(10):
            yaw_row = base["yaw_strata"][local_index // 2]
            seed_material = (
                f"farpoint-v010-holdout:{campaign['campaign_sha256']}:"
                f"{config['rollout_holdout']['seed']}:{object_id}:{local_index}"
            )
            seed = int.from_bytes(
                hashlib.sha256(seed_material.encode()).digest()[:8], "big"
            ) | (1 << 63)
            scene = _trial_record(
                config=config,
                base=base,
                campaign=campaign,
                object_id=object_id,
                yaw_row=yaw_row,
                region_band=holdout_regions[local_index],
                split="rollout_holdout",
                quota_ordinal=local_index,
                replacement_index=0,
                seed=seed,
                sample_ordinal=1000 + object_index * 10 + local_index,
                trial_prefix="holdout",
            )
            scene["scene_id"] = scene.pop("trial_id")
            scene.pop("variation_id")
            holdout_scenes.append(scene)
    holdout_seeds = {scene["seed"] for scene in holdout_scenes}
    if len(trials) != 200 or len(training_seeds) != 200:
        raise ValueError("formal plan must contain 200 unique variation seeds")
    if len(holdout_scenes) != 20 or len(holdout_seeds) != 20:
        raise ValueError("rollout holdout must contain 20 unique seeds")
    if training_seeds & holdout_seeds:
        raise ValueError("rollout holdout seeds overlap formal collection")

    plan = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": f"{campaign_id}_segment_000",
        "task_id": base["task_id"],
        "config_version": config["config_version"],
        "config_sha256": _sha256(config),
        "base_config_sha256": config["base_variation_config"]["config_sha256"],
        "campaign_sha256": campaign["campaign_sha256"],
        "campaign_contract": campaign,
        "pilot_authorization": deepcopy(authorization),
        "oracle_profile_id": base["oracle_profile_id"],
        "lighting_profile_id": base["lighting_profile_id"],
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
        "target": deepcopy(base["target"]),
        "table": deepcopy(base["table"]),
        "materials": deepcopy(base["materials"]),
        "trials": trials,
        "rollout_holdout": {
            **deepcopy(config["rollout_holdout"]),
            "scenes": holdout_scenes,
        },
        "collection": {
            "kind": FORMAL_KIND,
            "required_successes": len(trials),
            "maximum_attempts": len(trials) * 3,
            "attempt_policy": deepcopy(config["attempt_policy"]),
            "runtime_watchdog": deepcopy(config["runtime_watchdog"]),
            "self_healing_policy": deepcopy(config["self_healing_policy"]),
        },
        "coverage": {
            "objects": dict(sorted(Counter(row["object_variant_id"] for row in trials).items())),
            "regions": dict(sorted(Counter(row["region_band"] for row in trials).items())),
            "yaw_strata": dict(sorted(Counter(row["yaw_stratum_id"] for row in trials).items())),
            "splits": dict(sorted(Counter(row["split"] for row in trials).items())),
        },
    }
    plan["plan_sha256"] = _sha256(plan)
    return plan


def build_v010_replacement_plan(
    config: dict[str, Any],
    base_config: dict[str, Any],
    authorization: dict[str, Any],
    campaign: dict[str, Any],
    replacement_requests: list[dict[str, Any]],
    *,
    segment_id: str,
) -> dict[str, Any]:
    """Materialize immutable carryover and replacement requests as a new plan."""
    validate_v010_formal_config(config, base_config)
    if not replacement_requests:
        raise ValueError("replacement plan requires at least one request")
    campaign_errors = validate_campaign_semantics(campaign)
    if campaign_errors:
        raise ValueError("invalid campaign: " + "; ".join(campaign_errors))
    if (campaign.get("variation_contract") or {}).get("config_sha256") != _sha256(
        config
    ):
        raise ValueError("campaign does not belong to formal config")
    base = _formalized_base(config, base_config, authorization)
    yaw_rows = {row["yaw_stratum_id"]: row for row in base["yaw_strata"]}
    trials = []
    for request in replacement_requests:
        quota = request["quota"]
        trial = _trial_record(
            config=config,
            base=base,
            campaign=campaign,
            object_id=quota["object_variant_id"],
            yaw_row=yaw_rows[quota["yaw_stratum_id"]],
            region_band=quota["region_band"],
            split=quota["split"],
            quota_ordinal=int(quota["quota_ordinal"]),
            replacement_index=int(request["replacement_index"]),
            seed=int(request["variation_seed"]),
            sample_ordinal=int(quota["quota_ordinal"]),
            trial_prefix=segment_id,
        )
        trial["prior_attempt_count"] = int(request.get("prior_attempt_count", 0))
        trial["continuation_provenance"] = {
            "request_kind": request.get("request_kind", "replacement"),
            "source_segment_id": request.get("source_segment_id"),
            "source_variation_id": request.get(
                "source_variation_id", request.get("deferred_variation_id")
            ),
        }
        trials.append(trial)
    maximum_attempts = sum(
        3 - int(trial.get("prior_attempt_count", 0)) for trial in trials
    )
    plan = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": f"{campaign['campaign_id']}_{segment_id}",
        "task_id": base["task_id"],
        "config_version": config["config_version"],
        "config_sha256": _sha256(config),
        "base_config_sha256": config["base_variation_config"]["config_sha256"],
        "campaign_sha256": campaign["campaign_sha256"],
        "campaign_contract": deepcopy(campaign),
        "pilot_authorization": deepcopy(authorization),
        "oracle_profile_id": base["oracle_profile_id"],
        "lighting_profile_id": base["lighting_profile_id"],
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
        "target": deepcopy(base["target"]),
        "table": deepcopy(base["table"]),
        "materials": deepcopy(base["materials"]),
        "trials": trials,
        "rollout_holdout": deepcopy(campaign["rollout_holdout"]),
        "collection": {
            "kind": FORMAL_KIND,
            "required_successes": len(trials),
            "maximum_attempts": maximum_attempts,
            "attempt_policy": deepcopy(config["attempt_policy"]),
            "runtime_watchdog": deepcopy(config["runtime_watchdog"]),
            "self_healing_policy": deepcopy(config["self_healing_policy"]),
        },
        "coverage": {
            "objects": dict(sorted(Counter(row["object_variant_id"] for row in trials).items())),
            "regions": dict(sorted(Counter(row["region_band"] for row in trials).items())),
            "yaw_strata": dict(sorted(Counter(row["yaw_stratum_id"] for row in trials).items())),
            "splits": dict(sorted(Counter(row["split"] for row in trials).items())),
        },
        "replacement_requests": deepcopy(replacement_requests),
    }
    plan["plan_sha256"] = _sha256(plan)
    return plan


def initialize_v010_formal_campaign(
    root: str | Path, plan: dict[str, Any], *, git_commit: str
) -> dict[str, Any]:
    """Write the immutable campaign, first segment, plan, and evidence index."""
    if (plan.get("collection") or {}).get("kind") != FORMAL_KIND:
        raise ValueError("plan is not a v0.1.0 formal self-healing segment")
    destination = Path(root)
    paths = (
        destination / "campaign.json",
        destination / "segments/segment-000/segment.json",
        destination / "segments/segment-000/plan.json",
        destination / "evidence-index.json",
    )
    if any(path.exists() for path in paths):
        raise FileExistsError("v0.1.0 formal campaign declaration already exists")
    campaign = deepcopy(plan["campaign_contract"])
    segment = create_segment(
        {
            "campaign_id": campaign["campaign_id"],
            "campaign_sha256": campaign["campaign_sha256"],
            "segment_id": "segment-000",
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
    evidence_index = {
        "schema_version": "farpoint.campaign-evidence-index.v1",
        "campaign_id": campaign["campaign_id"],
        "segments": [
            {
                "segment": "segments/segment-000/segment.json",
                "plan": "segments/segment-000/plan.json",
                "manifest": "segments/segment-000/manifest.json",
            }
        ],
    }
    values = (campaign, segment, plan, evidence_index)
    for path, value in zip(paths, values, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return {"campaign": campaign, "segment": segment, "plan": plan}
