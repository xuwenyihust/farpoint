"""SO-101 v0.2.0 target/camera variation plans on the existing campaign engine."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from farpoint.campaign import create_campaign, create_segment, generic_variation_seed
from farpoint.camera_profiles import load_camera_profile, validate_camera_profile
from farpoint.scene_entities import bind_scene_entities
from farpoint.variation_engine import DeterministicLatinHypercubeSampler
from farpoint.v010_pilot import _archetype, _archetype_state, _variant_state, _variants, versioned_config


CONFIG_SCHEMA = "farpoint.so101-v020-config.v1"
PLAN_SCHEMA = "farpoint.so101-v020-plan.v1"
PLAN_MODES = {"pad-pilot", "combined-pilot", "formal"}
AUTHORIZATION_SCHEMA = "farpoint.so101-v020-pilot-authorization.v1"
YAW_STRATA = ("yaw00_18", "yaw18_36", "yaw36_54", "yaw54_72", "yaw72_90")
PILOT_ATTEMPT_BUDGETS = {"pad-pilot": 18, "combined-pilot": 45}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _add(left: list[float], right: list[float]) -> list[float]:
    return [round(float(a) + float(b), 9) for a, b in zip(left, right)]


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _view_angle_degrees(eye: list[float], look: list[float], base_eye: list[float], base_look: list[float]) -> float:
    first = [look[index] - eye[index] for index in range(3)]
    second = [base_look[index] - base_eye[index] for index in range(3)]
    cosine = sum(a * b for a, b in zip(first, second)) / (_norm(first) * _norm(second))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _look_at_quaternion_xyzw(eye: list[float], look: list[float]) -> list[float]:
    """Return an OpenGL camera rotation whose local -Z axis points at look."""
    forward = [look[index] - eye[index] for index in range(3)]
    length = _norm(forward)
    forward = [value / length for value in forward]
    world_up = [0.0, 0.0, 1.0]
    right = [
        forward[1] * world_up[2] - forward[2] * world_up[1],
        forward[2] * world_up[0] - forward[0] * world_up[2],
        forward[0] * world_up[1] - forward[1] * world_up[0],
    ]
    right_length = _norm(right)
    right = [value / right_length for value in right]
    up = [
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ]
    matrix = (
        (right[0], up[0], -forward[0]),
        (right[1], up[1], -forward[1]),
        (right[2], up[2], -forward[2]),
    )
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = [
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
            0.25 * scale,
        ]
    else:
        axis = max(range(3), key=lambda index: matrix[index][index])
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
            quaternion = [0.25 * scale, (matrix[0][1] + matrix[1][0]) / scale, (matrix[0][2] + matrix[2][0]) / scale, (matrix[2][1] - matrix[1][2]) / scale]
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
            quaternion = [(matrix[0][1] + matrix[1][0]) / scale, 0.25 * scale, (matrix[1][2] + matrix[2][1]) / scale, (matrix[0][2] - matrix[2][0]) / scale]
        else:
            scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
            quaternion = [(matrix[0][2] + matrix[2][0]) / scale, (matrix[1][2] + matrix[2][1]) / scale, 0.25 * scale, (matrix[1][0] - matrix[0][1]) / scale]
    norm = _norm(quaternion)
    return [round(value / norm, 9) for value in quaternion]


def load_v020_config(path: str | Path, *, project_root: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_v020_config(config, project_root=project_root)
    return config


def validate_v020_config(config: dict[str, Any], *, project_root: str | Path) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"v0.2.0 config must use {CONFIG_SCHEMA}")
    if config.get("target_successful_episodes") != 300 or config.get("maximum_attempts") != 450:
        raise ValueError("v0.2.0 must freeze a 300-success/450-attempt budget")
    if config.get("slots_per_cell") != 10 or config.get("split_per_cell") != {"train": 9, "validation": 1}:
        raise ValueError("v0.2.0 must freeze nine train and one validation slot per cell")
    variants = _variants(config)
    if set(variants) != {"blue-30mm-30g", "red-40mm-40g"}:
        raise ValueError("v0.2.0 requires the paired blue and red cube variants")
    targets = config.get("target_profiles") or []
    cameras = config.get("camera_profiles") or []
    if len(targets) != 3 or len({row["target_profile_id"] for row in targets}) != 3:
        raise ValueError("v0.2.0 requires three unique target profiles")
    if len(cameras) != 5 or len({row["camera_profile_id"] for row in cameras}) != 5:
        raise ValueError("v0.2.0 requires five unique camera profiles")
    if config.get("target_pad_candidates_m") != [[0.09, 0.09, 0.01], [0.10, 0.09, 0.01]]:
        raise ValueError("v0.2.0 target pad candidates drifted")
    base_eye = config["front_camera_base"]["eye_m"]
    base_look = config["front_camera_base"]["look_at_m"]
    for row in cameras:
        eye_offset = row["eye_offset_m"]
        look_offset = row["look_at_offset_m"]
        if _norm(eye_offset) > 0.025 + 1e-9 or _norm(look_offset) > 0.015 + 1e-9:
            raise ValueError("camera profile exceeds the frozen translation envelope")
        angle = _view_angle_degrees(_add(base_eye, eye_offset), _add(base_look, look_offset), base_eye, base_look)
        if row["camera_profile_id"] == "front-nominal":
            if angle > 1e-5:
                raise ValueError("nominal camera profile must not move")
        elif not 2.0 <= angle <= 4.0:
            raise ValueError(f"camera profile angle must be 2-4 degrees, got {angle:.3f}")
    load_camera_profile(Path(project_root) / config["camera_base_profile"])


def _target(config: dict[str, Any], profile: dict[str, Any], dimensions: list[float]) -> dict[str, Any]:
    return {
        "target_id": "green_rectangular_pad_v2",
        "asset_id": "green_rectangular_pad_v2",
        "entity_type": "pad", "representation": "procedural", "shape": "cuboid", "relation": "on",
        "position_m": deepcopy(profile["position_m"]),
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "dimensions_m": deepcopy(dimensions),
        "footprint_margin_m": float(config["footprint_margin_m"]),
        "rgba": [0.08, 0.70, 0.20, 1.0],
    }


def _effective_camera_profile(config: dict[str, Any], base: dict[str, Any], profile: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    eye = _add(config["front_camera_base"]["eye_m"], profile["eye_offset_m"])
    look = _add(config["front_camera_base"]["look_at_m"], profile["look_at_offset_m"])
    effective = deepcopy(base)
    effective["profile_id"] = f"so101-front-wrist-v020::{profile['camera_profile_id']}"
    front = effective["cameras"][0]
    front["mount"]["position_m"] = eye
    front["mount"]["orientation_xyzw"] = _look_at_quaternion_xyzw(eye, look)
    front["mount"]["look_at_m"] = look
    errors = validate_camera_profile(effective)
    if errors:
        raise ValueError("invalid resolved camera profile: " + "; ".join(errors))
    return effective, {"camera_profile_id": profile["camera_profile_id"], "eye_m": eye, "look_at_m": look}


def _region_band(x: float, y: float, config: dict[str, Any]) -> str:
    x0, x1 = config["sampler"]["x_bounds_m"]
    y0, y1 = config["sampler"]["y_bounds_m"]
    clearance = min(x - x0, x1 - x, y - y0, y1 - y)
    normalized = clearance / min((x1 - x0) / 2.0, (y1 - y0) / 2.0)
    return "outer" if normalized < 1 / 3 else ("middle" if normalized < 2 / 3 else "core")


def _campaign(config: dict[str, Any], *, plan_id: str, cells: list[tuple[str, str, str]], per_cell: int, mode: str) -> dict[str, Any]:
    quotas = []
    train = 0
    validation = 0
    for object_id, target_id, camera_id in cells:
        counts = {"train": per_cell - 1, "validation": 1} if per_cell == 10 else {"validation": per_cell}
        for split, count in counts.items():
            quotas.append({"object_variant_id": object_id, "target_profile_id": target_id, "camera_profile_id": camera_id, "split": split, "count": count})
            train += count if split == "train" else 0
            validation += count if split == "validation" else 0
    target_count = train + validation
    return create_campaign({
        "campaign_id": plan_id, "lineage_id": "farpoint-so101-v020", "task_id": config["task_id"],
        "campaign_version": config["config_version"], "campaign_kind": "formal" if mode == "formal" else "pilot",
        "target": {"successful_episodes": target_count, "splits": {key: value for key, value in (("train", train), ("validation", validation)) if value}},
        "quota_identity_fields": ["object_variant_id", "target_profile_id", "camera_profile_id", "split"],
        "quotas": quotas,
        "variation_contract": {"config_schema": CONFIG_SCHEMA, "config_sha256": canonical_sha256(config), "sampler_version": config["sampler"]["sampler_version"], "frozen_semantics": True},
        "attempt_policy": {
            "maximum_attempts_per_variation": 3,
            "global_attempt_limit": (
                450 if mode == "formal" else PILOT_ATTEMPT_BUDGETS[mode]
            ),
            "replacement_policy": "same_quota_new_variation_seed",
        },
        "watchdog_policy": {"path": config["runtime_watchdog"]},
        "rollout_holdout": {"scene_count": 20, "disjoint": True},
    })


def build_v020_plan(
    config: dict[str, Any],
    *,
    project_root: str | Path,
    plan_id: str,
    mode: str,
    pad_dimensions_m: list[float] | None = None,
    pilot_authorization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_v020_config(config, project_root=project_root)
    if mode not in PLAN_MODES or not plan_id:
        raise ValueError(f"mode must be one of {sorted(PLAN_MODES)} and plan_id non-empty")
    dimensions = deepcopy(pad_dimensions_m or config["target_pad_candidates_m"][0])
    if dimensions not in config["target_pad_candidates_m"]:
        raise ValueError("pad dimensions are outside the frozen candidates")
    if mode == "formal":
        validate_v020_pilot_authorization(
            pilot_authorization or {}, config=config, pad_dimensions_m=dimensions
        )
    objects = [row["variant_id"] for row in config["object_variants"]]
    targets = [row["target_profile_id"] for row in config["target_profiles"]]
    cameras = [row["camera_profile_id"] for row in config["camera_profiles"]]
    all_cells = [(obj, target, camera) for obj in objects for target in targets for camera in cameras]
    if mode == "pad-pilot":
        extreme = ("front-x-positive", "front-yz-negative")
        cells = [(obj, target, camera) for obj in objects for target in targets for camera in extreme]
        per_cell = 1
    elif mode == "combined-pilot":
        cells, per_cell = all_cells, 1
    else:
        cells, per_cell = all_cells, 10
    campaign = _campaign(config, plan_id=plan_id, cells=cells, per_cell=per_cell, mode=mode)
    variants = _variants(config)
    archetype = _archetype(config)
    target_rows = {row["target_profile_id"]: row for row in config["target_profiles"]}
    camera_rows = {row["camera_profile_id"]: row for row in config["camera_profiles"]}
    base_camera = load_camera_profile(Path(project_root) / config["camera_base_profile"])
    trials = []
    for cell_index, (object_id, target_id, camera_id) in enumerate(cells):
        cell_material = f"farpoint-v020-cell:{config['seed']}:{object_id}:{target_id}:{camera_id}".encode()
        cell_seed = int.from_bytes(hashlib.sha256(cell_material).digest()[:8], "big") & ((1 << 63) - 1)
        sampler = DeterministicLatinHypercubeSampler(
            bounds=(("x_m", *config["sampler"]["x_bounds_m"]), ("y_m", *config["sampler"]["y_bounds_m"]), ("yaw_degrees", *config["sampler"]["yaw_bounds_degrees"])),
            population=10, seed=cell_seed,
        )
        slots = range(10) if per_cell == 10 else [cell_index % 10]
        for local_index, slot in enumerate(slots):
            split = "validation" if per_cell == 1 or slot == 9 else "train"
            quota_ordinal = 0 if split == "validation" else slot
            quota = {"object_variant_id": object_id, "target_profile_id": target_id, "camera_profile_id": camera_id, "split": split}
            seed = generic_variation_seed(campaign["campaign_sha256"], quota=quota, quota_ordinal=quota_ordinal)
            sample = sampler.sample(slot)
            values = sample["values"]
            yaw = float(values["yaw_degrees"])
            half = math.radians(yaw) / 2.0
            variant = variants[object_id]
            object_state = {"shape": archetype.semantic_type, "asset_id": variant.asset_id, "dimensions_m": list(variant.dimensions_m), "position_m": [values["x_m"], values["y_m"], 0.032 + variant.dimensions_m[2] / 2], "orientation_xyzw": [0.0, 0.0, math.sin(half), math.cos(half)], "rgba": list(variant.rgba), "mass_kg": variant.mass_kg, **variant.object_material.to_dict()}
            target = _target(config, target_rows[target_id], dimensions)
            requested = bind_scene_entities(object_state, target)
            effective_camera, view = _effective_camera_profile(config, base_camera, camera_rows[camera_id])
            band = _region_band(values["x_m"], values["y_m"], config)
            yaw_id = YAW_STRATA[min(4, int(yaw // 18.0))]
            trial_id = f"v020_c{cell_index:02d}_s{slot:02d}_{object_id}_{target_id}_{camera_id}"
            compact = {"object_variant_id": object_id, "target_profile_id": target_id, "camera_profile_id": camera_id, "position_xy_m": [values["x_m"], values["y_m"]], "yaw_degrees": yaw, "region_band": band, "yaw_stratum_id": yaw_id}
            trials.append({
                "trial_id": trial_id, "variation_id": trial_id, "split": split, "seed": seed,
                "seed_material": {"campaign_sha256": campaign["campaign_sha256"], "cell_seed": cell_seed, "slot": slot, **quota},
                "quota_ordinal": quota_ordinal, "replacement_index": 0,
                "quota_identity_fields": ["object_variant_id", "target_profile_id", "camera_profile_id", "split", "quota_ordinal"],
                "requested": requested, "resolved": deepcopy(requested),
                "object_variant_id": object_id, "target_profile_id": target_id, "camera_profile_id": camera_id,
                "object_archetype": versioned_config(_archetype_state(archetype), config_version=archetype.version),
                "object_variant": versioned_config(_variant_state(variant), units={"dimensions_m": "m", "mass_kg": "kg"}, config_version=variant.version),
                "target_profile": versioned_config(target, units={"position_m": "m", "dimensions_m": "m"}, config_version=config["config_version"]),
                "camera_profile": {"requested": deepcopy(camera_rows[camera_id]), "resolved_profile": effective_camera, "config_version": config["config_version"], "config_sha256": canonical_sha256(effective_camera)},
                "front_camera_view": view,
                "feasible_region": versioned_config({"region_id": "v020-workspace", "frame_id": "isaac_world", "polygon_xy_m": [[0.14, -0.12], [0.26, -0.12], [0.26, -0.02], [0.14, -0.02]], "object_anchor": "center", "footprint_xy_m": list(variant.dimensions_m[:2])}, units={"polygon_xy_m": "m", "footprint_xy_m": "m"}, config_version=config["config_version"]),
                "sampler": versioned_config({"sampler_version": sample["sampler_version"], "population": 10, "slot": slot, "seed": cell_seed}, resolved=sample, units={"x_m": "m", "y_m": "m", "yaw_degrees": "degree"}, config_version=config["config_version"]),
                "region_band": band, "yaw_stratum_id": yaw_id, "object_yaw_degrees": yaw,
                "variation_requested": compact, "variation_resolved": deepcopy(compact),
                "variation_units": {"position_xy_m": "m", "yaw_degrees": "degree", "object_variant_id": "category", "target_profile_id": "category", "camera_profile_id": "category", "region_band": "category", "yaw_stratum_id": "category"},
                "mass_audit_tolerance_kg": 1e-6, "projection_cell_id": f"cube={object_id}|target={target_id}|camera={camera_id}",
            })
    plan = {
        "schema_version": PLAN_SCHEMA, "plan_id": plan_id, "task_id": config["task_id"], "config_version": config["config_version"], "config_sha256": canonical_sha256(config),
        "campaign_sha256": campaign["campaign_sha256"], "campaign_contract": campaign,
        "oracle_profile_id": config["oracle_profile_id"], "lighting_profile_id": config["lighting_profile_id"],
        "varied_axes": ["object_variant_id", "entities.pick_object.pose.position_m.x", "entities.pick_object.pose.position_m.y", "entities.pick_object.pose.orientation_xyzw", "entities.placement_target.pose.position_m", "camera.front.mount_transform"],
        "frozen_axes": ["entities.support_surface", "camera.intrinsics", "camera.wrist.mount_transform", "lighting.profile", "controller.profile", "success_criteria"],
        "target": _target(config, target_rows[targets[0]], dimensions), "table": deepcopy(config["table"]), "materials": deepcopy(config["materials"]), "trials": trials,
    }
    if mode == "formal":
        plan["collection"] = {"kind": "self_healing_campaign_segment", "required_successes": 300, "maximum_attempts": 450, "maximum_attempts_per_variation": 3, "fresh_nominal_only": True, "excluded_lineages": ["farpoint-so101-v010", "farpoint-so101-v011", "farpoint-so101-v012", "farpoint-so101-v013", "farpoint-so101-v014"]}
        plan["pilot_authorization"] = deepcopy(pilot_authorization)
    else:
        plan["pilot"] = {
            "kind": f"v020_{mode.replace('-', '_')}",
            "episode_contract": "farpoint.episode.v4",
            "required_successes": len(trials),
            "maximum_attempts": PILOT_ATTEMPT_BUDGETS[mode],
            "trial_ids": [row["trial_id"] for row in trials],
            "pad_dimensions_m": dimensions,
        }
    plan["coverage"] = {"cells": len(cells), "episodes": len(trials), "objects": dict(Counter(row["object_variant_id"] for row in trials)), "targets": dict(Counter(row["target_profile_id"] for row in trials)), "cameras": dict(Counter(row["camera_profile_id"] for row in trials)), "splits": dict(Counter(row["split"] for row in trials))}
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def validate_v020_pilot_authorization(
    authorization: dict[str, Any],
    *,
    config: dict[str, Any],
    pad_dimensions_m: list[float],
) -> None:
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ValueError("formal v0.2.0 requires a pilot authorization")
    if authorization.get("config_sha256") != canonical_sha256(config):
        raise ValueError("pilot authorization config hash mismatch")
    if authorization.get("selected_pad_dimensions_m") != pad_dimensions_m:
        raise ValueError("pilot authorization pad dimensions mismatch")
    pad = authorization.get("pad_pilot") or {}
    combined = authorization.get("combined_pilot") or {}
    if pad.get("pilot_status") != "PASS" or pad.get("success_count") != 12 or pad.get("required_successes") != 12:
        raise ValueError("pad pilot must pass 12/12")
    if combined.get("pilot_status") != "PASS" or combined.get("success_count") != 30 or combined.get("required_successes") != 30:
        raise ValueError("combined pilot must pass 30/30")
    for record in (pad, combined):
        for field in ("plan_sha256", "manifest_sha256", "report_sha256"):
            value = record.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"pilot authorization {field} must be a SHA256")


def build_v020_pilot_authorization(
    config: dict[str, Any],
    *,
    pad_plan: dict[str, Any],
    pad_manifest_path: str | Path,
    pad_report_path: str | Path,
    combined_plan: dict[str, Any],
    combined_manifest_path: str | Path,
    combined_report_path: str | Path,
) -> dict[str, Any]:
    """Bind exact immutable 12/12 and 30/30 evidence for formal planning."""
    pad_manifest_path = Path(pad_manifest_path)
    pad_report_path = Path(pad_report_path)
    combined_manifest_path = Path(combined_manifest_path)
    combined_report_path = Path(combined_report_path)
    pad_report = json.loads(pad_report_path.read_text(encoding="utf-8"))
    combined_report = json.loads(combined_report_path.read_text(encoding="utf-8"))
    selected_dimensions = deepcopy((pad_plan.get("pilot") or {}).get("pad_dimensions_m"))
    if (combined_plan.get("pilot") or {}).get("pad_dimensions_m") != selected_dimensions:
        raise ValueError("combined pilot did not use the pad selected by pad pilot")

    def record(plan: dict[str, Any], manifest_path: Path, report_path: Path, report: dict[str, Any]) -> dict[str, Any]:
        return {
            "plan_sha256": plan["plan_sha256"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "pilot_status": report.get("pilot_status"),
            "success_count": int(report.get("success_count", 0)),
            "required_successes": int(report.get("required_successes", 0)),
        }

    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "config_sha256": canonical_sha256(config),
        "selected_pad_dimensions_m": selected_dimensions,
        "pad_pilot": record(pad_plan, pad_manifest_path, pad_report_path, pad_report),
        "combined_pilot": record(
            combined_plan,
            combined_manifest_path,
            combined_report_path,
            combined_report,
        ),
    }
    validate_v020_pilot_authorization(
        authorization, config=config, pad_dimensions_m=selected_dimensions
    )
    authorization["authorization_sha256"] = canonical_sha256(authorization)
    return authorization


def initialize_v020_campaign(root: str | Path, plan: dict[str, Any], *, git_commit: str, segment_id: str = "segment-000", parent_manifest_sha256: str | None = None) -> dict[str, Any]:
    destination = Path(root)
    campaign = deepcopy(plan["campaign_contract"])
    segment_index = int(segment_id.rsplit("-", 1)[-1])
    segment = create_segment({"campaign_id": campaign["campaign_id"], "campaign_sha256": campaign["campaign_sha256"], "segment_id": segment_id, "segment_index": segment_index, "git_commit": git_commit, "plan_sha256": plan["plan_sha256"], "parent_manifest_sha256": parent_manifest_sha256, "oracle_profile_allowlist": [plan["oracle_profile_id"]], "execution_status": "RUNNING", "quality_status": "NOT_EVALUATED", "attempts": []})
    paths = {"campaign": destination / "campaign.json", "plan": destination / "segments" / segment_id / "plan.json", "segment": destination / "segments" / segment_id / "segment.json"}
    if paths["plan"].exists() or paths["segment"].exists():
        raise FileExistsError("v0.2.0 campaign declaration already exists")
    if segment_index == 0 and paths["campaign"].exists():
        raise FileExistsError("v0.2.0 campaign declaration already exists")
    if segment_index > 0:
        if not paths["campaign"].is_file():
            raise FileNotFoundError("continuation campaign.json is missing")
        existing = json.loads(paths["campaign"].read_text(encoding="utf-8"))
        if existing.get("campaign_sha256") != campaign["campaign_sha256"]:
            raise ValueError("continuation campaign hash differs from campaign.json")
    writes = [(paths["plan"], plan), (paths["segment"], segment)]
    if segment_index == 0:
        writes.insert(0, (paths["campaign"], campaign))
    for path, value in writes:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"campaign": campaign, "plan": plan, "segment": segment}


def remaining_v020_attempt_budget(campaign: dict[str, Any], total_attempts: int) -> int:
    """Return the frozen campaign budget remaining across continuation segments."""
    configured = (campaign.get("attempt_policy") or {}).get("global_attempt_limit")
    limit = 450 if configured is None else int(configured)
    remaining = limit - int(total_attempts)
    if remaining < 0:
        raise ValueError("prior attempts exceed the campaign global limit")
    return remaining


def build_v020_continuation_plan(
    config: dict[str, Any],
    *,
    project_root: str | Path,
    source_plan: dict[str, Any] | list[dict[str, Any]],
    requests: list[dict[str, Any]],
    segment_id: str,
    parent_manifest_sha256: str,
    remaining_global_attempts: int,
) -> dict[str, Any]:
    """Materialize only uncovered quotas without changing frozen semantics."""
    validate_v020_config(config, project_root=project_root)
    source_plans = source_plan if isinstance(source_plan, list) else [source_plan]
    primary_plan = source_plans[0]
    if any(row.get("schema_version") != PLAN_SCHEMA for row in source_plans):
        raise ValueError("v0.2.0 continuation requires a v0.2.0 source plan")
    if any(row.get("config_sha256") != canonical_sha256(config) for row in source_plans):
        raise ValueError("continuation config differs from the frozen source plan")
    if len({row.get("campaign_sha256") for row in source_plans}) != 1:
        raise ValueError("continuation source plans belong to different campaigns")
    if not isinstance(parent_manifest_sha256, str) or len(parent_manifest_sha256) != 64:
        raise ValueError("continuation requires a parent manifest SHA256")
    source_trials = {
        trial["variation_id"]: trial
        for segment_plan in source_plans
        for trial in segment_plan.get("trials") or []
    }
    trials = []
    for index, request in enumerate(requests):
        source_id = request["source_variation_id"]
        if source_id not in source_trials:
            raise ValueError(f"continuation source trial is unavailable: {source_id}")
        trial = deepcopy(source_trials[source_id])
        request_kind = request["request_kind"]
        if request_kind == "replacement":
            slot = int(trial["sampler"]["requested"]["slot"])
            sampler = DeterministicLatinHypercubeSampler(
                bounds=(("x_m", *config["sampler"]["x_bounds_m"]), ("y_m", *config["sampler"]["y_bounds_m"]), ("yaw_degrees", *config["sampler"]["yaw_bounds_degrees"])),
                population=10,
                seed=int(request["variation_seed"]),
            )
            sampled = sampler.sample_in_strata(
                trial["sampler"]["resolved"]["strata"], slot=slot
            )
            values = sampled["values"]
            yaw = float(values["yaw_degrees"])
            half = math.radians(yaw) / 2.0
            target = deepcopy(trial["target_profile"]["resolved"])
            object_state = deepcopy(trial["resolved"])
            object_state["position_m"] = [
                values["x_m"],
                values["y_m"],
                object_state["position_m"][2],
            ]
            object_state["orientation_xyzw"] = [
                0.0,
                0.0,
                math.sin(half),
                math.cos(half),
            ]
            trial["requested"] = bind_scene_entities(object_state, target)
            trial["resolved"] = deepcopy(trial["requested"])
            trial["region_band"] = _region_band(values["x_m"], values["y_m"], config)
            trial["yaw_stratum_id"] = YAW_STRATA[min(4, int(yaw // 18.0))]
            trial["object_yaw_degrees"] = yaw
            trial["variation_requested"].update(
                {
                    "position_xy_m": [values["x_m"], values["y_m"]],
                    "yaw_degrees": yaw,
                    "region_band": trial["region_band"],
                    "yaw_stratum_id": trial["yaw_stratum_id"],
                }
            )
            trial["variation_resolved"] = deepcopy(trial["variation_requested"])
            trial["sampler"] = versioned_config(
                {
                    "sampler_version": sampled["sampler_version"],
                    "population": 10,
                    "slot": slot,
                    "seed": int(request["variation_seed"]),
                },
                resolved=sampled,
                units={"x_m": "m", "y_m": "m", "yaw_degrees": "degree"},
                config_version=config["config_version"],
            )
        trial_id = f"v020_{segment_id}_q{index:03d}_r{int(request['replacement_index']):02d}"
        trial.update(
            {
                "trial_id": trial_id,
                "variation_id": trial_id,
                "seed": int(request["variation_seed"]),
                "replacement_index": int(request["replacement_index"]),
                "prior_attempt_count": int(request["prior_attempt_count"]),
                "continuation_provenance": {
                    "request_kind": request_kind,
                    "source_variation_id": source_id,
                    "parent_manifest_sha256": parent_manifest_sha256,
                    "source_plan_sha256": next(
                        row["plan_sha256"]
                        for row in source_plans
                        if any(
                            item["variation_id"] == source_id
                            for item in row.get("trials") or []
                        )
                    ),
                },
            }
        )
        trials.append(trial)
    remaining_budget = sum(int(row["remaining_attempt_count"]) for row in requests)
    maximum_attempts = min(remaining_budget, int(remaining_global_attempts))
    if requests and maximum_attempts < len(requests):
        raise ValueError("remaining global budget cannot reach all uncovered quotas")
    plan = deepcopy(primary_plan)
    plan["plan_id"] = f"{primary_plan['campaign_contract']['campaign_id']}_{segment_id}"
    plan["trials"] = trials
    plan["collection"] = {
        "kind": "self_healing_campaign_segment",
        "required_successes": len(trials),
        "maximum_attempts": maximum_attempts,
        "maximum_attempts_per_variation": 3,
        "global_attempts_remaining": int(remaining_global_attempts),
        "fresh_nominal_only": True,
    }
    # A continuation is a self-healing campaign segment, even when its source
    # was a pilot. Retaining the source pilot profile makes the generic runner
    # re-apply the original fixed-size pilot gate to the uncovered subset.
    plan.pop("pilot", None)
    plan["continuation"] = {
        "segment_id": segment_id,
        "parent_manifest_sha256": parent_manifest_sha256,
        "source_plan_sha256s": [row["plan_sha256"] for row in source_plans],
        "campaign_sha256": primary_plan["campaign_sha256"],
        "frozen_config_sha256": primary_plan["config_sha256"],
    }
    plan["coverage"] = {
        "uncovered_quotas": len(trials),
        "carryovers": sum(row["request_kind"] == "carryover" for row in requests),
        "replacements": sum(row["request_kind"] == "replacement" for row in requests),
    }
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan
