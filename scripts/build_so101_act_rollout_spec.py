#!/usr/bin/env python3
"""Materialize a policy-rollout spec from an immutable campaign holdout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

from farpoint.policy_rollout import load_rollout_spec
from farpoint.policy_training import canonical_sha256
from farpoint.v020_plan import build_v020_holdout_scenes, load_v020_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-limit", type=int)
    parser.add_argument(
        "--scene-indexes",
        help="comma-separated indexes into the immutable holdout scene order",
    )
    return parser.parse_args()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity_sha256(payload: dict, field: str) -> str:
    identity_payload = {key: value for key, value in payload.items() if key != field}
    encoded = json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _scene_record(source: dict) -> dict:
    obj = source["resolved"]
    record = {
        "scene_id": source["scene_id"],
        "seed": int(source["seed"]),
        "object_variant_id": source["object_variant_id"],
        "region_band": source["region_band"],
        "yaw_stratum_id": source["yaw_stratum_id"],
        "yaw_degrees": float(source["object_yaw_degrees"]),
        "object": {
            "shape": obj["shape"],
            "dimensions_m": obj["dimensions_m"],
            "position_m": obj["position_m"],
            "orientation_xyzw": obj["orientation_xyzw"],
            "rgba": obj["rgba"],
            "mass_kg": obj["mass_kg"],
        },
    }
    target_profile = source.get("target_profile") or {}
    target = target_profile.get("resolved")
    camera_view = source.get("front_camera_view")
    if target is not None:
        record["target_profile_id"] = source["target_profile_id"]
        record["target"] = {
            "position_m": target["position_m"],
            "dimensions_m": target["dimensions_m"],
            "footprint_margin_m": target["footprint_margin_m"],
        }
    if camera_view is not None:
        record["camera_profile_id"] = source["camera_profile_id"]
        record["front_camera_view"] = {
            "eye_m": camera_view["eye_m"],
            "look_at_m": camera_view["look_at_m"],
        }
    return record


def build_rollout_spec(
    template: dict,
    campaign_root: Path,
    *,
    scene_limit: int | None = None,
    scene_indexes: tuple[int, ...] | None = None,
) -> dict:
    if template.get("schema_version") != "farpoint.policy-rollout-template.v1":
        raise ValueError("unsupported rollout template schema")
    source = template["holdout_source"]
    campaign = _read(campaign_root / "campaign.json")
    if campaign.get("campaign_id") != source["campaign_id"]:
        raise ValueError("campaign identity does not match rollout template")
    if campaign.get("campaign_sha256") != source["campaign_sha256"]:
        raise ValueError("campaign SHA256 does not match rollout template")
    if _identity_sha256(campaign, "campaign_sha256") != source["campaign_sha256"]:
        raise ValueError("campaign content does not match its frozen SHA256")

    plan_paths = sorted((campaign_root / "segments").glob("*/plan.json"))
    if not plan_paths:
        raise ValueError("campaign contains no segment plans")
    source_plan = None
    excluded_seeds: set[int] = set()
    for path in plan_paths:
        plan = _read(path)
        plan_sha256 = plan.get("plan_sha256")
        if _identity_sha256(plan, "plan_sha256") != plan_sha256:
            raise ValueError(f"segment plan content does not match its SHA256: {path}")
        if plan.get("plan_sha256") == source["plan_sha256"]:
            source_plan = plan
        excluded_seeds.update(int(row["seed"]) for row in plan.get("trials", []))
    for path in sorted((campaign_root / "segments").glob("*/manifest.json")):
        manifest = _read(path)
        trial_seed_by_id = {
            str(row["trial_id"]): int(row["seed"])
            for row in _read(path.with_name("plan.json")).get("trials", [])
        }
        for attempt in manifest.get("attempts", []):
            trial_id = str(attempt.get("trial_id", ""))
            if trial_id in trial_seed_by_id:
                excluded_seeds.add(trial_seed_by_id[trial_id])
    if source_plan is None:
        raise ValueError("frozen holdout source plan is absent from campaign")
    holdout = source_plan.get("rollout_holdout") or {}
    if source.get("design") == "v020_cell_balanced":
        config_path = PROJECT_ROOT / source["variation_config"]
        config = load_v020_config(config_path, project_root=PROJECT_ROOT)
        if canonical_sha256(config) != source["variation_config_sha256"]:
            raise ValueError("v0.2.0 variation config SHA256 does not match template")
        if source_plan.get("config_sha256") != source["variation_config_sha256"]:
            raise ValueError("v0.2.0 source plan does not bind the variation config")
        source_scenes = build_v020_holdout_scenes(
            config,
            project_root=PROJECT_ROOT,
            source_plan_sha256=source_plan["plan_sha256"],
            replica_index=int(source["replica_index"]),
            replica_count=int(source["replica_count"]),
            pad_dimensions_m=source_plan["target"]["dimensions_m"],
        )
    else:
        source_scenes = holdout.get("scenes") or []
    if len(source_scenes) != source["scene_count"]:
        raise ValueError("campaign holdout scene count does not match template")
    holdout_seeds = [int(row["seed"]) for row in source_scenes]
    if len(set(holdout_seeds)) != len(holdout_seeds):
        raise ValueError("campaign holdout contains duplicate variation seeds")
    if set(holdout_seeds) & excluded_seeds:
        raise ValueError("campaign holdout overlaps collection or replacement seeds")

    if scene_limit is not None and scene_indexes is not None:
        raise ValueError("scene_limit and scene_indexes are mutually exclusive")
    evaluated = source_scenes
    if scene_limit is not None:
        if not 1 <= scene_limit <= len(source_scenes):
            raise ValueError("scene_limit must select part of the frozen holdout")
        # Spread smoke scenes across the immutable suite so a two-scene smoke
        # covers both object variants instead of selecting two adjacent rows.
        indexes = [
            math.floor(index * len(source_scenes) / scene_limit) for index in range(scene_limit)
        ]
        evaluated = [source_scenes[index] for index in indexes]
    elif scene_indexes is not None:
        if not scene_indexes or len(set(scene_indexes)) != len(scene_indexes):
            raise ValueError("scene_indexes must be non-empty and unique")
        if min(scene_indexes) < 0 or max(scene_indexes) >= len(source_scenes):
            raise ValueError("scene_indexes contains an out-of-range index")
        evaluated = [source_scenes[index] for index in scene_indexes]
    spec = copy.deepcopy(template)
    spec["schema_version"] = "farpoint.policy-rollout.v1"
    spec["holdout_source"] = {
        **source,
        "evaluated_scene_count": len(evaluated),
        "segment_count": len(plan_paths),
    }
    spec["scenes"] = [_scene_record(row) for row in evaluated]
    spec["acceptance"]["required_completed_episodes"] = len(evaluated)
    if scene_limit is not None or scene_indexes is not None:
        spec["suite_id"] = f"{spec['suite_id']}_smoke{len(evaluated)}"
        spec["task"]["evaluation_class"] = "independent_holdout_smoke"
    # Validate through the same path as runtime without retaining a temporary file.
    from farpoint.contracts import validate_contract

    errors = validate_contract(spec)
    if errors:
        raise ValueError("invalid generated rollout contract:\n" + "\n".join(errors))
    scene_ids = [row["scene_id"] for row in spec["scenes"]]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("generated rollout scene IDs are not unique")
    return spec


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"rollout spec output already exists: {args.output}")
    scene_indexes = None
    if args.scene_indexes:
        try:
            scene_indexes = tuple(int(value) for value in args.scene_indexes.split(","))
        except ValueError as error:
            raise ValueError("scene_indexes must contain integers") from error
    spec = build_rollout_spec(
        _read(args.template),
        args.campaign_root,
        scene_limit=args.scene_limit,
        scene_indexes=scene_indexes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    # Re-open through the runtime loader to catch semantic contract drift.
    load_rollout_spec(args.output)
    print(json.dumps({"output": str(args.output), "scene_count": len(spec["scenes"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
