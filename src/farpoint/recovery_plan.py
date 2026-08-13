"""Build deterministic training-only recovery plans from a frozen nominal plan."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from farpoint.campaign import canonical_sha256, create_campaign, create_segment


REGION_PATTERN = {
    "yaw00_18": ("core", "middle"),
    "yaw18_36": ("core", "outer"),
    "yaw36_54": ("core", "middle"),
    "yaw54_72": ("middle", "outer"),
    "yaw72_90": ("middle", "middle"),
}


def _rank(seed: int, trial: dict[str, Any]) -> bytes:
    material = f"{seed}:{trial['trial_id']}:{trial['seed']}"
    return hashlib.sha256(material.encode()).digest()


def build_recovery_plan(
    source_plan: dict[str, Any],
    config: dict[str, Any],
    *,
    campaign_id: str,
    scene_count: int = 20,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select balanced nominal train scenes and bind their ACT handoff runtime."""
    if scene_count not in (6, 20):
        raise ValueError("recovery plan scene_count must be 6 or 20")
    if config.get("target_successes") != 20:
        raise ValueError("v0.1.1 recovery config must target 20 successes")
    if config.get("yaw_region_pairs") != {
        key: list(value) for key, value in REGION_PATTERN.items()
    }:
        raise ValueError("recovery yaw/region allocation does not match the frozen policy")
    source_trials = [
        trial for trial in source_plan.get("trials", []) if trial.get("split") == "train"
    ]
    holdout_ids = {
        scene.get("scene_id")
        for scene in (source_plan.get("rollout_holdout") or {}).get("scenes", [])
    }
    if any(trial["trial_id"] in holdout_ids for trial in source_trials):
        raise ValueError("source train trials overlap rollout holdouts")

    requests = []
    for object_id in config["objects"]:
        for yaw_id, regions in REGION_PATTERN.items():
            for region in regions:
                requests.append((object_id, yaw_id, region))
    if scene_count == 6:
        requests = [requests[index] for index in (0, 3, 8, 10, 13, 18)]

    selected = []
    used = set()
    for object_id, yaw_id, region in requests:
        candidates = [
            trial
            for trial in source_trials
            if trial["object_variant_id"] == object_id
            and trial["yaw_stratum_id"] == yaw_id
            and trial["region_band"] == region
            and trial["trial_id"] not in used
        ]
        if not candidates:
            raise ValueError(f"source plan has no train scene for {object_id}/{yaw_id}/{region}")
        chosen = min(candidates, key=lambda row: _rank(int(config["selection_seed"]), row))
        selected.append(deepcopy(chosen))
        used.add(chosen["trial_id"])

    quotas = []
    counts = Counter(
        (
            trial["object_variant_id"],
            trial["yaw_stratum_id"],
            trial["region_band"],
            "train",
        )
        for trial in selected
    )
    for (object_id, yaw_id, region, split), count in sorted(counts.items()):
        quotas.append(
            {
                "object_variant_id": object_id,
                "yaw_stratum_id": yaw_id,
                "region_band": region,
                "split": split,
                "count": count,
            }
        )
    campaign = create_campaign(
        {
            "campaign_id": campaign_id,
            "lineage_id": "farpoint-so101-v011-recovery",
            "task_id": source_plan["task_id"],
            "campaign_version": config["config_version"],
            "campaign_kind": "pilot" if scene_count == 6 else "formal",
            "target": {"successful_episodes": scene_count, "splits": {"train": scene_count}},
            "quotas": quotas,
            "variation_contract": {
                "kind": "live_policy_recovery",
                "source_plan_sha256": source_plan["plan_sha256"],
                "selection_seed": config["selection_seed"],
                "demonstration_schema": "farpoint.demonstration.v1",
            },
            "attempt_policy": {
                "maximum_attempts_per_variation": 3,
                "global_attempt_limit": None,
                "replacement_policy": "same_quota_new_variation_seed",
            },
            "watchdog_policy": {"profile_id": "so101-v011-recovery-watchdog-v1"},
            "rollout_holdout": {
                "policy": "inherit_v010_holdout_excluded",
                "source_plan_sha256": source_plan["plan_sha256"],
            },
        }
    )
    plan = {
        "schema_version": "farpoint.so101-recovery-plan.v1",
        "plan_id": f"{campaign_id}_segment_000",
        "task_id": source_plan["task_id"],
        "config_version": config["config_version"],
        "config_sha256": canonical_sha256(config),
        "base_config_sha256": source_plan["base_config_sha256"],
        "campaign_sha256": campaign["campaign_sha256"],
        "campaign_contract": campaign,
        "oracle_profile_id": source_plan["oracle_profile_id"],
        "lighting_profile_id": source_plan["lighting_profile_id"],
        "varied_axes": deepcopy(source_plan["varied_axes"]),
        "frozen_axes": deepcopy(source_plan["frozen_axes"]),
        "target": deepcopy(source_plan["target"]),
        "table": deepcopy(source_plan["table"]),
        "materials": deepcopy(source_plan["materials"]),
        "trials": selected,
        "rollout_holdout": deepcopy(source_plan["rollout_holdout"]),
        "collection": {
            "kind": "self_healing_campaign_segment",
            "required_successes": scene_count,
            "maximum_attempts": scene_count * 3,
            "attempt_policy": deepcopy(campaign["attempt_policy"]),
        },
        "coverage": {
            "objects": dict(sorted(Counter(row["object_variant_id"] for row in selected).items())),
            "regions": dict(sorted(Counter(row["region_band"] for row in selected).items())),
            "yaw_strata": dict(sorted(Counter(row["yaw_stratum_id"] for row in selected).items())),
            "splits": {"train": scene_count},
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    runtime = {
        "schema_version": "farpoint.recovery-runtime.v1",
        "runtime_id": f"{campaign_id}-act-handoff",
        "source_policy": deepcopy(config["source_policy"]),
        "control": deepcopy(config["control"]),
        "trigger": deepcopy(config["trigger"]),
        "scenes": [
            {
                "variation_id": trial["variation_id"],
                "source_rollout_id": f"{campaign_id}-pre-handoff",
                "source_scene_id": trial["trial_id"],
                "source_partition": "train",
            }
            for trial in selected
        ],
    }
    return plan, runtime


def initialize_recovery_campaign(
    root: str | Path,
    plan: dict[str, Any],
    runtime: dict[str, Any],
    *,
    git_commit: str,
) -> dict[str, Any]:
    """Write one immutable recovery campaign declaration and runtime binding."""
    destination = Path(root)
    paths = {
        "campaign": destination / "campaign.json",
        "segment": destination / "segments/segment-000/segment.json",
        "plan": destination / "segments/segment-000/plan.json",
        "runtime": destination / "segments/segment-000/recovery-runtime.json",
        "evidence": destination / "evidence-index.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("recovery campaign declaration already exists")
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
    evidence = {
        "schema_version": "farpoint.campaign-evidence-index.v1",
        "campaign_id": campaign["campaign_id"],
        "segments": [
            {
                "segment": "segments/segment-000/segment.json",
                "plan": "segments/segment-000/plan.json",
                "manifest": "segments/segment-000/manifest.json",
                "recovery_runtime": "segments/segment-000/recovery-runtime.json",
            }
        ],
    }
    values = {
        "campaign": campaign,
        "segment": segment,
        "plan": plan,
        "runtime": runtime,
        "evidence": evidence,
    }
    for name, value in values.items():
        path = paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return {"campaign": campaign, "segment": segment, "plan": plan, "runtime": runtime}
