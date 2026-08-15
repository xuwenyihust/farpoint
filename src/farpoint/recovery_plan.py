"""Build deterministic training-only recovery plans from a frozen nominal plan."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

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


def _assign_campaign_quota_ordinals(
    selected: list[dict[str, Any]], *, source_plan_sha256: str
) -> None:
    """Map source trials onto dense quota ordinals for this recovery campaign."""
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for trial in selected:
        quota = (
            trial["object_variant_id"],
            trial["yaw_stratum_id"],
            trial["region_band"],
            trial["split"],
        )
        source = {
            "plan_sha256": source_plan_sha256,
            "trial_id": trial["trial_id"],
            "variation_id": trial["variation_id"],
        }
        if "quota_ordinal" in trial:
            source["quota_ordinal"] = int(trial["quota_ordinal"])
        trial["recovery_source"] = source
        trial["quota_ordinal"] = counts[quota]
        counts[quota] += 1


def _v012_region_slot_count(config: dict[str, Any]) -> int:
    trigger_classes = tuple(config.get("trigger_classes") or ())
    policy = config.get("quota_policy") or {}
    common = tuple(policy.get("region_slots") or ())
    by_class = policy.get("region_slots_by_trigger_class") or {}
    if by_class and set(by_class) != set(trigger_classes):
        raise ValueError("per-trigger region slots must match trigger classes exactly")
    counts = {
        len(tuple(by_class.get(trigger_class) or common))
        for trigger_class in trigger_classes
    }
    if len(counts) != 1:
        raise ValueError("every trigger class must declare the same region slot count")
    count = next(iter(counts), 0)
    if count <= 0:
        raise ValueError("multi-stage recovery region slots must not be empty")
    return count


def _v012_requests(config: dict[str, Any], scene_count: int) -> list[dict[str, str]]:
    """Expand a config-owned marginal policy into deterministic recovery slots."""
    trigger_classes = tuple(config.get("trigger_classes") or ())
    objects = tuple(config.get("objects") or ())
    yaw_strata = tuple(config.get("yaw_strata") or ())
    policy = config.get("quota_policy") or {}
    common_region_slots = tuple(policy.get("region_slots") or ())
    region_slots_by_class = policy.get("region_slots_by_trigger_class") or {}
    region_slot_count = _v012_region_slot_count(config)
    resolved_region_slots = {
        trigger_class: tuple(
            region_slots_by_class.get(trigger_class) or common_region_slots
        )
        for trigger_class in trigger_classes
    }
    validation_slots = {
        tuple(int(value) for value in pair) for pair in (policy.get("validation_slots") or ())
    }
    pilot_per_class = int(policy.get("pilot_per_trigger_class", 0))
    full_count = len(trigger_classes) * len(yaw_strata) * region_slot_count
    pilot_count = len(trigger_classes) * pilot_per_class
    if scene_count not in (pilot_count, full_count):
        raise ValueError(f"multi-stage recovery scene_count must be {pilot_count} or {full_count}")
    if not trigger_classes or set(trigger_classes) != set(config.get("trigger_profiles") or {}):
        raise ValueError("trigger classes and profiles must match exactly")
    if len(objects) < 2 or not yaw_strata or not region_slot_count:
        raise ValueError("multi-stage recovery quota axes must not be empty")
    if any(len(pair) != 2 for pair in validation_slots):
        raise ValueError("validation slots must be yaw/region-slot pairs")
    requests: list[dict[str, str]] = []
    for class_index, trigger_class in enumerate(trigger_classes):
        class_requests = []
        region_slots = resolved_region_slots[trigger_class]
        for yaw_index, yaw_id in enumerate(yaw_strata):
            for slot_index, region in enumerate(region_slots):
                object_id = objects[(yaw_index + slot_index + class_index) % len(objects)]
                split = "validation" if (yaw_index, slot_index) in validation_slots else "train"
                class_requests.append(
                    {
                        "trigger_class": trigger_class,
                        "object_variant_id": object_id,
                        "yaw_stratum_id": yaw_id,
                        "region_band": region,
                        "split": split,
                    }
                )
        if scene_count == pilot_count:
            pilot_indices = tuple(
                (class_index + len(yaw_strata) * offset) % len(class_requests)
                for offset in range(pilot_per_class)
            )
            requests.extend(class_requests[index] for index in pilot_indices)
        else:
            requests.extend(class_requests)
    if scene_count == full_count:
        declared_objects = policy.get("objects") or {}
        declared_regions = policy.get("regions") or {}
        declared_aggregate_regions = policy.get("aggregate_regions") or {}
        declared_regions_by_class = policy.get("regions_by_trigger_class") or {}
        if declared_regions_by_class and set(declared_regions_by_class) != set(
            trigger_classes
        ):
            raise ValueError(
                "declared per-trigger region quotas must match trigger classes exactly"
            )
        declared_splits = policy.get("splits") or {}
        declared_yaw_each = int(policy.get("yaw_strata_each", -1))
        declared_per_class = int(policy.get("per_trigger_class", -1))
        for trigger_class in trigger_classes:
            rows = [row for row in requests if row["trigger_class"] == trigger_class]
            observed = {
                "objects": Counter(row["object_variant_id"] for row in rows),
                "regions": Counter(row["region_band"] for row in rows),
                "splits": Counter(row["split"] for row in rows),
                "yaw": Counter(row["yaw_stratum_id"] for row in rows),
            }
            if len(rows) != declared_per_class:
                raise ValueError("declared per-trigger recovery quota does not match slots")
            if observed["objects"] != Counter(declared_objects):
                raise ValueError("declared object recovery quotas do not match slots")
            expected_regions = declared_regions_by_class.get(
                trigger_class, declared_regions
            )
            if observed["regions"] != Counter(expected_regions):
                raise ValueError("declared region recovery quotas do not match slots")
            if observed["splits"] != Counter(declared_splits):
                raise ValueError("declared split recovery quotas do not match slots")
            if set(observed["yaw"].values()) != {declared_yaw_each}:
                raise ValueError("declared yaw recovery quotas do not match slots")
        if declared_regions_by_class and Counter(
            row["region_band"] for row in requests
        ) != Counter(declared_aggregate_regions):
            raise ValueError("declared aggregate region recovery quotas do not match slots")
    return requests


def _runtime_payload(
    config: dict[str, Any],
    source_plan: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    runtime_id: str,
    source_rollout_id: str,
) -> dict[str, Any]:
    common = {
        "runtime_id": runtime_id,
        "source_policy": deepcopy(config["source_policy"]),
        "control": deepcopy(config["control"]),
        "oracle_handoff_profile": deepcopy(config["oracle_handoff_profile"]),
    }
    if config.get("schema_version") == "farpoint.recovery-plan-config.v2":
        return {
            "schema_version": "farpoint.recovery-runtime.v2",
            **common,
            "trigger_profiles": deepcopy(config["trigger_profiles"]),
            "task_context": {"target": deepcopy(source_plan["target"])},
            "scenes": [
                {
                    "variation_id": trial["variation_id"],
                    "source_rollout_id": source_rollout_id,
                    "source_scene_id": trial["recovery_source"].get("trial_id", trial["trial_id"]),
                    "source_partition": "train",
                    "trigger_class": trial["recovery_trigger_class"],
                }
                for trial in selected
            ],
        }
    return {
        "schema_version": "farpoint.recovery-runtime.v1",
        **common,
        "trigger": deepcopy(config["trigger"]),
        "scenes": [
            {
                "variation_id": trial["variation_id"],
                "source_rollout_id": source_rollout_id,
                "source_scene_id": trial["trial_id"],
                "source_partition": "train",
            }
            for trial in selected
        ],
    }


def build_recovery_plan(
    source_plan: dict[str, Any],
    config: dict[str, Any],
    *,
    campaign_id: str,
    scene_count: int = 20,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select balanced nominal train scenes and bind their ACT handoff runtime."""
    multistage = config.get("schema_version") == "farpoint.recovery-plan-config.v2"
    if multistage:
        expected = (
            len(config.get("trigger_classes") or ())
            * len(config.get("yaw_strata") or ())
            * _v012_region_slot_count(config)
        )
        if config.get("target_successes") != expected:
            raise ValueError("multi-stage recovery target does not match its quota slots")
        slot_requests = _v012_requests(config, scene_count)
    else:
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

    if not multistage:
        slot_requests = []
        for object_id in config["objects"]:
            for yaw_id, regions in REGION_PATTERN.items():
                for region in regions:
                    slot_requests.append(
                        {
                            "object_variant_id": object_id,
                            "yaw_stratum_id": yaw_id,
                            "region_band": region,
                            "split": "train",
                        }
                    )
        if scene_count == 6:
            slot_requests = [slot_requests[index] for index in (0, 3, 8, 10, 13, 18)]

    selected = []
    used = set()
    for request in slot_requests:
        object_id = request["object_variant_id"]
        yaw_id = request["yaw_stratum_id"]
        region = request["region_band"]
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
        chosen = deepcopy(chosen)
        chosen["recovery_source"] = {
            "plan_sha256": source_plan["plan_sha256"],
            "trial_id": chosen["trial_id"],
            "variation_id": chosen["variation_id"],
            **(
                {"quota_ordinal": int(chosen["quota_ordinal"])} if "quota_ordinal" in chosen else {}
            ),
        }
        chosen["split"] = request["split"]
        if multistage:
            chosen["recovery_trigger_class"] = request["trigger_class"]
        selected.append(chosen)
        used.add(chosen["trial_id"])
    _assign_campaign_quota_ordinals(selected, source_plan_sha256=source_plan["plan_sha256"])

    quotas = []
    counts = Counter(
        (
            trial["object_variant_id"],
            trial["yaw_stratum_id"],
            trial["region_band"],
            trial["split"],
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
            "lineage_id": (config["lineage_id"] if multistage else "farpoint-so101-v011-recovery"),
            "task_id": source_plan["task_id"],
            "campaign_version": config["config_version"],
            "campaign_kind": (
                "pilot" if scene_count != int(config["target_successes"]) else "formal"
            ),
            "target": {
                "successful_episodes": scene_count,
                "splits": dict(sorted(Counter(row["split"] for row in selected).items())),
            },
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
            "watchdog_policy": {
                "profile_id": (
                    config["watchdog_profile_id"]
                    if multistage
                    else "so101-v011-recovery-watchdog-v1"
                )
            },
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
            "splits": dict(sorted(Counter(row["split"] for row in selected).items())),
            **(
                {
                    "recovery_triggers": dict(
                        sorted(Counter(row["recovery_trigger_class"] for row in selected).items())
                    )
                }
                if multistage
                else {}
            ),
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    runtime = _runtime_payload(
        config,
        source_plan,
        selected,
        runtime_id=f"{campaign_id}-act-handoff",
        source_rollout_id=f"{campaign_id}-pre-handoff",
    )
    return plan, runtime


def build_recovery_continuation_plan(
    parent_plan: dict[str, Any],
    config: dict[str, Any],
    campaign: dict[str, Any],
    requests: list[dict[str, Any]],
    *,
    segment_id: str,
    replacement_materializer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize carryover/replacement requests as an immutable continuation.

    Recovery observations must begin from the exact same frozen scene until its
    three-attempt budget is exhausted.  A replacement is realized only through
    an explicit scene materializer; it is never created by rebinding a new seed
    to the old requested/resolved pose.
    """
    if parent_plan.get("schema_version") != "farpoint.so101-recovery-plan.v1":
        raise ValueError("parent plan is not an SO-101 recovery plan")
    if parent_plan.get("campaign_sha256") != campaign.get("campaign_sha256"):
        raise ValueError("parent plan does not belong to the recovery campaign")
    multistage = config.get("schema_version") == "farpoint.recovery-plan-config.v2"
    if multistage:
        expected = (
            len(config.get("trigger_classes") or ())
            * len(config.get("yaw_strata") or ())
            * _v012_region_slot_count(config)
        )
        if config.get("target_successes") != expected:
            raise ValueError("multi-stage recovery target does not match its quota slots")
    else:
        if config.get("target_successes") != 20:
            raise ValueError("v0.1.1 recovery config must target 20 successes")
        if config.get("yaw_region_pairs") != {
            key: list(value) for key, value in REGION_PATTERN.items()
        }:
            raise ValueError("recovery yaw/region allocation does not match the frozen policy")
    if not requests:
        raise ValueError("recovery continuation requires at least one request")

    by_variation = {trial["variation_id"]: trial for trial in parent_plan.get("trials") or []}
    if len(by_variation) != len(parent_plan.get("trials") or []):
        raise ValueError("parent recovery variation ids must be unique")
    trials = []
    for request in requests:
        source_id = request.get("source_variation_id")
        source = by_variation.get(source_id)
        if source is None:
            raise ValueError(f"unknown recovery source variation: {source_id}")
        quota = request.get("quota") or {}
        observed_quota = {
            "object_variant_id": source.get("object_variant_id"),
            "yaw_stratum_id": source.get("yaw_stratum_id"),
            "region_band": source.get("region_band"),
            "split": source.get("split"),
        }
        requested_bucket = {key: quota.get(key) for key in observed_quota}
        if observed_quota != requested_bucket:
            raise ValueError("recovery continuation quota does not match source trial")
        request_kind = request.get("request_kind")
        if request_kind == "carryover":
            if int(source["seed"]) != int(request["variation_seed"]):
                raise ValueError("carryover recovery seed does not match source trial")
            if int(source.get("replacement_index", 0)) != int(request.get("replacement_index", 0)):
                raise ValueError("carryover recovery replacement index mismatch")
            trial = deepcopy(source)
            source_quota_ordinal = int(trial["quota_ordinal"])
            trial["quota_ordinal"] = int(quota["quota_ordinal"])
        elif request_kind == "replacement":
            if replacement_materializer is None:
                raise ValueError(
                    "recovery replacement requests require a new-seed scene materializer"
                )
            trial = replacement_materializer(deepcopy(request))
            source_quota_ordinal = int(source["quota_ordinal"])
            if trial.get("variation_id") == source.get("variation_id"):
                raise ValueError("replacement materializer reused the source variation id")
            if int(trial.get("seed", -1)) == int(source.get("seed", -1)):
                raise ValueError("replacement materializer reused the source seed")
            if multistage:
                trial["recovery_trigger_class"] = source["recovery_trigger_class"]
        else:
            raise ValueError("recovery continuation request has an invalid kind")

        materialized_quota = {
            "object_variant_id": trial.get("object_variant_id"),
            "yaw_stratum_id": trial.get("yaw_stratum_id"),
            "region_band": trial.get("region_band"),
            "split": trial.get("split"),
            "quota_ordinal": int(trial.get("quota_ordinal", -1)),
        }
        if materialized_quota != quota:
            raise ValueError("materialized recovery scene changed its requested quota")
        if int(trial.get("seed", -1)) != int(request["variation_seed"]):
            raise ValueError("materialized recovery scene seed does not match request")
        if int(trial.get("replacement_index", 0)) != int(request.get("replacement_index", 0)):
            raise ValueError("materialized recovery replacement index mismatch")
        trial["prior_attempt_count"] = int(request["prior_attempt_count"])
        trial["continuation_provenance"] = {
            "request_kind": request_kind,
            "source_segment_id": request.get("source_segment_id"),
            "source_variation_id": source_id,
            "source_quota_ordinal": source_quota_ordinal,
        }
        if multistage and not trial.get("recovery_trigger_class"):
            raise ValueError("multi-stage recovery continuation lost its trigger class")
        trials.append(trial)

    holdout_seeds = {
        int(scene["seed"]) for scene in (parent_plan.get("rollout_holdout") or {}).get("scenes", [])
    }
    trial_seeds = [int(trial["seed"]) for trial in trials]
    if len(trial_seeds) != len(set(trial_seeds)):
        raise ValueError("recovery continuation contains duplicate variation seeds")
    if set(trial_seeds) & holdout_seeds:
        raise ValueError("recovery continuation overlaps rollout holdout seeds")

    from farpoint.campaign_recovery import validate_replacement_plan

    plan = deepcopy(parent_plan)
    plan["plan_id"] = f"{campaign['campaign_id']}_{segment_id}"
    plan["config_version"] = config["config_version"]
    plan["config_sha256"] = canonical_sha256(config)
    plan["campaign_contract"] = deepcopy(campaign)
    plan["trials"] = trials
    plan["collection"] = {
        "kind": "self_healing_campaign_segment",
        "required_successes": len(trials),
        "maximum_attempts": sum(int(request["remaining_attempt_count"]) for request in requests),
        "attempt_policy": deepcopy(campaign["attempt_policy"]),
    }
    plan["coverage"] = {
        "objects": dict(sorted(Counter(row["object_variant_id"] for row in trials).items())),
        "regions": dict(sorted(Counter(row["region_band"] for row in trials).items())),
        "yaw_strata": dict(sorted(Counter(row["yaw_stratum_id"] for row in trials).items())),
        "splits": dict(sorted(Counter(row["split"] for row in trials).items())),
        **(
            {
                "recovery_triggers": dict(
                    sorted(Counter(row["recovery_trigger_class"] for row in trials).items())
                )
            }
            if multistage
            else {}
        ),
    }
    plan["replacement_requests"] = deepcopy(requests)
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = canonical_sha256(plan)
    validate_replacement_plan(requests, plan)

    runtime = _runtime_payload(
        config,
        parent_plan,
        trials,
        runtime_id=f"{campaign['campaign_id']}-{segment_id}-act-handoff",
        source_rollout_id=f"{campaign['campaign_id']}-pre-handoff",
    )
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
                "episodes_root": "episodes",
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
