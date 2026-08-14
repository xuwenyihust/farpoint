import json
from pathlib import Path

from farpoint.campaign import validate_campaign_semantics
from farpoint.recovery_plan import (
    REGION_PATTERN,
    build_recovery_continuation_plan,
    build_recovery_plan,
    initialize_recovery_campaign,
)
from farpoint.recovery_runtime import load_recovery_runtime
from farpoint.v010_formal import (
    build_v010_formal_plan,
    load_v010_formal_config,
    materialize_v010_recovery_replacement_trial,
)


ROOT = Path(__file__).resolve().parents[1]


def source_plan():
    trials = []
    index = 0
    for object_id in ("red-40mm-40g", "blue-30mm-30g"):
        for yaw_id, regions in REGION_PATTERN.items():
            for region in sorted(set(regions)):
                for ordinal in range(2):
                    trials.append(
                        {
                            "trial_id": f"source-{index:03d}",
                            "variation_id": f"variation-{index:03d}",
                            "seed": 1000 + index,
                            "split": "train",
                            "object_variant_id": object_id,
                            "yaw_stratum_id": yaw_id,
                            "region_band": region,
                        }
                    )
                    index += 1
    return {
        "task_id": "so101_cube_pick_place",
        "plan_sha256": "1" * 64,
        "base_config_sha256": "2" * 64,
        "oracle_profile_id": "oracle-v1",
        "lighting_profile_id": "light-v1",
        "varied_axes": ["object_variant_id"],
        "frozen_axes": ["camera.profile"],
        "target": {"target_id": "pad"},
        "table": {"entity_id": "table"},
        "materials": {"object": {}},
        "trials": trials
        + [
            {
                **trials[0],
                "trial_id": "validation-scene",
                "variation_id": "validation-variation",
                "seed": 9999,
                "split": "validation",
            }
        ],
        "rollout_holdout": {"scenes": [{"scene_id": "holdout-only", "seed": 2**63}]},
    }


def config():
    return json.loads((ROOT / "configs/recovery/so101_v011_recovery20.json").read_text())


def multistage_config():
    return json.loads((ROOT / "configs/recovery/so101_v012_recovery80.json").read_text())


def multistage_source_plan():
    source = source_plan()
    expanded = []
    index = 0
    for object_id in ("red-40mm-40g", "blue-30mm-30g"):
        for yaw_id in multistage_config()["yaw_strata"]:
            for region in ("core", "middle", "outer"):
                for ordinal in range(8):
                    expanded.append(
                        {
                            "trial_id": f"v012-source-{index:04d}",
                            "variation_id": f"v012-variation-{index:04d}",
                            "seed": 50_000 + index,
                            "split": "train",
                            "object_variant_id": object_id,
                            "yaw_stratum_id": yaw_id,
                            "region_band": region,
                        }
                    )
                    index += 1
    source["trials"] = expanded
    return source


def test_recovery20_is_balanced_training_only_and_deterministic(tmp_path):
    first, runtime = build_recovery_plan(
        source_plan(), config(), campaign_id="recovery-test", scene_count=20
    )
    second, _ = build_recovery_plan(
        source_plan(), config(), campaign_id="recovery-test", scene_count=20
    )
    assert first == second
    assert len(first["trials"]) == 20
    assert first["coverage"] == {
        "objects": {"blue-30mm-30g": 10, "red-40mm-40g": 10},
        "regions": {"core": 6, "middle": 10, "outer": 4},
        "splits": {"train": 20},
        "yaw_strata": {
            "yaw00_18": 4,
            "yaw18_36": 4,
            "yaw36_54": 4,
            "yaw54_72": 4,
            "yaw72_90": 4,
        },
    }
    assert {trial["split"] for trial in first["trials"]} == {"train"}
    quota_ordinals = {}
    for trial in first["trials"]:
        quota = (
            trial["object_variant_id"],
            trial["yaw_stratum_id"],
            trial["region_band"],
            trial["split"],
        )
        quota_ordinals.setdefault(quota, []).append(trial["quota_ordinal"])
        assert trial["recovery_source"] == {
            "plan_sha256": source_plan()["plan_sha256"],
            "trial_id": trial["trial_id"],
            "variation_id": trial["variation_id"],
        }
    assert all(sorted(values) == list(range(len(values))) for values in quota_ordinals.values())
    assert validate_campaign_semantics(first["campaign_contract"]) == []
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime))
    assert load_recovery_runtime(runtime_path) == runtime


def test_recovery_pilot_is_six_distinct_train_scenes():
    plan, runtime = build_recovery_plan(
        source_plan(), config(), campaign_id="recovery-pilot", scene_count=6
    )
    assert len(plan["trials"]) == len(runtime["scenes"]) == 6
    assert plan["campaign_contract"]["campaign_kind"] == "pilot"
    assert len({row["source_scene_id"] for row in runtime["scenes"]}) == 6
    assert {row["source_partition"] for row in runtime["scenes"]} == {"train"}
    assert {trial["quota_ordinal"] for trial in plan["trials"]} == {0}


def test_v012_multistage_recovery80_has_exact_marginals_and_runtime_v2(tmp_path):
    plan, runtime = build_recovery_plan(
        multistage_source_plan(),
        multistage_config(),
        campaign_id="recovery80-test",
        scene_count=80,
    )
    assert len(plan["trials"]) == len(runtime["scenes"]) == 80
    assert plan["coverage"] == {
        "objects": {"blue-30mm-30g": 40, "red-40mm-40g": 40},
        "recovery_triggers": {
            "approach_miss": 20,
            "contact_without_lift": 20,
            "place_release_failure": 20,
            "transport_drift": 20,
        },
        "regions": {"core": 20, "middle": 40, "outer": 20},
        "splits": {"train": 72, "validation": 8},
        "yaw_strata": {
            "yaw00_18": 16,
            "yaw18_36": 16,
            "yaw36_54": 16,
            "yaw54_72": 16,
            "yaw72_90": 16,
        },
    }
    for trigger_class in plan["coverage"]["recovery_triggers"]:
        rows = [row for row in plan["trials"] if row["recovery_trigger_class"] == trigger_class]
        assert len(rows) == 20
        assert {
            key: sum(row["object_variant_id"] == key for row in rows)
            for key in multistage_config()["objects"]
        } == {
            "red-40mm-40g": 10,
            "blue-30mm-30g": 10,
        }
        assert {
            key: sum(row["region_band"] == key for row in rows)
            for key in ("core", "middle", "outer")
        } == {
            "core": 5,
            "middle": 10,
            "outer": 5,
        }
        assert sum(row["split"] == "validation" for row in rows) == 2
    assert runtime["schema_version"] == "farpoint.recovery-runtime.v2"
    assert runtime["task_context"]["target"] == multistage_source_plan()["target"]
    assert {row["trigger_class"] for row in runtime["scenes"]} == set(
        multistage_config()["trigger_classes"]
    )
    runtime_path = tmp_path / "runtime-v2.json"
    runtime_path.write_text(json.dumps(runtime))
    assert load_recovery_runtime(runtime_path) == runtime


def test_v012_pilot16_covers_every_trigger_object_region_and_yaw():
    plan, _runtime = build_recovery_plan(
        multistage_source_plan(),
        multistage_config(),
        campaign_id="recovery16-pilot",
        scene_count=16,
    )
    assert plan["campaign_contract"]["campaign_kind"] == "pilot"
    assert set(plan["coverage"]["recovery_triggers"].values()) == {4}
    assert set(plan["coverage"]["objects"]) == set(multistage_config()["objects"])
    assert set(plan["coverage"]["regions"]) == {"core", "middle", "outer"}
    assert set(plan["coverage"]["yaw_strata"]) == set(multistage_config()["yaw_strata"])


def test_v012_continuation_preserves_trigger_class_and_validation_split(tmp_path):
    parent, _runtime = build_recovery_plan(
        multistage_source_plan(),
        multistage_config(),
        campaign_id="recovery80-continuation",
        scene_count=80,
    )
    source = next(row for row in parent["trials"] if row["split"] == "validation")
    request = {
        "request_kind": "carryover",
        "source_segment_id": "segment-000",
        "source_variation_id": source["variation_id"],
        "quota": {
            "object_variant_id": source["object_variant_id"],
            "yaw_stratum_id": source["yaw_stratum_id"],
            "region_band": source["region_band"],
            "split": "validation",
            "quota_ordinal": 0,
        },
        "replacement_index": source.get("replacement_index", 0),
        "variation_seed": source["seed"],
        "prior_attempt_count": 1,
        "remaining_attempt_count": 2,
    }
    continuation, runtime = build_recovery_continuation_plan(
        parent,
        multistage_config(),
        parent["campaign_contract"],
        [request],
        segment_id="segment-001",
    )
    assert continuation["trials"][0]["recovery_trigger_class"] == source["recovery_trigger_class"]
    assert continuation["coverage"]["splits"] == {"validation": 1}
    assert runtime["scenes"][0]["trigger_class"] == source["recovery_trigger_class"]
    path = tmp_path / "runtime-v2-continuation.json"
    path.write_text(json.dumps(runtime))
    assert load_recovery_runtime(path) == runtime


def test_initialize_recovery_campaign_is_immutable(tmp_path):
    plan, runtime = build_recovery_plan(
        source_plan(), config(), campaign_id="recovery-init", scene_count=6
    )
    initialized = initialize_recovery_campaign(tmp_path, plan, runtime, git_commit="a" * 40)
    assert initialized["segment"]["plan_sha256"] == plan["plan_sha256"]
    assert load_recovery_runtime(tmp_path / "segments/segment-000/recovery-runtime.json") == runtime
    evidence = json.loads((tmp_path / "evidence-index.json").read_text())
    assert evidence["segments"][0]["episodes_root"] == "episodes"
    try:
        initialize_recovery_campaign(tmp_path, plan, runtime, git_commit="a" * 40)
    except FileExistsError:
        pass
    else:
        raise AssertionError("immutable campaign initialization accepted overwrite")


def test_recovery_continuation_preserves_scene_and_remaining_attempt_budget(tmp_path):
    parent, _ = build_recovery_plan(
        source_plan(), config(), campaign_id="recovery-continuation", scene_count=20
    )
    requests = []
    dense_ordinals = {}
    for index, source in enumerate(parent["trials"]):
        prior = 1 if index < 9 else 0
        bucket = (
            source["object_variant_id"],
            source["yaw_stratum_id"],
            source["region_band"],
            source["split"],
        )
        dense_ordinal = dense_ordinals.get(bucket, 0)
        dense_ordinals[bucket] = dense_ordinal + 1
        source["quota_ordinal"] += 2
        requests.append(
            {
                "request_kind": "carryover",
                "source_segment_id": "segment-000",
                "source_variation_id": source["variation_id"],
                "quota": {
                    "object_variant_id": source["object_variant_id"],
                    "yaw_stratum_id": source["yaw_stratum_id"],
                    "region_band": source["region_band"],
                    "split": source["split"],
                    "quota_ordinal": dense_ordinal,
                },
                "replacement_index": source.get("replacement_index", 0),
                "variation_seed": source["seed"],
                "prior_attempt_count": prior,
                "remaining_attempt_count": 3 - prior,
            }
        )

    continuation, runtime = build_recovery_continuation_plan(
        parent,
        config(),
        parent["campaign_contract"],
        requests,
        segment_id="segment-001",
    )

    assert continuation["collection"] == {
        "kind": "self_healing_campaign_segment",
        "required_successes": 20,
        "maximum_attempts": 51,
        "attempt_policy": parent["campaign_contract"]["attempt_policy"],
    }
    assert [row["trial_id"] for row in continuation["trials"]] == [
        row["trial_id"] for row in parent["trials"]
    ]
    assert [row["seed"] for row in continuation["trials"]] == [
        row["seed"] for row in parent["trials"]
    ]
    assert [row["quota_ordinal"] for row in continuation["trials"]] == [
        request["quota"]["quota_ordinal"] for request in requests
    ]
    assert [
        row["continuation_provenance"]["source_quota_ordinal"] for row in continuation["trials"]
    ] == [row["quota_ordinal"] for row in parent["trials"]]
    assert runtime["runtime_id"].endswith("segment-001-act-handoff")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime))
    assert load_recovery_runtime(runtime_path) == runtime


def test_recovery_continuation_rejects_unmaterialized_replacement_seed():
    parent, _ = build_recovery_plan(
        source_plan(), config(), campaign_id="recovery-replacement", scene_count=20
    )
    source = parent["trials"][0]
    request = {
        "request_kind": "replacement",
        "source_segment_id": "segment-000",
        "source_variation_id": source["variation_id"],
        "quota": {
            "object_variant_id": source["object_variant_id"],
            "yaw_stratum_id": source["yaw_stratum_id"],
            "region_band": source["region_band"],
            "split": source["split"],
            "quota_ordinal": source["quota_ordinal"],
        },
        "replacement_index": 1,
        "variation_seed": source["seed"] + 1,
        "prior_attempt_count": 0,
        "remaining_attempt_count": 3,
    }
    try:
        build_recovery_continuation_plan(
            parent,
            config(),
            parent["campaign_contract"],
            [request],
            segment_id="segment-001",
        )
    except ValueError as error:
        assert "new-seed scene materializer" in str(error)
    else:
        raise AssertionError("replacement seed was rebound to an old recovery scene")


def test_recovery_replacement_materializes_new_training_scene_deterministically():
    formal_config, base = load_v010_formal_config(
        ROOT / "configs/variations/so101_v010_formal200.json", project_root=ROOT
    )
    formal_source = build_v010_formal_plan(
        formal_config,
        base,
        formal_config["pilot_authorization"],
        campaign_id="formal-source",
    )
    parent, _ = build_recovery_plan(
        formal_source, config(), campaign_id="recovery-replacement", scene_count=20
    )
    source = parent["trials"][0]
    request = {
        "request_kind": "replacement",
        "source_segment_id": "segment-000",
        "source_variation_id": source["variation_id"],
        "quota": {
            "object_variant_id": source["object_variant_id"],
            "yaw_stratum_id": source["yaw_stratum_id"],
            "region_band": source["region_band"],
            "split": "train",
            "quota_ordinal": source["quota_ordinal"],
        },
        "replacement_index": 1,
        "variation_seed": source["seed"] + 1,
        "prior_attempt_count": 0,
        "remaining_attempt_count": 3,
    }

    def materialize(row):
        return materialize_v010_recovery_replacement_trial(
            formal_config,
            base,
            formal_config["pilot_authorization"],
            parent["campaign_contract"],
            row,
            segment_id="segment-001",
            source_plan_sha256=parent["campaign_contract"]["variation_contract"][
                "source_plan_sha256"
            ],
        )

    first, runtime = build_recovery_continuation_plan(
        parent,
        config(),
        parent["campaign_contract"],
        [request],
        segment_id="segment-001",
        replacement_materializer=materialize,
    )
    second, _ = build_recovery_continuation_plan(
        parent,
        config(),
        parent["campaign_contract"],
        [request],
        segment_id="segment-001",
        replacement_materializer=materialize,
    )
    assert first == second
    replacement = first["trials"][0]
    assert replacement["seed"] == request["variation_seed"]
    assert replacement["variation_id"] != source["variation_id"]
    assert replacement["requested"] != source["requested"]
    assert replacement["split"] == "train"
    assert replacement["continuation_provenance"] == {
        "request_kind": "replacement",
        "source_segment_id": "segment-000",
        "source_variation_id": source["variation_id"],
        "source_quota_ordinal": source["quota_ordinal"],
    }
    assert replacement["recovery_source"]["kind"] == "resampled_training_scene"
    assert runtime["scenes"][0]["source_partition"] == "train"


def test_recovery_replacement_materializer_rejects_non_training_quota():
    parent, _ = build_recovery_plan(
        source_plan(), config(), campaign_id="recovery-replacement", scene_count=20
    )
    source = parent["trials"][0]
    request = {
        "request_kind": "replacement",
        "source_segment_id": "segment-000",
        "source_variation_id": source["variation_id"],
        "quota": {
            "object_variant_id": source["object_variant_id"],
            "yaw_stratum_id": source["yaw_stratum_id"],
            "region_band": source["region_band"],
            "split": "validation",
            "quota_ordinal": source["quota_ordinal"],
        },
        "replacement_index": 1,
        "variation_seed": source["seed"] + 1,
        "prior_attempt_count": 0,
        "remaining_attempt_count": 3,
    }
    formal_config, base = load_v010_formal_config(
        ROOT / "configs/variations/so101_v010_formal200.json", project_root=ROOT
    )
    try:
        materialize_v010_recovery_replacement_trial(
            formal_config,
            base,
            formal_config["pilot_authorization"],
            parent["campaign_contract"],
            request,
            segment_id="segment-001",
            source_plan_sha256=parent["campaign_contract"]["variation_contract"][
                "source_plan_sha256"
            ],
        )
    except ValueError as error:
        assert "split is not allowed" in str(error)
    else:
        raise AssertionError("recovery materializer accepted a non-training quota")
