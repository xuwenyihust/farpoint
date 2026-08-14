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
    assert all(
        sorted(values) == list(range(len(values)))
        for values in quota_ordinals.values()
    )
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
    for index, source in enumerate(parent["trials"]):
        prior = 1 if index < 9 else 0
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
                    "quota_ordinal": source["quota_ordinal"],
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
