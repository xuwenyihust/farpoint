import json
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from farpoint.contracts import load_schema, validate_contract
from farpoint.policy_rollout import (
    constrain_policy_action,
    evaluate_rollout_acceptance,
    json_default,
    load_rollout_spec,
    resolve_replan_interval,
)


def test_json_default_converts_numpy_scalars_and_rejects_unknown_objects():
    assert json.dumps({"flag": np.bool_(True)}, default=json_default) == '{"flag": true}'
    assert json.dumps({"value": np.float32(1.25)}, default=json_default) == '{"value": 1.25}'
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps({"value": object()}, default=json_default)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "evaluations" / "so101_act_v0_0_3_rollout_smoke.json"
BASELINE_20K_CONFIG = (
    ROOT / "configs" / "evaluations" / "so101_act_v0_0_3_baseline_20k_rollout_smoke.json"
)
V010_TEMPLATE = ROOT / "configs" / "evaluations" / "so101_act_v0_1_0_holdout_template.json"
BUILDER_PATH = ROOT / "scripts" / "build_so101_act_rollout_spec.py"
BUILDER_SPEC = importlib.util.spec_from_file_location("build_rollout_spec", BUILDER_PATH)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
BUILDER_MODULE = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(BUILDER_MODULE)
build_rollout_spec = BUILDER_MODULE.build_rollout_spec


def _identity(payload, field):
    value = {key: item for key, item in payload.items() if key != field}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_campaign_fixture(tmp_path):
    campaign_root = tmp_path / "campaign"
    segment_root = campaign_root / "segments" / "segment-000"
    segment_root.mkdir(parents=True)
    scenes = []
    for index in range(20):
        large = index < 10
        scenes.append(
            {
                "scene_id": f"holdout_{index:02d}",
                "seed": 2**63 + index,
                "object_variant_id": "red_40mm_40g" if large else "blue_30mm_30g",
                "region_band": ("core", "middle", "outer")[index % 3],
                "yaw_stratum_id": f"yaw{(index % 5) * 18:02d}_{(index % 5 + 1) * 18:02d}",
                "object_yaw_degrees": float((index % 5) * 18 + 9),
                "resolved": {
                    "shape": "cube",
                    "dimensions_m": [0.04, 0.04, 0.04] if large else [0.03, 0.03, 0.03],
                    "position_m": [0.18, -0.08, 0.02 if large else 0.015],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "rgba": [0.9, 0.1, 0.1, 1.0] if large else [0.1, 0.1, 0.9, 1.0],
                    "mass_kg": 0.04 if large else 0.03,
                },
            }
        )
    plan = {
        "schema_version": "test-plan",
        "campaign_sha256": "0" * 64,
        "trials": [{"trial_id": "train", "seed": 41}],
        "rollout_holdout": {"scenes": scenes},
    }
    plan["plan_sha256"] = _identity(plan, "plan_sha256")
    (segment_root / "plan.json").write_text(json.dumps(plan))
    (segment_root / "manifest.json").write_text(
        json.dumps({"attempts": [{"trial_id": "train", "attempt_seed": 99}]})
    )
    campaign = {"campaign_id": "campaign-test", "segments": ["segment-000"]}
    campaign["campaign_sha256"] = _identity(campaign, "campaign_sha256")
    (campaign_root / "campaign.json").write_text(json.dumps(campaign))
    template = json.loads(V010_TEMPLATE.read_text())
    template["holdout_source"].update(
        {
            "campaign_id": campaign["campaign_id"],
            "campaign_sha256": campaign["campaign_sha256"],
            "plan_sha256": plan["plan_sha256"],
        }
    )
    return campaign_root, template


def test_rollout_smoke_contract_is_valid_and_excludes_test_data():
    spec = load_rollout_spec(CONFIG)
    assert validate_contract(spec) == []
    assert len(spec["scenes"]) == 5
    assert spec["checkpoint"]["dataset"]["excluded_test_episodes"] == "142:160"
    assert spec["acceptance"]["minimum_task_successes"] == 0
    assert load_schema("farpoint.policy-rollout.v1")["title"].endswith("v1")


def test_baseline_20k_rollout_reuses_frozen_smoke_scenes():
    pilot = load_rollout_spec(CONFIG)
    baseline = load_rollout_spec(BASELINE_20K_CONFIG)
    assert baseline["scenes"] == pilot["scenes"]
    assert baseline["control"] == pilot["control"]
    assert baseline["acceptance"] == pilot["acceptance"]
    assert baseline["checkpoint"]["step"] == 20_000
    assert baseline["checkpoint"]["training_run_id"] == ("act-v0.0.3-baseline-20k-4c062b8")
    assert baseline["checkpoint"]["dataset"] == pilot["checkpoint"]["dataset"]


def test_rollout_contract_rejects_duplicate_scene_identity(tmp_path):
    spec = json.loads(CONFIG.read_text())
    spec["scenes"][1]["scene_id"] = spec["scenes"][0]["scene_id"]
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="unique"):
        load_rollout_spec(path)


def test_policy_action_applies_hard_and_delta_safety_bounds():
    raw = np.asarray([110.0, -80.0, 5.0, 4.0, 3.0, 120.0])
    current = np.zeros(6)
    applied, diagnostics = constrain_policy_action(raw, current, max_delta=6.0)
    assert applied.tolist() == [6.0, -6.0, 5.0, 4.0, 3.0, 6.0]
    assert diagnostics["hard_range_violation_count"] == 2
    assert diagnostics["maximum_hard_range_excess_calibrated"] == 20.0
    assert diagnostics["delta_limited_count"] == 3
    assert diagnostics["maximum_applied_delta"] == 6.0
    with pytest.raises(ValueError, match="non-finite"):
        constrain_policy_action([0, 0, 0, 0, np.nan, 0], current, max_delta=6.0)


def test_replan_interval_defaults_to_checkpoint_and_validates_chunk_size():
    assert resolve_replan_interval(None, checkpoint_steps=100, chunk_size=100) == 100
    assert resolve_replan_interval(10, checkpoint_steps=100, chunk_size=100) == 10
    with pytest.raises(ValueError, match="positive"):
        resolve_replan_interval(0, checkpoint_steps=100, chunk_size=100)
    with pytest.raises(ValueError, match="chunk_size"):
        resolve_replan_interval(101, checkpoint_steps=100, chunk_size=100)


def test_interface_smoke_acceptance_reports_task_success_without_requiring_it():
    spec = load_rollout_spec(CONFIG)
    results = [
        {
            "scene_id": scene["scene_id"],
            "execution_status": "FINISHED",
            "task_success": index == 0,
            "nonfinite_action_count": 0,
            "hard_range_violation_count": 0,
        }
        for index, scene in enumerate(spec["scenes"])
    ]
    report = evaluate_rollout_acceptance(spec, results)
    assert report["status"] == "PASS"
    assert report["task_successes"] == 1
    assert report["task_success_rate"] == pytest.approx(0.2)
    results[0]["hard_range_violation_count"] = 1
    failed = evaluate_rollout_acceptance(spec, results)
    assert failed["status"] == "FAIL"
    assert "hard-range" in failed["acceptance_errors"][0]


def test_holdout_acceptance_gates_violation_magnitude_not_clipped_count(tmp_path):
    campaign_root, template = _write_campaign_fixture(tmp_path)
    spec = build_rollout_spec(template, campaign_root, scene_limit=2)
    results = [
        {
            "scene_id": scene["scene_id"],
            "execution_status": "FINISHED",
            "task_success": False,
            "terminal_reason": "lift_without_target_entry",
            "nonfinite_action_count": 0,
            "hard_range_violation_count": 100,
            "maximum_hard_range_excess_calibrated": 5.5,
        }
        for scene in spec["scenes"]
    ]
    accepted = evaluate_rollout_acceptance(spec, results)
    assert accepted["status"] == "PASS"
    assert accepted["hard_range_violation_count"] == 200
    assert accepted["maximum_hard_range_excess_calibrated"] == 5.5
    results[0]["maximum_hard_range_excess_calibrated"] = 6.01
    failed = evaluate_rollout_acceptance(spec, results)
    assert failed["status"] == "FAIL"
    assert "safety envelope" in failed["acceptance_errors"][0]


def test_v010_holdout_builder_preserves_high_seeds_and_stratifies_smoke(tmp_path):
    campaign_root, template = _write_campaign_fixture(tmp_path)
    spec = build_rollout_spec(template, campaign_root, scene_limit=2)
    assert validate_contract(spec) == []
    assert spec["task"]["evaluation_class"] == "independent_holdout_smoke"
    assert [scene["object_variant_id"] for scene in spec["scenes"]] == [
        "red_40mm_40g",
        "blue_30mm_30g",
    ]
    assert spec["scenes"][0]["seed"] == 2**63
    assert spec["scenes"][1]["seed"] == 2**63 + 10
    assert spec["holdout_source"]["evaluated_scene_count"] == 2
    assert spec["acceptance"]["required_completed_episodes"] == 2


def test_v010_holdout_builder_selects_explicit_balanced_diagnostic_indexes(tmp_path):
    campaign_root, template = _write_campaign_fixture(tmp_path)
    spec = build_rollout_spec(
        template,
        campaign_root,
        scene_indexes=(0, 1, 2, 10, 11, 12),
    )
    assert [scene["scene_id"] for scene in spec["scenes"]] == [
        "holdout_00",
        "holdout_01",
        "holdout_02",
        "holdout_10",
        "holdout_11",
        "holdout_12",
    ]
    assert {scene["object_variant_id"] for scene in spec["scenes"]} == {
        "red_40mm_40g",
        "blue_30mm_30g",
    }
    assert {scene["region_band"] for scene in spec["scenes"]} == {
        "core",
        "middle",
        "outer",
    }
    assert spec["acceptance"]["required_completed_episodes"] == 6


def test_v010_holdout_builder_rejects_invalid_explicit_indexes(tmp_path):
    campaign_root, template = _write_campaign_fixture(tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_rollout_spec(template, campaign_root, scene_limit=2, scene_indexes=(0, 10))
    with pytest.raises(ValueError, match="unique"):
        build_rollout_spec(template, campaign_root, scene_indexes=(0, 0))
    with pytest.raises(ValueError, match="out-of-range"):
        build_rollout_spec(template, campaign_root, scene_indexes=(20,))


def test_v010_holdout_builder_rejects_collection_overlap(tmp_path):
    campaign_root, template = _write_campaign_fixture(tmp_path)
    plan_path = campaign_root / "segments" / "segment-000" / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["trials"][0]["seed"] = 2**63
    plan["plan_sha256"] = _identity(plan, "plan_sha256")
    plan_path.write_text(json.dumps(plan))
    template["holdout_source"]["plan_sha256"] = plan["plan_sha256"]
    with pytest.raises(ValueError, match="overlaps collection"):
        build_rollout_spec(template, campaign_root)
