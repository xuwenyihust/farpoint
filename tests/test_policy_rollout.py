import json
from pathlib import Path

import numpy as np
import pytest

from farpoint.contracts import load_schema, validate_contract
from farpoint.policy_rollout import (
    constrain_policy_action,
    evaluate_rollout_acceptance,
    json_default,
    load_rollout_spec,
)


def test_json_default_converts_numpy_scalars_and_rejects_unknown_objects():
    assert json.dumps({"flag": np.bool_(True)}, default=json_default) == '{"flag": true}'
    assert json.dumps({"value": np.float32(1.25)}, default=json_default) == '{"value": 1.25}'
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps({"value": object()}, default=json_default)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "evaluations" / "so101_act_v0_0_3_rollout_smoke.json"


def test_rollout_smoke_contract_is_valid_and_excludes_test_data():
    spec = load_rollout_spec(CONFIG)
    assert validate_contract(spec) == []
    assert len(spec["scenes"]) == 5
    assert spec["checkpoint"]["dataset"]["excluded_test_episodes"] == "142:160"
    assert spec["acceptance"]["minimum_task_successes"] == 0
    assert load_schema("farpoint.policy-rollout.v1")["title"].endswith("v1")


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
    assert diagnostics["delta_limited_count"] == 3
    assert diagnostics["maximum_applied_delta"] == 6.0
    with pytest.raises(ValueError, match="non-finite"):
        constrain_policy_action([0, 0, 0, 0, np.nan, 0], current, max_delta=6.0)


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
