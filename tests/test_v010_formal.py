from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from farpoint.campaign_recovery import (
    build_replacement_requests,
    validate_replacement_plan,
)
from farpoint.so101_collection import create_manifest
from farpoint.v010_formal import (
    build_v010_formal_plan,
    build_v010_replacement_plan,
    initialize_v010_formal_campaign,
    load_v010_formal_config,
    validate_pilot_authorization,
    validate_v010_formal_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _values():
    config = json.loads(
        (ROOT / "configs/variations/so101_v010_formal200.json").read_text()
    )
    base = json.loads(
        (ROOT / config["base_variation_config"]["path"]).read_text()
    )
    return config, base


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authorized(tmp_path):
    config, base = _values()
    identity = config["pilot_authorization"]
    report = {
        "pilot_status": "PASS",
        "collection_id": identity["collection_id"],
        "git_commit": identity["git_commit"],
        "plan_sha256": identity["plan_sha256"],
        "required_cameras": ["front", "wrist"],
        "success_count": 12,
        "independent_episode_identity_count": 12,
        "acceptance_errors": [],
        "evidence_errors": [],
    }
    manifest = {
        "execution_status": "FINISHED",
        "quality_status": "PASS",
        "git_commit": identity["git_commit"],
        "plan_sha256": identity["plan_sha256"],
        "attempts": [{"attempt_id": str(index)} for index in range(12)],
    }
    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "manifest.json"
    _write_json(report_path, report)
    _write_json(manifest_path, manifest)
    config["pilot_authorization"]["report_sha256"] = _file_sha(report_path)
    config["pilot_authorization"]["manifest_sha256"] = _file_sha(manifest_path)
    authorization = validate_pilot_authorization(
        config, report_path=report_path, manifest_path=manifest_path
    )
    return config, base, authorization, report_path, manifest_path


def test_formal_config_freezes_exact_target_and_attempt_policy():
    config, base = _values()
    validate_v010_formal_config(config, base)
    invalid = deepcopy(config)
    invalid["target"]["splits"] = {"train": 179, "validation": 21}
    with pytest.raises(ValueError, match="train=180"):
        validate_v010_formal_config(invalid, base)
    loaded, loaded_base = load_v010_formal_config(
        ROOT / "configs/variations/so101_v010_formal200.json",
        project_root=ROOT,
    )
    assert loaded == config
    assert loaded_base == base


def test_pilot_authorization_rejects_any_evidence_mutation(tmp_path):
    config, _, _, report_path, manifest_path = _authorized(tmp_path)
    report = json.loads(report_path.read_text())
    report["success_count"] = 11
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="report SHA256"):
        validate_pilot_authorization(
            config, report_path=report_path, manifest_path=manifest_path
        )


def test_formal_plan_has_exact_quota_split_and_disjoint_holdout(tmp_path):
    config, base, authorization, _, _ = _authorized(tmp_path)
    plan = build_v010_formal_plan(
        config,
        base,
        authorization,
        campaign_id="so101-v010-formal-test",
    )
    trials = plan["trials"]
    holdout = plan["rollout_holdout"]["scenes"]

    assert len(trials) == 200
    assert len({row["variation_id"] for row in trials}) == 200
    assert len({row["seed"] for row in trials}) == 200
    assert Counter(row["split"] for row in trials) == Counter(
        {"train": 180, "validation": 20}
    )
    assert Counter(row["object_variant_id"] for row in trials) == Counter(
        {"red-40mm-40g": 100, "blue-30mm-30g": 100}
    )
    assert Counter(row["region_band"] for row in trials) == Counter(
        {"core": 50, "middle": 100, "outer": 50}
    )
    for object_id in ("red-40mm-40g", "blue-30mm-30g"):
        object_rows = [row for row in trials if row["object_variant_id"] == object_id]
        assert Counter(row["yaw_stratum_id"] for row in object_rows) == Counter(
            {
                "yaw00_18": 20,
                "yaw18_36": 20,
                "yaw36_54": 20,
                "yaw54_72": 20,
                "yaw72_90": 20,
            }
        )
        validation = [row for row in object_rows if row["split"] == "validation"]
        assert Counter(row["yaw_stratum_id"] for row in validation) == Counter(
            {
                "yaw00_18": 2,
                "yaw18_36": 2,
                "yaw36_54": 2,
                "yaw54_72": 2,
                "yaw72_90": 2,
            }
        )
        assert Counter(row["region_band"] for row in validation) == Counter(
            {"core": 2, "middle": 5, "outer": 3}
        )
    yaw_bounds = {
        row["yaw_stratum_id"]: (row["minimum_degrees"], row["maximum_degrees"])
        for row in base["yaw_strata"]
    }
    assert all(
        yaw_bounds[row["yaw_stratum_id"]][0]
        <= row["object_yaw_degrees"]
        < yaw_bounds[row["yaw_stratum_id"]][1]
        for row in trials
    )
    assert all(
        row["feasible_region"]["resolved"]["formal_eligible"] is True
        for row in trials
    )

    assert len(holdout) == 20
    assert len({row["seed"] for row in holdout}) == 20
    assert {row["seed"] for row in holdout}.isdisjoint(
        {row["seed"] for row in trials}
    )
    assert all(row["seed"] < 2**63 for row in trials)
    assert all(row["seed"] >= 2**63 for row in holdout)
    assert Counter(row["object_variant_id"] for row in holdout) == Counter(
        {"red-40mm-40g": 10, "blue-30mm-30g": 10}
    )
    for object_id in ("red-40mm-40g", "blue-30mm-30g"):
        object_holdout = [
            row for row in holdout if row["object_variant_id"] == object_id
        ]
        assert set(row["region_band"] for row in object_holdout) == {
            "core",
            "middle",
            "outer",
        }
        assert Counter(row["yaw_stratum_id"] for row in object_holdout) == Counter(
            {
                "yaw00_18": 2,
                "yaw18_36": 2,
                "yaw36_54": 2,
                "yaw54_72": 2,
                "yaw72_90": 2,
            }
        )

    manifest = create_manifest(
        plan, collection_id="segment-000", git_commit="abcdef1"
    )
    assert manifest["required_successes"] == 200
    assert manifest["maximum_attempts"] == 600


def test_formal_plan_is_deterministic_and_initialization_is_immutable(tmp_path):
    config, base, authorization, _, _ = _authorized(tmp_path)
    first = build_v010_formal_plan(
        config, base, authorization, campaign_id="so101-v010-formal-test"
    )
    second = build_v010_formal_plan(
        config, base, authorization, campaign_id="so101-v010-formal-test"
    )
    assert first == second

    root = tmp_path / "campaign"
    initialized = initialize_v010_formal_campaign(
        root, first, git_commit="abcdef1"
    )
    assert initialized["segment"]["parent_manifest_sha256"] is None
    assert json.loads((root / "campaign.json").read_text()) == first[
        "campaign_contract"
    ]
    assert len(json.loads((root / "segments/segment-000/plan.json").read_text())["trials"]) == 200
    with pytest.raises(FileExistsError):
        initialize_v010_formal_campaign(root, first, git_commit="abcdef1")


def test_formal_replacement_plan_changes_seed_without_changing_quota(tmp_path):
    config, base, authorization, _, _ = _authorized(tmp_path)
    parent_plan = build_v010_formal_plan(
        config, base, authorization, campaign_id="so101-v010-formal-test"
    )
    manifest = create_manifest(
        parent_plan, collection_id="segment-000", git_commit="abcdef1"
    )
    trial = parent_plan["trials"][0]
    for index in range(3):
        manifest["attempts"].append(
            {
                "attempt_id": f"{trial['trial_id']}__attempt{index:02d}",
                "trial_id": trial["trial_id"],
                "variation_id": trial["variation_id"],
                "split": trial["split"],
                "attempt_index": index,
                "attempt_seed": 100 + index,
                "episode_id": f"episode-{index}",
                "success": False,
                "dataset_valid": True,
                "failure_category": "oracle",
                "failure_reason": "bilateral_contact_lost:static_hold",
                "selected_for_dataset": False,
                "finished_at": "2026-08-11T00:00:00+00:00",
            }
        )
    requests = build_replacement_requests(
        parent_plan["campaign_contract"], parent_plan, manifest
    )
    continuation = build_v010_replacement_plan(
        config,
        base,
        authorization,
        parent_plan["campaign_contract"],
        requests,
        segment_id="segment-001",
    )

    assert len(continuation["trials"]) == 1
    validate_replacement_plan(requests, continuation)
    replacement = continuation["trials"][0]
    assert replacement["seed"] != trial["seed"]
    assert {
        key: replacement[key]
        for key in (
            "object_variant_id",
            "yaw_stratum_id",
            "region_band",
            "split",
            "quota_ordinal",
        )
    } == {
        key: trial[key]
        for key in (
            "object_variant_id",
            "yaw_stratum_id",
            "region_band",
            "split",
            "quota_ordinal",
        )
    }
