from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from farpoint.campaign import validate_campaign_semantics
from farpoint.so101_collection import create_manifest, create_pilot_manifest
from farpoint.v020_plan import (
    build_v020_continuation_plan,
    build_v020_plan,
    canonical_sha256,
    load_v020_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/variations/so101_v020_nominal300.json"


def _config():
    return load_v020_config(CONFIG, project_root=ROOT)


def _authorization(config):
    evidence = {
        "plan_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "report_sha256": "3" * 64,
    }
    return {
        "schema_version": "farpoint.so101-v020-pilot-authorization.v1",
        "config_sha256": canonical_sha256(config),
        "selected_pad_dimensions_m": [0.09, 0.09, 0.01],
        "pad_pilot": {
            **evidence,
            "pilot_status": "PASS",
            "success_count": 12,
            "required_successes": 12,
        },
        "combined_pilot": {
            **evidence,
            "pilot_status": "PASS",
            "success_count": 30,
            "required_successes": 30,
        },
    }


def test_v020_formal_plan_freezes_exact_cells_splits_and_continuous_lhs():
    config = _config()
    plan = build_v020_plan(
        config,
        project_root=ROOT,
        plan_id="v020-formal-test",
        mode="formal",
        pilot_authorization=_authorization(config),
    )
    assert len(plan["trials"]) == 300
    assert plan["coverage"]["cells"] == 30
    assert plan["coverage"]["splits"] == {"train": 270, "validation": 30}
    assert validate_campaign_semantics(plan["campaign_contract"]) == []

    cells = Counter(
        (
            trial["object_variant_id"],
            trial["target_profile_id"],
            trial["camera_profile_id"],
        )
        for trial in plan["trials"]
    )
    assert len(cells) == 30
    assert set(cells.values()) == {10}
    split_cells = defaultdict(Counter)
    strata = defaultdict(lambda: defaultdict(set))
    for trial in plan["trials"]:
        cell = (
            trial["object_variant_id"],
            trial["target_profile_id"],
            trial["camera_profile_id"],
        )
        split_cells[cell][trial["split"]] += 1
        for axis, value in trial["sampler"]["resolved"]["strata"].items():
            strata[cell][axis].add(value)
        assert 0.14 <= trial["variation_requested"]["position_xy_m"][0] <= 0.26
        assert -0.12 <= trial["variation_requested"]["position_xy_m"][1] <= -0.02
        assert 0.0 <= trial["object_yaw_degrees"] < 90.0
    assert set(tuple(sorted(value.items())) for value in split_cells.values()) == {
        (("train", 9), ("validation", 1))
    }
    assert all(values == set(range(10)) for cell in strata.values() for values in cell.values())

    manifest = create_manifest(
        plan, collection_id="v020-formal-test", git_commit="a" * 40
    )
    assert manifest["required_successes"] == 300
    assert manifest["maximum_attempts"] == 450


def test_v020_pilots_freeze_cell_successes_with_retry_budgets():
    for mode, expected, maximum in (
        ("pad-pilot", 12, 18),
        ("combined-pilot", 30, 45),
    ):
        plan = build_v020_plan(
            _config(), project_root=ROOT, plan_id=f"v020-{mode}", mode=mode
        )
        assert len(plan["trials"]) == expected
        manifest = create_pilot_manifest(
            plan, collection_id=plan["plan_id"], git_commit="b" * 40
        )
        assert manifest["required_successes"] == expected
        assert manifest["maximum_attempts"] == maximum
        assert manifest["completion_policy"] == "success_target"


def test_v020_plan_is_deterministic_and_records_target_camera_provenance():
    config = _config()
    first = build_v020_plan(
        config,
        project_root=ROOT,
        plan_id="v020-repeat",
        mode="formal",
        pilot_authorization=_authorization(config),
    )
    second = build_v020_plan(
        config,
        project_root=ROOT,
        plan_id="v020-repeat",
        mode="formal",
        pilot_authorization=_authorization(config),
    )
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["plan_sha256"] == second["plan_sha256"]
    trial = first["trials"][0]
    assert trial["target_profile"]["requested"]["position_m"]
    assert trial["front_camera_view"]["eye_m"]
    assert trial["front_camera_view"]["look_at_m"]
    assert trial["camera_profile"]["resolved_profile"]["cameras"][1]["mount"] == (
        json.loads((ROOT / "configs/cameras/so101_front_wrist_v1.json").read_text())["cameras"][1]["mount"]
    )


def test_v020_replacement_changes_jitter_but_preserves_quota_and_lhs_strata():
    config = _config()
    plan = build_v020_plan(
        config,
        project_root=ROOT,
        plan_id="v020-continuation",
        mode="formal",
        pilot_authorization=_authorization(config),
    )
    source = plan["trials"][0]
    request = {
        "request_kind": "replacement",
        "source_variation_id": source["variation_id"],
        "replacement_index": 1,
        "variation_seed": source["seed"] + 1,
        "prior_attempt_count": 0,
        "remaining_attempt_count": 3,
    }
    continuation = build_v020_continuation_plan(
        config,
        project_root=ROOT,
        source_plan=plan,
        requests=[request],
        segment_id="segment-001",
        parent_manifest_sha256="f" * 64,
        remaining_global_attempts=150,
    )
    replacement = continuation["trials"][0]
    assert replacement["quota_identity_fields"] == source["quota_identity_fields"]
    assert replacement["sampler"]["resolved"]["strata"] == source["sampler"]["resolved"]["strata"]
    assert replacement["variation_resolved"]["position_xy_m"] != source["variation_resolved"]["position_xy_m"]
    manifest = create_manifest(
        continuation, collection_id="v020-segment-001", git_commit="c" * 40
    )
    assert manifest["required_successes"] == 1
    assert manifest["maximum_attempts"] == 3


def test_v020_pilot_continuation_drops_source_pilot_profile():
    config = _config()
    plan = build_v020_plan(
        config,
        project_root=ROOT,
        plan_id="v020-combined-pilot-continuation",
        mode="combined-pilot",
    )
    source = plan["trials"][10]
    continuation = build_v020_continuation_plan(
        config,
        project_root=ROOT,
        source_plan=plan,
        requests=[
            {
                "request_kind": "carryover",
                "source_variation_id": source["variation_id"],
                "replacement_index": 0,
                "variation_seed": source["seed"],
                "prior_attempt_count": 1,
                "remaining_attempt_count": 2,
            }
        ],
        segment_id="segment-001",
        parent_manifest_sha256="f" * 64,
        remaining_global_attempts=15,
    )

    assert "pilot" not in continuation
    manifest = create_manifest(
        continuation,
        collection_id="v020-combined-pilot-segment-001",
        git_commit="d" * 40,
    )
    assert manifest["required_successes"] == 1
    assert manifest["maximum_attempts"] == 2
