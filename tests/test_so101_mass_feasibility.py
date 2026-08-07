from pathlib import Path

import pytest

from farpoint.object_variation import load_variation_config
from farpoint.so101_collection import create_gate_manifest
from farpoint.so101_mass_feasibility import (
    audit_resolved_mass,
    build_cube_mass_feasibility_plan,
    build_cube_mass_workspace_pilot_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def _plan(**kwargs):
    config = load_variation_config(
        ROOT / "configs/variations/so101_cube_pick_place_v1.json"
    )
    return build_cube_mass_feasibility_plan(
        config, profile_id="mass_003_test", **kwargs
    )


def test_mass_plan_freezes_matched_pairs_and_real_mass_axis():
    plan = _plan()

    assert len(plan["trials"]) == 10
    assert plan["varied_axes"] == ["entities.pick_object.physics.mass_kg"]
    assert {trial["resolved"]["mass_kg"] for trial in plan["trials"]} == {
        0.03,
        0.04,
    }
    assert len({trial["seed"] for trial in plan["trials"]}) == 10
    for pair_index in range(5):
        baseline, candidate = plan["trials"][pair_index * 2 : pair_index * 2 + 2]
        assert {baseline["mass_role"], candidate["mass_role"]} == {
            "baseline",
            "candidate",
        }
        assert baseline["environment_seed"] == candidate["environment_seed"]
        for trial in (baseline, candidate):
            mass = trial["resolved"]["mass_kg"]
            assert trial["requested"]["mass_kg"] == mass
            assert (
                trial["resolved"]["entities"]["pick_object"]["physics"]["mass_kg"]
                == mass
            )
    manifest = create_gate_manifest(
        plan, collection_id="mass_profile", git_commit="a" * 40
    )
    assert manifest["completion_policy"] == "all_planned_trials"
    assert manifest["maximum_attempts"] == 10


def test_mass_audit_is_fail_closed():
    verified = audit_resolved_mass(
        requested_mass_kg=0.03,
        resolved_mass_kg=0.03,
        physx_actual_mass_kg=0.03000001,
    )
    mismatch = audit_resolved_mass(
        requested_mass_kg=0.03,
        resolved_mass_kg=0.03,
        physx_actual_mass_kg=0.04,
    )

    assert verified["verified"] is True
    assert mismatch["verified"] is False
    with pytest.raises(ValueError, match="positive"):
        audit_resolved_mass(
            requested_mass_kg=0.03,
            resolved_mass_kg=0.03,
            physx_actual_mass_kg=0.0,
        )


def test_mass_plan_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="repetition budget"):
        _plan(repetitions_per_mass=2, minimum_successes_per_mass=3)


def _historical_baselines():
    return [
        {
            "episode_id": f"episode_baseline_{index}",
            "position_xy_m": [0.15 + 0.02 * index, -0.11 + 0.02 * index],
            "mass_kg": 0.04,
            "success": True,
        }
        for index in range(5)
    ]


def test_candidate_only_mass_workspace_plan_binds_historical_positions():
    config = load_variation_config(
        ROOT / "configs/variations/so101_cube_pick_place_v1.json"
    )
    plan = build_cube_mass_workspace_pilot_plan(
        config,
        pilot_id="candidate_workspace",
        candidate_mass_kg=0.03,
        edge_m=0.03,
        historical_baseline_commit="b" * 40,
        historical_baseline_collection_id="formal_v0_0_0",
        historical_baselines=_historical_baselines(),
    )

    assert len(plan["trials"]) == 5
    assert plan["gate"]["required_successes"] == 4
    assert plan["gate"]["historical_baseline"]["comparison_policy"] == (
        "solvable_position_reference_only"
    )
    assert {trial["resolved"]["mass_kg"] for trial in plan["trials"]} == {0.03}
    assert [trial["resolved"]["position_m"][:2] for trial in plan["trials"]] == [
        row["position_xy_m"] for row in _historical_baselines()
    ]
    manifest = create_gate_manifest(
        plan, collection_id="candidate_workspace", git_commit="c" * 40
    )
    assert manifest["maximum_attempts"] == 5
    assert manifest["required_successes"] == 4


def test_candidate_workspace_rejects_failed_historical_reference():
    config = load_variation_config(
        ROOT / "configs/variations/so101_cube_pick_place_v1.json"
    )
    baselines = _historical_baselines()
    baselines[0]["success"] = False
    with pytest.raises(ValueError, match="must be successful"):
        build_cube_mass_workspace_pilot_plan(
            config,
            pilot_id="candidate_workspace",
            candidate_mass_kg=0.03,
            edge_m=0.03,
            historical_baseline_commit="b" * 40,
            historical_baseline_collection_id="formal_v0_0_0",
            historical_baselines=baselines,
        )
