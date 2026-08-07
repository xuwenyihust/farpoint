from pathlib import Path

import pytest

from farpoint.object_variation import load_variation_config
from farpoint.so101_collection import create_gate_manifest
from farpoint.so101_mass_feasibility import (
    audit_resolved_mass,
    build_cube_mass_feasibility_plan,
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
