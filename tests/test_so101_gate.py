from pathlib import Path

import pytest

from farpoint.object_variation import load_variation_config
from farpoint.so101_collection import create_gate_manifest, next_attempt, record_attempt
from farpoint.so101_gate import (
    build_cube_workspace_matrix_plan,
    build_fixed_cube_gate_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_variation_config(
        ROOT / "configs/variations/so101_cube_pick_place_v1.json"
    )


def test_fixed_gate_freezes_scene_and_assigns_independent_uint32_seeds():
    plan = build_fixed_cube_gate_plan(
        _config(),
        gate_id="so101_40mm_fixed_gate",
        edge_m=0.04,
        position_xy_m=(0.150233, -0.114276),
        repetitions=20,
    )

    assert len(plan["trials"]) == 20
    assert len({trial["seed"] for trial in plan["trials"]}) == 20
    assert all(0 <= trial["seed"] <= 2**32 - 1 for trial in plan["trials"])
    assert {
        tuple(trial["resolved"]["position_m"]) for trial in plan["trials"]
    } == {(0.150233, -0.114276, 0.052000000000000005)}
    assert {
        tuple(trial["resolved"]["dimensions_m"]) for trial in plan["trials"]
    } == {(0.04, 0.04, 0.04)}

    manifest = create_gate_manifest(
        plan, collection_id="gate_run", git_commit="a" * 40
    )
    assert manifest["required_successes"] == 20
    assert manifest["maximum_attempts"] == 20
    assert manifest["release_status"] == "EXPERIMENTAL"
    assert next_attempt(manifest, plan)["trial_id"] == "so101_40mm_fixed_gate_rep00"


def test_fixed_gate_rejects_unconfigured_size():
    with pytest.raises(ValueError, match="configured cube sizes"):
        build_fixed_cube_gate_plan(
            _config(),
            gate_id="bad",
            edge_m=0.05,
            position_xy_m=(0.15, -0.11),
        )


def test_workspace_matrix_freezes_all_ten_cells_and_runs_all_before_scoring():
    positions = [
        (0.15, -0.11),
        (0.25, -0.11),
        (0.20, -0.095),
        (0.15, -0.08),
        (0.25, -0.08),
    ]
    plan = build_cube_workspace_matrix_plan(
        _config(),
        gate_id="workspace_gate",
        positions_xy_m=positions,
    )
    assert len(plan["trials"]) == 10
    assert plan["gate"]["required_successes"] == 9
    assert len({trial["cell_id"] for trial in plan["trials"]}) == 10
    assert len({trial["seed"] for trial in plan["trials"]}) == 10
    assert {
        tuple(trial["resolved"]["dimensions_m"]) for trial in plan["trials"]
    } == {(0.03, 0.03, 0.03), (0.04, 0.04, 0.04)}

    manifest = create_gate_manifest(
        plan, collection_id="workspace_run", git_commit="a" * 40
    )
    assert manifest["completion_policy"] == "all_planned_trials"
    for index in range(9):
        attempt = next_attempt(manifest, plan)
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=f"episode_{index}",
            success=True,
            dataset_valid=True,
        )
    assert manifest["execution_status"] == "RUNNING"
    assert next_attempt(manifest, plan)["variation_id"] == plan["trials"][9]["variation_id"]
    final_attempt = next_attempt(manifest, plan)
    record_attempt(
        manifest,
        plan,
        final_attempt,
        episode_id="episode_9",
        success=False,
        dataset_valid=True,
        failure_category="oracle",
        failure_reason="phase_timeout:close",
    )
    assert manifest["execution_status"] == "FINISHED"
    assert manifest["quality_status"] == "PASS"


def test_workspace_matrix_requires_five_unique_in_bounds_positions():
    with pytest.raises(ValueError, match="exactly five"):
        build_cube_workspace_matrix_plan(
            _config(), gate_id="bad", positions_xy_m=[(0.2, -0.07)]
        )
    with pytest.raises(ValueError, match="must be unique"):
        build_cube_workspace_matrix_plan(
            _config(), gate_id="bad", positions_xy_m=[(0.2, -0.07)] * 5
        )


def test_workspace_matrix_rejects_cube_footprint_overlapping_target_pad():
    config = _config()
    # Exercise the overlap guard with a deliberately adjacent target. The
    # production v4 target is now outside the configured pickup workspace.
    config["target"]["position_m"][1] = 0.02
    with pytest.raises(ValueError, match="overlaps the target pad"):
        build_cube_workspace_matrix_plan(
            config,
            gate_id="bad",
            positions_xy_m=[
                (0.15, -0.11),
                (0.25, -0.11),
                (0.20, -0.095),
                (0.15, -0.08),
                (0.25, -0.03),
            ],
        )
