from pathlib import Path

from farpoint.shape_pilot import (
    SENTINEL_CELLS,
    impossible_pilot_cell,
    pilot_acceptance,
    pilot_trials,
    scheduled_pilot_trials,
)
from farpoint.shape_position import generate_shape_position_plan, load_shape_position_config


CONFIG = Path(__file__).resolve().parents[1] / "configs/variations/farpoint_v0_0_1_cylinder_position.json"


def plan():
    return generate_shape_position_plan(load_shape_position_config(CONFIG))


def attempt(row, success):
    return {**row, "episode_id": "episode_" + row["trial_id"], "success": success, "dataset_valid": True}


def test_pilot_freezes_three_candidates_for_four_corners_and_center():
    rows = pilot_trials(plan())
    assert len(rows) == 15
    assert {row["cell_id"] for row in rows} == set(SENTINEL_CELLS)
    assert {row["slot"] for row in rows} == {0, 1, 2}


def test_pilot_advances_failures_and_stops_successful_cells():
    manifest = plan()
    first_round = scheduled_pilot_trials(manifest, [])[:5]
    attempts = [attempt(row, row["cell_id"] != "r00_c00") for row in first_round]
    remaining = scheduled_pilot_trials(manifest, attempts)
    assert {row["cell_id"] for row in remaining} == {"r00_c00"}
    assert remaining[0]["slot"] == 1
    attempts.append(attempt(remaining[0], True))
    assert pilot_acceptance(attempts)["accepted"] is True
    assert scheduled_pilot_trials(manifest, attempts) == []


def test_pilot_fails_when_three_candidates_for_a_cell_are_exhausted():
    manifest = plan()
    rows = [row for row in pilot_trials(manifest) if row["cell_id"] == "r00_c00"]
    attempts = [attempt(row, False) for row in rows]
    assert impossible_pilot_cell(manifest, attempts) == "r00_c00"
