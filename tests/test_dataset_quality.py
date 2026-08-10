from __future__ import annotations

from farpoint.dataset_quality import (
    color_label,
    count_axis,
    grid_cell,
    quaternion_yaw_degrees,
    select_representative_episodes,
    variation_record,
)
from farpoint.contracts import load_schema


def metadata(index=0, variation_id="cube_r00_c00_s0_k0", yaw=(0, 0, 0, 1)):
    return {
        "identity": {
            "dataset_episode_index": index,
            "episode_id": f"episode_{index}",
            "split": "train",
        },
        "variation": {
            "variation_id": variation_id,
            "resolved": {
                "position_m": [0.15 + index * 0.01, -0.1, 0.047],
                "orientation_xyzw": list(yaw),
                "dimensions_m": [0.03, 0.03, 0.03],
                "mass_kg": 0.04,
                "rgba": [0.85, 0.08, 0.06, 1.0],
                "shape": "cube",
            },
        },
    }


def test_variation_record_normalizes_axes():
    row = variation_record(metadata())
    assert row["row"] == 0
    assert row["column"] == 0
    assert row["yaw_deg"] == 0.0
    assert row["mass_kg"] == 0.04
    assert row["size_m"] == 0.03
    assert row["color"] == "red"


def test_grid_color_and_yaw_helpers():
    assert grid_cell("cube_r04_c01_s0_k1") == (4, 1)
    assert grid_cell("unstructured") == (None, None)
    assert color_label([0.05, 0.1, 0.9, 1]) == "blue"
    assert quaternion_yaw_degrees([0, 0, 0.2588190451, 0.9659258263]) == 30.0


def test_axis_counts_use_stable_numeric_labels():
    rows = [variation_record(metadata(0)), variation_record(metadata(1, "cube_r00_c01_s0_k0"))]
    assert count_axis(rows, "mass_kg") == {"0.04": 2}


def test_representative_selection_is_deterministic_and_diverse():
    rows = [
        variation_record(metadata(index, f"cube_r0{index}_c0{index}_s0_k0"))
        for index in range(5)
    ]
    first = select_representative_episodes(rows, 3)
    second = select_representative_episodes(list(reversed(rows)), 3)
    assert first == second
    assert first[0] == 0
    assert len(set(first)) == 3


def test_quality_report_schema_is_registered():
    schema = load_schema("farpoint.dataset-quality-report.v1")
    assert schema["properties"]["schema_version"]["const"] == (
        "farpoint.dataset-quality-report.v1"
    )
