import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_lerobot_v1_episode import (  # noqa: E402
    infer_fps,
    resolve_controlled_joint_names,
    select_joint_values,
)


ARM_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def test_infer_fps_from_source_timestamps():
    rows = [{"timestamp_seconds": value} for value in (0.0, 1 / 6, 2 / 6)]
    assert infer_fps(rows) == 6


def test_select_controlled_arm_and_gripper_values():
    names = ["finger_joint", *ARM_NAMES]
    row = {
        "joint_names": names,
        "joint_positions": list(range(7)),
        "action_joint_positions": list(range(10, 17)),
    }
    selected = resolve_controlled_joint_names({}, row)
    assert selected == [*ARM_NAMES, "finger_joint"]
    np.testing.assert_allclose(
        select_joint_values(row, selected, "joint_positions"),
        [1, 2, 3, 4, 5, 6, 0],
    )


def test_non_monotonic_timestamps_fail():
    rows = [{"timestamp_seconds": value} for value in (0.0, 0.2, 0.1)]
    try:
        infer_fps(rows)
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("expected non-monotonic timestamps to fail")
