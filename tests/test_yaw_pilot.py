from pathlib import Path

from farpoint.yaw_pilot import pilot_trials, yaw_audit_accepted
from farpoint.yaw_plan import generate_yaw_plan, load_yaw_config


def test_yaw_pilot_selects_three_cells_at_each_yaw():
    root = Path(__file__).resolve().parents[1]
    plan = generate_yaw_plan(load_yaw_config(root / "configs/variations/farpoint_v0_0_1_cube_yaw.json"))
    selected = pilot_trials(plan)
    assert len(selected) == 12
    assert {row["object_yaw_degrees"] for row in selected} == {0.0, 15.0, 30.0, 45.0}


def test_yaw_pilot_requires_visual_control_and_acceptable_audit():
    assert yaw_audit_accepted({"yaw_aware": {"control_source": "rgbd_cube_yaw", "alignment_stable": True, "audit_error_degrees": 10.0}})
    assert not yaw_audit_accepted({"yaw_aware": {"control_source": "task_ground_truth", "alignment_stable": True, "audit_error_degrees": 0.0}})
