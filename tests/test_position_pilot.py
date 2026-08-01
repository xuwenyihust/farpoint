import json
from pathlib import Path

from farpoint.position_pilot import audit_pilot_episode, pilot_trials
from farpoint.position_plan import generate_position_plan, load_position_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "variations" / "farpoint_v1_3_cube_position.json"


def test_pilot_selects_the_nine_edge_center_cross_product_cells():
    selected = pilot_trials(generate_position_plan(load_position_config(CONFIG)))
    assert [(item["row"], item["column"], item["slot"]) for item in selected] == [
        (row, column, 0) for row in (0, 2, 4) for column in (0, 2, 4)
    ]


def test_pilot_audit_requires_every_quality_and_artifact_gate(tmp_path):
    plan = generate_position_plan(load_position_config(CONFIG))
    trial = pilot_trials(plan)[0]
    episode_root = tmp_path / "episodes"
    episode = episode_root / "episode_1"
    episode.mkdir(parents=True)
    metadata = {
        "episode_id": "episode_1",
        "run_id": "run1",
        "trial_id": trial["trial_id"],
        "position_plan_sha256": plan["plan_sha256"],
        "variation": {"resolved": {"object_position_m": [*trial["object_position_xy_m"], 0.5]}},
    }
    metrics = {
        "success": True,
        "temporary_grasp_joint_created": False,
        "initial_object_perception_xy_error": 0.01,
        "object_lift_height": 0.16,
        "bilateral_contact_frames": 20,
        "max_continuous_transport_contact_frames": 120,
        "final_target_xy_distance": 0.04,
        "release_settle_frames": 120,
        "dataset_valid": True,
        "dataset_observation_count": 2,
    }
    (episode / "metadata.json").write_text(json.dumps(metadata))
    (episode / "metrics.json").write_text(json.dumps(metrics))
    for name in ("observations.jsonl", "trajectory.jsonl", "labels.jsonl", "phase_events.jsonl"):
        (episode / name).write_text("{}\n")
    for directory, suffix, count in (("preview", ".png", 10), ("observations/rgb", ".png", 2), ("observations/depth", ".npy", 2)):
        target = episode / directory
        target.mkdir(parents=True)
        for index in range(count):
            (target / f"{index}{suffix}").write_bytes(b"data")
    resources = episode_root / "_resources"
    resources.mkdir()
    (resources / "run1.csv").write_text("sample\n")
    (resources / "run1_summary.json").write_text("{}")

    result = audit_pilot_episode(episode, trial, plan_sha256=plan["plan_sha256"], episode_root=episode_root)
    assert result["accepted"] is True
    metrics["bilateral_contact_frames"] = 19
    (episode / "metrics.json").write_text(json.dumps(metrics))
    result = audit_pilot_episode(episode, trial, plan_sha256=plan["plan_sha256"], episode_root=episode_root)
    assert result["accepted"] is False
    assert "bilateral_contact" in result["errors"]
