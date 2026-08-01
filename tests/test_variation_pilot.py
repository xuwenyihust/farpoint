import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from create_variation_pilot import audit_episode, select_pilot  # noqa: E402


def make_episode(root, episode_id, variation, seed, success=True):
    episode = root / episode_id
    episode.mkdir(parents=True)
    metadata = {
        "episode_id": episode_id,
        "episode_seed": seed,
        "variation_id": variation,
        "run_id": "run-1",
        "finished_at": episode_id,
    }
    metrics = {"success": success, "failure_reason": None if success else "not_lifted"}
    (episode / "metadata.json").write_text(json.dumps(metadata))
    (episode / "metrics.json").write_text(json.dumps(metrics))
    for name in ("observations.jsonl", "trajectory.jsonl", "labels.jsonl", "phase_events.jsonl"):
        (episode / name).write_text("{}\n")
    preview = episode / "preview"
    preview.mkdir()
    for index in range(10):
        (preview / f"rgb_{index:04d}.png").write_bytes(b"png")
    resources = root / "_resources"
    resources.mkdir(exist_ok=True)
    (resources / "resource_run-1.csv").write_text("time,gpu\n0,1\n")


def test_audit_accepts_complete_failed_episode_with_reason(tmp_path):
    make_episode(tmp_path, "episode_1", "cube_position_left", 0, success=False)
    result = audit_episode(tmp_path / "episode_1")
    assert result["complete"] is True
    assert result["success"] is False


def test_audit_accepts_early_failed_episode_without_raw_frames(tmp_path):
    episode = tmp_path / "episode_early_fail"
    episode.mkdir()
    (episode / "metadata.json").write_text(json.dumps({"run_id": "run-2"}))
    (episode / "metrics.json").write_text(json.dumps({"success": False, "failure_reason": "no_grasp"}))
    (episode / "phase_events.jsonl").write_text("{}\n")
    preview = episode / "preview"
    preview.mkdir()
    for index in range(10):
        (preview / f"rgb_{index:04d}.png").write_bytes(b"png")
    resources = tmp_path / "_resources"
    resources.mkdir()
    (resources / "resource_run-2.csv").write_text("time,gpu\n0,1\n")
    assert audit_episode(episode)["complete"] is True


def test_select_pilot_reports_missing_profile_seed(tmp_path):
    make_episode(tmp_path, "episode_1", "cube_position_left", 0)
    config = {"profiles": [{"variation_id": "cube_position_left"}, {"variation_id": "cube_position_right"}], "task_name": "task"}
    trials, missing = select_pilot(tmp_path, config, [0, 1])
    assert len(trials) == 1
    assert "cube_position_left:seed=1" in missing
    assert "cube_position_right:seed=0" in missing
