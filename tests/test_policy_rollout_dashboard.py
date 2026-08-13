import json
from pathlib import Path

import pytest

from farpoint.policy_rollout_dashboard import PolicyRolloutDashboardIndex


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value), encoding="utf-8")


def _rollout(root: Path):
    rollout = root / "act-v010-holdout20"
    scene_id = "holdout_red_core_yaw00"
    spec = {
        "suite_id": "so101_act_v0_1_0_holdout20",
        "task": {
            "task_id": "so101_cube_pick_place",
            "evaluation_class": "independent_holdout",
        },
        "holdout_source": {"campaign_id": "formal200"},
        "acceptance": {"maximum_hard_range_excess_calibrated": 6.0},
        "scenes": [
            {
                "scene_id": scene_id,
                "object_variant_id": "red-40mm-40g",
                "region_band": "core",
                "yaw_stratum_id": "yaw00_18",
                "yaw_degrees": 9.0,
            }
        ],
    }
    report = {
        "suite_id": spec["suite_id"],
        "status": "FAIL",
        "created_at": "2026-08-13T05:00:00+00:00",
        "rollout_git_commit": "a" * 40,
        "checkpoint": {"step": 20_000, "model_sha256": "b" * 64},
        "holdout_source": spec["holdout_source"],
        "acceptance": {
            "completed_episodes": 1,
            "task_successes": 0,
            "task_success_rate": 0.0,
            "stage_progress": {"ever_cube_contact": 1, "ever_lifted": 1},
            "terminal_reason_counts": {"lift_without_target_entry": 1},
            "acceptance_errors": ["hard-range action excess exceeds the frozen safety envelope"],
            "nonfinite_action_count": 0,
            "hard_range_violation_count": 2,
            "maximum_hard_range_excess_calibrated": 11.12,
        },
        "episodes": [
            {
                "scene_id": scene_id,
                "task_success": False,
                "terminal_reason": "lift_without_target_entry",
                "policy_steps": 600,
                "stage_evidence": {"ever_cube_contact": True, "ever_lifted": True},
                "maximum_hard_range_excess_calibrated": 11.12,
                "videos": {
                    camera: {
                        "path": f"episodes/{scene_id}/{camera}.mp4",
                        "decoded_frames": 600,
                        "sha256": camera[0] * 64,
                    }
                    for camera in ("front", "wrist")
                },
            }
        ],
    }
    _write(rollout / "spec.json", spec)
    _write(rollout / "run" / "report.json", report)
    _write(rollout / "run" / "episodes" / scene_id / "front.mp4", b"front-video")
    _write(rollout / "run" / "episodes" / scene_id / "wrist.mp4", b"wrist-video")
    return rollout, scene_id


def test_policy_rollout_index_exposes_failed_evaluation_and_dual_video(tmp_path):
    rollout, scene_id = _rollout(tmp_path)
    index = PolicyRolloutDashboardIndex([tmp_path])
    rows = index.list_rollouts()
    assert len(rows) == 1
    assert rows[0]["rollout_id"] == rollout.name
    assert rows[0]["status"] == "FAIL"
    assert rows[0]["task_successes"] == 0
    assert rows[0]["maximum_hard_range_excess_calibrated"] == 11.12
    assert rows[0]["maximum_hard_range_excess_limit_calibrated"] == 6.0

    detail = index.detail(rollout.name)
    episode = detail["episodes"][0]
    assert episode["scene_id"] == scene_id
    assert set(episode["videos"]) == {"front", "wrist"}
    assert episode["videos"]["front"]["url"].endswith(f"/{scene_id}/front.mp4")
    assert index.video_path(rollout.name, scene_id, "wrist").read_bytes() == b"wrist-video"


def test_policy_rollout_index_rejects_asset_traversal_and_unreported_video(tmp_path):
    rollout, scene_id = _rollout(tmp_path)
    index = PolicyRolloutDashboardIndex([tmp_path])
    with pytest.raises(FileNotFoundError):
        index.video_path("../escape", scene_id, "front")
    with pytest.raises(FileNotFoundError):
        index.video_path(rollout.name, "../escape", "front")
    with pytest.raises(FileNotFoundError):
        index.video_path(rollout.name, scene_id, "side")
    unsafe = tmp_path / "unsafe'rollout"
    _write(unsafe / "spec.json", {"suite_id": "unsafe"})
    _write(unsafe / "run" / "report.json", {"suite_id": "unsafe"})
    assert [row["rollout_id"] for row in index.list_rollouts()] == [rollout.name]


def test_dashboard_html_has_policy_rollout_tab_and_dual_video_player():
    html = (
        Path(__file__).resolve().parents[1] / "dashboard" / "data-platform" / "index.html"
    ).read_text(encoding="utf-8")
    assert 'data-view="policy-rollouts"' in html
    assert 'api("/api/policy-rollouts")' in html
    assert 'id="rolloutFrontVideo"' in html
    assert 'id="rolloutWristVideo"' in html
