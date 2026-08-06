import hashlib
import json

import pytest

from farpoint.so101_episode_analysis import (
    analyze_so101_episodes,
    classify_so101_failure,
    render_so101_analysis_markdown,
    summarize_so101_episode,
)


def _write_episode(root, *, success=True, terminal_phase="retreat", duplicate=None):
    root.mkdir()
    rows = duplicate or [
        {
            "frame": 0,
            "timestamp_seconds": 0.0,
            "phase": "home",
            "joint_positions": [0.0] * 6,
            "action_joint_positions": [0.1] * 6,
            "rgb_path": "rgb/front_000000.png",
            "contact": {
                "cube_contact": False,
                "bilateral_cube_contact": False,
            },
            "truth": {"object_root_pose_xyzw": [0.1, 0.0, 0.05, 0, 0, 0, 1]},
        },
        {
            "frame": 1,
            "timestamp_seconds": 1 / 30,
            "phase": terminal_phase,
            "joint_positions": [0.0] * 6,
            "action_joint_positions": [0.1] * 6,
            "rgb_path": "rgb/front_000001.png",
            "contact": {
                "cube_contact": True,
                "bilateral_cube_contact": True,
            },
            "truth": {"object_root_pose_xyzw": [0.2, 0.1, 0.15, 0, 0, 0, 1]},
        },
    ]
    (root / "observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (root / "metadata.json").write_text(
        json.dumps({"variation": {"variation_id": "cube_0"}}), encoding="utf-8"
    )
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "success": success,
                "dataset_valid": success,
                "failure_reason": None if success else "phase_timeout:close",
            }
        ),
        encoding="utf-8",
    )
    return rows


def test_summarize_so101_episode_reports_physical_and_data_evidence(tmp_path):
    episode = tmp_path / "episode_0"
    _write_episode(episode)

    summary = summarize_so101_episode(episode)

    assert summary["success"] is True
    assert summary["observation_count"] == 2
    assert summary["terminal_phase"] == "retreat"
    assert summary["object_lift_above_initial_m"] == pytest.approx(0.1)
    assert summary["final_object_xy_displacement_m"] == pytest.approx([0.1, 0.1])
    assert summary["bilateral_contact_frames"] == 1
    assert summary["camera_frame_counts"] == {"front": 2}
    assert summary["state_dimensions"] == [6]
    assert summary["action_dimensions"] == [6]
    assert summary["timestamps_strictly_increasing"] is True
    assert summary["phase_ranges"] == [
        {"phase": "home", "start_frame": 0, "end_frame": 0, "frame_count": 1},
        {
            "phase": "retreat",
            "start_frame": 1,
            "end_frame": 1,
            "frame_count": 1,
        },
    ]
    assert summary["observations_sha256"] == hashlib.sha256(
        (episode / "observations.jsonl").read_bytes()
    ).hexdigest()


def test_analysis_flags_duplicate_observations_as_non_independent(tmp_path):
    first = tmp_path / "episode_first"
    rows = _write_episode(first)
    second = tmp_path / "episode_second"
    _write_episode(second, duplicate=rows)

    analysis = analyze_so101_episodes([first, second])

    assert analysis["episode_count"] == 2
    assert analysis["success_count"] == 2
    assert analysis["independent_observation_artifact_count"] == 1
    assert len(analysis["duplicate_observation_groups"]) == 1
    report = render_so101_analysis_markdown(analysis)
    assert "must not be counted as independent stability trials" in report


def test_analysis_counts_failure_reasons(tmp_path):
    episode = tmp_path / "episode_failed"
    _write_episode(episode, success=False, terminal_phase="close")

    analysis = analyze_so101_episodes([episode])

    assert analysis["failure_count"] == 1
    assert analysis["failure_reason_counts"] == {"phase_timeout:close": 1}
    assert analysis["failure_class_counts"] == {"phase_timeout": 1}


def test_summary_reports_proof_lift_tracking_and_close_recenter(tmp_path):
    episode = tmp_path / "episode_tracking"
    rows = _write_episode(episode)
    rows[0].update(
        {
            "phase": "close",
            "contact_forces_newtons": {"left_finger": 4.0, "right_finger": 0.0},
            "truth": {
                **rows[0]["truth"],
                "grasp_xy_correction_m": [0.003, 0.004],
            },
        }
    )
    rows[1].update(
        {
            "phase": "verify_contact",
            "grasp_evidence": {"proof_lift_m": 0.006},
            "truth": {
                **rows[1]["truth"],
                "proof_lift_target_m": 0.01,
                "gripper_link_pose_xyzw": [0.2, 0.1, 0.16, 0, 0, 0, 1],
            },
        }
    )
    rows.append(
        {
            **rows[1],
            "frame": 2,
            "timestamp_seconds": 2 / 30,
            "phase": "lift",
            "truth": {
                **rows[1]["truth"],
                "transport_lift_target_m": 0.075,
                "transport_lift_actual_m": 0.062,
            },
        }
    )
    (episode / "observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    summary = summarize_so101_episode(episode)

    assert summary["close_unilateral_contact_frames"] == 1
    assert summary["close_peak_unilateral_force_n"] == pytest.approx(4.0)
    assert summary["max_grasp_xy_recenter_m"] == pytest.approx(0.005)
    assert summary["proof_lift_tracking"] == {
        "target_max_m": pytest.approx(0.01),
        "actual_max_m": pytest.approx(0.006),
        "target_minus_actual_m": pytest.approx(0.004),
        "gripper_vertical_displacement_m": pytest.approx(0.0),
        "verify_frame_count": 1,
    }
    assert summary["transport_lift_tracking"] == {
        "target_max_m": pytest.approx(0.075),
        "actual_max_m": pytest.approx(0.062),
        "target_minus_actual_m": pytest.approx(0.013),
        "lift_frame_count": 1,
    }
    report = render_so101_analysis_markdown(
        analyze_so101_episodes([episode])
    )
    assert "Proof target/actual (mm)" in report
    assert "Transport target/actual (mm)" in report
    assert "10.00/6.00" in report
    assert "75.00/62.00" in report
    assert "5.00" in report


def test_failure_classification_is_stable_across_detailed_phase_names():
    assert (
        classify_so101_failure("bilateral_contact_lost:bilateral_settle", "oracle")
        == "bilateral_contact_lost"
    )
    assert classify_so101_failure("ConnectionError: lost", "runner") == "runner_error"


def test_summary_rejects_missing_episode_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        summarize_so101_episode(tmp_path)
