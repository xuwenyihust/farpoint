import json
from pathlib import Path

import pytest

from farpoint.demonstration import intervention_command_trace, recovery_demonstration
from farpoint.campaign import canonical_sha256
from farpoint.policy_rollout import load_rollout_spec
from farpoint.policy_training import file_sha256
from farpoint.recovery_replay import write_recovery_replay_bundle
from farpoint.recovery_replay_audit import build_recovery_replay_integrity_report
from farpoint.so101 import radians_to_lerobot

from test_episode_metadata_v4 import episode_v4


ACTION_SAFETY_CALIBRATION = {
    "reference_suite_id": "nominal-expert-replay",
    "reference_report_sha256": "f" * 64,
    "reference_minimum_delta_limited_actions_per_episode": 26,
    "reference_maximum_delta_limited_actions_per_episode": 32,
    "allowed_maximum_delta_limited_actions_per_episode": 45,
}


def _snapshot():
    return {
        "policy_step": 119,
        "joint_positions_rad": [-0.2, -0.6, -0.1, 1.5, -1.6, 1.7],
        "joint_velocities_rad_s": [0.0] * 6,
        "joint_position_target_rad": [-0.2, -0.6, -0.1, 1.5, -1.6, 1.7],
        "object_pose_xyzw": [0.2, -0.05, 0.02, 0.0, 0.0, 0.0, 1.0],
        "object_linear_velocity_mps": [0.0, 0.0, 0.0],
        "object_angular_velocity_rad_s": [0.0, 0.0, 0.0],
        "contact_forces_n": [0.0, 0.0],
        "applied_policy_action_calibrated": [0.0] * 6,
    }


def _write_episode(root, episode_id):
    root.mkdir()
    metadata = episode_v4()
    metadata["identity"]["episode_id"] = episode_id
    metadata["variation"]["resolved"].update(
        {"yaw_degrees": 9.0, "yaw_stratum_id": "yaw00_18", "region_band": "middle"}
    )
    snapshot = _snapshot()
    metadata["demonstration"] = recovery_demonstration(
        oracle_profile_id="oracle-v1",
        source_policy={
            "policy_type": "act",
            "checkpoint_step": 20_000,
            "model_sha256": "1" * 64,
            "training_run_id": "act-v010",
            "rollout_git_commit": "a" * 40,
        },
        trigger_id="stall-v1",
        failure_class="progress_stall",
        control_step=119,
        stage="pre_lift",
        trigger_evidence={"distance_m": 0.2},
        source_rollout_id="rollout-v1",
        source_scene_id="train-scene-1",
        state_snapshot=snapshot,
        recovery_strategy_id="regrasp-v1",
    )
    rows = [
        {
            "control_step": 0,
            "action_joint_positions": [-0.2, -0.6, -0.1, 1.5, -1.6, 1.7],
            "phase": "home",
        },
        {
            "control_step": 4,
            "action_joint_positions": [-0.19, -0.59, -0.09, 1.49, -1.59, 1.69],
            "phase": "home",
        },
    ]
    physics_rows = []
    for control_step in range(8):
        fraction = min(control_step, 4) / 4
        action = [
            float(first + fraction * (second - first))
            for first, second in zip(
                rows[0]["action_joint_positions"],
                rows[1]["action_joint_positions"],
            )
        ]
        physics_rows.append(
            {
                "control_step": control_step,
                "action_joint_positions": action,
                "phase": "home",
            }
        )
    trace_path = root / "oracle-commands.jsonl"
    trace_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in physics_rows))
    command_trace = intervention_command_trace(
        path=trace_path.name,
        sha256=file_sha256(trace_path),
        control_hz=120,
        sample_count=8,
        first_control_step=0,
        last_control_step=7,
        joint_order=metadata["recording"]["action_features"],
    )
    metadata["demonstration"]["intervention"]["command_trace"] = command_trace
    pre_handoff_rows = []
    for policy_step in range(120):
        pre_handoff_rows.append(
            {
                "policy_step": policy_step,
                "applied_action_calibrated": [0.0] * 6,
            }
        )
    (root / "pre-handoff-actions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pre_handoff_rows)
    )
    (root / "metadata.json").write_text(json.dumps(metadata))
    (root / "handoff.json").write_text(
        json.dumps(
            {
                "state_snapshot": snapshot,
                "command_trace": command_trace,
                "demonstration": metadata["demonstration"],
            }
        )
    )
    (root / "observations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_recovery_replay_binds_live_snapshot_and_exported_actions(tmp_path):
    first = tmp_path / "episode-1"
    second = tmp_path / "episode-2"
    _write_episode(first, "episode-1")
    _write_episode(second, "episode-2")
    selection = {
        "schema_version": "farpoint.export-selection.v1",
        "dataset_id": "recovery",
        "collection_id": "recovery-campaign",
        "episodes": [
            {"episode_dir": str(first), "split": "train"},
            {"episode_dir": str(second), "split": "train"},
        ],
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    root = Path(__file__).resolve().parents[1]
    template_path = root / "configs/evaluations/so101_act_v0_1_0_holdout_template.json"
    runtime = {
        "control": {
            "physics_hz": 120,
            "policy_hz": 30,
            "replan_interval_steps": 10,
            "action_safety_profile": {
                "schema_version": "farpoint.action-safety-profile.v1",
                "profile_id": "test",
                "joint_order": [
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                    "gripper.pos",
                ],
                "arm_max_command_speed_deg_s": 50.0,
                "gripper_max_command_slew_calibrated_per_step": 5.5,
                "source": {
                    "kind": "open_source_hardware_default",
                    "reference": "https://example.invalid/so101",
                    "resolved_revision": "test",
                    "statistic": "speed cap",
                },
            },
        }
    }
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime))
    output = tmp_path / "bundle"
    result = write_recovery_replay_bundle(
        selection_path,
        template_path,
        runtime_path,
        output,
        scene_count=2,
        suite_id="recovery_replay_test",
        action_safety_calibration=ACTION_SAFETY_CALIBRATION,
    )
    spec = load_rollout_spec(output / "spec.json")
    replay = json.loads((output / "replay-manifest.json").read_text())
    assert result["scene_count"] == 2
    assert spec["task"]["evaluation_class"] == "recovery_expert_replay"
    assert spec["acceptance"]["minimum_task_successes"] == 2
    assert spec["acceptance"]["maximum_delta_limited_actions"] == 90
    assert spec["recovery_replay_source"]["action_safety_calibration"] == ACTION_SAFETY_CALIBRATION
    assert spec["recovery_replay_source"]["selection_sha256"] == file_sha256(selection_path)
    assert spec["scenes"][0]["seed"] == episode_v4()["identity"]["attempt_seed"]
    assert spec["scenes"][0]["seed"] != episode_v4()["identity"]["variation_seed"]
    assert "initial_state" not in spec["scenes"][0]
    assert len(replay["scenes"][0]["actions_calibrated"]) == 122
    assert len(replay["scenes"][0]["physics_action_groups_radians"]) == 122
    assert all(len(group) == 4 for group in replay["scenes"][0]["physics_action_groups_radians"])
    assert replay["physics_replay"] == {
        "mode": "exact_trace",
        "unit": "radian",
        "physics_hz": 120,
        "policy_hz": 30,
        "maximum_targets_per_policy_step": 4,
    }
    assert spec["recovery_replay_source"]["state_restore"] == (
        "reset_plus_full_command_history_v1"
    )
    assert spec["recovery_replay_source"]["command_replay"] == (
        "policy_history_then_physics_rate_trace_v1"
    )
    assert replay["scenes"][0]["source_pre_handoff_trace"]["sample_count"] == 120
    assert replay["scenes"][0]["source_values_clipped_by_exporter"] == 0
    assert replay["scenes"][0]["source_physics_values_clipped_by_exporter"] == 0


def test_recovery_replay_rejects_non_train_episode(tmp_path):
    episode = tmp_path / "episode"
    _write_episode(episode, "episode")
    metadata = json.loads((episode / "metadata.json").read_text())
    metadata["identity"]["split"] = "validation"
    metadata["variation"]["split"] = "validation"
    (episode / "metadata.json").write_text(json.dumps(metadata))
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": "farpoint.export-selection.v1",
                "collection_id": "recovery",
                "episodes": [{"episode_dir": str(episode), "split": "validation"}],
            }
        )
    )
    root = Path(__file__).resolve().parents[1]
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"control": {"physics_hz": 120, "policy_hz": 30}}))
    try:
        write_recovery_replay_bundle(
            selection_path,
            root / "configs/evaluations/so101_act_v0_1_0_holdout_template.json",
            runtime,
            tmp_path / "out",
            scene_count=1,
            suite_id="invalid",
            action_safety_calibration=ACTION_SAFETY_CALIBRATION,
        )
    except ValueError as error:
        assert "train" in str(error)
    else:
        raise AssertionError("validation recovery episode was accepted")


def test_recovery_replay_rejects_snapshot_hash_mismatch(tmp_path):
    episode = tmp_path / "episode"
    _write_episode(episode, "episode")
    handoff = json.loads((episode / "handoff.json").read_text())
    handoff["state_snapshot"]["policy_step"] += 1
    (episode / "handoff.json").write_text(json.dumps(handoff))
    selection = {
        "schema_version": "farpoint.export-selection.v1",
        "collection_id": "recovery",
        "episodes": [{"episode_dir": str(episode), "split": "train"}],
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps({"control": {"physics_hz": 120, "policy_hz": 30}}))
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        write_recovery_replay_bundle(
            selection_path,
            root / "configs/evaluations/so101_act_v0_1_0_holdout_template.json",
            runtime_path,
            tmp_path / "out",
            scene_count=1,
            suite_id="invalid_snapshot",
            action_safety_calibration=ACTION_SAFETY_CALIBRATION,
        )


def test_recovery_replay_rejects_command_trace_hash_mismatch(tmp_path):
    episode = tmp_path / "episode"
    _write_episode(episode, "episode")
    (episode / "oracle-commands.jsonl").write_text("{}\n")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": "farpoint.export-selection.v1",
                "collection_id": "recovery",
                "episodes": [{"episode_dir": str(episode), "split": "train"}],
            }
        )
    )
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps({"control": {"physics_hz": 120, "policy_hz": 30}}))
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="command trace hash mismatch"):
        write_recovery_replay_bundle(
            selection_path,
            root / "configs/evaluations/so101_act_v0_1_0_holdout_template.json",
            runtime_path,
            tmp_path / "out",
            scene_count=1,
            suite_id="invalid_command_trace",
            action_safety_calibration=ACTION_SAFETY_CALIBRATION,
        )


def test_recovery_replay_rejects_safety_bound_below_reference(tmp_path):
    episode = tmp_path / "episode"
    _write_episode(episode, "episode")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": "farpoint.export-selection.v1",
                "collection_id": "recovery",
                "episodes": [{"episode_dir": str(episode), "split": "train"}],
            }
        )
    )
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps({"control": {"physics_hz": 120, "policy_hz": 30}}))
    invalid = {**ACTION_SAFETY_CALIBRATION}
    invalid["allowed_maximum_delta_limited_actions_per_episode"] = 31
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="calibration bounds"):
        write_recovery_replay_bundle(
            selection_path,
            root / "configs/evaluations/so101_act_v0_1_0_holdout_template.json",
            runtime_path,
            tmp_path / "out",
            scene_count=1,
            suite_id="invalid_bounds",
            action_safety_calibration=invalid,
        )


def test_recovery_replay_integrity_passes_with_diagnostic_task_failure(tmp_path):
    episode = tmp_path / "episode"
    _write_episode(episode, "episode")
    (episode / "metrics.json").write_text(
        json.dumps({"success": True, "dataset_valid": True})
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": "farpoint.export-selection.v1",
                "collection_id": "recovery",
                "episodes": [{"episode_dir": str(episode), "split": "train"}],
            }
        )
    )
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "control": {
                    "physics_hz": 120,
                    "policy_hz": 30,
                    "action_safety_profile": {
                        "schema_version": "farpoint.action-safety-profile.v1",
                        "profile_id": "test",
                        "joint_order": [
                            "shoulder_pan.pos",
                            "shoulder_lift.pos",
                            "elbow_flex.pos",
                            "wrist_flex.pos",
                            "wrist_roll.pos",
                            "gripper.pos",
                        ],
                        "arm_max_command_speed_deg_s": 50.0,
                        "gripper_max_command_slew_calibrated_per_step": 5.5,
                        "source": {
                            "kind": "open_source_hardware_default",
                            "reference": "https://example.invalid/so101",
                            "resolved_revision": "test",
                            "statistic": "speed cap",
                        },
                    },
                }
            }
        )
    )
    root = Path(__file__).resolve().parents[1]
    bundle = tmp_path / "bundle"
    write_recovery_replay_bundle(
        selection_path,
        root / "configs/evaluations/so101_act_v0_1_0_holdout_template.json",
        runtime_path,
        bundle,
        scene_count=1,
        suite_id="recovery_integrity_test",
        action_safety_calibration=ACTION_SAFETY_CALIBRATION,
    )
    spec = json.loads((bundle / "spec.json").read_text())
    replay = json.loads((bundle / "replay-manifest.json").read_text())
    source = replay["scenes"][0]
    scene_id = source["scene_id"]
    snapshot = _snapshot()
    episode_root = tmp_path / "run" / "episodes" / scene_id
    episode_root.mkdir(parents=True)
    rows = []
    for index, group in enumerate(source["physics_action_groups_radians"]):
        rows.append(
            {
                "policy_step": index,
                "state_calibrated": radians_to_lerobot(
                    snapshot["joint_positions_rad"], clip=True
                ).tolist(),
                "target_radians": snapshot["joint_position_target_rad"],
                "policy_execution": {"physics_actions_radians": group},
                "cube_pose_xyzw": snapshot["object_pose_xyzw"],
                "cube_velocity_mps": snapshot["object_linear_velocity_mps"],
                "contact_forces_n": snapshot["contact_forces_n"],
            }
        )
    trace_path = episode_root / "actions.jsonl"
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    videos = {}
    for camera in ("front", "wrist"):
        path = episode_root / f"{camera}.mp4"
        path.write_bytes(camera.encode())
        videos[camera] = {
            "path": str(path.relative_to(tmp_path / "run")),
            "sha256": file_sha256(path),
            "decoded_frames": len(rows),
            "width": 640,
            "height": 480,
            "avg_frame_rate": "30/1",
        }
    commit = "a" * 40
    report = {
        "rollout_git_commit": commit,
        "spec_sha256": canonical_sha256(spec),
        "recovery_replay_source": spec["recovery_replay_source"],
        "policy_server": {
            "action_execution": {
                "replay_manifest_sha256": file_sha256(bundle / "replay-manifest.json")
            }
        },
        "episodes": [
            {
                "scene_id": scene_id,
                "task_success": False,
                "terminal_reason": "contact_without_lift",
                "policy_steps": len(rows),
                "trace": str(trace_path.relative_to(tmp_path / "run")),
                "videos": videos,
                "nonfinite_action_count": 0,
                "hard_range_violation_count": 0,
            }
        ],
    }
    (tmp_path / "run" / "report.json").write_text(json.dumps(report))
    audit = build_recovery_replay_integrity_report(
        selection_path=selection_path,
        spec_path=bundle / "spec.json",
        replay_manifest_path=bundle / "replay-manifest.json",
        run_root=tmp_path / "run",
        expected_git_commit=commit,
    )
    assert audit["status"] == "PASS"
    assert audit["source_success_count"] == 1
    assert audit["playback_task_success_count"] == 0
    assert audit["evidence_errors"] == []
