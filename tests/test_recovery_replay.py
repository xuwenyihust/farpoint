import json
from pathlib import Path

import pytest

from farpoint.demonstration import recovery_demonstration
from farpoint.policy_rollout import load_rollout_spec
from farpoint.policy_training import file_sha256
from farpoint.recovery_replay import write_recovery_replay_bundle

from test_episode_metadata_v4 import episode_v4


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
    (root / "metadata.json").write_text(json.dumps(metadata))
    (root / "handoff.json").write_text(json.dumps({"state_snapshot": snapshot}))
    rows = [
        {"action_joint_positions": [-0.2, -0.6, -0.1, 1.5, -1.6, 1.7], "phase": "home"},
        {"action_joint_positions": [-0.19, -0.59, -0.09, 1.49, -1.59, 1.69], "phase": "home"},
    ]
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
    )
    spec = load_rollout_spec(output / "spec.json")
    replay = json.loads((output / "replay-manifest.json").read_text())
    assert result["scene_count"] == 2
    assert spec["task"]["evaluation_class"] == "recovery_expert_replay"
    assert spec["acceptance"]["minimum_task_successes"] == 2
    assert spec["recovery_replay_source"]["selection_sha256"] == file_sha256(selection_path)
    assert spec["scenes"][0]["initial_state"]["policy_step"] == 119
    assert len(replay["scenes"][0]["actions_calibrated"]) == 2
    assert replay["scenes"][0]["source_values_clipped_by_exporter"] == 0


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
    runtime.write_text(json.dumps({"control": {}}))
    try:
        write_recovery_replay_bundle(
            selection_path,
            root / "configs/evaluations/so101_act_v0_1_0_holdout_template.json",
            runtime,
            tmp_path / "out",
            scene_count=1,
            suite_id="invalid",
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
    runtime_path.write_text(json.dumps({"control": {}}))
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        write_recovery_replay_bundle(
            selection_path,
            root / "configs/evaluations/so101_act_v0_1_0_holdout_template.json",
            runtime_path,
            tmp_path / "out",
            scene_count=1,
            suite_id="invalid_snapshot",
        )
