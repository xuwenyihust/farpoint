"""Audit recovery expert replay integrity independently of contact outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from farpoint.campaign import canonical_sha256
from farpoint.policy_training import file_sha256
from farpoint.so101 import lerobot_to_radians


BOUNDARY_LIMITS = {
    "joint_position_rad": 1e-6,
    "object_pose": 1e-5,
    "object_linear_velocity_mps": 2e-4,
    "contact_force_n": 1e-3,
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _maximum_error(first: Any, second: Any) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return float("inf")
    return float(np.max(np.abs(left - right), initial=0.0))


def _source_episode_map(selection: dict[str, Any]) -> dict[str, Path]:
    result = {}
    for episode in selection.get("episodes") or []:
        root = Path(episode["episode_dir"])
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        episode_id = metadata["identity"]["episode_id"]
        if episode_id in result:
            raise ValueError("recovery replay selection contains duplicate episode IDs")
        result[episode_id] = root
    return result


def build_recovery_replay_integrity_report(
    *,
    selection_path: Path,
    spec_path: Path,
    replay_manifest_path: Path,
    run_root: Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Verify replay provenance, exact commands, boundary state, and videos.

    Task outcome remains diagnostic: rigid-contact playback can diverge under
    sub-micron numerical differences even when the recorded commands and
    handoff state are reproduced. The live, same-process source episode is the
    authoritative physical-success evidence.
    """
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
    rollout = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    source_roots = _source_episode_map(selection)
    errors: list[str] = []
    selection_sha = file_sha256(selection_path)
    replay_sha = file_sha256(replay_manifest_path)
    if spec["recovery_replay_source"]["selection_sha256"] != selection_sha:
        errors.append("spec_selection_sha256_mismatch")
    if replay["source"]["selection_sha256"] != selection_sha:
        errors.append("replay_selection_sha256_mismatch")
    if rollout["recovery_replay_source"]["selection_sha256"] != selection_sha:
        errors.append("report_selection_sha256_mismatch")
    if rollout.get("spec_sha256") != canonical_sha256(spec):
        errors.append("report_spec_sha256_mismatch")
    if (rollout.get("policy_server") or {}).get("action_execution", {}).get(
        "replay_manifest_sha256"
    ) != replay_sha:
        errors.append("report_replay_manifest_sha256_mismatch")
    if rollout.get("rollout_git_commit") != expected_git_commit:
        errors.append("report_git_commit_mismatch")
    if spec["recovery_replay_source"].get("state_restore") != (
        "reset_plus_full_command_history_v1"
    ):
        errors.append("full_history_state_restore_not_frozen")
    if spec["recovery_replay_source"].get("command_replay") != (
        "policy_history_then_physics_rate_trace_v1"
    ):
        errors.append("full_history_command_replay_not_frozen")

    replay_scenes = {scene["scene_id"]: scene for scene in replay["scenes"]}
    result_scenes = {scene["scene_id"]: scene for scene in rollout["episodes"]}
    if set(replay_scenes) != set(result_scenes) or set(replay_scenes) != {
        scene["scene_id"] for scene in spec["scenes"]
    }:
        errors.append("scene_identity_set_mismatch")
    audits = []
    for scene_id, source in replay_scenes.items():
        prefix = f"{scene_id}:"
        source_root = source_roots.get(source["source_recovery_episode_id"])
        result = result_scenes.get(scene_id)
        if source_root is None or result is None:
            errors.append(prefix + "missing_source_or_result")
            continue
        metadata_path = source_root / "metadata.json"
        handoff_path = source_root / "handoff.json"
        observations_path = source_root / "observations.jsonl"
        metrics_path = source_root / "metrics.json"
        for key, path in (
            ("source_metadata_sha256", metadata_path),
            ("source_handoff_sha256", handoff_path),
            ("source_observations_sha256", observations_path),
        ):
            if source.get(key) != file_sha256(path):
                errors.append(prefix + key + "_mismatch")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not metrics.get("success") or not metrics.get("dataset_valid"):
            errors.append(prefix + "source_episode_not_success_valid")
        pre = source["source_pre_handoff_trace"]
        pre_path = source_root / pre["path"]
        if file_sha256(pre_path) != pre["sha256"]:
            errors.append(prefix + "pre_handoff_trace_sha256_mismatch")
        command = source["source_command_trace"]
        command_path = source_root / command["path"]
        if file_sha256(command_path) != command["sha256"]:
            errors.append(prefix + "oracle_command_trace_sha256_mismatch")

        trace_path = run_root / result["trace"]
        trace = _rows(trace_path)
        groups = source["physics_action_groups_radians"]
        if len(trace) != result["policy_steps"]:
            errors.append(prefix + "rollout_trace_length_mismatch")
        if len(trace) < len(groups) and not result.get("task_success"):
            errors.append(prefix + "failed_playback_ended_before_source_trace")
        compared = len(trace)
        maximum_target_error = 0.0
        for index in range(compared):
            execution = trace[index].get("policy_execution") or {}
            expected_group = (
                groups[index]
                if index < len(groups)
                else [groups[-1][-1]]
                * int(replay["physics_replay"]["maximum_targets_per_policy_step"])
            )
            error = _maximum_error(
                execution.get("physics_actions_radians"), expected_group
            )
            maximum_target_error = max(maximum_target_error, error)
        if maximum_target_error > 1e-7:
            errors.append(prefix + "physics_target_sequence_mismatch")

        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))["state_snapshot"]
        boundary = int(pre["sample_count"])
        if boundary >= len(trace) or boundary < 1:
            errors.append(prefix + "handoff_boundary_missing")
            continue
        before = trace[boundary - 1]
        after = trace[boundary]
        boundary_errors = {
            "joint_position_rad": _maximum_error(
                lerobot_to_radians(after["state_calibrated"], clip=True),
                handoff["joint_positions_rad"],
            ),
            "object_pose": _maximum_error(
                before["cube_pose_xyzw"], handoff["object_pose_xyzw"]
            ),
            "object_linear_velocity_mps": _maximum_error(
                before["cube_velocity_mps"], handoff["object_linear_velocity_mps"]
            ),
            "contact_force_n": _maximum_error(
                before["contact_forces_n"], handoff["contact_forces_n"]
            ),
        }
        for key, error in boundary_errors.items():
            if error > BOUNDARY_LIMITS[key]:
                errors.append(prefix + f"handoff_{key}_error")
        if _maximum_error(before["target_radians"], handoff["joint_position_target_rad"]) > 1e-7:
            errors.append(prefix + "handoff_target_error")
        videos_ok = True
        for camera in ("front", "wrist"):
            evidence = (result.get("videos") or {}).get(camera) or {}
            video_path = run_root / str(evidence.get("path") or "")
            ok = (
                video_path.is_file()
                and file_sha256(video_path) == evidence.get("sha256")
                and evidence.get("decoded_frames") == result.get("policy_steps")
                and evidence.get("width") == 640
                and evidence.get("height") == 480
                and evidence.get("avg_frame_rate") == "30/1"
            )
            videos_ok = videos_ok and ok
            if not ok:
                errors.append(prefix + f"{camera}_video_integrity_error")
        if result.get("nonfinite_action_count") != 0:
            errors.append(prefix + "nonfinite_actions")
        if result.get("hard_range_violation_count") != 0:
            errors.append(prefix + "hard_range_violations")
        audits.append(
            {
                "scene_id": scene_id,
                "source_episode_id": source["source_recovery_episode_id"],
                "source_success_dataset_valid": bool(
                    metrics.get("success") and metrics.get("dataset_valid")
                ),
                "playback_task_success": bool(result.get("task_success")),
                "playback_terminal_reason": result.get("terminal_reason"),
                "physics_target_count_compared": compared,
                "maximum_physics_target_error_rad": maximum_target_error,
                "handoff_boundary_errors": boundary_errors,
                "dual_video_integrity": videos_ok,
            }
        )
    return {
        "schema_version": "farpoint.recovery-replay-integrity-report.v1",
        "status": "PASS" if not errors else "FAIL",
        "suite_id": spec["suite_id"],
        "rollout_git_commit": expected_git_commit,
        "selection_sha256": selection_sha,
        "spec_file_sha256": file_sha256(spec_path),
        "spec_canonical_sha256": canonical_sha256(spec),
        "replay_manifest_sha256": replay_sha,
        "scene_count": len(audits),
        "source_success_count": sum(
            int(row["source_success_dataset_valid"]) for row in audits
        ),
        "playback_task_success_count": sum(int(row["playback_task_success"]) for row in audits),
        "playback_task_outcome_policy": "diagnostic_only_due_to_rigid_contact_nondeterminism",
        "limits": BOUNDARY_LIMITS,
        "scenes": audits,
        "evidence_errors": sorted(set(errors)),
    }
