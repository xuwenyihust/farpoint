#!/usr/bin/env python3
"""Generate raw SO-101 cube demonstrations from an Isaac Lab oracle."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import os
import signal
import sys
import traceback
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.campaign_live import LiveCampaignPublisher  # noqa: E402
from farpoint.episode_v4 import (  # noqa: E402
    build_so101_episode_v4,
    is_v010_episode_plan,
)
from farpoint.episode_video import seal_rgb_video  # noqa: E402
from farpoint.so101_runtime import resolve_headless_mode  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

# USD limits observed from the pinned SO-101 asset (radians), kept explicit so
# controller targets cannot be normalized or extrapolated by a backend.
SO101_JOINT_LIMITS = np.asarray(
    [[-1.920, 1.920], [-1.745, 1.745], [-1.745, 1.571],
     [-1.658, 1.658], [-2.793, 2.793]], dtype=np.float32
)
SO101_HOME_JOINTS = np.asarray(
    [-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, 1.7453], dtype=np.float32
)
SO101_GRASP_POSTURE = np.asarray([0.50, 0.50], dtype=np.float32)
# Safe-height reference only.  The 5-DOF arm cannot preserve a full world
# quaternion throughout translation, so production targeting uses the current
# measured orientation and gripper-local aperture on every control step.
SO101_CAPTURE_ORIENTATION_XYZW = np.asarray(
    [-0.6417147, 0.1408973, 0.0826372, -0.7493473], dtype=np.float32
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("headless", "viewer"), default="headless")
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "outputs/so101_variation_plan.json")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "outputs/so101_collection/manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/episodes")
    parser.add_argument("--max-attempts-this-run", type=int, default=150)
    parser.add_argument(
        "--collection-id",
        help="Explicit identity for a non-gate collection; required for formal profiles.",
    )
    parser.add_argument(
        "--watchdog-policy",
        type=Path,
        help=(
            "Evaluate a frozen SO-101 watchdog policy after each completed "
            "attempt and stop before starting another attempt when required."
        ),
    )
    collection_mode = parser.add_mutually_exclusive_group()
    collection_mode.add_argument(
        "--gate-plan",
        action="store_true",
        help="Use a frozen repeatability-gate plan instead of the 100-trial collection.",
    )
    collection_mode.add_argument(
        "--pilot-plan",
        action="store_true",
        help="Use a frozen bounded code-review or diagnostic pilot plan.",
    )
    parser.add_argument("--diagnose-jacobian", action="store_true")
    parser.add_argument("--diagnose-grasp-postures", action="store_true")
    parser.add_argument(
        "--diagnostic-jaw-targets",
        nargs="+",
        type=float,
        default=(1.40,),
        help=(
            "Rotary-jaw targets to evaluate with --diagnose-grasp-postures; "
            "values must be within the pinned USD joint range."
        ),
    )
    parser.add_argument("--diagnose-grasp-offsets", action="store_true")
    parser.add_argument("--diagnose-grasp-grid", action="store_true")
    parser.add_argument("--diagnose-grasp-paths", action="store_true")
    parser.add_argument("--diagnose-calibrated-grasp", action="store_true")
    parser.add_argument(
        "--enable-wrist-camera",
        action="store_true",
        help="Opt in to the wrist camera while preserving legacy v0 front-only defaults.",
    )
    parser.add_argument(
        "--require-dual-camera",
        action="store_true",
        help="Require the frozen v0.1.0 front+wrist profile and fail on config drift.",
    )
    parser.add_argument(
        "--camera-profile",
        type=Path,
        default=PROJECT_ROOT / "configs/cameras/so101_front_wrist_v1.json",
        help="Versioned dual-camera profile used by v0.1.0 collection and diagnostics.",
    )
    parser.add_argument(
        "--recovery-runtime",
        type=Path,
        help=(
            "Run a frozen ACT policy until a versioned live-handoff trigger, "
            "then continue the same PhysX state with the Oracle."
        ),
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        help="Campaign directory for atomic status, heartbeat, preview, and events.",
    )
    parser.add_argument("--campaign-id")
    parser.add_argument("--segment-id")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.require_dual_camera:
        args.enable_wrist_camera = True
    campaign_values = (args.campaign_root, args.campaign_id, args.segment_id)
    if any(value is not None for value in campaign_values) and not all(
        value is not None for value in campaign_values
    ):
        parser.error(
            "--campaign-root, --campaign-id, and --segment-id must be provided together"
        )
    args.headless = resolve_headless_mode(
        args.mode,
        args.livestream,
        livestream_env=os.environ.get("LIVESTREAM"),
    )
    return args


args_cli = parse_args()
live_publisher = None
if args_cli.campaign_root is not None:
    live_publisher = LiveCampaignPublisher(
        args_cli.campaign_root,
        args_cli.campaign_id,
        args_cli.segment_id,
    )
    # This intentionally precedes AppLauncher: RTX/Kit startup can take
    # minutes or fail, and the campaign must still become immediately visible.
    live_publisher.start(
        payload={
            "git_commit": os.environ.get("FARPOINT_GIT_COMMIT", "unknown"),
            "startup_phase": "isaac_app_launcher",
        }
    )
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import farpoint_so101_env  # noqa: E402,F401
from farpoint_so101_env.mdp import (  # noqa: E402
    SO101_GRIPPER_DYNAMIC_FRICTION,
    SO101_GRIPPER_RESTITUTION,
    SO101_GRIPPER_STATIC_FRICTION,
)
from farpoint.contracts import validate_contract, validate_episode_semantics  # noqa: E402
from farpoint.demonstration import (  # noqa: E402
    intervention_command_trace,
    recovery_demonstration,
)
from farpoint.camera_profiles import (  # noqa: E402
    build_camera_records,
    camera_cfg_drift_errors,
    load_camera_profile,
    resolved_mounts_from_profile,
)
from farpoint.control import (  # noqa: E402
    advance_so101_slow_close_target,
    bounded_position_target,
    collision_safe_pregrasp_waypoints,
    force_controlled_rotary_jaw_target,
    relative_object_grasp_servo_target,
    settle_release_separation_target,
    so101_release_object_target,
    so101_approach_jaw_target,
    so101_capture_admission_ready,
    so101_bilateral_capture_ready,
    so101_capture_contact_loss_grace_s,
    so101_capture_jaw_backoff_force_n,
    so101_slow_close_bilateral_brake_force_n,
    so101_slow_close_backoff_step_rad,
    so101_balanced_capture_close_step,
    so101_imbalanced_capture_close_step,
    so101_proof_entry_force_floor,
    so101_cube_contact_handoff,
    so101_minimum_safe_descent_fraction,
    so101_adaptive_pre_capture_recenter_limit,
    so101_post_capture_recenter_step,
    so101_reset_support_is_stable,
    unilateral_contact_recenter_target,
    unsafe_so101_approach_contact,
)
from farpoint.grasp_oracle import (  # noqa: E402
    ContactAwareGraspStateMachine,
    ControlRecordingSchedule,
    GraspDecision,
    GraspEvidence,
    GraspPhase,
    advance_proof_lift_command,
    cartesian_motion_command_base,
    capture_hold_preload_for_force,
    capture_retention_recenter_fallback_active,
    capture_retention_reopen_active,
    capture_retention_force_floor,
    capture_preload_force_floor,
    captured_force_imbalance_requires_recenter,
    captured_force_imbalance_requires_squeeze_pause,
    contact_constrained_joint_step_limit,
    capture_aperture_laterally_aligned,
    contact_force_vectors_opposed,
    grasp_phase_allows_unilateral_recenter,
    gripper_target_for_object_local_offset,
    gripper_xy_target_for_object_local_offset,
    latch_pre_capture_recenter_object_reference,
    point_in_local_frame,
    proof_lift_recovery_holds_xy,
    quaternion_rotation_matrix_xyzw,
    rotary_jaw_capture_hold_target,
    so101_recenter_contact_memory,
    unilateral_contact_requires_recenter,
)
from farpoint.oracle import (  # noqa: E402
    OracleObservation,
    OraclePhase,
    OracleStateMachine,
    damped_least_squares,
    oriented_box_footprint_inside_target,
    quaternion_direction_error,
)
from farpoint.policy_rollout import (  # noqa: E402
    constrain_policy_action,
    resolve_action_safety_profile,
)
from farpoint.recovery_runtime import (  # noqa: E402
    RecoveryTriggerDetector,
    load_recovery_runtime,
    recovery_descent_duration_seconds,
    recovery_oracle_command_continuity_enabled,
    recovery_oracle_entry_phase,
    recovery_oracle_slew_limits,
    recovery_trigger_for_scene,
    scene_binding,
    slew_recovery_oracle_target,
)
from farpoint.scene_entities import bind_scene_entities  # noqa: E402
from farpoint.so101 import (  # noqa: E402
    LEROBOT_JOINT_NAMES,
    SIM_JOINT_NAMES,
    lerobot_to_radians,
    mapping_metadata,
    radians_to_lerobot,
)
from farpoint.so101_grasp_geometry import (  # noqa: E402
    SO101_APERTURE_REFERENCE_IN_GRIPPER_M,
    SO101_CAPTURE_CLOSING_AXIS_LOCAL,
    SO101_RUNTIME_QUATERNION_ORDER,
    SO101_WORKSHOP_ASSET_SHA256,
    SO101_WORKSHOP_COMMIT,
    posture_geometry_diagnostics,
    so101_capture_aperture_reference,
    so101_capture_channel_direction_world,
    so101_level_capture_orientation_xyzw,
    so101_pre_capture_recenter_aperture_reference,
)
from farpoint.so101_collection import (  # noqa: E402
    CollectionSignalAbort,
    abort_attempt_run_state,
    abort_collection_manifest,
    build_attempt_run_state,
    build_export_selection,
    create_gate_manifest,
    create_manifest,
    create_pilot_manifest,
    collection_interruption_reason,
    episode_id_for_attempt,
    finish_diagnostic_manifest,
    load_manifest,
    next_attempt,
    record_attempt,
    raise_collection_signal_abort,
    write_manifest,
)
from farpoint.so101_watchdog import (  # noqa: E402
    evaluate_so101_collection,
    load_watchdog_policy,
)


CONTROL_RECORDING_SCHEDULE = ControlRecordingSchedule(
    control_hz=120, recording_hz=30
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evaluate_watchdog(plan, manifest, policy):
    report = evaluate_so101_collection(
        plan,
        manifest,
        policy,
        episodes_root=args_cli.output_root,
    )
    _write_json(args_cli.manifest.with_name("watchdog.json"), report)
    decision = report["decision"]
    print(
        f"SO101_WATCHDOG decision={decision} "
        f"reasons={','.join(report['reasons']) or 'none'}",
        flush=True,
    )
    if decision in {"STOP", "INVALID"} and manifest.get("execution_status") == "RUNNING":
        reason = ";".join(report["reasons"] or report["errors"]) or "watchdog_stop"
        abort_collection_manifest(
            manifest, f"watchdog:{decision.lower()}:{reason}"
        )
        write_manifest(args_cli.manifest, manifest)
    return decision


def _torch_pose(position, device, orientation_xyzw=(0.0, 0.0, 0.0, 1.0)):
    return torch.tensor(
        [[*position, *orientation_xyzw]], dtype=torch.float32, device=device
    )


def _numpy(value):
    """Convert Isaac Lab tensor or NumPy backend values to a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _move_object(obj, position, device, orientation_xyzw=(0.0, 0.0, 0.0, 1.0)):
    obj.write_root_pose_to_sim(_torch_pose(position, device, orientation_xyzw))
    obj.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=device))


def _move_static_frame(frame, position, device):
    """Move a non-rigid AssetBase through Isaac Lab's frame-view API."""
    positions = torch.tensor([position], dtype=torch.float32, device=device)
    frame.set_world_poses(positions=positions)


def _image(camera, device):
    return np.asarray(_numpy(camera.data.output["rgb"][0, ..., :3]), dtype=np.uint8)


def _policy_request(path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    policy_url = os.environ.get("FARPOINT_ACT_POLICY_URL", "http://127.0.0.1:8766")
    request = urllib.request.Request(
        f"{policy_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def _jpeg_payload(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _policy_action(state: np.ndarray, images: dict[str, np.ndarray], task: str):
    response = _policy_request(
        "/action",
        {
            "state": state.tolist(),
            "images_jpeg": {
                f"observation.images.{camera_id}": _jpeg_payload(image)
                for camera_id, image in images.items()
            },
            "task": task,
        },
    )
    return np.asarray(response["action"], dtype=np.float32), response.get("execution", {})


def _aim_front_camera(scene, device, camera_view=None) -> None:
    """Set the front camera from an explicit, episode-bound eye/look-at pair."""
    camera_view = camera_view or {}
    eye = torch.tensor(
        [camera_view.get("eye_m", [0.42, -0.38, 0.34])],
        dtype=torch.float32,
        device=device,
    )
    target = torch.tensor(
        [camera_view.get("look_at_m", [0.20, 0.02, 0.06])],
        dtype=torch.float32,
        device=device,
    )
    scene["front_camera"].set_world_poses_from_view(eye, target)


def _contact(sensors) -> bool:
    if not isinstance(sensors, (tuple, list)):
        sensors = (sensors,)
    for sensor in sensors:
        if hasattr(sensor.data, "force_matrix_w"):
            forces = sensor.data.force_matrix_w
            if bool(torch.linalg.vector_norm(forces, dim=-1).max().item() > 2.0):
                return True
        elif hasattr(sensor.data, "net_forces_w"):
            if bool(torch.linalg.vector_norm(sensor.data.net_forces_w, dim=-1).max().item() > 2.0):
                return True
    return False


def _bilateral_contact(sensors, threshold_n: float = 2.0) -> bool:
    """Require cube-filtered force on both the moving and fixed fingers."""
    for sensor in sensors:
        if not hasattr(sensor.data, "force_matrix_w"):
            return False
        force = float(torch.linalg.vector_norm(sensor.data.force_matrix_w, dim=-1).max().item())
        if force <= threshold_n:
            return False
    return True


def _contact_debug(sensors) -> list[dict[str, float]]:
    values = []
    for sensor in sensors:
        item = {}
        if hasattr(sensor.data, "force_matrix_w"):
            vectors = sensor.data.force_matrix_w.reshape(-1, 3)
            norms = torch.linalg.vector_norm(vectors, dim=-1)
            max_index = int(torch.argmax(norms).item())
            item["cube_filtered_max_n"] = float(norms[max_index].item())
            item["cube_filtered_vector_n"] = _numpy(vectors[max_index]).tolist()
        if hasattr(sensor.data, "net_forces_w"):
            item["all_contacts_max_n"] = float(
                torch.linalg.vector_norm(sensor.data.net_forces_w, dim=-1).max().item()
            )
        values.append(item)
    return values


def _cube_contact_forces(sensors) -> tuple[float, float]:
    """Return peak cube-filtered force for the fixed and moving fingers."""
    forces = []
    for sensor in sensors:
        force = 0.0
        if hasattr(sensor.data, "force_matrix_w"):
            force = float(
                torch.linalg.vector_norm(sensor.data.force_matrix_w, dim=-1).max().item()
            )
        forces.append(force)
    if len(forces) != 2:
        raise RuntimeError(f"expected two SO-101 finger sensors, got {len(forces)}")
    return forces[0], forces[1]


def _cube_contact_force_vectors(sensors) -> tuple[np.ndarray, np.ndarray]:
    """Return peak cube-filtered force vectors for both finger sensors."""
    vectors = []
    for sensor in sensors:
        peak = np.zeros(3, dtype=np.float32)
        if hasattr(sensor.data, "force_matrix_w"):
            candidates = sensor.data.force_matrix_w.reshape(-1, 3)
            norms = torch.linalg.vector_norm(candidates, dim=-1)
            peak = _numpy(candidates[int(torch.argmax(norms).item())]).astype(
                np.float32
            )
        vectors.append(peak)
    if len(vectors) != 2:
        raise RuntimeError(f"expected two SO-101 finger sensors, got {len(vectors)}")
    return vectors[0], vectors[1]


def _body_index(robot) -> int:
    indexes, _names = robot.find_bodies("gripper")
    if len(indexes) != 1:
        raise RuntimeError(f"expected one SO-101 gripper body, got {indexes}")
    return int(indexes[0])


def _run_recovery_handoff(
    env, trial, active_object, sensors, root, runtime, *, oracle_profile_id
):
    """Execute ACT to a measured pre-lift trigger without resetting the scene."""
    binding = scene_binding(runtime, trial["variation_id"])
    policy = runtime["source_policy"]
    runtime_git_commit = os.environ.get("FARPOINT_GIT_COMMIT", "")
    if len(runtime_git_commit) != 40:
        raise RuntimeError("recovery runtime requires an exact 40-character git commit")
    policy_provenance = {
        **policy,
        "rollout_git_commit": runtime_git_commit,
    }
    health = _policy_request("/health")
    if health.get("status") != "ready":
        raise RuntimeError("recovery source policy server is not ready")
    if health.get("model_sha256") != policy["model_sha256"]:
        raise RuntimeError("recovery source policy checkpoint identity mismatch")
    policy_execution = health.get("action_execution") or {}
    if int(policy_execution.get("replan_interval_steps", -1)) != int(
        runtime["control"]["replan_interval_steps"]
    ):
        raise RuntimeError("recovery source policy replan interval mismatch")
    if health.get("camera_features") != [
        "observation.images.front",
        "observation.images.wrist",
    ]:
        raise RuntimeError("recovery source policy camera feature mismatch")
    _policy_request("/reset", {"scene_id": binding["source_scene_id"]})

    scene = env.scene
    robot = scene["robot"]
    body_index = _body_index(robot)
    control = runtime["control"]
    safety_profile = resolve_action_safety_profile(control)
    trigger_profile = recovery_trigger_for_scene(runtime, trial["variation_id"])
    detector = RecoveryTriggerDetector(trigger_profile)
    physics_steps = int(control["physics_hz"]) // int(control["policy_hz"])
    previous_applied = radians_to_lerobot(
        _numpy(robot.data.joint_pos[0]).astype(np.float32), clip=True
    )
    initial_z = float(_numpy(active_object.data.root_pos_w[0])[2])
    rows = []
    trigger = None
    snapshot = None
    maximum_steps = int(trigger_profile["maximum_policy_steps_before_handoff"])
    target_spec = (runtime.get("task_context") or {}).get("target") or {
        "position_m": [0.20, 0.10, 0.037],
        "dimensions_m": [0.16, 0.14, 0.01],
        "footprint_margin_m": 0.005,
    }
    object_spec = trial["resolved"]
    for policy_step in range(maximum_steps):
        images = {
            "front": _image(scene["front_camera"], env.device),
            "wrist": _image(scene["wrist_camera"], env.device),
        }
        joint_radians = _numpy(robot.data.joint_pos[0]).astype(np.float32)
        state = radians_to_lerobot(joint_radians, clip=True)
        raw_action, execution = _policy_action(
            state,
            images,
            "Pick up the cube and place it on the green target pad.",
        )
        applied, safety = constrain_policy_action(
            raw_action,
            state,
            action_safety_profile=safety_profile,
            previous_applied_action=previous_applied,
        )
        previous_applied = applied.copy()
        target_radians = lerobot_to_radians(applied, clip=True)
        target = torch.tensor([target_radians], dtype=torch.float32, device=env.device)
        for _ in range(physics_steps):
            env.step(target)
        object_pose = _numpy(active_object.data.root_pose_w[0]).astype(np.float32)
        object_velocity = _numpy(active_object.data.root_lin_vel_w[0]).astype(np.float32)
        gripper_pose = _numpy(
            robot.data.body_link_pose_w.torch[0, body_index]
        ).astype(np.float32)
        forces = _cube_contact_forces(sensors)
        lifted = max(forces) >= float(
            trigger_profile.get("contact_force_threshold_n", 0.1)
        ) and float(object_pose[2]) > initial_z + float(
            trigger_profile.get("lift_threshold_m", 0.005)
        )
        cube_in_target = oriented_box_footprint_inside_target(
            object_pose[:3],
            object_spec["dimensions_m"],
            object_pose[3:7],
            target_spec["position_m"],
            target_spec["dimensions_m"],
            margin_m=float(target_spec.get("footprint_margin_m", 0.005)),
        )
        gripper_released = abs(
            float(robot.data.joint_pos[0, 5].item()) - float(SO101_HOME_JOINTS[5])
        ) < 0.08
        cube_stable = float(np.linalg.norm(object_velocity)) < 0.03
        trigger = detector.observe(
            policy_step=policy_step,
            gripper_position_m=gripper_pose[:3],
            object_position_m=object_pose[:3],
            cube_lifted=lifted,
            hard_range_violation_count=safety["hard_range_violation_count"],
            command_slew_limited_count=safety["command_slew_limited_count"],
            contact_forces_n=forces,
            target_position_m=target_spec["position_m"],
            cube_in_target=cube_in_target,
            gripper_released=gripper_released,
            cube_stable=cube_stable,
        )
        rows.append(
            {
                "policy_step": policy_step,
                "state_calibrated": state.tolist(),
                "raw_action_calibrated": raw_action.tolist(),
                "applied_action_calibrated": applied.tolist(),
                "policy_execution": execution,
                "action_safety": safety,
                "object_pose_xyzw": object_pose.tolist(),
                "object_velocity_mps": object_velocity.tolist(),
                "contact_forces_n": list(forces),
                "triggered": trigger is not None,
            }
        )
        if trigger is not None:
            snapshot = {
                "policy_step": policy_step,
                "joint_positions_rad": _numpy(robot.data.joint_pos[0]).tolist(),
                "joint_velocities_rad_s": _numpy(robot.data.joint_vel[0]).tolist(),
                "joint_position_target_rad": _numpy(
                    robot.data.joint_pos_target[0]
                ).tolist(),
                "object_pose_xyzw": object_pose.tolist(),
                "object_linear_velocity_mps": object_velocity.tolist(),
                "object_angular_velocity_rad_s": _numpy(
                    active_object.data.root_ang_vel_w[0]
                ).tolist(),
                "contact_forces_n": list(forces),
                "applied_policy_action_calibrated": applied.tolist(),
            }
            break
    (root / "pre-handoff-actions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    if trigger is None or snapshot is None:
        raise RuntimeError(
            "recovery handoff was not admitted before the bounded stage deadline"
        )
    trigger["oracle_entry_phase"] = recovery_oracle_entry_phase(trigger)
    snapshot["trigger"] = copy.deepcopy(trigger)
    demonstration = recovery_demonstration(
        oracle_profile_id=oracle_profile_id,
        source_policy=policy_provenance,
        trigger_id=trigger_profile["trigger_id"],
        failure_class=trigger["failure_class"],
        control_step=int(trigger["policy_step"]),
        handoff_stage=trigger["handoff_stage"],
        trigger_reason=trigger["trigger_reason"],
        trigger_evidence=trigger,
        source_rollout_id=binding["source_rollout_id"],
        source_scene_id=binding["source_scene_id"],
        state_snapshot=snapshot,
        recovery_strategy_id=(
            trigger_profile.get("strategy_id")
            or runtime["oracle_handoff_profile"].get("strategy_id")
            or "regrasp_from_live_state_v1"
        ),
    )
    _write_json(
        root / "handoff.json",
        {
            "runtime_id": runtime["runtime_id"],
            "binding": binding,
            "trigger": trigger,
            "state_snapshot": snapshot,
            "demonstration": demonstration,
        },
    )
    return demonstration, snapshot


def _ik_action(
    robot,
    ee_frame,
    target,
    commanded,
    body_index,
    device,
    posture_target=None,
    orientation_target=None,
    orientation_local_axis=(0.0, 0.0, 1.0),
    orientation_weight=0.03,
    nullspace_gain=0.20,
    max_joint_step=0.02,
    lock_wrist=False,
    control_point_position=None,
    control_point_offset_world=None,
    position_weights=None,
):
    # Use the articulation's body-link pose and Jacobian from the same frame;
    # FrameTransformer target offsets can lag one simulation tick on reset.
    ee = _numpy(robot.data.body_link_pose_w.torch[0, body_index, :3])
    jacobians = robot.data.body_link_jacobian_w.torch
    # The fixed-base root body has no Jacobian row. Body-pose indices after
    # the root are therefore shifted by one in Isaac Lab's articulation view.
    jacobi_body_index = body_index - 1
    if not getattr(_ik_action, "_jac_printed", False) and float(np.linalg.norm(np.asarray(target) - ee)) > 0.02:
        print(
            f"SO101_JAC_DEBUG body_index={body_index} shape={tuple(jacobians.shape)} "
            f"max={float(torch.abs(jacobians).max().item())} "
            f"body_norms={torch.linalg.vector_norm(jacobians[0, :, :3, :], dim=(1, 2)).detach().cpu().tolist()}",
            flush=True,
        )
        _ik_action._jac_printed = True
    spatial_jacobian = _numpy(jacobians[0, jacobi_body_index, :6, :5])
    if control_point_position is None:
        controlled_position = ee
        position_jacobian = spatial_jacobian[:3]
    else:
        if control_point_offset_world is None:
            raise ValueError(
                "control_point_offset_world is required with control_point_position"
            )
        controlled_position = np.asarray(
            control_point_position, dtype=np.float32
        )
        point_offset = np.asarray(
            control_point_offset_world, dtype=np.float32
        )
        skew_offset = np.asarray(
            (
                (0.0, -point_offset[2], point_offset[1]),
                (point_offset[2], 0.0, -point_offset[0]),
                (-point_offset[1], point_offset[0], 0.0),
            ),
            dtype=np.float32,
        )
        # For a rigid point p = x + r attached to the end effector,
        # p_dot = v + omega x r = (Jv - skew(r) Jw) q_dot.
        position_jacobian = (
            spatial_jacobian[:3]
            - skew_offset @ spatial_jacobian[3:]
        )
    position_error = np.asarray(target) - controlled_position
    measured = _numpy(robot.data.joint_pos[0]).astype(np.float32)
    posture_error = np.zeros(5, dtype=np.float32)
    if posture_target is None:
        posture_target = SO101_GRASP_POSTURE
    posture_error[3] = float(posture_target[0]) - measured[3]
    posture_error[4] = float(posture_target[1]) - measured[4]
    if orientation_target is None:
        # Keep Cartesian position as the primary task and the grasp posture as
        # a compliant null-space preference. Making both wrist joints primary
        # produced high unilateral contact loads and removed the compliance
        # present in the physically successful run-171 trajectory. The safe
        # overhead route now prevents the direct-path collision that prompted
        # the earlier five-dimensional experiment.
        jacobian = position_jacobian
        task_error = position_error
        nullspace_error = posture_error
    else:
        orientation_weight = float(orientation_weight)
        if not np.isfinite(orientation_weight) or orientation_weight <= 0.0:
            raise ValueError("orientation_weight must be positive and finite")
        # SO-101 has five arm joints: position plus full orientation would be
        # a six-dimensional, over-constrained task. Align one selected local
        # axis (three rows, rank two) and leave rotation around that axis to
        # the wrist-roll posture regularizer.
        orientation_error = quaternion_direction_error(
            orientation_target,
            _numpy(robot.data.body_link_pose_w.torch[0, body_index, 3:7]),
            local_axis=orientation_local_axis,
        )
        jacobian = spatial_jacobian.copy()
        jacobian[3:] *= orientation_weight
        task_error = np.concatenate(
            (position_error, orientation_weight * orientation_error)
        )
        nullspace_error = posture_error
    if position_weights is not None:
        if orientation_target is not None:
            raise ValueError("position_weights requires position-only IK")
        weights = np.asarray(position_weights, dtype=np.float32)
        if weights.shape != (3,) or np.any(weights <= 0.0):
            raise ValueError("position_weights must contain three positive values")
        jacobian = jacobian * weights[:, None]
        task_error = task_error * weights
    # Once physical contact is established the wrist posture is part of the
    # grasp constraint.  Do not let DLS "spend" wrist motion to solve the
    # Cartesian task and then overwrite those same joints below: that makes
    # the modelled update differ from the command and can stall transport.
    controlled_joint_count = 3 if lock_wrist else 5
    controlled_jacobian = jacobian[:, :controlled_joint_count]
    controlled_nullspace_error = (
        None
        if lock_wrist or nullspace_error is None
        else nullspace_error[:controlled_joint_count]
    )
    delta = damped_least_squares(
        controlled_jacobian,
        task_error,
        damping=0.06,
        nullspace_error=controlled_nullspace_error,
        nullspace_gain=nullspace_gain,
    )
    action = np.asarray(commanded, dtype=np.float32).copy()
    # Integrate resolved-rate IK in command space.  Re-basing on measured
    # joints every frame erases accumulated position error and caps the drive
    # torque below what is needed to move the gravity-loaded arm.
    # Preserve the DLS direction when limiting resolved-rate motion. Clipping
    # every joint independently turns large solutions into an equal-magnitude
    # sign vector and can drive the arm into a completely different branch.
    joint_step_limit = float(max_joint_step)
    if joint_step_limit <= 0.0:
        raise ValueError("max_joint_step must be positive")
    max_delta = float(np.max(np.abs(delta)))
    if max_delta > joint_step_limit:
        delta = delta * (joint_step_limit / max_delta)
    action[:controlled_joint_count] = (
        action[:controlled_joint_count] + delta
    )
    action[:5] = np.clip(action[:5], measured[:5] - 0.30, measured[:5] + 0.30)
    # Keep the generated target inside the pinned USD joint limits.  The
    # position action manager does not clamp targets when offset-free control
    # is enabled, and some PhysX articulations report wider soft limits.
    action[:5] = np.clip(action[:5], SO101_JOINT_LIMITS[:, 0], SO101_JOINT_LIMITS[:, 1])
    if not getattr(_ik_action, "_printed", False) and float(np.linalg.norm(np.asarray(target) - ee)) > 0.02:
        print(f"SO101_IK_DEBUG ee={ee.tolist()} delta={delta.tolist()} commanded={np.asarray(commanded).tolist()} action={action.tolist()}", flush=True)
        _ik_action._printed = True
    return torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)


def _write_frame(root: Path, frame: int, state, action, front, wrist=None):
    front_path = root / "rgb" / f"front_{frame:06d}.png"
    front_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(front, mode="RGB").save(front_path)
    row = {
        "frame": frame,
        "timestamp_seconds": frame / 30.0,
        "phase": "",
        "rgb_path": str(front_path.relative_to(root)),
        "joint_names": list(SIM_JOINT_NAMES),
        "controlled_joint_names": list(SIM_JOINT_NAMES),
        "joint_positions": [float(value) for value in state],
        "joint_velocities": [],
        "action_joint_positions": [float(value) for value in action],
        "contact_forces_newtons": {"left_finger": 0.0, "right_finger": 0.0},
        "object_pose_estimate": {},
    }
    if wrist is not None:
        wrist_path = root / "rgb" / f"wrist_{frame:06d}.png"
        Image.fromarray(wrist, mode="RGB").save(wrist_path)
        row["wrist_rgb_path"] = str(wrist_path.relative_to(root))
    return row


def _variant_name(trial):
    edge = trial["resolved"]["dimensions_m"][0]
    color = trial["resolved"]["rgba"]
    size = "small" if edge < 0.035 else "large"
    color_name = "red" if color[0] > color[2] else "blue"
    return f"cube_{size}_{color_name}"


def run_jacobian_diagnostic(env) -> None:
    """Compare reported translational Jacobian columns with driven joint motion."""
    robot = env.scene["robot"]
    body_index = _body_index(robot)
    device = env.device
    records = []
    for joint_index, joint_name in enumerate(SIM_JOINT_NAMES[:5]):
        env.reset()
        home = torch.tensor(SO101_HOME_JOINTS[None, :], dtype=torch.float32, device=device)
        for _ in range(90):
            env.step(home)
        baseline = _numpy(robot.data.body_link_pose_w.torch[0, body_index, :3]).copy()
        baseline_joint = float(robot.data.joint_pos[0, joint_index].item())
        jacobians = _numpy(robot.data.body_link_jacobian_w.torch[0, :, :3, joint_index]).copy()
        perturbed = home.clone()
        perturbed[0, joint_index] += 0.04
        for _ in range(30):
            env.step(perturbed)
        observed = _numpy(robot.data.body_link_pose_w.torch[0, body_index, :3]) - baseline
        joint_delta = float(robot.data.joint_pos[0, joint_index].item()) - baseline_joint
        empirical = observed / joint_delta if abs(joint_delta) > 1e-5 else np.zeros(3)
        records.append(
            {
                "joint": joint_name,
                "joint_delta": joint_delta,
                "analytic_by_body": jacobians.tolist(),
                "empirical": empirical.tolist(),
                "observed": observed.tolist(),
                "cosine_by_body": [
                    float(np.dot(row, empirical) / (np.linalg.norm(row) * np.linalg.norm(empirical)))
                    if np.linalg.norm(row) > 1e-8 and np.linalg.norm(empirical) > 1e-8
                    else 0.0
                    for row in jacobians
                ],
            }
        )
    print("SO101_JACOBIAN_DIAGNOSTIC " + json.dumps(records, sort_keys=True), flush=True)


def run_grasp_posture_diagnostic(env, output_root: Path) -> None:
    """Screen top-down aperture postures without touching the table or cube."""
    scene = env.scene
    robot = scene["robot"]
    body_index = _body_index(robot)
    device = env.device
    target = np.asarray((0.165, -0.11, 0.16), dtype=np.float32)
    object_position = np.asarray((0.1589218, -0.1071630, 0.052), dtype=np.float32)
    object_edge_m = 0.040
    active_name = "cube_large_blue"
    cube_names = (
        "cube_small_red",
        "cube_small_blue",
        "cube_large_red",
        "cube_large_blue",
    )
    destination = output_root / "grasp_posture_diagnostic"
    destination.mkdir(parents=True, exist_ok=True)
    # Historical run171/run173 used pitch~=1.3 and put the real aperture below
    # the table when aligned to the cube.  The existing front-camera sweep
    # identifies 0.5 and 0.0 rad as the two plausible table-safe branches;
    # sweep roll only within a bounded neighborhood around each branch.
    jaw_targets = tuple(float(value) for value in args_cli.diagnostic_jaw_targets)
    if not jaw_targets or any(not -0.1746 <= value <= 1.7453 for value in jaw_targets):
        raise ValueError(
            "diagnostic jaw targets must be within the pinned [-0.1746, 1.7453] rad range"
        )
    candidates = tuple(
        (pitch, roll, jaw_target)
        for pitch in (0.50, 0.0)
        for roll in (-0.50, 0.0, 0.50)
        for jaw_target in jaw_targets
    )
    results = []
    diagnostic_seed = 0
    np.random.seed(diagnostic_seed)
    torch.manual_seed(diagnostic_seed)
    for index, (pitch, roll, jaw_target) in enumerate(candidates):
        env.farpoint_active_cube = active_name
        env.reset(seed=diagnostic_seed)
        for inactive_index, name in enumerate(cube_names):
            _move_object(
                scene[name],
                object_position
                if name == active_name
                else (-10.0 - inactive_index, 0.0, 0.1),
                device,
            )
        _aim_front_camera(scene, device)
        command = SO101_HOME_JOINTS.copy()
        command[4] = 0.0
        action = torch.tensor(command[None, :], dtype=torch.float32, device=device)
        env.step(action)
        print(
            f"SO101_GRASP_POSTURE_START pitch_target={pitch} roll_target={roll} "
            f"jaw_target={jaw_target}",
            flush=True,
        )
        for _ in range(180):
            action = _ik_action(
                robot,
                scene["ee_frame"],
                target,
                command,
                body_index,
                device,
                posture_target=(pitch, roll),
            )
            # Match the 40 mm episode pre-shape while remaining safely above
            # the workspace.  This evaluates the actual rotary-jaw geometry,
            # not the fully open link origins used by the old diagnostic.
            action[0, 5] = jaw_target
            command = _numpy(action[0]).astype(np.float32).copy()
            env.step(action)
        front = _image(scene["front_camera"], device)
        gripper_pose = _numpy(
            robot.data.body_link_pose_w.torch[0, body_index]
        ).copy()
        jaw_pose = _numpy(
            robot.data.body_link_pose_w.torch[
                0, robot.body_names.index("jaw")
            ]
        ).copy()
        geometry = posture_geometry_diagnostics(
            gripper_pose,
            jaw_pose,
            object_position,
            object_half_height_m=object_edge_m * 0.5,
        )
        label = (
            f"posture_{index}_pitch_{pitch:+.2f}_roll_{roll:+.3f}"
            f"_jaw_{jaw_target:+.3f}"
        )
        front_path = destination / f"{label}_front.png"
        Image.fromarray(front, mode="RGB").save(front_path)
        if args_cli.enable_wrist_camera:
            wrist = _image(scene["wrist_camera"], device)
            Image.fromarray(wrist, mode="RGB").save(destination / f"{label}_wrist.png")
        position_error = target - gripper_pose[:3]
        record = {
            "candidate_index": index,
            "pitch_target_rad": pitch,
            "roll_target_rad": roll,
            "jaw_target_rad": jaw_target,
            "joints_rad": _numpy(robot.data.joint_pos[0]).tolist(),
            "gripper_pose_xyzw": gripper_pose.tolist(),
            "jaw_pose_xyzw": jaw_pose.tolist(),
            "position_error_m": position_error.tolist(),
            "kinematically_reached": bool(np.linalg.norm(position_error) < 0.010),
            "front_image": front_path.name,
            "geometry": geometry,
        }
        results.append(record)
        print(
            "SO101_GRASP_POSTURE "
            + json.dumps(record, sort_keys=True),
            flush=True,
        )
    report = {
        "schema_version": "farpoint.so101_aperture_posture_diagnostic.v1",
        "workshop_commit": SO101_WORKSHOP_COMMIT,
        "asset_sha256": SO101_WORKSHOP_ASSET_SHA256,
        "runtime_quaternion_order": SO101_RUNTIME_QUATERNION_ORDER,
        "environment_seed": diagnostic_seed,
        "camera_features": ["front"],
        "wrist_camera_enabled": bool(args_cli.enable_wrist_camera),
        "object": {
            "shape": "cube",
            "edge_m": object_edge_m,
            "center_world_m": object_position.tolist(),
        },
        "aperture_reference_in_gripper_m": (
            SO101_APERTURE_REFERENCE_IN_GRIPPER_M.tolist()
        ),
        "candidates": results,
    }
    (destination / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_calibrated_grasp_diagnostic(env, output_root: Path) -> None:
    """Run the production 40 mm slow-close and proof-lift calibration."""
    scene = env.scene
    robot = scene["robot"]
    body_index = _body_index(robot)
    jaw_body_index = robot.body_names.index("jaw")
    device = env.device
    active_name = "cube_large_blue"
    inactive = [
        name
        for name in (
            "cube_small_red",
            "cube_small_blue",
            "cube_large_red",
            "cube_large_blue",
        )
        if name != active_name
    ]
    object_position = np.asarray((0.1589218, -0.1071630, 0.052), dtype=np.float32)
    object_orientation = np.asarray(
        (0.0, 0.0, np.sin(np.pi / 8.0), np.cos(np.pi / 8.0)),
        dtype=np.float32,
    )
    object_edge_m = 0.040
    posture_target = np.asarray((0.50, 0.50), dtype=np.float32)
    expected_safe_orientation = np.asarray(
        (-0.6417147, 0.1408973, 0.0826372, -0.7493473), dtype=np.float32
    )
    jaw_preshape = so101_approach_jaw_target(object_edge_m)
    aperture_reference = so101_capture_aperture_reference(jaw_preshape)
    closed_jaw = float(np.deg2rad(-10.0))
    bilateral_threshold_n = 0.10
    seed = 0
    destination = output_root / "calibrated_grasp_diagnostic"
    destination.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    env.farpoint_active_cube = active_name
    env.reset(seed=seed)
    for index, name in enumerate(inactive):
        _move_object(scene[name], (-10.0 - index, 0.0, 0.1), device)
    _move_object(
        scene[active_name], object_position, device, object_orientation
    )
    _aim_front_camera(scene, device)
    command = SO101_HOME_JOINTS.copy()
    action = torch.tensor(command[None, :], dtype=torch.float32, device=device)
    env.step(action)

    # Reach the calibrated orientation well above the object before descending
    # vertically.  The final position is recomputed from the live cube center.
    safe_target = np.asarray((0.165, -0.11, 0.16), dtype=np.float32)
    for _ in range(240):
        action = _ik_action(
            robot,
            scene["ee_frame"],
            safe_target,
            command,
            body_index,
            device,
            posture_target=posture_target,
        )
        action[0, 5] = jaw_preshape
        command = _numpy(action[0]).astype(np.float32).copy()
        env.step(action)

    safe_pose = _numpy(robot.data.body_link_pose_w.torch[0, body_index]).copy()
    settled_cube = _numpy(scene[active_name].data.root_pos_w[0, :3]).copy()
    aligned_target = gripper_target_for_object_local_offset(
        settled_cube, safe_pose[3:7], aperture_reference
    )
    feed_distance_m = 0.070
    # Route above a distal, tip-first pregrasp.  The fixed finger extends along
    # gripper-local -Z, so +Z translation places the cube beyond both tips.
    for _ in range(180):
        live_pose = _numpy(
            robot.data.body_link_pose_w.torch[0, body_index]
        ).copy()
        aligned_target = gripper_target_for_object_local_offset(
            settled_cube,
            live_pose[3:7],
            aperture_reference,
        )
        local_z_world = quaternion_rotation_matrix_xyzw(live_pose[3:7])[:, 2]
        above_target = (
            aligned_target
            + feed_distance_m * local_z_world
            + np.asarray((0.0, 0.0, 0.070), dtype=np.float32)
        )
        action = _ik_action(
            robot,
            scene["ee_frame"],
            above_target,
            command,
            body_index,
            device,
            posture_target=posture_target,
        )
        action[0, 5] = jaw_preshape
        command = _numpy(action[0]).astype(np.float32).copy()
        env.step(action)

    # Descend to the distal pregrasp while the cube remains outside the tips.
    for _ in range(180):
        live_pose = _numpy(
            robot.data.body_link_pose_w.torch[0, body_index]
        ).copy()
        aligned_target = gripper_target_for_object_local_offset(
            settled_cube,
            live_pose[3:7],
            aperture_reference,
        )
        local_z_world = quaternion_rotation_matrix_xyzw(live_pose[3:7])[:, 2]
        pregrasp_target = aligned_target + feed_distance_m * local_z_world
        action = _ik_action(
            robot,
            scene["ee_frame"],
            pregrasp_target,
            command,
            body_index,
            device,
            posture_target=posture_target,
        )
        action[0, 5] = jaw_preshape
        command = _numpy(action[0]).astype(np.float32).copy()
        env.step(action)

    contacts = (scene["contact_jaw"], scene["contact_gripper"])
    unexpected_descent_contact = False
    first_contact_fraction = None
    descent_steps_completed = 0
    for step in range(280):
        fraction = (step + 1) / 280
        live_pose = _numpy(
            robot.data.body_link_pose_w.torch[0, body_index]
        ).copy()
        aligned_target = gripper_target_for_object_local_offset(
            settled_cube,
            live_pose[3:7],
            aperture_reference,
        )
        local_z_world = quaternion_rotation_matrix_xyzw(live_pose[3:7])[:, 2]
        target = aligned_target + (
            feed_distance_m * (1.0 - fraction) * local_z_world
        )
        action = _ik_action(
            robot,
            scene["ee_frame"],
            target,
            command,
            body_index,
            device,
            posture_target=posture_target,
            max_joint_step=0.01,
        )
        action[0, 5] = jaw_preshape
        command = _numpy(action[0]).astype(np.float32).copy()
        env.step(action)
        descent_steps_completed = step + 1
        if _contact(contacts):
            first_contact_fraction = fraction
            # Contact in the final 10% is the intended transition from
            # contact-free insertion to contact alignment.  Earlier contact
            # means a finger swept the cube before it entered the aperture.
            unexpected_descent_contact = fraction < 0.90
            break

    Image.fromarray(_image(scene["front_camera"], device), mode="RGB").save(
        destination / "aligned_open_front.png"
    )
    alignment_pose = _numpy(
        robot.data.body_link_pose_w.torch[0, body_index]
    ).copy()
    cube_before_close = _numpy(scene[active_name].data.root_pos_w[0]).copy()
    close_force_history = []
    bilateral_steps = 0
    overload = False
    contact_jaw_target = None
    if not unexpected_descent_contact:
        for _ in range(900):
            forces = _cube_contact_forces(contacts)
            close_force_history.append(forces)
            overload = max(forces) > 30.0
            if overload:
                break
            bilateral_steps = (
                bilateral_steps + 1
                if min(forces) >= bilateral_threshold_n
                else 0
            )
            if bilateral_steps >= 15:
                contact_jaw_target = float(command[5])
                break
            action = _ik_action(
                robot,
                scene["ee_frame"],
                alignment_pose[:3],
                command,
                body_index,
                device,
                posture_target=posture_target,
                max_joint_step=0.005,
            )
            command[5] = max(closed_jaw, float(command[5]) - 0.001)
            action[0, 5] = float(command[5])
            command = _numpy(action[0]).astype(np.float32).copy()
            env.step(action)

    Image.fromarray(_image(scene["front_camera"], device), mode="RGB").save(
        destination / "closed_front.png"
    )
    hold_bilateral_steps = 0
    if contact_jaw_target is not None:
        for _ in range(60):
            action = _ik_action(
                robot,
                scene["ee_frame"],
                alignment_pose[:3],
                command,
                body_index,
                device,
                posture_target=posture_target,
                max_joint_step=0.005,
            )
            action[0, 5] = contact_jaw_target
            command = _numpy(action[0]).astype(np.float32).copy()
            env.step(action)
            if _bilateral_contact(contacts, threshold_n=bilateral_threshold_n):
                hold_bilateral_steps += 1

    lift_start_cube = _numpy(scene[active_name].data.root_pos_w[0]).copy()
    lift_bilateral_steps = 0
    if hold_bilateral_steps >= 45:
        for step in range(180):
            lift_target = alignment_pose[:3] + np.asarray(
                (0.0, 0.0, 0.025 * (step + 1) / 180), dtype=np.float32
            )
            action = _ik_action(
                robot,
                scene["ee_frame"],
                lift_target,
                command,
                body_index,
                device,
                posture_target=posture_target,
                max_joint_step=0.005,
            )
            action[0, 5] = contact_jaw_target
            command = _numpy(action[0]).astype(np.float32).copy()
            env.step(action)
            if _bilateral_contact(contacts, threshold_n=bilateral_threshold_n):
                lift_bilateral_steps += 1

    final_cube = _numpy(scene[active_name].data.root_pos_w[0]).copy()
    Image.fromarray(_image(scene["front_camera"], device), mode="RGB").save(
        destination / "proof_lift_front.png"
    )
    lifted_m = float(final_cube[2] - lift_start_cube[2])
    success = bool(
        not unexpected_descent_contact
        and not overload
        and bilateral_steps >= 15
        and hold_bilateral_steps >= 45
        and lift_bilateral_steps >= 135
        and lifted_m >= 0.015
    )
    report = {
        "schema_version": "farpoint.so101_calibrated_grasp_diagnostic.v1",
        "success": success,
        "environment_seed": seed,
        "camera_features": ["front"],
        "wrist_camera_enabled": False,
        "asset_sha256": SO101_WORKSHOP_ASSET_SHA256,
        "object_edge_m": object_edge_m,
        "object_orientation_xyzw": object_orientation.tolist(),
        "aperture_center_local_m": aperture_reference.tolist(),
        "posture_target_rad": posture_target.tolist(),
        "expected_safe_orientation_xyzw": expected_safe_orientation.tolist(),
        "measured_safe_pose_xyzw": safe_pose.tolist(),
        "safe_orientation_error_norm": float(
            np.linalg.norm(safe_pose[3:7] - expected_safe_orientation)
        ),
        "jaw_preshape_rad": jaw_preshape,
        "bilateral_threshold_n": bilateral_threshold_n,
        "aligned_target_m": np.asarray(aligned_target).tolist(),
        "alignment_pose_xyzw": alignment_pose.tolist(),
        "jaw_pose_xyzw": _numpy(
            robot.data.body_link_pose_w.torch[0, jaw_body_index]
        ).tolist(),
        "descent_steps_completed": descent_steps_completed,
        "first_contact_fraction": first_contact_fraction,
        "unexpected_descent_contact": unexpected_descent_contact,
        "overload": overload,
        "bilateral_close_steps": bilateral_steps,
        "bilateral_hold_steps": hold_bilateral_steps,
        "bilateral_lift_steps": lift_bilateral_steps,
        "contact_jaw_target_rad": contact_jaw_target,
        "maximum_close_forces_n": (
            np.max(np.asarray(close_force_history), axis=0).tolist()
            if close_force_history
            else [0.0, 0.0]
        ),
        "cube_before_close_m": cube_before_close.tolist(),
        "cube_lift_start_m": lift_start_cube.tolist(),
        "cube_final_m": final_cube.tolist(),
        "cube_lift_m": lifted_m,
        "final_contact": _contact_debug(contacts),
    }
    _write_json(destination / "results.json", report)
    print("SO101_CALIBRATED_GRASP " + json.dumps(report, sort_keys=True), flush=True)


def run_grasp_offset_diagnostic(env, output_root: Path) -> None:
    """Render an open gripper at candidate tool offsets around one cube."""
    scene = env.scene
    robot = scene["robot"]
    body_index = _body_index(robot)
    device = env.device
    object_position = np.asarray((0.1589218, -0.1071630, 0.047), dtype=np.float32)
    destination = output_root / "grasp_offset_diagnostic"
    destination.mkdir(parents=True, exist_ok=True)
    active_name = "cube_small_red"
    env.farpoint_active_cube = active_name
    contact = (scene["contact_jaw"], scene["contact_gripper"])
    for index, x_offset in enumerate((-0.060, -0.050, -0.040, -0.030, -0.020)):
        env.reset()
        for name in (
            "cube_small_red",
            "cube_small_blue",
            "cube_large_red",
            "cube_large_blue",
        ):
            _move_object(
                scene[name],
                object_position if name == active_name else (-10.0, 0.0, 0.1),
                device,
            )
        _aim_front_camera(scene, device)
        command = SO101_HOME_JOINTS.copy()
        command[4] = 0.0
        action = torch.tensor(command[None, :], dtype=torch.float32, device=device)
        env.step(action)
        target = object_position + np.asarray((x_offset, -0.025, 0.030), dtype=np.float32)
        for _ in range(180):
            action = _ik_action(
                robot,
                scene["ee_frame"],
                target,
                command,
                body_index,
                device,
                posture_target=SO101_GRASP_POSTURE,
            )
            action[0, 5] = float(SO101_HOME_JOINTS[5])
            command = _numpy(action[0]).astype(np.float32).copy()
            env.step(action)
        label = f"x_{index}_{x_offset:+.3f}"
        Image.fromarray(_image(scene["front_camera"], device), mode="RGB").save(
            destination / f"{label}_front.png"
        )
        if args_cli.enable_wrist_camera:
            Image.fromarray(_image(scene["wrist_camera"], device), mode="RGB").save(
                destination / f"{label}_wrist.png"
            )
        print(
            "SO101_GRASP_OFFSET "
            + json.dumps(
                {
                    "x_offset_m": x_offset,
                    "target_m": target.tolist(),
                    "gripper_pose_xyzw": _numpy(
                        robot.data.body_link_pose_w.torch[0, body_index]
                    ).tolist(),
                    "cube_position_m": _numpy(scene[active_name].data.root_pos_w[0]).tolist(),
                    "contact": _contact_debug(contact),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def run_grasp_grid_diagnostic(env, output_root: Path) -> None:
    """Search contact-only Cartesian offsets without restarting Isaac Sim."""
    scene = env.scene
    robot = scene["robot"]
    body_index = _body_index(robot)
    device = env.device
    object_position = np.asarray((0.1589218, -0.1071630, 0.047), dtype=np.float32)
    active_name = "cube_small_red"
    inactive = [
        name
        for name in (
            "cube_small_red",
            "cube_small_blue",
            "cube_large_red",
            "cube_large_blue",
        )
        if name != active_name
    ]
    contact = (scene["contact_jaw"], scene["contact_gripper"])
    destination = output_root / "grasp_grid_diagnostic"
    destination.mkdir(parents=True, exist_ok=True)
    candidates = [
        (x_offset, -0.018, z_offset)
        for z_offset in (0.035,)
        for x_offset in (-0.024, -0.016, -0.008)
    ]
    results = []
    for candidate_index, offset in enumerate(candidates):
        env.farpoint_active_cube = active_name
        env.reset()
        for index, name in enumerate(inactive):
            _move_object(scene[name], (-10.0 - index, 0.0, 0.1), device)
        _move_object(scene[active_name], object_position, device)
        command = SO101_HOME_JOINTS.copy()
        command[4] = 0.0
        action = torch.tensor(command[None, :], dtype=torch.float32, device=device)
        env.step(action)

        outside_y_offset = 0.080
        pregrasp = object_position + np.asarray(
            (offset[0], outside_y_offset, 0.150)
        )
        for _ in range(180):
            action = _ik_action(
                robot,
                scene["ee_frame"],
                pregrasp,
                command,
                body_index,
                device,
                posture_target=SO101_GRASP_POSTURE,
            )
            action[0, 5] = 0.80
            command = _numpy(action[0]).astype(np.float32).copy()
            env.step(action)

        descent_contact = False
        descent_steps = 160
        for step in range(descent_steps):
            fraction = (step + 1) / descent_steps
            target = pregrasp.copy()
            target[2] = object_position[2] + (
                (1.0 - fraction) * 0.150 + fraction * offset[2]
            )
            action = _ik_action(
                robot,
                scene["ee_frame"],
                target,
                command,
                body_index,
                device,
                posture_target=SO101_GRASP_POSTURE,
            )
            action[0, 5] = 0.80
            command = _numpy(action[0]).astype(np.float32).copy()
            env.step(action)
            if _contact(contact):
                descent_contact = True
                break

        # At grasp height, insert along -Y with the finger tips trailing the
        # gripper link. Approaching from -Y makes the fixed finger push the
        # cube ahead of the aperture instead of letting it enter the throat.
        if not descent_contact:
            approach_steps = 120
            for step in range(approach_steps):
                fraction = (step + 1) / approach_steps
                target = object_position + np.asarray(
                    (
                        offset[0],
                        (1.0 - fraction) * outside_y_offset
                        + fraction * offset[1],
                        offset[2],
                    )
                )
                action = _ik_action(
                    robot,
                    scene["ee_frame"],
                    target,
                    command,
                    body_index,
                    device,
                    posture_target=SO101_GRASP_POSTURE,
                )
                action[0, 5] = 0.80
                command = _numpy(action[0]).astype(np.float32).copy()
                env.step(action)
                if _contact(contact):
                    descent_contact = True
                    break

        hold_pose = _numpy(robot.data.body_link_pose_w.torch[0, body_index, :3]).copy()
        hold_orientation = _numpy(
            robot.data.body_link_pose_w.torch[0, body_index, 3:7]
        ).copy()
        first_contact_debug = _contact_debug(contact)
        first_contact_cube = _numpy(scene[active_name].data.root_pos_w[0]).copy()
        bilateral_steps = 0
        best_bilateral_forces = (0.0, 0.0)
        for _ in range(60):
            action = _ik_action(
                robot,
                scene["ee_frame"],
                hold_pose,
                command,
                body_index,
                device,
                posture_target=SO101_GRASP_POSTURE,
                orientation_target=hold_orientation,
            )
            action[0, 5] = max(
                float(np.deg2rad(-10.0)),
                float(robot.data.joint_pos[0, 5].item()) - 0.04,
            )
            command = _numpy(action[0]).astype(np.float32).copy()
            env.step(action)
            if _bilateral_contact(contact):
                bilateral_steps += 1
                best_bilateral_forces = _cube_contact_forces(contact)
            else:
                bilateral_steps = 0
            if bilateral_steps >= 5:
                break

        closed_pose = _numpy(robot.data.body_link_pose_w.torch[0, body_index]).copy()
        closed_cube = _numpy(scene[active_name].data.root_pos_w[0]).copy()
        closed_contact_debug = _contact_debug(contact)
        lift_start_z = float(closed_cube[2])
        retained_contact_steps = 0
        if bilateral_steps >= 5:
            held_jaw = float(robot.data.joint_pos[0, 5].item())
            for step in range(80):
                lift_target = hold_pose + np.asarray(
                    (0.0, 0.0, 0.020 * (step + 1) / 80), dtype=np.float32
                )
                action = _ik_action(
                    robot,
                    scene["ee_frame"],
                    lift_target,
                    command,
                    body_index,
                    device,
                    posture_target=SO101_GRASP_POSTURE,
                    orientation_target=hold_orientation,
                )
                action[0, 5] = held_jaw
                command = _numpy(action[0]).astype(np.float32).copy()
                env.step(action)
                if _bilateral_contact(contact):
                    retained_contact_steps += 1

        final_cube = _numpy(scene[active_name].data.root_pos_w[0]).copy()
        record = {
            "candidate_index": candidate_index,
            "offset_m": list(offset),
            "descent_contact": descent_contact,
            "first_contact": first_contact_debug,
            "first_contact_cube_m": first_contact_cube.tolist(),
            "bilateral_steps": bilateral_steps,
            "bilateral_forces_n": list(best_bilateral_forces),
            "retained_contact_steps": retained_contact_steps,
            "cube_lift_m": float(final_cube[2] - lift_start_z),
            "closed_gripper_pose_xyzw": closed_pose.tolist(),
            "closed_cube_m": closed_cube.tolist(),
            "final_cube_m": final_cube.tolist(),
            "closed_contact": closed_contact_debug,
        }
        results.append(record)
        Image.fromarray(_image(scene["front_camera"], device), mode="RGB").save(
            destination / f"candidate_{candidate_index:02d}_front.png"
        )
        print("SO101_GRASP_GRID " + json.dumps(record, sort_keys=True), flush=True)
    _write_json(destination / "results.json", {"candidates": results})


def run_grasp_path_diagnostic(env, output_root: Path) -> None:
    """Screen two-stage 40 mm insertion paths before contact-only closing."""
    scene = env.scene
    robot = scene["robot"]
    body_index = _body_index(robot)
    device = env.device
    active_name = "cube_large_blue"
    inactive = [
        name
        for name in (
            "cube_small_red",
            "cube_small_blue",
            "cube_large_red",
            "cube_large_blue",
        )
        if name != active_name
    ]
    object_position = np.asarray((0.1589218, -0.1071630, 0.052), dtype=np.float32)
    object_orientation = np.asarray(
        (0.0, 0.0, np.sin(np.pi / 8.0), np.cos(np.pi / 8.0)),
        dtype=np.float32,
    )
    jaw_preshape = so101_approach_jaw_target(0.040)
    aperture_reference = so101_capture_aperture_reference(jaw_preshape)
    outside_distance_m = 0.080
    vertical_clearance_m = 0.080
    contact_threshold_n = 0.10
    # Exact jaw=1.7 mesh contact-pair direction in the gripper frame.  The
    # first path screen showed that the horizontal channel perpendicular to
    # this axis is the only route that gets close to the 40 mm cube.  This
    # The prior screen showed wrist-roll regularization alone does not level
    # the closing axis. Keep its best roll branch, directly level the measured
    # end-effector axis above the cube, and retain a small height screen.
    candidates = tuple(
        (0.25, aperture_height_offset_m)
        for aperture_height_offset_m in (-0.010, 0.0, 0.010)
    )
    destination = output_root / "grasp_path_diagnostic"
    destination.mkdir(parents=True, exist_ok=True)
    results = []

    def live_aligned_target(aperture_height_offset_m):
        pose = _numpy(robot.data.body_link_pose_w.torch[0, body_index]).copy()
        cube = _numpy(scene[active_name].data.root_pos_w[0, :3]).copy()
        return gripper_target_for_object_local_offset(
            cube
            + np.asarray(
                (0.0, 0.0, aperture_height_offset_m), dtype=np.float32
            ),
            pose[3:7],
            aperture_reference,
        )

    for candidate_index, (roll, aperture_height_offset_m) in enumerate(candidates):
        posture_target = np.asarray((0.50, roll), dtype=np.float32)
        label = f"roll_{roll:+.2f}_height_{aperture_height_offset_m:+.3f}"
        env.farpoint_active_cube = active_name
        env.reset(seed=0)
        for index, name in enumerate(inactive):
            _move_object(scene[name], (-10.0 - index, 0.0, 0.1), device)
        _move_object(
            scene[active_name], object_position, device, object_orientation
        )
        _aim_front_camera(scene, device)
        command = SO101_HOME_JOINTS.copy()
        action = torch.tensor(command[None, :], dtype=torch.float32, device=device)
        env.step(action)
        cube_start = _numpy(scene[active_name].data.root_pos_w[0, :3]).copy()
        first_contact_segment = None
        first_contact_fraction = None
        first_contact_forces = (0.0, 0.0)
        contacts = (scene["contact_jaw"], scene["contact_gripper"])
        orientation_target = None

        def advance_segment(segment, steps, target_for_fraction):
            nonlocal command
            nonlocal first_contact_segment
            nonlocal first_contact_fraction
            nonlocal first_contact_forces
            segment_start_cube = _numpy(
                scene[active_name].data.root_pos_w[0, :3]
            ).copy()
            for step in range(steps):
                fraction = (step + 1) / steps
                target = target_for_fraction(fraction)
                action = _ik_action(
                    robot,
                    scene["ee_frame"],
                    target,
                    command,
                    body_index,
                    device,
                    posture_target=posture_target,
                    orientation_target=orientation_target,
                    orientation_local_axis=SO101_CAPTURE_CLOSING_AXIS_LOCAL,
                    orientation_weight=0.12,
                    max_joint_step=0.01,
                )
                action[0, 5] = jaw_preshape
                command = _numpy(action[0]).astype(np.float32).copy()
                env.step(action)
                forces = _cube_contact_forces(contacts)
                if max(forces) >= contact_threshold_n:
                    first_contact_segment = segment
                    first_contact_fraction = fraction
                    first_contact_forces = forces
                    return False
                cube_now = _numpy(
                    scene[active_name].data.root_pos_w[0, :3]
                ).copy()
                if float(np.linalg.norm(cube_now - segment_start_cube)) >= 0.003:
                    first_contact_segment = segment
                    first_contact_fraction = fraction
                    first_contact_forces = forces
                    return False
            return True

        safe_target = np.asarray((0.165, -0.11, 0.18), dtype=np.float32)
        contact_free = advance_segment(
            "safe_posture", 180, lambda _fraction: safe_target
        )
        unlevelled_safe_pose = _numpy(
            robot.data.body_link_pose_w.torch[0, body_index]
        ).copy()
        orientation_target = so101_level_capture_orientation_xyzw(
            unlevelled_safe_pose[3:7]
        )
        if contact_free:
            contact_free = advance_segment(
                "level_capture_axis", 240, lambda _fraction: safe_target
            )
        safe_pose = _numpy(
            robot.data.body_link_pose_w.torch[0, body_index]
        ).copy()
        closing_axis_world = (
            quaternion_rotation_matrix_xyzw(safe_pose[3:7])
            @ SO101_CAPTURE_CLOSING_AXIS_LOCAL
        )
        # Use the side that v9 showed to be reachable.  The opposite side
        # drove the arm across the cube during the outside descent.
        direction = so101_capture_channel_direction_world(
            orientation_target
        )
        if contact_free:
            contact_free = advance_segment(
                "above_outside",
                180,
                lambda _fraction: (
                    live_aligned_target(aperture_height_offset_m)
                    + outside_distance_m * direction
                    + np.asarray((0.0, 0.0, vertical_clearance_m), dtype=np.float32)
                ),
            )
        if contact_free:
            contact_free = advance_segment(
                "outside_descent",
                180,
                lambda fraction: (
                    live_aligned_target(aperture_height_offset_m)
                    + outside_distance_m * direction
                    + np.asarray(
                        (0.0, 0.0, vertical_clearance_m * (1.0 - fraction)),
                        dtype=np.float32,
                    )
                ),
            )
        if contact_free:
            contact_free = advance_segment(
                "lateral_insert",
                240,
                lambda fraction: (
                    live_aligned_target(aperture_height_offset_m)
                    + outside_distance_m * (1.0 - fraction) * direction
                ),
            )

        final_pose = _numpy(
            robot.data.body_link_pose_w.torch[0, body_index]
        ).copy()
        final_cube = _numpy(scene[active_name].data.root_pos_w[0, :3]).copy()
        final_target = gripper_target_for_object_local_offset(
            final_cube
            + np.asarray(
                (0.0, 0.0, aperture_height_offset_m), dtype=np.float32
            ),
            final_pose[3:7],
            aperture_reference,
        )
        final_position_error_m = float(
            np.linalg.norm(final_target - final_pose[:3])
        )
        final_closing_axis_world = (
            quaternion_rotation_matrix_xyzw(final_pose[3:7])
            @ SO101_CAPTURE_CLOSING_AXIS_LOCAL
        )
        record = {
            "candidate_index": candidate_index,
            "label": label,
            "posture_target_rad": posture_target.tolist(),
            "aperture_height_offset_m": aperture_height_offset_m,
            "unlevelled_closing_axis_world": (
                quaternion_rotation_matrix_xyzw(unlevelled_safe_pose[3:7])
                @ SO101_CAPTURE_CLOSING_AXIS_LOCAL
            ).tolist(),
            "levelled_orientation_target_xyzw": orientation_target.tolist(),
            "closing_axis_world_at_safe": closing_axis_world.tolist(),
            "closing_axis_world_final": final_closing_axis_world.tolist(),
            "outside_direction_world": direction.tolist(),
            "contact_free": bool(
                contact_free
                and final_position_error_m < 0.010
                and abs(float(final_closing_axis_world[2])) < 0.05
            ),
            "first_contact_segment": first_contact_segment,
            "first_contact_fraction": first_contact_fraction,
            "first_contact_forces_n": list(first_contact_forces),
            "final_position_error_m": final_position_error_m,
            "cube_displacement_m": (final_cube - cube_start).tolist(),
            "final_contact": _contact_debug(contacts),
        }
        results.append(record)
        Image.fromarray(_image(scene["front_camera"], device), mode="RGB").save(
            destination / f"candidate_{candidate_index:02d}_{label}_front.png"
        )
        print("SO101_GRASP_PATH " + json.dumps(record, sort_keys=True), flush=True)

    report = {
        "schema_version": "farpoint.so101_grasp_path_diagnostic.v3",
        "git_commit": os.environ.get("FARPOINT_GIT_COMMIT", "unknown"),
        "asset_sha256": SO101_WORKSHOP_ASSET_SHA256,
        "camera_features": ["front"],
        "wrist_camera_enabled": False,
        "object_edge_m": 0.040,
        "object_orientation_xyzw": object_orientation.tolist(),
        "jaw_preshape_rad": jaw_preshape,
        "aperture_center_local_m": aperture_reference.tolist(),
        "closing_axis_local": SO101_CAPTURE_CLOSING_AXIS_LOCAL.tolist(),
        "outside_distance_m": outside_distance_m,
        "vertical_clearance_m": vertical_clearance_m,
        "contact_threshold_n": contact_threshold_n,
        "maximum_closing_axis_abs_z": 0.05,
        "successful_candidate_labels": [
            item["label"] for item in results if item["contact_free"]
        ],
        "candidates": results,
    }
    _write_json(destination / "results.json", report)


def _runtime_camera_records(scene, camera_profile):
    intrinsics = {}
    for camera_id in ("front", "wrist"):
        sensor = scene[f"{camera_id}_camera"]
        matrix = getattr(sensor.data, "intrinsic_matrices", None)
        if matrix is None:
            raise RuntimeError(f"{camera_id} camera did not expose runtime intrinsics")
        intrinsics[camera_id] = _numpy(matrix[0]).tolist()
    return build_camera_records(
        camera_profile,
        resolved_intrinsics=intrinsics,
        resolved_mounts=resolved_mounts_from_profile(camera_profile),
    )


def run_attempt(
    env,
    trial,
    output_root: Path,
    git_commit: str,
    collection_id: str,
    camera_profile=None,
    live_publisher=None,
    v010_context=None,
    recovery_runtime=None,
):
    from farpoint.so101_mass_feasibility import audit_resolved_mass

    device = env.device
    scene = env.scene
    robot = scene["robot"]
    ee_frame = scene["ee_frame"]
    contact = (scene["contact_jaw"], scene["contact_gripper"])
    active_name = _variant_name(trial)
    inactive = [name for name in ("cube_small_red", "cube_small_blue", "cube_large_red", "cube_large_blue") if name != active_name]
    env.farpoint_active_cube = active_name
    episode_seed = int(trial["attempt_seed"])
    environment_seed = int(trial.get("environment_seed", episode_seed)) % (2**32)
    root = output_root / episode_id_for_attempt(collection_id, trial["attempt_id"])
    if root.exists():
        raise FileExistsError(f"episode output already exists: {root}")
    run_state = build_attempt_run_state(
        trial, collection_id=collection_id, git_commit=git_commit
    )
    if v010_context is not None:
        run_state["recording"]["cameras"] = [
            "observation.images.front",
            "observation.images.wrist",
        ]
    _write_json(root / "run-state.json", run_state)
    np.random.seed(environment_seed)
    torch.manual_seed(environment_seed)
    env.reset(seed=environment_seed)
    object_spec = trial["resolved"]
    grasp_posture = SO101_GRASP_POSTURE.copy()
    active_object = scene[active_name]
    resolved_mass_kg = float(object_spec["mass_kg"])
    active_object.set_masses_index(
        masses=torch.tensor(
            [[resolved_mass_kg]], dtype=torch.float32, device=device
        )
    )
    physx_actual_mass_kg = float(active_object.data.body_mass.torch[0, 0].item())
    mass_audit = audit_resolved_mass(
        requested_mass_kg=float(trial["requested"]["mass_kg"]),
        resolved_mass_kg=resolved_mass_kg,
        physx_actual_mass_kg=physx_actual_mass_kg,
        tolerance_kg=float(trial.get("mass_audit_tolerance_kg", 1e-6)),
    )
    run_state["physics_audit"] = {"mass": copy.deepcopy(mass_audit)}
    _write_json(root / "run-state.json", run_state)
    if not mass_audit["verified"]:
        raise RuntimeError(
            "PhysX cube mass audit failed: "
            f"requested={mass_audit['requested_mass_kg']}, "
            f"resolved={mass_audit['resolved_mass_kg']}, "
            f"actual={mass_audit['physx_actual_mass_kg']}, "
            f"tolerance={mass_audit['tolerance_kg']}"
        )
    # Hold the configured workshop pose on the first physics step.  Starting
    # from the authored all-zero pose leaves the arm horizontal and produces a
    # large gravity transient before the oracle receives its first observation.
    initial_joints = torch.tensor(
        [SO101_HOME_JOINTS.tolist()],
        dtype=torch.float32,
        device=device,
    )
    print(f"SO101_RESET_DEBUG after_reset={_numpy(robot.data.joint_pos[0]).tolist()}", flush=True)
    for index, name in enumerate(inactive):
        _move_object(scene[name], (-10.0 - index, 0.0, 0.1), device)
    expected_object_position = np.asarray(object_spec["position_m"], dtype=np.float32)
    reset_spawn_position = expected_object_position.copy()
    reset_spawn_position[2] += 0.002
    # Advance one manager step so FrameTransformer data reflects the reset
    # articulation before choosing the HOME waypoint.  Without this sync,
    # the first observation can be a stale pre-reset pose.
    # Send the same explicit pose as the first manager action; using the stale
    # tensor cached before write_joint_state would restore the USD default.
    camera_view = copy.deepcopy(trial.get("front_camera_view") or {})
    _aim_front_camera(scene, device, camera_view)
    target_spec = copy.deepcopy(
        (trial.get("target_profile") or {}).get("resolved")
        or v010_context["plan"]["target"]
    )
    target_position = np.asarray(target_spec["position_m"], dtype=np.float32)
    target_dimensions = np.asarray(target_spec["dimensions_m"], dtype=np.float32)
    _move_static_frame(scene["target_pad"], target_position.tolist(), device)
    # Spawn just above the table and let PhysX establish support before the
    # oracle reads its first target. A failed audit is a deterministic scene
    # error and must not be hidden by retrying the same pose.
    _move_object(
        scene[active_name],
        reset_spawn_position,
        device,
        object_spec["orientation_xyzw"],
    )
    for _ in range(8):
        env.step(initial_joints)
    measured_object_position = _numpy(active_object.data.root_pos_w[0])
    measured_object_orientation = _numpy(active_object.data.root_quat_w[0])
    measured_object_velocity = _numpy(active_object.data.root_lin_vel_w[0])
    reset_support_verified = so101_reset_support_is_stable(
        expected_object_position,
        measured_object_position,
        measured_object_velocity,
    )
    reset_support_audit = {
        "spawn_clearance_m": 0.002,
        "settling_control_steps": 8,
        "expected_position_m": expected_object_position.tolist(),
        "measured_position_m": measured_object_position.tolist(),
        "measured_linear_velocity_mps": measured_object_velocity.tolist(),
        "maximum_xy_error_m": 0.002,
        "maximum_z_error_m": 0.001,
        "maximum_speed_mps": 0.05,
        "verified": reset_support_verified,
    }
    run_state["physics_audit"]["reset_support"] = copy.deepcopy(
        reset_support_audit
    )
    _write_json(root / "run-state.json", run_state)
    if not reset_support_verified:
        raise RuntimeError(
            "SO-101 cube failed reset support validation: "
            f"expected={expected_object_position.tolist()}, "
            f"measured={measured_object_position.tolist()}, "
            f"linear_velocity={measured_object_velocity.tolist()}"
        )
    # The settling steps above are for the cube, not part of the oracle
    # trajectory. Restore the articulation state without advancing physics so
    # every episode begins from the same authored HOME pose as the validated
    # collector. Keeping the arm motion accumulated during cube settling
    # changed reachability and introduced deterministic PREPLACE regressions.
    robot.write_joint_state_to_sim(
        initial_joints,
        torch.zeros_like(initial_joints),
    )
    robot.set_joint_position_target(initial_joints)
    env.sim.forward()
    scene.update(0.0)
    restored_joints = _numpy(robot.data.joint_pos[0])
    maximum_home_error_rad = float(
        np.max(np.abs(restored_joints - SO101_HOME_JOINTS))
    )
    reset_support_audit["arm_restored_to_home"] = bool(
        maximum_home_error_rad <= 1e-5
    )
    reset_support_audit["maximum_home_error_rad"] = maximum_home_error_rad
    run_state["physics_audit"]["reset_support"] = copy.deepcopy(
        reset_support_audit
    )
    _write_json(root / "run-state.json", run_state)
    if not reset_support_audit["arm_restored_to_home"]:
        raise RuntimeError(
            "SO-101 arm failed reset HOME restoration: "
            f"maximum_error_rad={maximum_home_error_rad}"
        )
    body_index = _body_index(robot)
    home_ee = _numpy(robot.data.body_link_pose_w.torch[0, body_index, :3]).copy()
    demonstration = None
    recovery_snapshot = None
    if recovery_runtime is not None:
        demonstration, recovery_snapshot = _run_recovery_handoff(
            env,
            trial,
            active_object,
            contact,
            root,
            recovery_runtime,
            oracle_profile_id=v010_context["plan"]["oracle_profile_id"],
        )
        run_state["recovery"] = {
            "runtime_id": recovery_runtime["runtime_id"],
            "handoff": copy.deepcopy(demonstration["intervention"]["handoff"]),
        }
        _write_json(root / "run-state.json", run_state)
    print(
        "SO101_RESET_SUPPORT "
        f"position={measured_object_position.tolist()} "
        f"linear_velocity={measured_object_velocity.tolist()} "
        f"maximum_home_error_rad={maximum_home_error_rad}",
        flush=True,
    )
    print(f"SO101_RESET_DEBUG after_env_step={_numpy(robot.data.joint_pos[0]).tolist()}", flush=True)
    action_term = env.action_manager.get_term("joint_positions")
    print(
        "SO101_ACTION_DEBUG "
        f"raw={_numpy(action_term.raw_actions[0]).tolist()} "
        f"processed={_numpy(action_term.processed_actions[0]).tolist()} "
        f"joint_targets={_numpy(robot.data.joint_pos_target[0]).tolist()}",
        flush=True,
    )
    # Recovery can hand off while ACT is already closing. Release success must
    # still mean the authored fully-open jaw, not the incidental handoff angle.
    open_jaw = float(SO101_HOME_JOINTS[5])
    closed_jaw = float(np.deg2rad(-10.0))
    object_position = (
        np.asarray(recovery_snapshot["object_pose_xyzw"][:3], dtype=np.float32)
        if recovery_snapshot is not None
        else np.asarray(object_spec["position_m"], dtype=np.float32)
    )
    approach_jaw = so101_approach_jaw_target(object_spec["dimensions_m"][0])
    capture_object_in_gripper = so101_capture_aperture_reference(approach_jaw)
    pre_capture_recenter_aperture_reference = (
        so101_pre_capture_recenter_aperture_reference(
            approach_jaw,
            object_spec["dimensions_m"][0],
        )
    )
    # Release above the raised target pad instead of driving the fingertips
    # down to the cube's resting height. At the lower target the cube contacts
    # the pad first and can slip while the jaw tries to open.
    release_position = target_position.copy()
    release_position[2] = float(target_position[2]) + 0.043
    # Separate capture admission from contact persistence. A 0.1 N bilateral
    # sample is enough to preserve an already captured cube, but it is too
    # weak to freeze the rotary jaw: light cubes can briefly touch both long
    # fingers while still sliding out of the aperture. Require the same 2 N,
    # three-tick confirmation used by CLOSE before entering bilateral settle.
    schedule = CONTROL_RECORDING_SCHEDULE
    recovery_entry_phase = (
        OraclePhase(recovery_snapshot["trigger"]["oracle_entry_phase"])
        if recovery_snapshot is not None
        and recovery_runtime.get("schema_version") == "farpoint.recovery-runtime.v2"
        else (OraclePhase.PREGRASP if recovery_runtime is not None else OraclePhase.HOME)
    )
    machine = OracleStateMachine(
        phase=recovery_entry_phase,
        phase_timeout_steps=schedule.steps_for_seconds(40.0),
        required_contact_steps=1,
        required_stable_steps=schedule.steps_for_seconds(0.5),
    )
    grasp_machine = ContactAwareGraspStateMachine(
        control_hz=schedule.control_hz,
        object_width_m=object_spec["dimensions_m"][0],
        # Low-force bilateral contact is sufficient to enter the quasi-static
        # hold; rigidity and the later physical proof lift remain mandatory.
        # The formal workspace run showed stable 30 mm captures oscillating
        # around 0.1--1.9 N before being rejected by the former 2 N floor.
        minimum_contact_force_n=0.10,
        capture_contact_force_n=2.0,
        # Require an evidence-bounded, size-aware bilateral preload before
        # contact-bound motion. A fixed 4 N floor correctly blocked the 40 mm
        # c08 ejection but also rejected a stable 30 mm capture at 3.61 N.
        # This changes only proof entry readiness, not persistence, force
        # ceilings, or the independent 5 mm physical proof.
        minimum_proof_entry_force_n=so101_proof_entry_force_floor(
            object_spec["dimensions_m"][0],
        ),
        maximum_force_n=30.0,
        bilateral_settle_s=0.125,
        static_hold_s=0.20,
        proof_lift_hold_s=0.10,
        phase_timeout_s=20.0,
        # Give the bounded force/recenter controller enough time to restore a
        # missing side after first contact. The smaller cube needs three more
        # 30 Hz ticks for the rotary jaw to finish closing; 40 mm cubes retain
        # the validated 0.20 s window. Stable timers still reset on every
        # unilateral sample and physical proof lift remains mandatory.
        maximum_contact_loss_s=so101_capture_contact_loss_grace_s(
            object_spec["dimensions_m"][0]
        ),
    )
    commanded_joints = _numpy(robot.data.joint_pos[0]).astype(np.float32).copy()
    recovery_oracle_previous_target = None
    recovery_oracle_maximum_delta = None
    if recovery_snapshot is not None and recovery_oracle_command_continuity_enabled(
        recovery_runtime
    ):
        recovery_oracle_previous_target = np.asarray(
            recovery_snapshot["joint_position_target_rad"], dtype=np.float32
        )
        commanded_joints = recovery_oracle_previous_target.copy()
        recovery_oracle_maximum_delta = recovery_oracle_slew_limits(recovery_runtime)
    cube_was_lifted = False
    grasp_hold_pose = None
    grasp_hold_nominal_pose = None
    grasp_hold_nominal_y = None
    grasp_hold_posture = None
    grasp_offset = None
    grasp_jaw_hold = None
    grasp_jaw_reference = None
    release_hold_pose = None
    placement_grasp_offset = None
    transport_object_target = None
    transport_recovering_height = False
    transport_recovery_xy = None
    placement_descent_initialized = False
    lift_object_start_position = None
    transport_lift_target_m = 0.0
    verify_bilateral_steps = 0
    verify_capture_latched = False
    close_capture_confirmed = False
    verify_capture_wait_steps = 0
    verify_grasp_armed = False
    verify_lift_height = 0.0
    verify_object_start_z = None
    grasp_relative_reference = None
    previous_object_in_gripper = None
    capture_recenter_side = None
    pre_capture_recenter_object_reference = None
    capture_object_minus_grasp = None
    descent_lateral_correction = 0.0
    pregrasp_route_index = 0
    if recovery_entry_phase in {
        OraclePhase.PREPLACE,
        OraclePhase.OPEN,
        OraclePhase.SETTLE,
    }:
        current_joints = _numpy(robot.data.joint_pos[0]).astype(np.float32)
        gripper_pose = _numpy(robot.data.body_link_pose_w.torch[0, body_index])
        live_object_position = _numpy(scene[active_name].data.root_pos_w[0])
        grasp_hold_pose = gripper_pose[:3].copy()
        grasp_hold_nominal_pose = grasp_hold_pose.copy()
        grasp_hold_nominal_y = float(grasp_hold_pose[1])
        grasp_hold_posture = current_joints[3:5].copy()
        grasp_offset = grasp_hold_pose - live_object_position
        grasp_jaw_hold = float(current_joints[5])
        grasp_jaw_reference = grasp_jaw_hold
        grasp_relative_reference = point_in_local_frame(
            gripper_pose, live_object_position
        )
        previous_object_in_gripper = grasp_relative_reference.copy()
        capture_object_minus_grasp = live_object_position - grasp_hold_pose
        cube_was_lifted = bool(
            recovery_snapshot["trigger"].get(
                "ever_lifted", recovery_snapshot["trigger"].get("cube_lifted", False)
            )
        )
    rows = []
    physics_command_rows = []
    for control_step in range(schedule.steps_for_seconds(120.0)):
        phase = machine.phase
        phase_motion_complete = True
        descent_fraction = None
        recovery_oracle_safety = None
        current = robot.data.joint_pos[0]
        ee_position = _numpy(robot.data.body_link_pose_w.torch[0, body_index, :3]).copy()
        control_point_position = None
        control_point_offset_world = None
        if phase is OraclePhase.CLOSE and grasp_hold_pose is None:
            # Stop the Cartesian descent at first finger contact. Holding
            # this pose while the jaw closes avoids driving the fingertips
            # through the cube before grasping starts.
            grasp_hold_pose = ee_position.copy()
            grasp_hold_nominal_pose = grasp_hold_pose.copy()
            grasp_hold_nominal_y = float(grasp_hold_pose[1])
            # Approach posture avoids an early table/cube sweep, while the
            # physically successful enclosure settles at a different wrist
            # roll. Transition to that calibrated capture posture during CLOSE
            # and latch the measured result only after bilateral confirmation.
            grasp_hold_posture = grasp_posture.copy()
        if phase is OraclePhase.VERIFY_CONTACT and grasp_offset is None:
            grasp_offset = ee_position - _numpy(scene[active_name].data.root_pos_w[0])
            # CLOSE integrates Cartesian commands while the fingers are
            # physically constrained by the cube.  Rebase on the very first
            # VERIFY frame, before waiting for the force window, so none of
            # that constrained-motion command tail reaches the object.
            commanded_joints = _numpy(current).astype(np.float32).copy()
            grasp_hold_pose = ee_position.copy()
            grasp_hold_nominal_pose = grasp_hold_pose.copy()
            if grasp_jaw_hold is None:
                grasp_jaw_hold = float(current[5].item())
            if grasp_jaw_reference is None:
                grasp_jaw_reference = grasp_jaw_hold
        balanced_forces = None
        if grasp_hold_pose is not None and phase in {
            OraclePhase.CLOSE,
            OraclePhase.VERIFY_CONTACT,
            OraclePhase.LIFT,
            OraclePhase.PREPLACE,
            OraclePhase.PLACE_DESCEND,
        }:
            balanced_forces = _cube_contact_forces(contact)
        jaw_force_action = None
        capture_admissible = so101_capture_admission_ready(
            float(current[5].item()),
            object_spec["dimensions_m"][0],
        )
        settling_capture = (
            phase is OraclePhase.CLOSE
            and grasp_machine.phase
            in {GraspPhase.BILATERAL_SETTLE, GraspPhase.STATIC_HOLD}
        )
        settling_force_imbalanced = bool(
            settling_capture
            and balanced_forces is not None
            and captured_force_imbalance_requires_squeeze_pause(
                *balanced_forces,
                minimum_force_n=grasp_machine.minimum_contact_force_n,
                proof_entry_force_n=grasp_machine.minimum_proof_entry_force_n,
            )
        )
        retention_preload_fallback = bool(
            settling_capture
            and balanced_forces is not None
            and capture_retention_recenter_fallback_active(
                grasp_machine.phase,
                grasp_machine.phase_steps,
                *balanced_forces,
                grasp_machine.minimum_proof_entry_force_n,
            )
        )
        if (
            grasp_jaw_hold is not None
            and balanced_forces is not None
            and (
                not verify_capture_latched
                or phase is OraclePhase.VERIFY_CONTACT
                or settling_capture
            )
        ):
            jaw_update = force_controlled_rotary_jaw_target(
                grasp_jaw_hold,
                float(current[5].item()),
                *balanced_forces,
                open_position=open_jaw,
                closed_position=closed_jaw,
                # The edge-yaw traces entered capture above 2 N on both
                # fingers, then decayed through 1.4 N before the old 0.5 N
                # controller reacted. Retain 90% of the unchanged admission
                # force so the rotary jaw closes while contact still exists;
                # persistence and maximum-force validation remain independent.
                min_force=capture_retention_force_floor(
                    (
                        capture_preload_force_floor(
                            grasp_machine.capture_contact_force_n
                        )
                        if settling_capture
                        else 3.0
                    ),
                    capture_validation_active=(
                        settling_capture or phase is OraclePhase.VERIFY_CONTACT
                    ),
                ),
                # Back off before the independent 30 N safety validator can
                # trip on a one-control-tick unilateral force spike. Keep
                # the validated 30 mm path at 20 N while the larger contact
                # geometry starts backing off earlier.
                max_force=so101_capture_jaw_backoff_force_n(
                    object_spec["dimensions_m"][0],
                ),
                close_step=(
                    so101_imbalanced_capture_close_step(
                        object_spec["dimensions_m"][0],
                    )
                    if settling_force_imbalanced
                    else so101_balanced_capture_close_step(
                        object_spec["dimensions_m"][0],
                    )
                    if settling_capture
                    else (0.001 if phase is OraclePhase.VERIFY_CONTACT else 0.002)
                ),
                # Capture transition already applies a bounded preload
                # relief.  Repeating another 1 mrad opening impulse during
                # large-cube settle shed the weak-side contact in four ticks
                # (r11 q014). Preserve the validated 30 mm path and taper
                # only the settle backoff to zero at 40 mm.
                backoff_step=(
                    so101_slow_close_backoff_step_rad(
                        object_spec["dimensions_m"][0],
                        small_cube_step=0.001,
                    )
                    if settling_capture
                    else 0.001
                ),
                # The immutable c26 r24 trace showed that the state-driven
                # retention fallback improved both finger forces while the
                # jaw command remained pinned at the normal 12 mrad preload
                # cap.  Give only that already-gated recovery path another
                # 6 mrad; normal capture and the independent force/range
                # safety limits remain unchanged.
                max_preload_error=(
                    0.018
                    if retention_preload_fallback
                    else 0.012
                    if settling_capture
                    else 0.030
                ),
                preload_reference_position=grasp_jaw_reference,
            )
            grasp_jaw_hold = float(jaw_update["position"])
            jaw_force_action = str(jaw_update["action"])
        recenter_active = False
        relative_recenter_active = False
        recenter_used_memory = False
        recenter_forces = balanced_forces
        contact_geometry_valid = False
        if balanced_forces is not None:
            contact_force_vectors = _cube_contact_force_vectors(contact)
            contact_geometry_valid = contact_force_vectors_opposed(
                *contact_force_vectors,
                minimum_force_n=grasp_machine.minimum_contact_force_n,
            )
        closing_alignment = phase is OraclePhase.CLOSE and (
            grasp_phase_allows_unilateral_recenter(grasp_machine.phase)
        )
        verification_alignment = (
            phase is OraclePhase.VERIFY_CONTACT
            and grasp_phase_allows_unilateral_recenter(grasp_machine.phase)
        )
        # A proof lift is still a contact-constrained capture phase.  The r5
        # c22 trace retained a rigid unilateral enclosure after one finger
        # dropped, but VERIFY disabled the existing bounded XY recovery and
        # spent the entire contact-loss grace only squeezing the jaw.  Carry
        # the same contact memory and 2 mm recenter corridor through proof;
        # force, lift, rigidity, and timeout gates remain unchanged.
        if (
            (closing_alignment or settling_capture or verification_alignment)
            and balanced_forces is not None
        ):
            recenter_memory = so101_recenter_contact_memory(
                *balanced_forces,
                capture_recenter_side,
                minimum_force_n=grasp_machine.minimum_contact_force_n,
            )
            recenter_forces = recenter_memory["forces"]
            capture_recenter_side = recenter_memory["side"]
            recenter_used_memory = bool(recenter_memory["used_memory"])
        elif phase not in {
            OraclePhase.DESCEND,
            OraclePhase.CLOSE,
            OraclePhase.VERIFY_CONTACT,
        }:
            capture_recenter_side = None
        unilateral_recenter = bool(
            recenter_forces is not None
            and unilateral_contact_requires_recenter(
                *recenter_forces,
                minimum_force_n=grasp_machine.minimum_contact_force_n,
            )
        )
        proof_force_imbalance_recenter = bool(
            settling_capture
            and balanced_forces is not None
            and captured_force_imbalance_requires_recenter(
                *balanced_forces,
                minimum_force_n=grasp_machine.minimum_contact_force_n,
                proof_entry_force_n=grasp_machine.minimum_proof_entry_force_n,
            )
        )
        # A bilateral force imbalance pauses jaw squeeze.  Pre-capture
        # calibration remains useful while CLOSE is still finding the
        # enclosure, but is not a valid post-capture recenter reference.  Once
        # BILATERAL_SETTLE records a physical capture, recover that measured
        # object/gripper offset through settle and proof lift.  The immutable
        # v0.2.0 c22 trace showed that chasing the old calibration after proof
        # contact loss followed the sliding cube for 2.16 mm without restoring
        # the weak finger.
        capture_retention_fallback = bool(
            recenter_forces is not None
            and capture_retention_recenter_fallback_active(
                grasp_machine.phase,
                grasp_machine.phase_steps,
                *recenter_forces,
                grasp_machine.minimum_proof_entry_force_n,
                # Frozen segment-005 traces separated successful captures
                # (at most 234 unilateral slow-close frames) from the
                # structural timeout cluster (370--581 frames). Preserve the
                # biased pre-capture center for the first 10 seconds, then
                # reuse the existing calibrated aperture fallback instead of
                # spending the remaining phase budget pinned one-sided.
                minimum_slow_close_steps=schedule.steps_for_seconds(10.0),
            )
        )
        capture_retention_reopen = False
        capture_recenter_required = (
            unilateral_recenter
            or capture_retention_fallback
            or proof_force_imbalance_recenter
        )
        if (
            grasp_hold_pose is not None
            and (closing_alignment or settling_capture or verification_alignment)
            and recenter_forces is not None
            and capture_recenter_required
            or (
                grasp_hold_pose is not None
                and (closing_alignment or verification_alignment)
                and recenter_forces is not None
                and min(recenter_forces)
                >= grasp_machine.minimum_contact_force_n
                and not contact_geometry_valid
            )
        ):
            if proof_force_imbalance_recenter:
                jaw_center = _numpy(
                    robot.data.body_link_pose_w.torch[
                        0, robot.body_names.index("jaw"), :3
                    ]
                )
                gripper_center = _numpy(
                    robot.data.body_link_pose_w.torch[0, body_index, :3]
                )
                left_force, right_force = balanced_forces
                directional_forces = (
                    (left_force, 0.0)
                    if left_force >= right_force
                    else (0.0, right_force)
                )
                proof_recenter_limit_m = 0.004
                recenter = unilateral_contact_recenter_target(
                    grasp_hold_pose,
                    grasp_hold_nominal_pose,
                    {"center": jaw_center.tolist()},
                    {"center": gripper_center.tolist()},
                    *directional_forces,
                    min_force=grasp_machine.minimum_contact_force_n,
                    step=so101_post_capture_recenter_step(
                        maximum_correction_m=proof_recenter_limit_m,
                    ),
                    max_correction=proof_recenter_limit_m,
                    # r14 proved that the transport-style "away from strong"
                    # convention reduced q014's weak side from 5.64 N to
                    # 2.00 N.  Sensor-side labels and this rotated aperture's
                    # Cartesian finger axis have opposite handedness here;
                    # use the measured strong side with the capture convention
                    # that preserves the weak-side enclosure.  r15 exhausted
                    # the earlier 2 mm corridor while improving the weak side
                    # from 2.00 N to 2.35 N; this proof-only path therefore
                    # receives one additional bounded 2 mm of travel.  The
                    # ordinary unilateral and pre-capture corridors remain
                    # unchanged.
                    move_toward_contact=True,
                )
            elif closing_alignment:
                # A finger-side label alone does not determine which world
                # direction centers this rotated, long-finger aperture.  Use
                # simulator truth and the measured gripper orientation to
                # recover the calibrated object-in-aperture XY offset while
                # preserving the contact handoff height.
                live_gripper_pose = _numpy(
                    robot.data.body_link_pose_w.torch[0, body_index]
                )
                live_object_world = _numpy(
                    scene[active_name].data.root_pos_w[0, :3]
                )
                capture_retention_reopen = capture_retention_reopen_active(
                    capture_retention_fallback,
                    point_in_local_frame(live_gripper_pose, live_object_world),
                    capture_object_in_gripper,
                )
                # The stalled fallback changes the aperture reference, not
                # the world anchor.  Targeted q021 evidence showed that using
                # the live, already sliding cube made the servo chase it from
                # (+16, +16) mm to (-18, +18) mm, worsening local alignment
                # from 17 mm to 41 mm despite eventually touching both
                # fingers. Keep the first-contact object anchor immutable so
                # the fallback recenters the gripper instead of following a
                # cube displaced by unilateral force.
                object_world = (
                    pre_capture_recenter_object_reference
                    if pre_capture_recenter_object_reference is not None
                    else live_object_world
                )
                active_recenter_reference = pre_capture_recenter_aperture_reference
                if capture_retention_reopen:
                    active_recenter_reference = capture_object_in_gripper
                desired_gripper = gripper_xy_target_for_object_local_offset(
                    object_world,
                    live_gripper_pose,
                    active_recenter_reference,
                )
                desired_object_minus_grasp = object_world - desired_gripper
                current_xy_correction = (
                    np.asarray(grasp_hold_pose[:2], dtype=np.float64)
                    - np.asarray(grasp_hold_nominal_pose[:2], dtype=np.float64)
                )
                correction_limit = so101_adaptive_pre_capture_recenter_limit(
                    object_spec["dimensions_m"][0],
                    current_xy_correction,
                    unilateral_contact=capture_recenter_required,
                )
                if capture_retention_reopen:
                    # Frozen r4/q021 and r6/q014 evidence separate a genuinely
                    # off-aperture stall (17.35 mm) from a small residual
                    # alignment error (about 3 mm). Expand only while the jaw
                    # is re-opening around the former; otherwise preserve the
                    # validated 16 mm path.
                    # Frozen r4 q021 evidence left the fixed-anchor servo
                    # saturated at (+16, +16) mm while the cube still had a
                    # 17.35 mm aperture-local Y error.  The earlier 18 mm
                    # diagnostic failed because it also chased the sliding
                    # live cube; with the first-contact world anchor now
                    # immutable, expand only this 10-second stalled fallback.
                    # The 30 mm endpoint remains 9 mm and ordinary unilateral
                    # capture remains capped by the validated 16 mm path.
                    correction_limit = so101_adaptive_pre_capture_recenter_limit(
                        object_spec["dimensions_m"][0],
                        current_xy_correction,
                        unilateral_contact=True,
                        maximum_correction_m=0.018,
                        large_width_fraction=0.45,
                    )
                aligned = relative_object_grasp_servo_target(
                    object_world,
                    desired_object_minus_grasp,
                    ee_position,
                    grasp_hold_nominal_pose,
                    max_step=0.000125,
                    max_correction=(
                        correction_limit,
                        correction_limit,
                        0.0,
                    ),
                )
                recenter = {
                    "position": aligned["position"],
                    "active": float(np.linalg.norm(aligned["error"][:2])) > 1e-6,
                }
            elif proof_lift_recovery_holds_xy(
                proof_lift_armed=verify_grasp_armed,
                unilateral_contact=unilateral_recenter,
            ):
                # The moving jaw can still recover inside the unchanged
                # contact-loss grace. Holding XY prevents the retained fixed
                # finger from dragging the cube away at the same rate that
                # the jaw closes toward it.
                recenter = {
                    "position": grasp_hold_pose,
                    "active": False,
                }
            elif capture_object_minus_grasp is not None:
                object_world = _numpy(
                    scene[active_name].data.root_pos_w[0, :3]
                )
                # This is an XY enclosure repair, not a second lift command.
                # Project the object's Z onto the current proof-lift base so
                # the relative servo cannot fold measured proof motion back
                # into grasp_hold_pose before verify_lift_height is added.
                planar_object_world = object_world.copy()
                planar_object_world[2] = (
                    float(capture_object_minus_grasp[2])
                    + float(grasp_hold_pose[2])
                )
                aligned = relative_object_grasp_servo_target(
                    planar_object_world,
                    capture_object_minus_grasp,
                    grasp_hold_pose,
                    grasp_hold_nominal_pose,
                    max_step=so101_post_capture_recenter_step(),
                    max_correction=(0.002, 0.002, 0.0),
                )
                recenter = {
                    "position": aligned["position"],
                    "active": float(np.linalg.norm(aligned["error"][:2]))
                    > 1e-6,
                }
            else:
                jaw_center = _numpy(
                    robot.data.body_link_pose_w.torch[
                        0, robot.body_names.index("jaw"), :3
                    ]
                )
                gripper_center = _numpy(
                    robot.data.body_link_pose_w.torch[0, body_index, :3]
                )
                recenter = unilateral_contact_recenter_target(
                    grasp_hold_pose,
                    grasp_hold_nominal_pose,
                    {"center": jaw_center.tolist()},
                    {"center": gripper_center.tolist()},
                    *recenter_forces,
                    min_force=grasp_machine.minimum_contact_force_n,
                    step=so101_post_capture_recenter_step(),
                    max_correction=0.002,
                    move_toward_contact=True,
                )
            grasp_hold_pose = np.asarray(recenter["position"], dtype=np.float32)
            recenter_active = bool(recenter["active"])
        if (
            not recenter_active
            and settling_capture
            and capture_object_minus_grasp is not None
            and grasp_hold_nominal_pose is not None
            and balanced_forces is not None
            and min(balanced_forces)
            < grasp_machine.minimum_proof_entry_force_n
        ):
            # Contact forces can remain balanced while a constrained arm
            # drifts far enough to fail the independent rigidity check.  In
            # that case force-side recentering has no direction.  Track the
            # physical capture offset from the measured EE pose, with the same
            # conservative 0.125 mm/tick rate and a bounded 6 mm recovery box.
            object_world = _numpy(scene[active_name].data.root_pos_w[0, :3])
            relative_recenter = relative_object_grasp_servo_target(
                object_world,
                capture_object_minus_grasp,
                ee_position,
                grasp_hold_nominal_pose,
                max_step=so101_post_capture_recenter_step(),
                max_correction=(0.006, 0.006, 0.006),
            )
            if float(np.linalg.norm(relative_recenter["error"])) > 0.002:
                grasp_hold_pose = np.asarray(
                    relative_recenter["position"], dtype=np.float32
                )
                recenter_active = True
                relative_recenter_active = True
        if phase is OraclePhase.HOME:
            target = home_ee
            jaw = open_jaw
        elif phase in {OraclePhase.PREGRASP, OraclePhase.DESCEND}:
            # Recompute the aperture target from the measured gripper
            # orientation.  A fixed safe-height quaternion is invalid for this
            # 5-DOF arm because its world orientation changes with shoulder and
            # elbow configuration even when both wrist targets stay fixed.
            object_world = _numpy(scene[active_name].data.root_pos_w[0, :3])
            live_gripper_pose = _numpy(
                robot.data.body_link_pose_w.torch[0, body_index]
            )
            capture_target = gripper_target_for_object_local_offset(
                object_world,
                live_gripper_pose[3:7],
                capture_object_in_gripper,
            )
            local_z_world = quaternion_rotation_matrix_xyzw(
                live_gripper_pose[3:7]
            )[:, 2]
            feed_distance_m = 0.070
            distal_pregrasp = capture_target + feed_distance_m * local_z_world
            if phase is OraclePhase.PREGRASP:
                final_target = distal_pregrasp
            else:
                insertion_steps = schedule.steps_for_seconds(
                    recovery_descent_duration_seconds(recovery_runtime)
                )
                descent_fraction = min(
                    1.0, (machine.phase_steps + 1) / insertion_steps
                )
                final_target = capture_target + (
                    feed_distance_m * (1.0 - descent_fraction) * local_z_world
                )
                phase_motion_complete = descent_fraction >= 1.0
            if phase is OraclePhase.PREGRASP and pregrasp_route_index < 3:
                # The formal run proved that translating directly toward the
                # former 0.16 m staging pose sweeps a long finger through some
                # 40 mm cubes. Lift at the home XY first, translate at a
                # reachable 0.19 m clearance, then descend outside the fingers.
                safe_route = collision_safe_pregrasp_waypoints(
                    home_ee,
                    distal_pregrasp,
                    clearance_z=max(
                        0.19,
                        float(home_ee[2]) + 0.04,
                        float(distal_pregrasp[2]) + 0.04,
                    ),
                )
                route = tuple(
                    np.asarray(waypoint, dtype=np.float32)
                    for waypoint in (*safe_route, distal_pregrasp.tolist())
                )
                target = route[pregrasp_route_index]
                phase_motion_complete = False
                route_position_tolerance = 0.008
                route_posture_tolerance = (
                    0.10 if pregrasp_route_index == 0 else 0.20
                )
                route_posture_ready = (
                    abs(float(current[3].item()) - float(grasp_posture[0]))
                    < route_posture_tolerance
                    and abs(
                        float(current[4].item()) - float(grasp_posture[1])
                    )
                    < route_posture_tolerance
                )
                if (
                    float(np.linalg.norm(ee_position - target))
                    < route_position_tolerance
                    and route_posture_ready
                ):
                    pregrasp_route_index += 1
                    target = (
                        route[pregrasp_route_index]
                        if pregrasp_route_index < len(route)
                        else final_target
                    )
                    phase_motion_complete = pregrasp_route_index == len(route)
                    print(
                        f"SO101_ORACLE_PREGRASP_ROUTE_CLEAR index={pregrasp_route_index} "
                        f"ee={ee_position.tolist()} next_target={target.tolist()}",
                        flush=True,
                    )
            else:
                target = final_target
            # Pre-shape the rotary jaw just wider than the object. Starting
            # from the 110-degree mechanical maximum makes the long moving
            # finger sweep the cube away before the fixed finger can engage.
            jaw = approach_jaw
        elif phase is OraclePhase.CLOSE:
            finger_forces = balanced_forces
            target = grasp_hold_pose
            if grasp_jaw_hold is not None:
                # A constrained Cartesian hold must be resolved from measured
                # joints on every 120 Hz tick. Accumulating IK targets after
                # contact produced a 7.7 mrad command tail in smoke-007 and
                # peeled the fixed finger away despite a latched aperture.
                commanded_joints = _numpy(current).astype(np.float32).copy()
                jaw = grasp_jaw_hold
                gripper_control = "measured_rebase_capture_hold"
            elif so101_bilateral_capture_ready(
                *finger_forces,
                capture_admissible,
                object_width_m=object_spec["dimensions_m"][0],
                capture_contact_force_n=grasp_machine.capture_contact_force_n,
            ) and contact_geometry_valid:
                # Both cube sidewalls constrain the arm before the grasp state
                # machine has completed its force *and* relative-speed window.
                # Rebase the joint command on every such sample to discard the
                # pre-contact IK tail and keep closing gently, but do not latch
                # a capture here.  The state machine below is the single owner
                # of capture admission and overrides this action in the exact
                # tick that BILATERAL_SETTLE is entered.  A second collector-
                # local latch used to ignore the relative-speed gate, freeze a
                # still-moving 40 mm cube, and leave the state machine stuck in
                # SLOW_CLOSE after bilateral force decayed.
                commanded_joints = _numpy(current).astype(np.float32).copy()
                target = ee_position.copy()
                # A single bilateral sample can be an edge-impact pulse.
                # Continue closing with the same bounded preload used after
                # capture until the shared state machine observes its full
                # confirmation window at or below the capture-speed gate.
                # Using a smaller confirmation-only preload produced a stable
                # 40 mm limit cycle immediately around the unchanged 90%
                # force floor, so six consecutive samples could never accrue.
                jaw = rotary_jaw_capture_hold_target(
                    float(current[5].item()),
                    closed_position=closed_jaw,
                    open_position=open_jaw,
                )
                # Hold the wrists at the measured bilateral-contact posture
                # immediately.  Waiting until BILATERAL_SETTLE let the generic
                # CLOSE posture override keep chasing the pre-grasp 0.5/0.5
                # target for one extra actuator-response window.  The frozen
                # v0.2.0 c10 r29c trace showed that tail peeling both contacts
                # away after six otherwise-valid confirmation samples.
                grasp_hold_posture = (
                    _numpy(current[3:5]).astype(np.float32).copy()
                )
                gripper_control = "confirm_bilateral"
                target = grasp_hold_pose
            else:
                # The validated calibration stops Cartesian insertion at first
                # contact and closes quasi-statically at the measured pose.
                # Continuing to chase the displaced cube recreates the old
                # unilateral wedge failure.
                target = grasp_hold_pose
                if capture_retention_reopen:
                    # Once a stalled cube is materially off aperture, further
                    # rotary closure only sweeps it away. Re-open at the same
                    # zero-impact rate used by the small-cube force backoff
                    # while the Cartesian servo recenters, then resume the
                    # unchanged slow-close controller after alignment.
                    jaw = min(
                        approach_jaw,
                        float(current[5].item()) + 0.002,
                    )
                    gripper_control = "stalled_aperture_reopen"
                else:
                    jaw_update = advance_so101_slow_close_target(
                        float(commanded_joints[5]),
                        float(current[5].item()),
                        *finger_forces,
                        open_position=open_jaw,
                        closed_position=closed_jaw,
                        max_force=so101_slow_close_bilateral_brake_force_n(
                            object_spec["dimensions_m"][0],
                        ),
                        unilateral_backoff_force=so101_capture_jaw_backoff_force_n(
                            object_spec["dimensions_m"][0],
                        ),
                        backoff_step=so101_slow_close_backoff_step_rad(
                            object_spec["dimensions_m"][0],
                        ),
                        capture_admissible=capture_admissible,
                    )
                    jaw = float(jaw_update["position"])
                    gripper_control = f"calibrated_slow_{jaw_update['action']}"
        elif phase is OraclePhase.VERIFY_CONTACT:
            # Prove the grasp with a gentle test lift. A direct 8 cm target
            # change produces enough acceleration to shed a small cube before
            # contact persistence can be evaluated.
            just_armed_proof_lift = False
            bilateral_capture = (
                min(balanced_forces) >= 0.10 and contact_geometry_valid
            )
            if bilateral_capture and not verify_capture_latched:
                # Re-capture the measured aperture as soon as recentering has
                # restored both contacts. Continuing toward the older, tighter
                # jaw target can squeeze the cube back out on the next tick.
                verify_capture_latched = True
                verify_capture_wait_steps = 0
                grasp_jaw_hold = max(
                    closed_jaw, float(current[5].item()) - 0.005
                )
                grasp_jaw_reference = grasp_jaw_hold
                commanded_joints = _numpy(current).astype(np.float32).copy()
                grasp_hold_pose = ee_position.copy()
                grasp_hold_posture = _numpy(current[3:5]).astype(np.float32).copy()
                verify_bilateral_steps += 1
            elif bilateral_capture and verify_capture_latched:
                verify_bilateral_steps += 1
            if verify_capture_latched:
                verify_capture_wait_steps += 1
            if not verify_grasp_armed:
                # While the rotary jaw settles, hold the arm exactly where it
                # is, apart from a bounded Cartesian recenter step, instead of
                # integrating pose error against a constrained contact.
                commanded_joints = _numpy(current).astype(np.float32).copy()
                if not recenter_active:
                    grasp_hold_pose = ee_position.copy()
            if (
                verify_capture_latched
                and not close_capture_confirmed
                and not verify_grasp_armed
                and verify_capture_wait_steps >= 3
            ):
                # The first bilateral sample can be a closing impact. If the
                # measured-aperture hold does not reproduce it after settling,
                # release the latch and resume bounded Cartesian recentering.
                verify_capture_latched = False
                verify_capture_wait_steps = 0
                grasp_jaw_hold = float(current[5].item())
                grasp_jaw_reference = grasp_jaw_hold
            # Let force control and Cartesian recentering settle first. Two
            # strong bilateral samples arm one fixed, modest lift target; the
            # state machine still rejects the grasp if contact is not retained.
            if (
                (verify_bilateral_steps >= 3 or close_capture_confirmed)
                and bilateral_capture
                and not verify_grasp_armed
                and verify_capture_wait_steps >= 3
            ):
                verify_grasp_armed = True
                just_armed_proof_lift = True
                # Contact forces can prevent the arm from exactly reaching its
                # Cartesian hold target while IK continues integrating in
                # command space.  Discard that accumulated command error before
                # the proof lift; otherwise the first lift samples include a
                # lateral transient that can peel the weak finger off the cube.
                commanded_joints = _numpy(current).astype(np.float32).copy()
                grasp_hold_pose = ee_position.copy()
                verify_object_start_z = float(scene[active_name].data.root_pos_w[0, 2].item())
            if verify_grasp_armed:
                # Preserve the command-space integration that starts from the
                # one-time rebase above. Rebasing on measured joints on every
                # tick caps the gravity-loaded arm at one 5 mrad servo step;
                # the failed workspace runs then moved the EE only 1.6 mm and
                # never achieved the required 5 mm physical proof lift.
                commanded_joints, verify_lift_height = advance_proof_lift_command(
                    commanded_joints,
                    _numpy(current),
                    verify_lift_height,
                    just_armed=just_armed_proof_lift,
                    contact_retained=bilateral_capture,
                )
            target = grasp_hold_pose + np.asarray((0.0, 0.0, verify_lift_height))
            jaw = grasp_jaw_hold if grasp_jaw_hold is not None else closed_jaw
            gripper_control = "verify_hold"
        elif phase is OraclePhase.LIFT:
            # Continue from the proof-lift height without a target discontinuity.
            # A 0.25 mm/control-step ramp stays within the demonstrated stable
            # contact envelope and still reaches 8 cm inside the phase timeout.
            lift_height = min(
                0.08, verify_lift_height + 0.0000625 * (machine.phase_steps + 1)
            )
            transport_lift_target_m = lift_height - verify_lift_height
            phase_motion_complete = lift_height >= 0.08
            object_world_position = _numpy(
                scene[active_name].data.root_pos_w[0]
            )
            if lift_object_start_position is None:
                lift_object_start_position = object_world_position.copy()
            target = lift_object_start_position + np.asarray(
                (0.0, 0.0, lift_height - verify_lift_height),
                dtype=np.float32,
            )
            control_point_position = object_world_position
            control_point_offset_world = object_world_position - ee_position
            jaw = grasp_jaw_hold if grasp_jaw_hold is not None else closed_jaw
            # Rebase exactly once when transport begins, then retain command-
            # space integration. The former per-tick rebase moved a validated
            # 30 mm grasp only 35.7 mm before the 40 s LIFT timeout instead of
            # reaching the requested 80 mm clearance.
            commanded_joints = cartesian_motion_command_base(
                commanded_joints,
                _numpy(current),
                entering_motion=machine.phase_steps == 0,
            )
        elif phase is OraclePhase.PREPLACE:
            object_world_position = _numpy(
                scene[active_name].data.root_pos_w[0]
            )
            if transport_object_target is None:
                commanded_joints = _numpy(current).astype(np.float32).copy()
                # Move only as far onto the target pad as success requires. The
                # SO-101 has five arm DoF and cannot retain this physical
                # wrist posture at an unnecessarily distant pad-centre
                # waypoint. Preserve the already-proven lift height and pick
                # the nearest point 10 mm inside the valid XY region. Use the
                # box half-diagonal so the footprint stays inside even if the
                # cube changes which face is down during transport.
                conservative_footprint_radius = 0.5 * float(
                    np.linalg.norm(object_spec["dimensions_m"])
                )
                valid_half_extent = (
                    0.5 * target_dimensions[:2]
                    - conservative_footprint_radius
                    - 0.005
                )
                interior_margin = np.minimum(
                    0.010, 0.5 * valid_half_extent
                )
                lower = target_position[:2] - valid_half_extent + interior_margin
                upper = target_position[:2] + valid_half_extent - interior_margin
                transport_object_target = np.asarray(
                    (
                        np.clip(object_world_position[0], lower[0], upper[0]),
                        np.clip(object_world_position[1], lower[1], upper[1]),
                        object_world_position[2],
                    ),
                    dtype=np.float32,
                )
            # Preserve vertical clearance over the raised target pad. Use
            # hysteresis so lateral transport pauses whenever load-induced sag
            # erodes that clearance, then resumes after recovering margin.
            if object_world_position[2] < 0.086:
                if not transport_recovering_height:
                    transport_recovery_xy = object_world_position[:2].copy()
                    # Start recovery from the measured configuration, then
                    # retain the integrated command until clearance returns.
                    commanded_joints = _numpy(current).astype(np.float32).copy()
                transport_recovering_height = True
            elif (
                transport_recovering_height
                and object_world_position[2] >= 0.092
            ):
                transport_recovering_height = False
                transport_recovery_xy = None
                commanded_joints = _numpy(current).astype(np.float32).copy()
            if transport_recovering_height:
                if transport_recovery_xy is None:
                    transport_recovery_xy = object_world_position[:2].copy()
                recovery_object_target = np.asarray(
                    (
                        transport_recovery_xy[0],
                        transport_recovery_xy[1],
                        0.105,
                    ),
                    dtype=np.float32,
                )
                target = (
                    recovery_object_target
                    + ee_position
                    - object_world_position
                )
            else:
                target = (
                    transport_object_target
                    + ee_position
                    - object_world_position
                )
            jaw = grasp_jaw_hold if grasp_jaw_hold is not None else closed_jaw
            # Transport with the same measured-joint resolved-rate control as
            # LIFT.  Accumulating a distant Cartesian target in command space
            # lets the arm target run ahead of its measured pose and produces
            # a lateral acceleration that shears the cube from the fingers.
        elif phase is OraclePhase.PLACE_DESCEND:
            if not placement_descent_initialized:
                placement_descent_initialized = True
                commanded_joints = _numpy(current).astype(np.float32).copy()
            if placement_grasp_offset is None:
                placement_grasp_offset = (
                    ee_position - _numpy(scene[active_name].data.root_pos_w[0])
                )
            release_object_position = np.asarray(
                so101_release_object_target(
                    transport_object_target,
                    release_position[2],
                ),
                dtype=np.float32,
            )
            current_object_position = _numpy(
                scene[active_name].data.root_pos_w[0]
            )
            target = (
                release_object_position
                + ee_position
                - current_object_position
            )
            jaw = grasp_jaw_hold if grasp_jaw_hold is not None else closed_jaw
        elif phase is OraclePhase.OPEN:
            if release_hold_pose is None:
                release_hold_pose = ee_position.copy()
            target = release_hold_pose
            jaw = open_jaw
        elif phase is OraclePhase.SETTLE:
            if release_hold_pose is None:
                release_hold_pose = ee_position.copy()
            target = np.asarray(
                settle_release_separation_target(
                    release_hold_pose,
                    machine.phase_steps,
                    control_hz=schedule.control_hz,
                ),
                dtype=np.float32,
            )
            jaw = open_jaw
        else:
            target = (
                release_hold_pose
                if release_hold_pose is not None
                else ee_position
            ) + np.asarray((0.0, 0.0, 0.09))
            jaw = open_jaw

        if phase not in {OraclePhase.CLOSE, OraclePhase.VERIFY_CONTACT}:
            gripper_control = "phase_target"
        if jaw_force_action is not None:
            gripper_control = f"force_{jaw_force_action}"
        if recenter_active:
            gripper_control += "+recenter"
        if control_step == 0:
            print(
                f"SO101_ORACLE_START phase={phase.value} "
                f"ee={_numpy(robot.data.body_link_pose_w.torch[0, body_index, :3]).tolist()} "
                f"target={np.asarray(target).tolist()}",
                flush=True,
            )
        if phase is OraclePhase.PREGRASP and machine.phase_steps == 0:
            print(
                f"SO101_ORACLE_PREGRASP_START ee={_numpy(robot.data.body_link_pose_w.torch[0, body_index, :3]).tolist()} "
                f"target={np.asarray(target).tolist()}",
                flush=True,
            )
        if phase is OraclePhase.HOME:
            posture_target = SO101_HOME_JOINTS[3:5]
            action = torch.tensor(
                SO101_HOME_JOINTS[None, :], dtype=torch.float32, device=device
            )
        else:
            if grasp_hold_posture is not None and phase in {
                OraclePhase.CLOSE,
                OraclePhase.VERIFY_CONTACT,
                OraclePhase.LIFT,
                OraclePhase.PREPLACE,
                OraclePhase.PLACE_DESCEND,
                OraclePhase.OPEN,
                OraclePhase.SETTLE,
            }:
                # Freeze the two wrist joints at first physical contact. A
                # direction-only quaternion task leaves rotation around the
                # approach axis unconstrained; that roll drift was enough to
                # lever the cube out of the rotary jaw during VERIFY.
                posture_target = grasp_hold_posture
            else:
                posture_target = grasp_posture
            # Position plus the two calibrated wrist targets is the reachable
            # 5-DOF task.  A fixed world quaternion over-constrains translation
            # and was the cause of the pre-v6 single-finger collision.
            active_orientation_target = None
            action = _ik_action(
                robot,
                ee_frame,
                target,
                commanded_joints,
                body_index,
                device,
                posture_target=posture_target,
                orientation_target=active_orientation_target,
                # Wrist posture regularization is useful while approaching,
                # but after contact it can trade Cartesian accuracy for a
                # null-space motion that shears a small cube out of the jaw.
                nullspace_gain=(
                    0.20
                    if phase
                    in {
                        OraclePhase.PREPLACE,
                        OraclePhase.PLACE_DESCEND,
                    }
                    else 0.0
                    if phase
                    in {
                        OraclePhase.CLOSE,
                        OraclePhase.VERIFY_CONTACT,
                        OraclePhase.LIFT,
                        OraclePhase.OPEN,
                        OraclePhase.SETTLE,
                        OraclePhase.RETREAT,
                    }
                    else 0.20
                ),
                max_joint_step=contact_constrained_joint_step_limit(
                    0.02 * schedule.recording_hz / schedule.control_hz,
                    proof_lift_armed=(
                        phase is OraclePhase.VERIFY_CONTACT
                        and verify_grasp_armed
                    ),
                ),
                lock_wrist=(
                    grasp_hold_posture is not None
                    and phase
                    in {
                        OraclePhase.CLOSE,
                        OraclePhase.VERIFY_CONTACT,
                        OraclePhase.LIFT,
                        OraclePhase.OPEN,
                        OraclePhase.SETTLE,
                    }
                ),
                control_point_position=control_point_position,
                control_point_offset_world=control_point_offset_world,
                position_weights=(
                    (1.0, 1.0, 4.0)
                    if phase
                    in {OraclePhase.PREPLACE, OraclePhase.PLACE_DESCEND}
                    else (1.0, 1.0, 2.0)
                    if phase is OraclePhase.LIFT
                    else None
                ),
            )
            if grasp_hold_posture is not None and phase in {
                OraclePhase.CLOSE,
                OraclePhase.VERIFY_CONTACT,
                OraclePhase.LIFT,
                OraclePhase.OPEN,
                OraclePhase.SETTLE,
            }:
                # Cartesian transport is intentionally rebased on measured
                # joints every frame to avoid shearing the grasp. That leaves
                # only a 0.02 rad servo error on the wrists, which is too weak
                # to resist unilateral fingertip loads. Apply the latched
                # contact posture directly, while retaining the same bounded
                # 0.30 rad actuator-error envelope used by the IK path.
                action[0, 3] = bounded_position_target(
                    grasp_hold_posture[0], float(current[3].item()), 0.30
                )
                action[0, 4] = bounded_position_target(
                    grasp_hold_posture[1], float(current[4].item()), 0.30
                )
            elif grasp_hold_posture is not None and phase in {
                OraclePhase.PREPLACE,
                OraclePhase.PLACE_DESCEND,
            }:
                # Transport needs some wrist freedom to stay in the 5-DoF
                # reachable set, but an unconstrained position-only solve can
                # flip Wrist_Pitch to its limit and rotate the grasp offset
                # faster than the object-space servo can follow. Keep both
                # wrist axes inside a small neighborhood of the physically
                # validated capture posture.
                wrist_excursion = 0.35
                action[0, 3] = torch.clamp(
                    action[0, 3],
                    float(grasp_hold_posture[0] - wrist_excursion),
                    float(grasp_hold_posture[0] + wrist_excursion),
                )
                action[0, 4] = torch.clamp(
                    action[0, 4],
                    float(grasp_hold_posture[1] - wrist_excursion),
                    float(grasp_hold_posture[1] + wrist_excursion),
                )
        action[0, 5] = jaw
        commanded_joints = _numpy(action[0]).astype(np.float32).copy()
        has_contact = _contact(contact)
        bilateral_contact = _bilateral_contact(contact)
        contact_forces = _cube_contact_forces(contact)
        descent_cube_contact = so101_cube_contact_handoff(*contact_forces)
        if phase is OraclePhase.DESCEND and descent_cube_contact:
            # DESCEND can hand off on a one-tick fingertip contact. Remember
            # that side before CLOSE begins so a zero-force sample on the next
            # tick still has a deterministic aperture-recenter direction.
            capture_recenter_side = so101_recenter_contact_memory(
                *contact_forces,
                capture_recenter_side,
                minimum_force_n=grasp_machine.minimum_contact_force_n,
            )["side"]
        object_pose = _numpy(scene[active_name].data.root_pose_w[0])
        if (
            phase in {OraclePhase.DESCEND, OraclePhase.CLOSE}
            and max(contact_forces) >= grasp_machine.minimum_contact_force_n
        ):
            pre_capture_recenter_object_reference = (
                latch_pre_capture_recenter_object_reference(
                    object_pose[:3],
                    pre_capture_recenter_object_reference,
                )
            )
        gripper_pose = _numpy(
            robot.data.body_link_pose_w.torch[0, body_index]
        )
        object_in_gripper = point_in_local_frame(
            gripper_pose, object_pose[:3]
        )
        if grasp_relative_reference is None and min(contact_forces) >= 0.10:
            grasp_relative_reference = object_in_gripper.copy()
        relative_translation_error = (
            float("inf")
            if grasp_relative_reference is None
            else float(
                np.linalg.norm(object_in_gripper - grasp_relative_reference)
            )
        )
        relative_speed = (
            float("inf")
            if previous_object_in_gripper is None
            else float(
                np.linalg.norm(object_in_gripper - previous_object_in_gripper)
                * schedule.control_hz
            )
        )
        previous_object_in_gripper = object_in_gripper.copy()
        cube_z = float(scene[active_name].data.root_pos_w[0, 2].item())
        # The cube root starts at half-height above the table. Five additional
        # millimetres means its bottom face has physically cleared the table.
        cube_lifted = cube_z > object_position[2] + 0.005
        grasp_proof_lifted = (
            verify_grasp_armed
            and verify_object_start_z is not None
            and cube_z > verify_object_start_z + 0.003
        )
        state_machine_cube_lifted = (
            grasp_proof_lifted
            if phase is OraclePhase.VERIFY_CONTACT
            else cube_lifted
        )
        # A closing fingertip can briefly kick the cube more than 5 mm before
        # a grasp exists.  That impact is not lift proof and must not arm the
        # later drop detector.  Only VERIFY's contact-retaining proof lift (or
        # an already-entered transport phase) establishes that the cube was
        # genuinely carried above the table.
        if phase is OraclePhase.VERIFY_CONTACT:
            cube_was_lifted = cube_was_lifted or grasp_proof_lifted
        elif phase in {
            OraclePhase.LIFT,
            OraclePhase.PREPLACE,
            OraclePhase.PLACE_DESCEND,
            OraclePhase.OPEN,
            OraclePhase.SETTLE,
            OraclePhase.RETREAT,
        }:
            cube_was_lifted = cube_was_lifted or cube_lifted
        # Static-body filtering is not supported by PhysX GPU contact-pair
        # reporting. Table clearance is instead guaranteed by the validated
        # grasp posture and Cartesian waypoint envelope.
        unexpected_collision = unsafe_so101_approach_contact(
            phase,
            has_contact,
            descent_fraction,
            # The fixed 45-degree yaw makes the leading corner of a 40 mm cube
            # reach the moving finger earlier than a 30 mm cube. The frozen
            # size-aware threshold keeps PREGRASP contact unsafe while allowing
            # low-force DESCEND contact to hand off to calibrated slow close.
            minimum_safe_descent_fraction=so101_minimum_safe_descent_fraction(
                object_spec["dimensions_m"][0]
            ),
        )
        cube_dropped = (
            cube_was_lifted
            and phase
            in {OraclePhase.VERIFY_CONTACT, OraclePhase.LIFT, OraclePhase.PREPLACE}
            and not has_contact
            and cube_z < object_position[2] + 0.002
        )
        grasp_evidence = GraspEvidence(
                left_force_n=float(contact_forces[0]),
                right_force_n=float(contact_forces[1]),
                # The SO-101 fingers are long along local Z. Enclosure needs
                # tight alignment across the aperture plane; physical depth
                # is validated separately by bilateral force, rigidity and
                # proof lift. A 3-D norm incorrectly rejected stable fingertip
                # captures at short-reach workspace positions.
                aperture_aligned=capture_aperture_laterally_aligned(
                    object_in_gripper,
                    capture_object_in_gripper,
                ),
                capture_admissible=capture_admissible,
                contact_geometry_valid=contact_geometry_valid,
                relative_translation_error_m=relative_translation_error,
                relative_speed_mps=relative_speed,
                proof_lift_m=(
                    0.0
                    if verify_object_start_z is None
                    else cube_z - verify_object_start_z
                ),
                collision=unexpected_collision,
            )
        if phase in {
            OraclePhase.DESCEND,
            OraclePhase.CLOSE,
            OraclePhase.VERIFY_CONTACT,
        }:
            grasp_decision = grasp_machine.step(grasp_evidence)
        else:
            grasp_decision = GraspDecision(
                phase=grasp_machine.phase,
                entered_phase=False,
                rebase_joint_command=False,
                hold_cartesian_pose=False,
                failure_reason=grasp_machine.failure_reason,
            )
        if grasp_decision.rebase_joint_command:
            commanded_joints = _numpy(current).astype(np.float32).copy()
            if grasp_hold_pose is not None:
                grasp_hold_pose = gripper_pose[:3].copy()
        if grasp_decision.rebase_relative_tracking:
            # The first weak bilateral sample can occur while slow-close or
            # unilateral recenter is still moving the cube. Rigidity is
            # measured from a state-machine-approved stable capture candidate
            # and again from the exact capture entering BILATERAL_SETTLE, not
            # permanently from that earlier transient.
            grasp_relative_reference = object_in_gripper.copy()
            previous_object_in_gripper = object_in_gripper.copy()
        if (
            grasp_decision.entered_phase
            and grasp_decision.phase is GraspPhase.BILATERAL_SETTLE
        ):
            # Freeze the exact physical capture before sending the CLOSE
            # command computed earlier in this tick. Continuing to close for
            # even one 120 Hz step can turn bilateral contact into a one-sided
            # squeeze on the rotary jaw.
            # A zero-error position target lets contact force relax to zero
            # immediately. Retain the full 8 mrad buildup target unless both
            # fingers are proof-ready and the stronger side has at least 1 N
            # overload margin, where 2 mrad relief prevents over-compression.
            # The immutable v0.2.0 c22 trace entered around 5.4/4.3 N and
            # passed with relief. c26 around 4.7/4.5 N shed both contacts when
            # a 4 mrad transition opened the jaw; its subsequent correction is
            # independently slowed by the size-aware settle controller. Force
            # control and all proof gates remain otherwise unchanged.
            grasp_jaw_hold = rotary_jaw_capture_hold_target(
                float(current[5].item()),
                closed_position=closed_jaw,
                open_position=open_jaw,
                relative_speed_mps=relative_speed,
                preload_rad=capture_hold_preload_for_force(
                    *finger_forces,
                    proof_entry_force_n=(
                        grasp_machine.minimum_proof_entry_force_n
                    ),
                ),
            )
            grasp_jaw_reference = grasp_jaw_hold
            grasp_hold_pose = gripper_pose[:3].copy()
            grasp_hold_nominal_pose = grasp_hold_pose.copy()
            capture_object_minus_grasp = object_pose[:3] - gripper_pose[:3]
            grasp_hold_nominal_y = float(grasp_hold_pose[1])
            grasp_hold_posture = _numpy(current[3:5]).astype(np.float32).copy()
            verify_capture_latched = True
            # Discard the arm IK command already computed earlier in this
            # transition tick, then apply only the bounded jaw preload.
            action[0, :5] = current[:5]
            action[0, 5] = grasp_jaw_hold
            commanded_joints = _numpy(current).astype(np.float32).copy()
            commanded_joints[5] = grasp_jaw_hold
            gripper_control = "bilateral_capture_preload_hold"
        if grasp_decision.phase is GraspPhase.PROOF_LIFT:
            close_capture_confirmed = True
        if grasp_decision.phase is GraspPhase.FAILED:
            machine.fail(grasp_decision.failure_reason or "grasp_failed")

        if recovery_oracle_previous_target is not None:
            bounded, recovery_oracle_safety = slew_recovery_oracle_target(
                recovery_oracle_previous_target,
                _numpy(action[0]),
                recovery_oracle_maximum_delta,
            )
            action = torch.tensor([bounded], dtype=torch.float32, device=device)
            recovery_oracle_previous_target = bounded.copy()
            if not grasp_decision.rebase_joint_command:
                commanded_joints = bounded.copy()

        if recovery_snapshot is not None:
            physics_command_rows.append(
                {
                    "control_step": control_step,
                    "timestamp_seconds": control_step / schedule.control_hz,
                    "phase": phase.value,
                    "grasp_phase": grasp_decision.phase.value,
                    "action_joint_positions": _numpy(action[0]).tolist(),
                    "command_safety": copy.deepcopy(recovery_oracle_safety),
                }
            )

        stable_grasp_contact = (
            grasp_decision.phase
            in {GraspPhase.PROOF_LIFT, GraspPhase.VALIDATED}
            if phase is OraclePhase.CLOSE
            else grasp_decision.phase is GraspPhase.VALIDATED
        )

        if schedule.should_record(control_step):
            frame = schedule.frame_index(control_step)
            front = _image(scene["front_camera"], device)
            wrist = (
                _image(scene["wrist_camera"], device)
                if args_cli.enable_wrist_camera
                else None
            )
            if live_publisher is not None and live_publisher.preview_due():
                preview = io.BytesIO()
                Image.fromarray(front, mode="RGB").save(
                    preview, format="JPEG", quality=75, optimize=False
                )
                live_publisher.publish_preview(preview.getvalue())
            row = _write_frame(
                root,
                frame,
                _numpy(current),
                _numpy(action[0]),
                front,
                wrist,
            )
            row["phase"] = phase.value
            row["control_step"] = control_step
            row["grasp_phase"] = grasp_decision.phase.value
            row["contact"] = {
                "cube_contact": has_contact,
                "bilateral_cube_contact": bilateral_contact,
                "unexpected_collision": unexpected_collision,
                "sensors": _contact_debug(contact),
            }
            row["contact_forces_newtons"] = {
                "left_finger": float(contact_forces[0]),
                "right_finger": float(contact_forces[1]),
            }
            row["grasp_evidence"] = {
                "object_in_gripper_m": object_in_gripper.tolist(),
                "relative_translation_error_m": relative_translation_error,
                "relative_speed_mps": relative_speed,
                "proof_lift_m": (
                    0.0
                    if verify_object_start_z is None
                    else cube_z - verify_object_start_z
                ),
            }
            if recovery_oracle_safety is not None:
                row["recovery_oracle_command_safety"] = recovery_oracle_safety
            row["truth"] = {
                "object_root_pose_xyzw": object_pose.tolist(),
                "object_linear_velocity_mps": _numpy(
                    scene[active_name].data.root_lin_vel_w[0]
                ).tolist(),
                "gripper_link_pose_xyzw": gripper_pose.tolist(),
                "jaw_link_pose_xyzw": _numpy(
                    robot.data.body_link_pose_w.torch[
                        0, robot.body_names.index("jaw")
                    ]
                ).tolist(),
                "gripper_control": gripper_control,
                "recenter_contact_memory_side": capture_recenter_side,
                "recenter_used_contact_memory": recenter_used_memory,
                "pre_capture_recenter_object_reference_m": (
                    None
                    if pre_capture_recenter_object_reference is None
                    else pre_capture_recenter_object_reference.tolist()
                ),
                "contact_geometry_valid": contact_geometry_valid,
                "relative_grasp_recenter_active": relative_recenter_active,
                "descent_lateral_correction_m": descent_lateral_correction,
                "grasp_lateral_correction_m": (
                    0.0
                    if grasp_hold_nominal_y is None
                    else float(grasp_hold_pose[1] - grasp_hold_nominal_y)
                ),
                "grasp_xy_correction_m": (
                    [0.0, 0.0]
                    if grasp_hold_nominal_pose is None
                    else (
                        np.asarray(grasp_hold_pose[:2])
                        - np.asarray(grasp_hold_nominal_pose[:2])
                    ).tolist()
                ),
                "approach_jaw_target_rad": approach_jaw,
                "capture_aperture_reference_local_m": (
                    capture_object_in_gripper.tolist()
                ),
                "pre_capture_recenter_aperture_reference_local_m": (
                    pre_capture_recenter_aperture_reference.tolist()
                ),
                "proof_lift_target_m": float(verify_lift_height),
                "transport_lift_target_m": float(transport_lift_target_m),
                "transport_lift_actual_m": (
                    0.0
                    if lift_object_start_position is None
                    else float(object_pose[2] - lift_object_start_position[2])
                ),
            }
            rows.append(row)
        posture_tolerance = (
            0.20
            if phase is OraclePhase.PREGRASP
            else 0.10
            if phase is OraclePhase.DESCEND
            else 0.05
        )
        posture_ready = (
            abs(float(current[3].item()) - float(posture_target[0]))
            < posture_tolerance
            and abs(float(current[4].item()) - float(posture_target[1]))
            < posture_tolerance
        )
        if phase in {OraclePhase.HOME, OraclePhase.RETREAT}:
            orientation_ready = True
        elif active_orientation_target is not None:
            orientation_ready = float(
                np.linalg.norm(
                    quaternion_direction_error(
                        active_orientation_target,
                        gripper_pose[3:7],
                    )
                )
            ) < 0.10
        else:
            orientation_ready = posture_ready
        position_tolerance = 0.003 if phase is OraclePhase.DESCEND else 0.012
        reached_position = (
            ee_position
            if control_point_position is None
            else control_point_position
        )
        position_reached = (
            float(
                np.linalg.norm(
                    reached_position - target
                )
            )
            < position_tolerance
            and orientation_ready
            and phase_motion_complete
        )
        cube_in_target = oriented_box_footprint_inside_target(
            object_pose[:3],
            object_spec["dimensions_m"],
            object_pose[3:7],
            target_position,
            target_dimensions,
            margin_m=0.005,
        )
        obs = OracleObservation(
            # An open two-finger gripper can surround the cube without either
            # finger touching it. Reaching the calibrated side-entry waypoint
            # permits CLOSE; that phase still requires sustained bilateral
            # contact before lift verification. Early contact also stops entry.
            reached_target=(
                (
                    position_reached
                    and phase
                    not in {
                        OraclePhase.PREPLACE,
                        OraclePhase.PLACE_DESCEND,
                    }
                )
                # The generic contact signal requires 2 N and may include
                # non-cube contacts. Stop insertion on the first cube-filtered
                # fingertip contact so the jaw cannot push the object away
                # while waiting for collision-level force.
                or (phase is OraclePhase.DESCEND and descent_cube_contact)
                # Under load, the 5-DOF arm can retain a small Cartesian pose
                # residual even after the cube has ample target-pad clearance.
                # Advance on the physical transport condition, not an
                # unreachable unloaded end-effector target.
                or (
                    phase is OraclePhase.LIFT
                    and bilateral_contact
                    and cube_z >= object_position[2] + 0.050
                )
                or (
                    phase is OraclePhase.PREPLACE
                    and bilateral_contact
                    and cube_in_target
                    and transport_object_target is not None
                    and float(
                        np.linalg.norm(
                            _numpy(
                                scene[active_name].data.root_pos_w[0, :2]
                            )
                            - transport_object_target[:2]
                        )
                    )
                    <= 0.008
                )
                or (
                    phase is OraclePhase.PLACE_DESCEND
                    and cube_in_target
                    and cube_z <= float(release_position[2]) + 0.005
                )
            ),
            has_contact=stable_grasp_contact,
            cube_lifted=state_machine_cube_lifted,
            cube_in_target=cube_in_target,
            gripper_released=abs(float(current[5].item()) - open_jaw) < 0.08,
            cube_stable=float(torch.linalg.vector_norm(scene[active_name].data.root_lin_vel_w[0]).item()) < 0.03,
            collision=unexpected_collision,
            cube_dropped=cube_dropped,
        )
        machine.step(obs)
        env.step(action)
        if machine.phase in {OraclePhase.SUCCEEDED, OraclePhase.FAILED}:
            break
    success = machine.phase is OraclePhase.SUCCEEDED
    if not success:
        ee = _numpy(robot.data.body_link_pose_w.torch[0, body_index, :3])
        jacobian = _numpy(robot.data.body_link_jacobian_w.torch[0, body_index - 1, :3, :5])
        terminal_error = np.asarray(target) - ee
        terminal_delta = damped_least_squares(jacobian, terminal_error, damping=0.06)
        body_poses = {
            name: _numpy(robot.data.body_link_pose_w.torch[0, index]).tolist()
            for index, name in enumerate(robot.body_names)
            if name in {"wrist", "gripper", "jaw"}
        }
        print(
            "SO101_ORACLE_DEBUG "
            f"phase={machine.phase.value} reason={machine.failure_reason} "
            f"ee={ee.tolist()} "
            f"target={np.asarray(target).tolist()} "
            f"error={terminal_error.tolist()} delta={terminal_delta.tolist()} "
            f"joints={_numpy(robot.data.joint_pos[0]).tolist()} "
            f"targets={_numpy(robot.data.joint_pos_target[0]).tolist()} "
            f"contact={_contact(contact)} cube={_numpy(scene[active_name].data.root_pos_w[0]).tolist()} "
            f"contact_forces={json.dumps(_contact_debug(contact), sort_keys=True)} "
            f"body_poses={json.dumps(body_poses, sort_keys=True)}",
            flush=True,
        )
    if recovery_snapshot is not None:
        if demonstration is None or not physics_command_rows:
            raise RuntimeError("recovery command trace was not captured")
        trace_path = root / "oracle-commands.jsonl"
        trace_bytes = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in physics_command_rows
        ).encode("utf-8")
        trace_path.write_bytes(trace_bytes)
        trace = intervention_command_trace(
            path=trace_path.name,
            sha256=hashlib.sha256(trace_bytes).hexdigest(),
            control_hz=int(schedule.control_hz),
            sample_count=len(physics_command_rows),
            first_control_step=int(physics_command_rows[0]["control_step"]),
            last_control_step=int(physics_command_rows[-1]["control_step"]),
            joint_order=LEROBOT_JOINT_NAMES,
        )
        demonstration["intervention"]["command_trace"] = trace
        handoff_path = root / "handoff.json"
        handoff = _read_json(handoff_path)
        handoff["demonstration"] = copy.deepcopy(demonstration)
        handoff["command_trace"] = copy.deepcopy(trace)
        _write_json(handoff_path, handoff)
        run_state["recovery"]["command_trace"] = copy.deepcopy(trace)
        _write_json(root / "run-state.json", run_state)
    scene_object = {
        "shape": object_spec["shape"],
        "asset_id": object_spec["asset_id"],
        "dimensions_m": object_spec["dimensions_m"],
        "initial_pose": {
            "position_m": object_spec["position_m"],
            "orientation_xyzw": object_spec["orientation_xyzw"],
        },
        "rgba": object_spec["rgba"],
        "mass_kg": object_spec["mass_kg"],
        "static_friction": object_spec["static_friction"],
        "dynamic_friction": object_spec["dynamic_friction"],
        "restitution": object_spec["restitution"],
        "mass_audit": copy.deepcopy(mass_audit),
        "reset_support_audit": copy.deepcopy(reset_support_audit),
    }
    scene_target = {
        "target_id": "green_rectangular_pad_v1",
        "asset_id": "green_rectangular_pad_v1",
        "entity_type": "pad",
        "representation": "procedural",
        "shape": "cuboid",
        "relation": "on",
        "type": "raised_rectangular_pad",
        "position_m": target_position.tolist(),
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "dimensions_m": target_dimensions.tolist(),
        "footprint_margin_m": 0.005,
        "rgba": [0.08, 0.70, 0.20, 1.0],
    }
    requested_variation = copy.deepcopy(trial["requested"])
    if "entities" not in requested_variation:
        requested_variation = bind_scene_entities(requested_variation, scene_target)
    resolved_variation = bind_scene_entities(object_spec, scene_target)
    resolved_variation["entities"]["pick_object"]["physics"]["mass_audit"] = (
        copy.deepcopy(mass_audit)
    )
    episode_camera_profile = (
        (trial.get("camera_profile") or {}).get("resolved_profile")
        or camera_profile
    )
    resolved_camera_records = (
        _runtime_camera_records(scene, episode_camera_profile)
        if episode_camera_profile is not None
        else None
    )
    if resolved_camera_records is not None:
        if v010_context is not None:
            video_artifacts = {
                camera_id: seal_rgb_video(
                    root,
                    camera_id=camera_id,
                    frame_count=len(rows),
                    fps=int(schedule.recording_hz),
                )
                for camera_id in ("front", "wrist")
            }
            for camera_record in resolved_camera_records:
                camera_record["video_artifact"] = video_artifacts[
                    camera_record["camera_id"]
                ]
        _write_json(
            root / "camera-evidence.json",
            {
                "profile": copy.deepcopy(episode_camera_profile),
                "resolved_cameras": copy.deepcopy(resolved_camera_records),
                "same_control_tick": True,
                "recording_stride": schedule.recording_stride,
            },
        )
    legacy_metadata = {
        "schema_version": "farpoint.episode.v3",
        "identity": {"episode_id": root.name, "trial_id": trial["trial_id"], "task_id": "so101_cube_pick_place", "split": trial["split"], "episode_seed": environment_seed},
        "provenance": {"collection_id": collection_id, "git_commit": git_commit, "simulator": "Isaac Sim", "simulator_image": "nvcr.io/nvidia/isaac-sim:6.0.0", "physics_engine": "PhysX", "asset_commit": "ce807d99724cb65671abec01f908a2fcb4a6eab7", "variation_seed": int(trial["seed"]), "attempt_seed": episode_seed, "environment_seed": environment_seed},
        "task": {"task_id": "so101_cube_pick_place", "instruction": f"Pick up the {object_spec['shape']} and place it on the green target pad.", "object_shape": object_spec["shape"], "success_criteria_id": "contact_pick_place_footprint_v2", "manipulated_entity_id": "pick_object", "target_entity_id": "placement_target", "acceptance_region_id": "placement_region"},
        "embodiment": {"robot": "so101", "gripper": "so101_jaw", "arm_dof": 5, "gripper_dof": 1, "controller": "contact_aware_local_frame_dls_v0", "control_mode": "joint_position", "grasp_mode": "contact_only", "joint_mapping": mapping_metadata(), "finger_physics_material": {"static_friction": SO101_GRIPPER_STATIC_FRICTION, "dynamic_friction": SO101_GRIPPER_DYNAMIC_FRICTION, "restitution": SO101_GRIPPER_RESTITUTION, "friction_combine_mode": "max"}},
        "scene": {"coordinate_frame": "isaac_world", "object": scene_object, "target": scene_target, "entities": list(resolved_variation["entities"].values()), "cameras": (resolved_camera_records if resolved_camera_records is not None else ([{"name": "observation.images.front", "resolution": [640, 480]}] + ([{"name": "observation.images.wrist", "resolution": [640, 480]}] if args_cli.enable_wrist_camera else []))), "lighting_profile_id": "fixed_default"},
        "variation": {"schema_version": "farpoint.variation.v3", "variation_id": trial["variation_id"], "varied_axes": copy.deepcopy(trial["varied_axes"]), "frozen_axes": copy.deepcopy(trial["frozen_axes"]), "requested": requested_variation, "resolved": resolved_variation, "split": trial["split"]},
        "recording": {"fps": schedule.recording_hz, "control_hz": schedule.control_hz, "recording_stride": schedule.recording_stride, "cameras": (["observation.images.front", "observation.images.wrist"] if args_cli.enable_wrist_camera else ["observation.images.front"]), "frame_count": len(rows), "state_features": list(LEROBOT_JOINT_NAMES), "action_features": list(LEROBOT_JOINT_NAMES), "state_unit": "radian", "action_unit": "radian", "sampling_semantics": "state_before_action_at_control_step; image_latest_30hz_render"},
        "outcome": {"success": success, "dataset_valid": bool(rows), "failure_category": None if success else "oracle", "failure_reason": None if success else machine.failure_reason},
    }
    if v010_context is None:
        metadata = legacy_metadata
    else:
        if resolved_camera_records is None:
            raise ValueError("episode v4 requires resolved front and wrist cameras")
        resolved_object_state = copy.deepcopy(object_spec)
        resolved_object_state["position_m"] = measured_object_position.tolist()
        resolved_object_state["orientation_xyzw"] = (
            measured_object_orientation.tolist()
        )
        resolved_object_state["mass_kg"] = float(
            mass_audit["physx_actual_mass_kg"]
        )
        joint_mapping = mapping_metadata()
        joint_mapping["joint_order"] = list(LEROBOT_JOINT_NAMES)
        metadata = build_so101_episode_v4(
            episode_id=root.name,
            campaign=v010_context["campaign"],
            segment=v010_context["segment"],
            plan=v010_context["plan"],
            trial=trial,
            attempt_seed=episode_seed,
            git_commit=git_commit,
            simulator_image_digest=v010_context["simulator_image_digest"],
            resolved_object=resolved_object_state,
            target=scene_target,
            table=v010_context["plan"]["table"],
            camera_records=resolved_camera_records,
            embodiment={
                "robot": "so101",
                "gripper": "so101_jaw",
                "arm_dof": 5,
                "gripper_dof": 1,
                "controller": "contact_aware_local_frame_dls_v0",
                "control_mode": "joint_position",
                "grasp_mode": "contact_only",
                "joint_mapping": joint_mapping,
            },
            frame_count=len(rows),
            control_hz=schedule.control_hz,
            success=success,
            dataset_valid=bool(rows),
            failure_category=None if success else "oracle",
            failure_reason=None if success else machine.failure_reason,
            physics_audit={
                "mass": copy.deepcopy(mass_audit),
                "reset_support": copy.deepcopy(reset_support_audit),
            },
            demonstration=demonstration,
        )
    errors = validate_contract(metadata)
    if errors:
        raise ValueError("invalid SO-101 episode metadata: " + "; ".join(errors))
    semantic_errors = validate_episode_semantics(metadata)
    if semantic_errors:
        raise ValueError(
            "invalid SO-101 episode semantics: " + "; ".join(semantic_errors)
        )
    _write_json(root / "metadata.json", metadata)
    (root / "observations.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    _write_json(root / "metrics.json", {"success": success, "dataset_valid": bool(rows), "failure_category": metadata["outcome"]["failure_category"], "failure_reason": metadata["outcome"]["failure_reason"], "observation_count": len(rows), "physics_audit": {"mass": mass_audit, "reset_support": reset_support_audit}})
    run_state["execution_status"] = "FINISHED"
    run_state["recording"]["frame_count"] = len(rows)
    run_state["outcome"] = copy.deepcopy(metadata["outcome"])
    _write_json(root / "run-state.json", run_state)
    return root.name, success, bool(rows), metadata["outcome"]["failure_category"], metadata["outcome"]["failure_reason"]


def main():
    from farpoint.object_variation import generate_variation_plan, load_variation_config

    config = load_variation_config(PROJECT_ROOT / "configs/variations/so101_cube_pick_place_v1.json")
    camera_profile = (
        load_camera_profile(args_cli.camera_profile)
        if args_cli.require_dual_camera
        else None
    )
    plan = _read_json(args_cli.plan) if args_cli.plan.exists() else generate_variation_plan(config)
    recovery_runtime = (
        load_recovery_runtime(args_cli.recovery_runtime)
        if args_cli.recovery_runtime is not None
        else None
    )
    v010_context = None
    if is_v010_episode_plan(plan):
        if not args_cli.require_dual_camera:
            raise ValueError("v0.1.0 collection requires --require-dual-camera")
        if args_cli.campaign_root is None:
            raise ValueError("v0.1.0 collection requires campaign identity arguments")
        campaign = _read_json(args_cli.campaign_root / "campaign.json")
        segment = _read_json(
            args_cli.campaign_root
            / "segments"
            / args_cli.segment_id
            / "segment.json"
        )
        if campaign.get("campaign_id") != args_cli.campaign_id:
            raise ValueError("v0.1.0 campaign id does not match campaign.json")
        simulator_image_digest = os.environ.get(
            "FARPOINT_SIMULATOR_IMAGE_DIGEST", ""
        )
        if not simulator_image_digest.startswith("sha256:"):
            raise ValueError("v0.1.0 collection requires simulator image digest")
        v010_context = {
            "campaign": campaign,
            "segment": segment,
            "plan": plan,
            "simulator_image_digest": simulator_image_digest,
        }
    if recovery_runtime is not None:
        if v010_context is None:
            raise ValueError("recovery collection requires an episode v4 campaign plan")
        if not args_cli.require_dual_camera:
            raise ValueError("recovery collection requires synchronized front+wrist cameras")
        plan_variations = {trial["variation_id"] for trial in plan["trials"]}
        runtime_variations = {
            scene["variation_id"] for scene in recovery_runtime["scenes"]
        }
        if plan_variations != runtime_variations:
            raise ValueError(
                "recovery runtime scene bindings must exactly match plan variations"
            )
    watchdog_policy = (
        load_watchdog_policy(args_cli.watchdog_policy)
        if args_cli.watchdog_policy is not None
        else None
    )
    args_cli.plan.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args_cli.plan, plan)
    if args_cli.manifest.exists():
        manifest = load_manifest(args_cli.manifest, plan)
    else:
        if args_cli.gate_plan:
            manifest = create_gate_manifest(
                plan,
                collection_id=plan["plan_id"],
                git_commit=os.environ.get("FARPOINT_GIT_COMMIT", "unknown"),
            )
        elif args_cli.pilot_plan:
            manifest = create_pilot_manifest(
                plan,
                collection_id=plan["plan_id"],
                git_commit=os.environ.get("FARPOINT_GIT_COMMIT", "unknown"),
            )
        else:
            if plan.get("collection") and not args_cli.collection_id:
                raise ValueError("formal collection profiles require --collection-id")
            manifest = create_manifest(
                plan,
                collection_id=(
                    args_cli.collection_id or "so101_cube_pick_place_pilot"
                ),
                git_commit=os.environ.get("FARPOINT_GIT_COMMIT", "unknown"),
            )
    # Persist RUNNING before Isaac environment construction. A SIGINT/SIGTERM
    # during the expensive RTX startup must still leave a terminal manifest.
    write_manifest(args_cli.manifest, manifest)
    if live_publisher is not None:
        live_publisher.update_status(
            collection_id=manifest["collection_id"],
            target_successful_episodes=int(manifest["required_successes"]),
            startup_phase="environment_construction",
        )
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, raise_collection_signal_abort)
    env = None
    active_attempt = None
    try:
        # Isaac Lab 3.0 keeps the config entry point in the Gym registration,
        # but Gymnasium does not instantiate that config automatically.
        from farpoint_so101_env.env_cfg import SO101CubePickPlaceEnvCfg

        env_cfg = SO101CubePickPlaceEnvCfg()
        if is_v010_episode_plan(plan):
            target_dimensions = (plan.get("target") or {}).get("dimensions_m")
            if target_dimensions is not None:
                env_cfg.scene.target_pad.spawn.size = tuple(
                    float(value) for value in target_dimensions
                )
        # Seed the construction path as well as every reset. Isaac Lab warns
        # and may initialize manager state nondeterministically when this is None.
        env_cfg.seed = 0
        if not args_cli.enable_wrist_camera:
            # Both entries are removed before construction, so v0 neither
            # spawns nor renders the wrist sensor.
            env_cfg.scene.wrist_camera = None
            env_cfg.observations.policy.wrist_rgb = None
        if camera_profile is not None:
            camera_errors = camera_cfg_drift_errors(camera_profile, env_cfg.scene)
            if camera_errors:
                raise ValueError(
                    "Isaac camera config does not match profile: "
                    + "; ".join(camera_errors)
                )
        env = gym.make("Farpoint-SO101-PickPlace-Cube-v0", cfg=env_cfg).unwrapped
        if args_cli.diagnose_jacobian:
            run_jacobian_diagnostic(env)
            finish_diagnostic_manifest(manifest, "jacobian", succeeded=True)
            write_manifest(args_cli.manifest, manifest)
            return
        if args_cli.diagnose_grasp_postures:
            try:
                run_grasp_posture_diagnostic(env, args_cli.output_root)
            except Exception as error:
                details = traceback.format_exc()
                print("SO101_GRASP_POSTURE_ERROR\n" + details, flush=True)
                _write_json(
                    args_cli.output_root / "diagnostic_error.json",
                    {"error": repr(error), "traceback": details},
                )
                finish_diagnostic_manifest(
                    manifest, "grasp_postures", succeeded=False
                )
                write_manifest(args_cli.manifest, manifest)
                raise
            finish_diagnostic_manifest(
                manifest, "grasp_postures", succeeded=True
            )
            write_manifest(args_cli.manifest, manifest)
            return
        if args_cli.diagnose_calibrated_grasp:
            try:
                run_calibrated_grasp_diagnostic(env, args_cli.output_root)
            except Exception as error:
                details = traceback.format_exc()
                print("SO101_CALIBRATED_GRASP_ERROR\n" + details, flush=True)
                _write_json(
                    args_cli.output_root / "diagnostic_error.json",
                    {"error": repr(error), "traceback": details},
                )
                finish_diagnostic_manifest(
                    manifest, "calibrated_grasp", succeeded=False
                )
                write_manifest(args_cli.manifest, manifest)
                raise
            finish_diagnostic_manifest(
                manifest, "calibrated_grasp", succeeded=True
            )
            write_manifest(args_cli.manifest, manifest)
            return
        if args_cli.diagnose_grasp_offsets:
            run_grasp_offset_diagnostic(env, args_cli.output_root)
            finish_diagnostic_manifest(manifest, "grasp_offsets", succeeded=True)
            write_manifest(args_cli.manifest, manifest)
            return
        if args_cli.diagnose_grasp_grid:
            run_grasp_grid_diagnostic(env, args_cli.output_root)
            finish_diagnostic_manifest(manifest, "grasp_grid", succeeded=True)
            write_manifest(args_cli.manifest, manifest)
            return
        if args_cli.diagnose_grasp_paths:
            run_grasp_path_diagnostic(env, args_cli.output_root)
            finish_diagnostic_manifest(manifest, "grasp_paths", succeeded=True)
            write_manifest(args_cli.manifest, manifest)
            return
        watchdog_decision = (
            _evaluate_watchdog(plan, manifest, watchdog_policy)
            if watchdog_policy is not None
            else "CONTINUE"
        )
        for _ in range(args_cli.max_attempts_this_run):
            if watchdog_decision in {"STOP", "INVALID"}:
                break
            attempt = next_attempt(manifest, plan)
            if attempt is None:
                break
            active_attempt = attempt
            if live_publisher is not None:
                live_publisher.attempt_started(
                    attempt["attempt_id"], attempt["variation_id"]
                )
            try:
                episode_id, success, valid, category, reason = run_attempt(
                    env,
                    attempt,
                    args_cli.output_root,
                    os.environ.get("FARPOINT_GIT_COMMIT", "unknown"),
                    manifest["collection_id"],
                    camera_profile,
                    live_publisher,
                    v010_context,
                    recovery_runtime,
                )
            except Exception as error:
                details = traceback.format_exc()
                print(
                    f"SO101_ATTEMPT_ERROR attempt={attempt['attempt_id']}\n{details}",
                    flush=True,
                )
                failed_episode_id = episode_id_for_attempt(
                    manifest["collection_id"], attempt["attempt_id"]
                )
                error_root = args_cli.output_root / failed_episode_id
                run_state_path = error_root / "run-state.json"
                if isinstance(error, FileExistsError):
                    failed_run_state = None
                elif run_state_path.exists():
                    failed_run_state = _read_json(run_state_path)
                elif not error_root.exists():
                    failed_run_state = build_attempt_run_state(
                        attempt,
                        collection_id=manifest["collection_id"],
                        git_commit=os.environ.get("FARPOINT_GIT_COMMIT", "unknown"),
                    )
                    if v010_context is not None:
                        failed_run_state["recording"]["cameras"] = [
                            "observation.images.front",
                            "observation.images.wrist",
                        ]
                else:
                    failed_run_state = None
                if failed_run_state is not None:
                    failed_run_state["execution_status"] = "FAILED"
                    failed_run_state["outcome"] = {
                        "success": False,
                        "dataset_valid": False,
                        "failure_category": "runner",
                        "failure_reason": f"{type(error).__name__}: {error}",
                    }
                    _write_json(run_state_path, failed_run_state)
                runner_error_path = (
                    error_root / "runner_error.json"
                    if failed_run_state is not None
                    else args_cli.output_root
                    / f"runner_error_{attempt['attempt_id']}.json"
                )
                _write_json(
                    runner_error_path,
                    {"error": repr(error), "traceback": details},
                )
                episode_id, success, valid, category, reason = (
                    failed_episode_id if failed_run_state is not None else None,
                    False,
                    False,
                    "runner",
                    f"{type(error).__name__}: {error}",
                )
            record_attempt(manifest, plan, attempt, episode_id=episode_id, success=success, dataset_valid=valid, failure_category=category, failure_reason=reason)
            write_manifest(args_cli.manifest, manifest)
            if live_publisher is not None:
                live_publisher.attempt_completed(
                    attempt_id=attempt["attempt_id"],
                    variation_id=attempt["variation_id"],
                    success=success,
                    dataset_valid=valid,
                    episode_id=episode_id,
                    failure_reason=reason,
                )
            print(f"SO101_ATTEMPT {attempt['attempt_id']} success={success} phase={category or 'complete'}", flush=True)
            active_attempt = None
            if watchdog_policy is not None:
                watchdog_decision = _evaluate_watchdog(
                    plan, manifest, watchdog_policy
                )
                if watchdog_decision in {"STOP", "INVALID"}:
                    break
    except (KeyboardInterrupt, CollectionSignalAbort) as error:
        reason = collection_interruption_reason(error)
        if manifest.get("execution_status") == "RUNNING":
            abort_collection_manifest(manifest, reason)
        if active_attempt is not None:
            interrupted_episode_id = episode_id_for_attempt(
                manifest["collection_id"], active_attempt["attempt_id"]
            )
            run_state_path = (
                args_cli.output_root / interrupted_episode_id / "run-state.json"
            )
            if run_state_path.exists():
                interrupted_run_state = _read_json(run_state_path)
                if interrupted_run_state.get("execution_status") == "RUNNING":
                    abort_attempt_run_state(interrupted_run_state, reason)
                    _write_json(run_state_path, interrupted_run_state)
        write_manifest(args_cli.manifest, manifest)
        print(
            f"SO101_COLLECTION_ABORTED reason={reason} "
            f"attempt={None if active_attempt is None else active_attempt['attempt_id']}",
            flush=True,
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        if live_publisher is not None:
            execution_status = manifest.get("execution_status")
            live_publisher.finish(
                execution_status=(
                    "FINISHED" if execution_status == "FINISHED" else "PAUSED"
                ),
                quality_status=manifest.get("quality_status", "NOT_EVALUATED"),
            )
        if env is not None:
            env.close()
        simulation_app.close()
    if manifest["quality_status"] == "PASS" and not args_cli.gate_plan:
        _write_json(args_cli.manifest.with_name("export_selection.json"), build_export_selection(manifest, str(args_cli.output_root)))
    print(f"SO101_COLLECTION status={manifest['quality_status']} selected={len(manifest['selected_variations'])}", flush=True)


if __name__ == "__main__":
    main()
