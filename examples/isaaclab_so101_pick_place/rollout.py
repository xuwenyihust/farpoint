#!/usr/bin/env python3
"""Run a frozen SO-101 ACT checkpoint in closed-loop Isaac Lab simulation."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.so101_runtime import resolve_headless_mode  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("headless", "viewer"), default="headless")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = resolve_headless_mode(
        args.mode, args.livestream, livestream_env=os.environ.get("LIVESTREAM")
    )
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import farpoint_so101_env  # noqa: E402,F401
from farpoint.oracle import oriented_box_footprint_inside_target  # noqa: E402
from farpoint.policy_rollout import (  # noqa: E402
    constrain_policy_action,
    evaluate_rollout_acceptance,
    load_rollout_spec,
)
from farpoint.policy_training import canonical_sha256, file_sha256  # noqa: E402
from farpoint.so101 import lerobot_to_radians, radians_to_lerobot  # noqa: E402
from farpoint.control import so101_reset_support_is_stable  # noqa: E402


SO101_HOME_JOINTS = np.asarray(
    [-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, 1.7453], dtype=np.float32
)
TARGET_POSITION = np.asarray([0.20, 0.10, 0.037], dtype=np.float32)
TARGET_DIMENSIONS = np.asarray([0.16, 0.14, 0.01], dtype=np.float32)
TARGET_MARGIN_M = 0.005


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _aim_front_camera(scene, device) -> None:
    eye = torch.tensor([[0.42, -0.38, 0.34]], dtype=torch.float32, device=device)
    target = torch.tensor([[0.20, 0.02, 0.06]], dtype=torch.float32, device=device)
    scene["front_camera"].set_world_poses_from_view(eye, target)


def _move_object(obj, position, device, orientation_xyzw=(0.0, 0.0, 0.0, 1.0)):
    pose = torch.tensor(
        [[*position, *orientation_xyzw]], dtype=torch.float32, device=device
    )
    obj.write_root_pose_to_sim(pose)
    obj.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=device))


def _image(scene) -> np.ndarray:
    return np.asarray(_numpy(scene["front_camera"].data.output["rgb"][0, ..., :3]), dtype=np.uint8)


def _variant_name(scene_spec: dict) -> str:
    obj = scene_spec["object"]
    size = "small" if obj["dimensions_m"][0] < 0.035 else "large"
    color = "red" if obj["rgba"][0] > obj["rgba"][2] else "blue"
    return f"cube_{size}_{color}"


def _cube_contact_forces(scene) -> tuple[float, float]:
    values = []
    for name in ("contact_jaw", "contact_gripper"):
        sensor = scene[name]
        force = float(
            torch.linalg.vector_norm(sensor.data.force_matrix_w, dim=-1).max().item()
        )
        values.append(force)
    return values[0], values[1]


class VideoWriter:
    def __init__(self, path: Path, fps: int):
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480",
                "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, image: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg video pipe is closed")
        self.process.stdin.write(np.ascontiguousarray(image).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg video writer exited {return_code}")


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


def _policy_action(state: np.ndarray, front: np.ndarray, task: str) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(front, mode="RGB").save(buffer, format="JPEG", quality=95)
    response = _policy_request(
        "/action",
        {
            "state": state.tolist(),
            "front_jpeg": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "task": task,
        },
    )
    return np.asarray(response["action"], dtype=np.float32)


def _reset_scene(env, scene_spec: dict) -> tuple[str, dict]:
    device = env.device
    scene = env.scene
    robot = scene["robot"]
    seed = int(scene_spec["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.reset(seed=seed)
    active_name = _variant_name(scene_spec)
    env.farpoint_active_cube = active_name
    inactive_names = [
        name
        for name in (
            "cube_small_red",
            "cube_small_blue",
            "cube_large_red",
            "cube_large_blue",
        )
        if name != active_name
    ]
    for index, name in enumerate(inactive_names):
        _move_object(scene[name], (-10.0 - index, 0.0, 0.1), device)
    obj = scene_spec["object"]
    active = scene[active_name]
    active.set_masses_index(
        masses=torch.tensor([[obj["mass_kg"]]], dtype=torch.float32, device=device)
    )
    actual_mass = float(active.data.body_mass.torch[0, 0].item())
    if abs(actual_mass - float(obj["mass_kg"])) > 1e-6:
        raise RuntimeError(f"PhysX mass mismatch: {actual_mass} != {obj['mass_kg']}")
    initial_joints = torch.tensor([SO101_HOME_JOINTS], dtype=torch.float32, device=device)
    spawn_position = np.asarray(obj["position_m"], dtype=np.float32).copy()
    spawn_position[2] += 0.002
    _aim_front_camera(scene, device)
    _move_object(active, spawn_position, device, obj["orientation_xyzw"])
    for _ in range(8):
        env.step(initial_joints)
    measured_position = _numpy(active.data.root_pos_w[0])
    measured_velocity = _numpy(active.data.root_lin_vel_w[0])
    if not so101_reset_support_is_stable(
        np.asarray(obj["position_m"]), measured_position, measured_velocity
    ):
        raise RuntimeError(
            f"cube reset support unstable: position={measured_position.tolist()} "
            f"velocity={measured_velocity.tolist()}"
        )
    robot.write_joint_state_to_sim(initial_joints, torch.zeros_like(initial_joints))
    robot.set_joint_position_target(initial_joints)
    env.sim.forward()
    scene.update(0.0)
    home_error = float(np.max(np.abs(_numpy(robot.data.joint_pos[0]) - SO101_HOME_JOINTS)))
    if home_error > 1e-5:
        raise RuntimeError(f"arm HOME restoration failed: {home_error}")
    return active_name, {
        "requested_mass_kg": float(obj["mass_kg"]),
        "actual_mass_kg": actual_mass,
        "measured_position_m": measured_position.tolist(),
        "measured_velocity_mps": measured_velocity.tolist(),
        "maximum_home_error_rad": home_error,
    }


def _run_episode(env, scene_spec, spec, root):
    episode_root = root / "episodes" / scene_spec["scene_id"]
    episode_root.mkdir(parents=True)
    active_name, reset_audit = _reset_scene(env, scene_spec)
    _policy_request("/reset", {})
    scene = env.scene
    robot = scene["robot"]
    active = scene[active_name]
    obj = scene_spec["object"]
    control = spec["control"]
    physics_steps_per_policy = control["physics_hz"] // control["policy_hz"]
    initial_z = float(obj["position_m"][2])
    ever_contact = False
    ever_bilateral = False
    ever_lifted = False
    ever_in_target = False
    stable_steps = 0
    hard_range_violations = 0
    delta_limited = 0
    nonfinite = 0
    policy_steps = 0
    task_success = False
    writer = VideoWriter(episode_root / "front.mp4", control["policy_hz"])
    trace_file = (episode_root / "actions.jsonl").open("w", encoding="utf-8")
    try:
        for step in range(control["max_policy_steps"]):
            front = _image(scene)
            writer.write(front)
            joint_radians = _numpy(robot.data.joint_pos[0]).astype(np.float32)
            state = radians_to_lerobot(joint_radians, clip=True)
            raw_action = _policy_action(state, front, spec["task"]["instruction"])
            if raw_action.shape != (6,) or not np.all(np.isfinite(raw_action)):
                nonfinite += 1
                raise RuntimeError(f"invalid policy action at step {step}: {raw_action}")
            applied, safety = constrain_policy_action(
                raw_action, state, max_delta=control["max_delta_calibrated"]
            )
            hard_range_violations += safety["hard_range_violation_count"]
            delta_limited += safety["delta_limited_count"]
            target_radians = lerobot_to_radians(applied, clip=True)
            target = torch.tensor([target_radians], dtype=torch.float32, device=env.device)
            for _ in range(physics_steps_per_policy):
                env.step(target)
            forces = _cube_contact_forces(scene)
            cube_pose = _numpy(active.data.root_pose_w[0])
            cube_velocity = _numpy(active.data.root_lin_vel_w[0])
            contact = max(forces) >= 0.10
            bilateral = min(forces) >= 0.10
            lifted = contact and float(cube_pose[2]) > initial_z + 0.005
            in_target = oriented_box_footprint_inside_target(
                cube_pose[:3],
                obj["dimensions_m"],
                cube_pose[3:7],
                TARGET_POSITION,
                TARGET_DIMENSIONS,
                margin_m=TARGET_MARGIN_M,
            )
            released = float(robot.data.joint_pos[0, 5].item()) >= SO101_HOME_JOINTS[5] - 0.10
            stable = bool(float(np.linalg.norm(cube_velocity)) < 0.03)
            ever_contact = ever_contact or contact
            ever_bilateral = ever_bilateral or bilateral
            ever_lifted = ever_lifted or lifted
            ever_in_target = ever_in_target or in_target
            valid_settle = ever_lifted and in_target and released and stable
            stable_steps = stable_steps + 1 if valid_settle else 0
            task_success = stable_steps >= control["stable_steps"]
            trace_row = {
                "policy_step": step,
                "state_calibrated": state.tolist(),
                "raw_action_calibrated": raw_action.tolist(),
                "applied_action_calibrated": applied.tolist(),
                "target_radians": target_radians.tolist(),
                "action_safety": safety,
                "cube_pose_xyzw": cube_pose.tolist(),
                "cube_velocity_mps": cube_velocity.tolist(),
                "contact_forces_n": list(forces),
                "cube_in_target": in_target,
                "gripper_released": released,
                "cube_stable": stable,
            }
            trace_file.write(json.dumps(trace_row, sort_keys=True) + "\n")
            policy_steps += 1
            if policy_steps % control["policy_hz"] == 0:
                trace_file.flush()
            if task_success:
                break
    finally:
        trace_file.close()
        writer.close()
    if task_success:
        terminal_reason = "success"
    elif not ever_contact:
        terminal_reason = "no_cube_contact"
    elif not ever_lifted:
        terminal_reason = "contact_without_lift"
    elif not ever_in_target:
        terminal_reason = "lift_without_target_entry"
    else:
        terminal_reason = "target_entry_without_stable_release"
    result = {
        "scene_id": scene_spec["scene_id"],
        "execution_status": "FINISHED",
        "task_success": task_success,
        "terminal_reason": terminal_reason,
        "policy_steps": policy_steps,
        "video": str((episode_root / "front.mp4").relative_to(root)),
        "trace": str((episode_root / "actions.jsonl").relative_to(root)),
        "reset_audit": reset_audit,
        "stage_evidence": {
            "ever_cube_contact": ever_contact,
            "ever_bilateral_contact": ever_bilateral,
            "ever_lifted": ever_lifted,
            "ever_entered_target": ever_in_target,
            "maximum_stable_release_steps": stable_steps,
        },
        "nonfinite_action_count": nonfinite,
        "hard_range_violation_count": hard_range_violations,
        "delta_limited_count": delta_limited,
    }
    _write_json(episode_root / "result.json", result)
    return result


def main() -> int:
    spec = load_rollout_spec(args_cli.spec)
    if args_cli.output_root.exists():
        raise FileExistsError(f"rollout output root already exists: {args_cli.output_root}")
    model_file = args_cli.checkpoint / "model.safetensors"
    if file_sha256(model_file) != spec["checkpoint"]["model_sha256"]:
        raise RuntimeError("checkpoint model SHA256 does not match the frozen rollout spec")
    rollout_git_commit = os.environ.get("FARPOINT_GIT_COMMIT", "")
    isaac_image_id = os.environ.get("FARPOINT_ISAAC_IMAGE_ID", "")
    policy_image_id = os.environ.get("FARPOINT_POLICY_IMAGE_ID", "")
    base_image_id = os.environ.get("FARPOINT_SO101_BASE_IMAGE_ID", "")
    if len(rollout_git_commit) != 40:
        raise RuntimeError("FARPOINT_GIT_COMMIT must bind rollout evidence")
    if not isaac_image_id.startswith("sha256:"):
        raise RuntimeError("FARPOINT_ISAAC_IMAGE_ID must bind rollout evidence")
    if policy_image_id != spec["checkpoint"]["training_image_id"]:
        raise RuntimeError("policy server image ID does not match checkpoint provenance")
    if base_image_id != spec["environment"]["isaac_base_image_id"]:
        raise RuntimeError("Isaac base image ID does not match the frozen rollout spec")
    args_cli.output_root.mkdir(parents=True)
    policy_health = _policy_request("/health")
    if policy_health.get("status") != "ready":
        raise RuntimeError("policy server is not ready")
    if policy_health.get("model_sha256") != spec["checkpoint"]["model_sha256"]:
        raise RuntimeError("policy server checkpoint identity mismatch")
    if policy_health.get("policy_image_id") != policy_image_id:
        raise RuntimeError("policy server image identity mismatch")
    if policy_health.get("lerobot_version") != spec["environment"]["lerobot_version"]:
        raise RuntimeError("policy server LeRobot version mismatch")
    from farpoint_so101_env.env_cfg import SO101CubePickPlaceEnvCfg

    env_cfg = SO101CubePickPlaceEnvCfg()
    env_cfg.seed = 0
    env_cfg.scene.wrist_camera = None
    env_cfg.observations.policy.wrist_rgb = None
    env = gym.make(spec["environment"]["gym_id"], cfg=env_cfg).unwrapped
    results = []
    try:
        for scene_spec in spec["scenes"]:
            result = _run_episode(env, scene_spec, spec, args_cli.output_root)
            results.append(result)
            print(
                f"SO101_ACT_ROLLOUT scene={result['scene_id']} "
                f"success={result['task_success']} reason={result['terminal_reason']}",
                flush=True,
            )
    finally:
        env.close()
    acceptance = evaluate_rollout_acceptance(spec, results)
    report = {
        "schema_version": "farpoint.policy-rollout-report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_id": spec["suite_id"],
        "status": acceptance["status"],
        "rollout_git_commit": rollout_git_commit,
        "isaac_image_id": isaac_image_id,
        "policy_server": policy_health,
        "isaac_base_image_id": base_image_id,
        "spec_sha256": canonical_sha256(spec),
        "checkpoint": spec["checkpoint"],
        "data_policy": {
            "training_episodes": spec["checkpoint"]["dataset"]["train_episodes"],
            "validation_episodes": spec["checkpoint"]["dataset"]["validation_episodes"],
            "test_episodes_consumed": False,
            "excluded_test_episodes": spec["checkpoint"]["dataset"]["excluded_test_episodes"],
        },
        "acceptance": acceptance,
        "episodes": results,
        "interpretation": (
            "Interface smoke: task success is reported but not required. "
            "This suite validates closed-loop observation, action scaling, control, and evidence."
        ),
    }
    _write_json(args_cli.output_root / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
