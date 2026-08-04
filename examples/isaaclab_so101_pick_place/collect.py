#!/usr/bin/env python3
"""Generate raw SO-101 cube demonstrations from an Isaac Lab oracle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# USD limits observed from the pinned SO-101 asset (radians), kept explicit so
# controller targets cannot be normalized or extrapolated by a backend.
SO101_JOINT_LIMITS = np.asarray(
    [[-1.920, 1.920], [-1.745, 1.745], [-1.745, 1.571],
     [-1.658, 1.658], [-2.793, 2.793]], dtype=np.float32
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("headless", "viewer"), default="headless")
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "outputs/so101_variation_plan.json")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "outputs/so101_collection/manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/episodes")
    parser.add_argument("--max-attempts-this-run", type=int, default=150)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = args.mode == "headless"
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import farpoint_so101_env  # noqa: E402,F401
from farpoint.contracts import validate_contract  # noqa: E402
from farpoint.oracle import OracleObservation, OraclePhase, OracleStateMachine, damped_least_squares  # noqa: E402
from farpoint.so101 import LEROBOT_JOINT_NAMES, SIM_JOINT_NAMES, mapping_metadata  # noqa: E402
from farpoint.so101_collection import (  # noqa: E402
    build_export_selection,
    create_manifest,
    load_manifest,
    next_attempt,
    record_attempt,
    write_manifest,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _torch_pose(position, device):
    return torch.tensor([[*position, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)


def _numpy(value):
    """Convert Isaac Lab tensor or NumPy backend values to a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _move_object(obj, position, device):
    obj.write_root_pose_to_sim(_torch_pose(position, device))
    obj.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=device))


def _image(camera, device):
    return np.asarray(_numpy(camera.data.output["rgb"][0, ..., :3]), dtype=np.uint8)


def _contact(sensor) -> bool:
    if hasattr(sensor.data, "force_matrix_w"):
        forces = sensor.data.force_matrix_w
        return bool(torch.linalg.vector_norm(forces, dim=-1).max().item() > 2.0)
    if hasattr(sensor.data, "net_forces_w"):
        return bool(torch.linalg.vector_norm(sensor.data.net_forces_w, dim=-1).max().item() > 2.0)
    return False


def _body_index(robot) -> int:
    indexes, _names = robot.find_bodies("gripper")
    if len(indexes) != 1:
        raise RuntimeError(f"expected one SO-101 gripper body, got {indexes}")
    # The USD robot is fixed-base, so Isaac Lab's public Jacobian excludes the
    # fixed root row (jacobi_body_idx == body_idx - 1).
    return int(indexes[0]) - 1


def _ik_action(robot, ee_frame, target, current, body_index, device):
    ee = _numpy(ee_frame.data.target_pos_w[0, 0])
    jacobians = robot.data.body_link_jacobian_w.torch
    jacobian = _numpy(jacobians[0, body_index, :3, :5])
    delta = damped_least_squares(jacobian, np.asarray(target) - ee, damping=0.06)
    action = _numpy(current).astype(np.float32).copy()
    action[:5] = action[:5] + np.clip(delta, -0.01, 0.01)
    # Keep the generated target inside the pinned USD joint limits.  The
    # position action manager does not clamp targets when offset-free control
    # is enabled, and some PhysX articulations report wider soft limits.
    action[:5] = np.clip(action[:5], SO101_JOINT_LIMITS[:, 0], SO101_JOINT_LIMITS[:, 1])
    return torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)


def _write_frame(root: Path, frame: int, state, action, front, wrist):
    front_path = root / "rgb" / f"front_{frame:06d}.png"
    wrist_path = root / "rgb" / f"wrist_{frame:06d}.png"
    front_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(front, mode="RGB").save(front_path)
    Image.fromarray(wrist, mode="RGB").save(wrist_path)
    return {
        "frame": frame,
        "timestamp_seconds": frame / 30.0,
        "phase": "",
        "rgb_path": str(front_path.relative_to(root)),
        "wrist_rgb_path": str(wrist_path.relative_to(root)),
        "joint_names": list(SIM_JOINT_NAMES),
        "controlled_joint_names": list(SIM_JOINT_NAMES),
        "joint_positions": [float(value) for value in state],
        "joint_velocities": [],
        "action_joint_positions": [float(value) for value in action],
        "contact_forces_newtons": {"left_finger": 0.0, "right_finger": 0.0},
        "object_pose_estimate": {},
    }


def _variant_name(trial):
    edge = trial["resolved"]["dimensions_m"][0]
    color = trial["resolved"]["rgba"]
    size = "small" if edge < 0.035 else "large"
    color_name = "red" if color[0] > color[2] else "blue"
    return f"cube_{size}_{color_name}"


def run_attempt(env, trial, output_root: Path, git_commit: str):
    device = env.device
    scene = env.scene
    robot = scene["robot"]
    ee_frame = scene["ee_frame"]
    contact = scene["contact_grasp"]
    active_name = _variant_name(trial)
    inactive = [name for name in ("cube_small_red", "cube_small_blue", "cube_large_red", "cube_large_blue") if name != active_name]
    env.farpoint_active_cube = active_name
    env.reset()
    # The workshop USD carries a non-zero default state and Isaac Lab 3.0 may
    # apply it after manager reset.  Re-assert the documented neutral pose so
    # every episode starts from the same physical configuration.
    robot.write_joint_state_to_sim(
        torch.tensor(
            [[-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, 1.7453]],
            dtype=torch.float32,
            device=device,
        ),
        torch.zeros((1, 6), dtype=torch.float32, device=device),
    )
    for index, name in enumerate(inactive):
        _move_object(scene[name], (-10.0 - index, 0.0, 0.1), device)
    object_spec = trial["resolved"]
    _move_object(scene[active_name], object_spec["position_m"], device)
    env.sim.step()
    home_ee = _numpy(ee_frame.data.target_pos_w[0, 0]).copy()
    body_index = _body_index(robot)
    open_jaw = float(robot.data.joint_pos[0, 5].item())
    closed_jaw = float(np.deg2rad(-8.0))
    object_position = np.asarray(object_spec["position_m"], dtype=np.float32)
    target_position = np.asarray([0.22, 0.10, 0.060], dtype=np.float32)
    machine = OracleStateMachine()
    rows = []
    root = output_root / f"episode_{trial['attempt_id']}"
    if root.exists():
        raise FileExistsError(f"episode output already exists: {root}")
    for frame in range(900):
        phase = machine.phase
        if phase is OraclePhase.HOME:
            target = home_ee
            jaw = open_jaw
        elif phase in {OraclePhase.PREGRASP, OraclePhase.DESCEND, OraclePhase.CLOSE}:
            target = object_position + np.asarray((0.0, 0.0, 0.075 if phase is OraclePhase.PREGRASP else 0.035))
            jaw = closed_jaw if phase is OraclePhase.CLOSE else open_jaw
        elif phase is OraclePhase.VERIFY_CONTACT:
            target = object_position + np.asarray((0.0, 0.0, 0.12))
            jaw = closed_jaw
        elif phase in {OraclePhase.LIFT, OraclePhase.PREPLACE}:
            target = target_position + np.asarray((0.0, 0.0, 0.13)) if phase is OraclePhase.PREPLACE else object_position + np.asarray((0.0, 0.0, 0.13))
            jaw = closed_jaw
        elif phase in {OraclePhase.PLACE_DESCEND, OraclePhase.OPEN, OraclePhase.SETTLE}:
            target = target_position + np.asarray((0.0, 0.0, 0.045))
            jaw = open_jaw if phase is not OraclePhase.PLACE_DESCEND else closed_jaw
        else:
            target = target_position + np.asarray((0.0, 0.0, 0.14))
            jaw = open_jaw

        current = robot.data.joint_pos[0]
        if frame == 0:
            print(
                f"SO101_ORACLE_START phase={phase.value} "
                f"ee={_numpy(ee_frame.data.target_pos_w[0, 0]).tolist()} "
                f"target={np.asarray(target).tolist()}",
                flush=True,
            )
        if phase is OraclePhase.PREGRASP and machine.phase_steps == 0:
            print(
                f"SO101_ORACLE_PREGRASP_START ee={_numpy(ee_frame.data.target_pos_w[0, 0]).tolist()} "
                f"target={np.asarray(target).tolist()}",
                flush=True,
            )
        action = _ik_action(robot, ee_frame, target, current, body_index, device)
        action[0, 5] = jaw
        state = _numpy(current)
        front = _image(scene["front_camera"], device)
        wrist = _image(scene["wrist_camera"], device)
        row = _write_frame(root, frame, state, _numpy(action[0]), front, wrist)
        row["phase"] = phase.value
        rows.append(row)
        obs = OracleObservation(
            reached_target=float(np.linalg.norm(_numpy(ee_frame.data.target_pos_w[0, 0]) - target)) < 0.012,
            has_contact=_contact(contact),
            cube_lifted=float(scene[active_name].data.root_pos_w[0, 2].item()) > object_position[2] + 0.025,
            cube_in_target=float(torch.linalg.vector_norm(scene[active_name].data.root_pos_w[0, :2] - torch.tensor(target_position[:2], device=device)).item()) < 0.035,
            gripper_released=abs(float(current[5].item()) - open_jaw) < 0.08,
            cube_stable=float(torch.linalg.vector_norm(scene[active_name].data.root_lin_vel_w[0]).item()) < 0.03,
        )
        machine.step(obs)
        env.step(action)
        if machine.phase in {OraclePhase.SUCCEEDED, OraclePhase.FAILED}:
            break
    success = machine.phase is OraclePhase.SUCCEEDED
    if not success:
        print(
            "SO101_ORACLE_DEBUG "
            f"phase={machine.phase.value} reason={machine.failure_reason} "
            f"ee={_numpy(ee_frame.data.target_pos_w[0, 0]).tolist()} "
            f"target={np.asarray(target).tolist()} "
            f"joints={_numpy(robot.data.joint_pos[0]).tolist()}",
            flush=True,
        )
    metadata = {
        "schema_version": "farpoint.episode.v3",
        "identity": {"episode_id": root.name, "trial_id": trial["trial_id"], "task_id": "so101_cube_pick_place", "split": trial["split"], "episode_seed": int(trial["attempt_seed"])},
        "provenance": {"git_commit": git_commit, "simulator": "Isaac Sim", "simulator_image": "nvcr.io/nvidia/isaac-sim:6.0.0", "physics_engine": "PhysX", "asset_commit": "ce807d99724cb65671abec01f908a2fcb4a6eab7"},
        "task": {"task_id": "so101_cube_pick_place", "instruction": "Pick up the cube and place it in the green tray.", "object_shape": "cube", "success_criteria_id": "contact_pick_place_v1"},
        "embodiment": {"robot": "so101", "gripper": "so101_jaw", "arm_dof": 5, "gripper_dof": 1, "controller": "damped_least_squares_ik", "control_mode": "joint_position", "grasp_mode": "contact_only", "joint_mapping": mapping_metadata()},
        "scene": {"coordinate_frame": "isaac_world", "object": {"shape": "cube", "asset_id": object_spec["asset_id"], "dimensions_m": object_spec["dimensions_m"], "initial_pose": {"position_m": object_spec["position_m"], "orientation_xyzw": object_spec["orientation_xyzw"]}, "rgba": object_spec["rgba"], "mass_kg": object_spec["mass_kg"], "static_friction": object_spec["static_friction"], "dynamic_friction": object_spec["dynamic_friction"], "restitution": object_spec["restitution"]}, "target": {"target_id": "fixed_green_tray_v1", "position_m": target_position.tolist()}, "cameras": [{"name": "observation.images.front", "resolution": [640, 480]}, {"name": "observation.images.wrist", "resolution": [640, 480]}], "lighting_profile_id": "fixed_default"},
        "variation": {"schema_version": "farpoint.variation.v3", "variation_id": trial["variation_id"], "varied_axes": ["object.position_m.x", "object.position_m.y", "object.dimensions_m", "object.rgba"], "frozen_axes": ["object.shape", "object.mass_kg", "object.static_friction", "object.dynamic_friction"], "requested": trial["requested"], "resolved": object_spec, "split": trial["split"]},
        "recording": {"fps": 30, "cameras": ["observation.images.front", "observation.images.wrist"], "frame_count": len(rows), "state_features": list(LEROBOT_JOINT_NAMES), "action_features": list(LEROBOT_JOINT_NAMES), "state_unit": "radian", "action_unit": "radian"},
        "outcome": {"success": success, "dataset_valid": bool(rows), "failure_category": None if success else "oracle", "failure_reason": None if success else machine.failure_reason},
    }
    errors = validate_contract(metadata)
    if errors:
        raise ValueError("invalid SO-101 episode metadata: " + "; ".join(errors))
    _write_json(root / "metadata.json", metadata)
    (root / "observations.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    _write_json(root / "metrics.json", {"success": success, "dataset_valid": bool(rows), "failure_category": metadata["outcome"]["failure_category"], "failure_reason": metadata["outcome"]["failure_reason"], "observation_count": len(rows)})
    return root.name, success, bool(rows), metadata["outcome"]["failure_category"], metadata["outcome"]["failure_reason"]


def main():
    from farpoint.object_variation import generate_variation_plan, load_variation_config

    config = load_variation_config(PROJECT_ROOT / "configs/variations/so101_cube_pick_place_v1.json")
    plan = _read_json(args_cli.plan) if args_cli.plan.exists() else generate_variation_plan(config)
    args_cli.plan.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args_cli.plan, plan)
    if args_cli.manifest.exists():
        manifest = load_manifest(args_cli.manifest, plan)
    else:
        manifest = create_manifest(plan, collection_id="so101_cube_pick_place_pilot", git_commit=os.environ.get("FARPOINT_GIT_COMMIT", "unknown"))
    # Isaac Lab 3.0 keeps the config entry point in the Gym registration, but
    # Gymnasium does not instantiate that config automatically.  Construct it
    # explicitly so this works both from the launcher and from Python callers.
    from farpoint_so101_env.env_cfg import SO101CubePickPlaceEnvCfg

    env_cfg = SO101CubePickPlaceEnvCfg()
    env = gym.make("Farpoint-SO101-PickPlace-Cube-v0", cfg=env_cfg).unwrapped
    try:
        for _ in range(args_cli.max_attempts_this_run):
            attempt = next_attempt(manifest, plan)
            if attempt is None:
                break
            try:
                episode_id, success, valid, category, reason = run_attempt(env, attempt, args_cli.output_root, os.environ.get("FARPOINT_GIT_COMMIT", "unknown"))
            except Exception as error:
                episode_id, success, valid, category, reason = None, False, False, "runner", str(error)
            record_attempt(manifest, plan, attempt, episode_id=episode_id, success=success, dataset_valid=valid, failure_category=category, failure_reason=reason)
            write_manifest(args_cli.manifest, manifest)
            print(f"SO101_ATTEMPT {attempt['attempt_id']} success={success} phase={category or 'complete'}", flush=True)
    finally:
        env.close()
        simulation_app.close()
    if manifest["quality_status"] == "PASS":
        _write_json(args_cli.manifest.with_name("export_selection.json"), build_export_selection(manifest, str(args_cli.output_root)))
    print(f"SO101_COLLECTION status={manifest['quality_status']} selected={len(manifest['selected_variations'])}", flush=True)


if __name__ == "__main__":
    main()
