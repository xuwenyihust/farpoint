"""Observation terms for the SO-101 manager-based environment."""

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg


def joint_positions(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return env.scene[asset_cfg.name].data.joint_pos


def end_effector_position(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")):
    return env.scene[sensor_cfg.name].data.target_pos_w[..., 0, :]


def camera_rgb(env, sensor_cfg: SceneEntityCfg):
    return env.scene[sensor_cfg.name].data.output["rgb"][..., :3]


def active_cube_position(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube_small_red")):
    return env.scene[asset_cfg.name].data.root_pos_w


def disable_workshop_camera_mount_collision(env, env_ids=None):
    """Disable only the unused physical camera bracket's collider.

    Farpoint spawns its own massless wrist camera. The workshop bracket hangs
    below the open fingers in this tabletop mounting and otherwise contacts the
    table before either grasping surface reaches a 30 mm cube.
    """
    del env_ids
    path = env.scene["robot"].cfg.prim_path + "/gripper/collisions/camera_mount"
    for prim in sim_utils.find_matching_prims(path):
        attribute = prim.GetAttribute("physics:collisionEnabled")
        if not attribute.IsValid():
            raise RuntimeError(f"camera mount collision attribute not found: {prim.GetPath()}")
        attribute.Set(False)
