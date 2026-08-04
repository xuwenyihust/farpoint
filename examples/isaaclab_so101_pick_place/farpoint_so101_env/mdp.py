"""Observation terms for the SO-101 manager-based environment."""

from isaaclab.managers import SceneEntityCfg


def joint_positions(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return env.scene[asset_cfg.name].data.joint_pos


def end_effector_position(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")):
    return env.scene[sensor_cfg.name].data.target_pos_w[..., 0, :]


def camera_rgb(env, sensor_cfg: SceneEntityCfg):
    return env.scene[sensor_cfg.name].data.output["rgb"][..., :3]


def active_cube_position(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube_small_red")):
    return env.scene[asset_cfg.name].data.root_pos_w
