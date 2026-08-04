"""Gym registration for the Farpoint SO-101 Isaac Lab environment."""

import gymnasium as gym


gym.register(
    id="Farpoint-SO101-PickPlace-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "farpoint_so101_env.env_cfg:SO101CubePickPlaceEnvCfg",
    },
)
