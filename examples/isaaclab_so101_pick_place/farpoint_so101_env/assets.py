"""Pinned SO-101 articulation configuration for Isaac Lab 3.0."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


WORKSHOP_COMMIT = "ce807d99724cb65671abec01f908a2fcb4a6eab7"
WORKSHOP_ASSET_SHA256 = "11f5f0bb5f2fae3eefebbcd07dfafc6b14602f6c4e5dae8f21a4a46892991006"


def so101_usd_path() -> str:
    explicit = os.environ.get("FARPOINT_SO101_USD")
    if explicit:
        return explicit
    root = Path(__file__).resolve().parents[3]
    return str(root / ".cache/farpoint/assets/so101" / WORKSHOP_COMMIT / "SO-ARM101-USD.usd")


SO101_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=so101_usd_path(),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "Rotation": -0.2736,
            "Pitch": -0.6109,
            "Elbow": -0.0745,
            "Wrist_Pitch": 1.5148,
            "Wrist_Roll": -1.6034,
            "Jaw": 1.7453,
        },
        pos=(-0.05, 0.0, 0.0),
        rot=(0.70710678, 0.0, 0.0, 0.70710678),
    ),
    actuators={
        "rotation": ImplicitActuatorCfg(joint_names_expr=["Rotation"], effort_limit_sim=30, stiffness=55, damping=0.7),
        "pitch": ImplicitActuatorCfg(joint_names_expr=["Pitch"], effort_limit_sim=30, stiffness=30, damping=0.8),
        "elbow": ImplicitActuatorCfg(joint_names_expr=["Elbow"], effort_limit_sim=30, stiffness=25, damping=0.7),
        "wrist_pitch": ImplicitActuatorCfg(joint_names_expr=["Wrist_Pitch"], effort_limit_sim=30, stiffness=12, damping=0.5),
        "wrist_roll": ImplicitActuatorCfg(joint_names_expr=["Wrist_Roll"], effort_limit_sim=30, stiffness=7, damping=0.5),
        "gripper": ImplicitActuatorCfg(joint_names_expr=["Jaw"], effort_limit_sim=30, stiffness=4, damping=0.3),
    },
)
