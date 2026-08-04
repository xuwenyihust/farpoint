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
        # The printed SO-101 fingertips are rubberized.  The workshop USD has
        # no explicit physics material, so Isaac otherwise falls back to a
        # low-friction default that releases the cube during lift.
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=1.5,
            restitution=0.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
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
        # Mount the fixed base at the table edge and on the table surface.
        # Keeping z=0 would intersect the 4 cm table collider at reset.
        # The workshop USD assumes a raised mounting pedestal.  At the table
        # surface the pitched arm intersects the tabletop; 26.7 cm preserves
        # clearance while keeping the cube-height grasp inside the workspace.
        pos=(-0.05, 0.0, 0.267),
        rot=(0.70710678, 0.0, 0.0, 0.70710678),
    ),
    actuators={
        # Isaac Sim 6.0 needs more damping than the workshop's Isaac Sim 5.1
        # values at this task's 30 Hz control rate.  Keep the same 30 N-m
        # effort limits while suppressing the reset transient.
        "rotation": ImplicitActuatorCfg(joint_names_expr=["Rotation"], effort_limit_sim=30, stiffness=100, damping=10),
        "pitch": ImplicitActuatorCfg(joint_names_expr=["Pitch"], effort_limit_sim=30, stiffness=80, damping=8),
        "elbow": ImplicitActuatorCfg(joint_names_expr=["Elbow"], effort_limit_sim=30, stiffness=70, damping=7),
        "wrist_pitch": ImplicitActuatorCfg(joint_names_expr=["Wrist_Pitch"], effort_limit_sim=30, stiffness=50, damping=5),
        "wrist_roll": ImplicitActuatorCfg(joint_names_expr=["Wrist_Roll"], effort_limit_sim=30, stiffness=30, damping=3),
        "gripper": ImplicitActuatorCfg(joint_names_expr=["Jaw"], effort_limit_sim=30, stiffness=20, damping=2),
    },
)
