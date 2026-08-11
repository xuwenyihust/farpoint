"""Observation terms for the SO-101 manager-based environment."""

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from pxr import Usd, UsdGeom, UsdShade


SO101_GRIPPER_STATIC_FRICTION = 2.0
SO101_GRIPPER_DYNAMIC_FRICTION = 1.5
SO101_GRIPPER_RESTITUTION = 0.0


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
        # Setting collisionEnabled on a descendant mesh is too late for the
        # workshop asset's merged collision hierarchy: PhysX still parses the
        # camera bracket into the gripper rigid body. Deactivate only this
        # collision-purpose subtree before startup; the visual bracket and the
        # separately spawned Farpoint wrist camera remain intact.
        prim.SetActive(False)
        if prim.IsActive():
            raise RuntimeError(f"failed to deactivate camera mount collider: {prim.GetPath()}")


def bind_so101_gripper_material(env, env_ids=None):
    """Bind a high-friction material directly to instantiated finger colliders."""
    del env_ids
    material_path = "/World/PhysicsMaterials/SO101Gripper"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=SO101_GRIPPER_STATIC_FRICTION,
        dynamic_friction=SO101_GRIPPER_DYNAMIC_FRICTION,
        restitution=SO101_GRIPPER_RESTITUTION,
        friction_combine_mode="max",
    )
    material_cfg.func(material_path, material_cfg)
    robot_path = env.scene["robot"].cfg.prim_path
    matched = []
    for link_name in ("gripper", "jaw"):
        collider_path = f"{robot_path}/{link_name}/collisions"
        for prim in sim_utils.find_matching_prims(collider_path):
            # The workshop asset marks each collision subtree instanceable.
            # De-instance only these two small subtrees so USD can author a
            # per-finger material binding without modifying the pinned file.
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
            matched.append(str(prim.GetPath()))
            bind_targets = [prim]
            bind_targets.extend(
                descendant
                for descendant in Usd.PrimRange(prim)
                if descendant != prim and descendant.IsA(UsdGeom.Mesh)
            )
            material = UsdShade.Material(prim.GetStage().GetPrimAtPath(material_path))
            for target in bind_targets:
                binding_api = UsdShade.MaterialBindingAPI.Apply(target)
                binding_api.Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
                relationship = target.GetRelationship("material:binding:physics")
                if not relationship.IsValid() or not relationship.GetTargets():
                    raise RuntimeError(f"physics material binding failed: {target.GetPath()}")
    if len(matched) != 2:
        raise RuntimeError(f"expected gripper and jaw collider roots, got {matched}")
