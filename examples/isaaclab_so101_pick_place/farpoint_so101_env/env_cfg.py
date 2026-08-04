"""Manager-based Isaac Lab environment for SO-101 cube pick-and-place."""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as lab_mdp
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.utils import configclass

from . import mdp
from .assets import SO101_CFG


def _cube(name: str, edge: float, color: tuple[float, float, float]):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(edge, edge, edge),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=2.0,
                max_angular_velocity=8.0,
                solver_position_iteration_count=16,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-10.0, 0.0, 0.1)),
    )


@configclass
class SO101CubeSceneCfg(InteractiveSceneCfg):
    num_envs = 1
    env_spacing = 2.0
    replicate_physics = False
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.50, 0.42, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.42)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.20, 0.0, 0.012)),
    )
    tray_base = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tray/Base",
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.08, 0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.70, 0.20)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.22, 0.10, 0.037)),
    )
    tray_wall_x0 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tray/WallX0",
        spawn=sim_utils.CuboidCfg(size=(0.01, 0.08, 0.025), collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.17, 0.10, 0.050)),
    )
    tray_wall_x1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tray/WallX1",
        spawn=sim_utils.CuboidCfg(size=(0.01, 0.08, 0.025), collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.27, 0.10, 0.050)),
    )
    tray_wall_y0 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tray/WallY0",
        spawn=sim_utils.CuboidCfg(size=(0.10, 0.01, 0.025), collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.22, 0.06, 0.050)),
    )
    tray_wall_y1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tray/WallY1",
        spawn=sim_utils.CuboidCfg(size=(0.10, 0.01, 0.025), collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.22, 0.14, 0.050)),
    )
    cube_small_red = _cube("CubeSmallRed", 0.03, (0.85, 0.08, 0.06))
    cube_small_blue = _cube("CubeSmallBlue", 0.03, (0.04, 0.20, 0.85))
    cube_large_red = _cube("CubeLargeRed", 0.04, (0.85, 0.08, 0.06))
    cube_large_blue = _cube("CubeLargeBlue", 0.04, (0.04, 0.20, 0.85))
    robot = SO101_CFG
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper", name="gripper")
        ],
    )
    contact_jaw = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/CubeSmallRed",
            "{ENV_REGEX_NS}/CubeSmallBlue",
            "{ENV_REGEX_NS}/CubeLargeRed",
            "{ENV_REGEX_NS}/CubeLargeBlue",
        ],
    )
    contact_gripper = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/CubeSmallRed",
            "{ENV_REGEX_NS}/CubeSmallBlue",
            "{ENV_REGEX_NS}/CubeLargeRed",
            "{ENV_REGEX_NS}/CubeLargeBlue",
        ],
    )
    front_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraFront",
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=0.35),
        # Isaac Lab 3.0 camera offsets use xyzw quaternions.  The collector
        # also sets this static camera with set_world_poses_from_view after
        # reset, so its optical axis is defined by an explicit look-at point.
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.42, -0.38, 0.34),
            rot=(-0.3815, -0.2510, -0.4317, 0.7829),
            convention="opengl",
        ),
    )
    wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/FarpointWristCamera",
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=13.5, focus_distance=0.08),
        # The original workshop-style value was written as wxyz.  OffsetCfg
        # in the pinned Isaac Lab release expects xyzw; leaving it unconverted
        # points the optical axis back into the gripper body.
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.005, 0.06, -0.062),
            rot=(-0.3826834, 0.0, 0.0, 0.9238795),
            convention="opengl",
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(0.95, 0.95, 0.95)),
    )


@configclass
class ActionsCfg:
    joint_positions = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_positions = ObsTerm(func=mdp.joint_positions)
        end_effector_position = ObsTerm(func=mdp.end_effector_position)
        cube_position = ObsTerm(func=mdp.active_cube_position)
        front_rgb = ObsTerm(func=mdp.camera_rgb, params={"sensor_cfg": SceneEntityCfg("front_camera")})
        wrist_rgb = ObsTerm(func=mdp.camera_rgb, params={"sensor_cfg": SceneEntityCfg("wrist_camera")})

        def __post_init__(self):
            self.concatenate_terms = False
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EmptyManagerCfg:
    """Placeholder for managers not needed by the oracle MVP."""


@configclass
class EventsCfg:
    disable_camera_mount_collision = EventTerm(
        func=mdp.disable_workshop_camera_mount_collision,
        mode="prestartup",
    )
    reset_scene_to_default = EventTerm(
        func=lab_mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class SO101CubePickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: SO101CubeSceneCfg = SO101CubeSceneCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: EmptyManagerCfg = EmptyManagerCfg()
    terminations: EmptyManagerCfg = EmptyManagerCfg()
    commands: EmptyManagerCfg = EmptyManagerCfg()
    curriculum: EmptyManagerCfg = EmptyManagerCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.render.rendering_mode = "balanced"
        self.viewer.eye = (0.45, -0.45, 0.35)
        self.viewer.lookat = (0.20, 0.02, 0.07)
