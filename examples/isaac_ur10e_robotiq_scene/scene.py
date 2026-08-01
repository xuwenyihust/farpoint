import ast
import contextlib
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


PROJECT_ROOT = Path("/workspace/project")
TASK_PATH = Path(
    os.environ.get(
        "FARPOINT_TASK_PATH",
        str(PROJECT_ROOT / "examples" / "isaac_ur10e_robotiq_scene" / "task.yaml"),
    )
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.benchmark import classify_failure, randomize_task
from farpoint.control import (
    apply_place_hover_guard,
    bilateral_grasp_ready,
    bounded_position_target,
    cartesian_tracking_servo_target,
    contact_pair_force_summary,
    filtered_contact_force,
    force_controlled_gripper_target,
    grasp_validation_decision,
    gripper_aperture_alignment,
    integral_visual_servo_grasp_target,
    merge_contact_group_samples,
    placement_converged,
    rate_limit_revolute_joint_targets,
    rmpflow_world_target,
    simulation_stop_reason,
    tactile_contact_hold_target,
    tactile_search_active,
    temporal_contact_confirmed,
    track_observed_pick_target,
    transport_grasp_support,
    unilateral_contact_recenter_target,
    undirected_axis_angle_error_degrees,
    update_contact_loss_streak,
)
from farpoint.dataset import validate_episode_dataset
from farpoint.episode_metadata import validate_simulator_metadata_v2
from farpoint.perception import (
    PerceptionError,
    estimate_dominant_color_pose,
    look_at_calibration,
    xy_error,
)
from farpoint.position_plan import apply_position_trial, load_position_plan
from farpoint.variation import load_variation_config, resolve_variation


ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def parse_simple_yaml(path):
    data = {}
    stack = [(0, data)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, raw_value = raw_line.strip().partition(":")
        value = raw_value.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value == "":
            child = {}
            current[key] = child
            stack.append((indent + 2, child))
            continue
        if value.startswith('"') and value.endswith('"'):
            current[key] = value[1:-1]
            continue
        normalized_value = value.lower()
        if normalized_value in {"true", "false"}:
            current[key] = normalized_value == "true"
            continue
        if normalized_value in {"null", "none"}:
            current[key] = None
            continue
        try:
            current[key] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            current[key] = value
    return data


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def write_json(path, payload):
    path.write_text(
        json.dumps(
            payload,
            default=json_default,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def utc_now():
    return datetime.now(timezone.utc)


def sha256_json(payload):
    encoded = json.dumps(
        payload,
        default=json_default,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rotation_matrix_quaternion_xyzw(matrix):
    """Convert a 3x3 rotation matrix to a normalized XYZW quaternion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.tolist()


def measured_joint_smoothness(joint_history):
    if len(joint_history) < 3:
        return None
    accelerations = []
    for index in range(2, len(joint_history)):
        previous = np.asarray(joint_history[index - 2], dtype=np.float64)
        current = np.asarray(joint_history[index - 1], dtype=np.float64)
        following = np.asarray(joint_history[index], dtype=np.float64)
        width = min(len(previous), len(current), len(following))
        accelerations.append(
            float(np.linalg.norm(following[:width] - 2.0 * current[:width] + previous[:width]))
        )
    return sum(accelerations) / len(accelerations) if accelerations else None


def append_phase(path, phase, **fields):
    payload = {"time": utc_now().isoformat(), "phase": phase, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                payload,
                default=json_default,
                sort_keys=True,
            )
            + "\n"
        )


def make_visual_shape(
    Cube,
    Cylinder,
    PreviewSurfaceMaterial,
    path,
    config,
    object_type="cube",
):
    material = PreviewSurfaceMaterial(f"/World/Materials/{Path(path).name}")
    material.set_input_values("diffuseColor", config["color"])
    if object_type == "cylinder":
        # Keep the cylinder within the 2F-85 aperture while giving RGB-D enough
        # pixels for stable pose estimation at the pilot camera resolution.
        radius_scale = float(config.get("cylinder_radius_scale", 0.72))
        shape = Cylinder(
            paths=config["path"],
            positions=config["position"],
            radii=float(config["size"]) * radius_scale,
            heights=float(config["size"]),
            axes="Z",
            scales=config["scale"],
        )
    else:
        shape = Cube(
            paths=config["path"],
            positions=config["position"],
            sizes=config["size"],
            scales=config["scale"],
        )
    shape.apply_visual_materials(material)
    return shape


def count_prim_subtree(stage, root_path):
    return sum(1 for prim in stage.Traverse() if str(prim.GetPath()).startswith(root_path))


def set_cube_transform(stage, path, center, scale):
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(path)
    xform = UsdGeom.Xformable(prim)
    translate_attr = prim.GetAttribute("xformOp:translate")
    scale_attr = prim.GetAttribute("xformOp:scale")
    if translate_attr:
        translate_attr.Set(tuple(center))
    else:
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(tuple(center))
    if scale_attr:
        scale_attr.Set(tuple(scale))
    else:
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(tuple(scale))


def apply_physics_body(stage, path, mass=None, kinematic=False):
    from pxr import PhysxSchema, UsdPhysics

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise ValueError(f"physics prim not found: {path}")
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    if mass is not None:
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    if hasattr(physx_body, "CreateKinematicEnabledAttr"):
        physx_body.CreateKinematicEnabledAttr(bool(kinematic))


def set_rigid_body_kinematic(stage, path, enabled):
    from pxr import PhysxSchema, Sdf

    prim = stage.GetPrimAtPath(path)
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    attr = prim.GetAttribute("physxRigidBody:kinematicEnabled")
    if attr and attr.IsValid():
        attr.Set(bool(enabled))
    elif hasattr(physx_body, "CreateKinematicEnabledAttr"):
        physx_body.CreateKinematicEnabledAttr(bool(enabled))
    else:
        prim.CreateAttribute("physxRigidBody:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(bool(enabled))


def set_drive_attr(drive_api, create_method_name, get_method_name, value):
    create_method = getattr(drive_api, create_method_name, None)
    get_method = getattr(drive_api, get_method_name, None)
    attr = get_method() if get_method else None
    if attr and attr.IsValid():
        attr.Set(value)
    elif create_method:
        create_method(value)


def find_joint_prim_path(stage, robot_root, joint_name):
    exact_paths = [
        f"{robot_root}/Joints/{joint_name}",
        f"{robot_root}/{joint_name}",
    ]
    for path in exact_paths:
        if stage.GetPrimAtPath(path):
            return path
    matches = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(robot_root) and path.rsplit("/", 1)[-1] == joint_name:
            matches.append(path)
    if not matches:
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(robot_root) and joint_name in path:
                matches.append(path)
    return sorted(matches, key=len)[0] if matches else None


def configure_arm_articulation_drives(stage, robot_root, arm_joint_names):
    from pxr import UsdPhysics

    drive_records = []
    for joint_name in arm_joint_names:
        joint_path = find_joint_prim_path(stage, robot_root, joint_name)
        if not joint_path:
            drive_records.append({"joint": joint_name, "path": None, "configured": False})
            continue
        prim = stage.GetPrimAtPath(joint_path)
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        set_drive_attr(drive, "CreateTypeAttr", "GetTypeAttr", "acceleration")
        set_drive_attr(drive, "CreateStiffnessAttr", "GetStiffnessAttr", 22000.0)
        set_drive_attr(drive, "CreateDampingAttr", "GetDampingAttr", 2600.0)
        set_drive_attr(drive, "CreateMaxForceAttr", "GetMaxForceAttr", 1000000000.0)
        drive_records.append(
            {
                "joint": joint_name,
                "path": joint_path,
                "configured": True,
                "drive": "angular",
                "type": "acceleration",
                "stiffness": 22000.0,
                "damping": 2600.0,
                "max_force": 1000000000.0,
            }
        )
    return drive_records


def create_fixed_grasp_joint(stage, joint_path, body0_path, body1_path, anchor_world_position=None):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    if stage.GetPrimAtPath(joint_path):
        stage.RemovePrim(joint_path)
    body0 = stage.GetPrimAtPath(body0_path)
    body1 = stage.GetPrimAtPath(body1_path)
    if not body0 or not body0.IsValid():
        raise ValueError(f"grasp body not found: {body0_path}")
    if not body1 or not body1.IsValid():
        raise ValueError(f"object body not found: {body1_path}")

    cache = UsdGeom.XformCache()
    body0_world = cache.GetLocalToWorldTransform(body0)
    body1_world = cache.GetLocalToWorldTransform(body1)
    anchor_position = anchor_world_position or body1_world.ExtractTranslation()
    local_pos0 = body0_world.GetInverse().Transform(anchor_position)
    local_pos1 = body1_world.GetInverse().Transform(anchor_position)
    joint_world_rotation = Gf.Quatd(1.0)
    local_rot0 = body0_world.ExtractRotationQuat().GetInverse() * joint_world_rotation
    local_rot1 = body1_world.ExtractRotationQuat().GetInverse() * joint_world_rotation

    joint = UsdPhysics.Joint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateLocalPos0Attr().Set((float(local_pos0[0]), float(local_pos0[1]), float(local_pos0[2])))
    joint.CreateLocalPos1Attr().Set((float(local_pos1[0]), float(local_pos1[1]), float(local_pos1[2])))
    imag0 = local_rot0.GetImaginary()
    imag1 = local_rot1.GetImaginary()
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(local_rot0.GetReal()), Gf.Vec3f(float(imag0[0]), float(imag0[1]), float(imag0[2]))))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(float(local_rot1.GetReal()), Gf.Vec3f(float(imag1[0]), float(imag1[1]), float(imag1[2]))))
    for axis in ["transX", "transY", "transZ", "rotX", "rotY", "rotZ"]:
        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
        limit.CreateLowAttr(0.0)
        limit.CreateHighAttr(0.0)
    return joint


def remove_grasp_joint(stage, joint_path):
    if stage.GetPrimAtPath(joint_path):
        stage.RemovePrim(joint_path)


def prim_has_mimic_signal(prim):
    if "mimic" in str(prim.GetPath()).lower():
        return True
    for attr in prim.GetAttributes():
        if "mimic" in attr.GetName().lower():
            return True
    return False


def prim_world_position(stage, path):
    from pxr import UsdGeom

    if not path:
        return None
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None
    translation = UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()
    return [float(translation[0]), float(translation[1]), float(translation[2])]


def prim_world_pose(stage, path):
    from pxr import UsdGeom

    if not path:
        return None, None
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None, None
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    translation = transform.ExtractTranslation()
    rotation = transform.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    return (
        [float(translation[0]), float(translation[1]), float(translation[2])],
        [
            float(rotation.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ],
    )


def prim_world_bounds(stage, path):
    from pxr import Usd, UsdGeom

    if not path:
        return None
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None
    bound = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    ).ComputeWorldBound(prim)
    aligned = bound.ComputeAlignedRange()
    minimum = aligned.GetMin()
    maximum = aligned.GetMax()
    return {
        "minimum": [float(minimum[i]) for i in range(3)],
        "maximum": [float(maximum[i]) for i in range(3)],
        "center": [
            float((minimum[i] + maximum[i]) * 0.5)
            for i in range(3)
        ],
    }


def select_variant(prim, variant_set_name, preferred_selection):
    variant_set = prim.GetVariantSets().GetVariantSet(variant_set_name)
    if not variant_set or not variant_set.IsValid():
        return None, []
    names = list(variant_set.GetVariantNames())
    if preferred_selection in names:
        variant_set.SetVariantSelection(preferred_selection)
        return preferred_selection, names
    preferred_lower = preferred_selection.lower()
    for name in names:
        if name.lower() == preferred_lower:
            variant_set.SetVariantSelection(name)
            return name, names
    return variant_set.GetVariantSelection(), names


def find_end_effector_prim_path(stage, robot_root, hint):
    exact_robotiq_base = f"{robot_root}/Robotiq_2F_85/base_link"
    if stage.GetPrimAtPath(exact_robotiq_base):
        return exact_robotiq_base
    candidates = []
    keywords = [hint.lower(), "robotiq", "base_link", "ee_link", "tool0", "flange"]
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(robot_root):
            continue
        lower = path.lower()
        if any(keyword and keyword in lower for keyword in keywords):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (0 if "robotiq" in item.lower() and "base_link" in item.lower() else 1, -len(item)))
    return candidates[0]


def read_recorded_frames(path):
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def joint_ranges_degrees(history, names):
    if not history:
        return {name: 0.0 for name in names}
    ranges = {}
    for index, name in enumerate(names):
        values = [row[index] for row in history]
        ranges[name] = round(math.degrees(max(values) - min(values)), 3)
    return ranges


def radians_to_degrees(values):
    return [round(math.degrees(float(value)), 3) for value in values]


def gripper_target(
    frame,
    attach_frame,
    release_frame,
    close_position=47.0,
    close_start_frame=None,
    close_end_frame=None,
):
    close_start = (
        max(0, attach_frame - 50)
        if close_start_frame is None
        else max(0, int(close_start_frame))
    )
    close_end = (
        max(close_start + 1, attach_frame - 10)
        if close_end_frame is None
        else max(close_start + 1, int(close_end_frame))
    )
    if frame < close_start:
        return 0.0
    if frame < close_end:
        return float(close_position) * smoothstep(
            (frame - close_start) / max(close_end - close_start, 1)
        )
    if frame < release_frame:
        return float(close_position)
    return float(close_position) * (
        1.0 - smoothstep((frame - release_frame) / 30.0)
    )


def configure_contact_material(
    stage,
    material_path,
    body_paths,
    static_friction,
    dynamic_friction,
    restitution,
):
    from pxr import UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, material_path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(float(static_friction))
    physics.CreateDynamicFrictionAttr(float(dynamic_friction))
    physics.CreateRestitutionAttr(float(restitution))
    bound_paths = []
    for path in body_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material,
            materialPurpose="physics",
        )
        bound_paths.append(path)
    return bound_paths


def annotator_array(annotator):
    data = annotator.get_data()
    if isinstance(data, dict):
        data = data.get("data")
    if data is None:
        return None
    return np.asarray(data)


def capture_rgbd(rgb_annotator, depth_annotator):
    rgb = annotator_array(rgb_annotator)
    depth = annotator_array(depth_annotator)
    if rgb is None or depth is None or rgb.size == 0 or depth.size == 0:
        raise PerceptionError("RGB-D annotators did not return a complete frame")
    return rgb[:, :, :3].astype(np.uint8), depth.astype(np.float32)


def read_contact_sample(
    sensor,
    required_body_path=None,
    physics_dt=1.0 / 60.0,
):
    if sensor is None:
        return 0.0, False
    reading = sensor.get_sensor_reading()
    if not reading.is_valid:
        return 0.0, False
    if required_body_path is not None:
        frame = sensor.get_data()
        filtered = filtered_contact_force(
            frame.get("contacts", []),
            required_body_path,
            physics_dt=physics_dt,
        )
        return max(0.0, float(filtered["force"])), True
    return max(0.0, float(reading.value)), True


def read_contact_group(
    sensors,
    required_body_path=None,
    physics_dt=1.0 / 60.0,
):
    samples = [
        read_contact_sample(
            sensor,
            required_body_path=required_body_path,
            physics_dt=physics_dt,
        )
        for sensor in (sensors or [])
    ]
    valid_samples = [force for force, valid in samples if valid]
    raw_pair_forces = {}
    for sensor in sensors or []:
        frame = sensor.get_data()
        for record in contact_pair_force_summary(
            frame.get("contacts", []),
            physics_dt=physics_dt,
        ):
            key = (record["body0"], record["body1"])
            existing = raw_pair_forces.get(key)
            if existing is None or record["force"] > existing["force"]:
                raw_pair_forces[key] = dict(record)
    return (
        max(valid_samples, default=0.0),
        bool(valid_samples),
        [force for force, _ in samples],
        sorted(
            raw_pair_forces.values(),
            key=lambda record: record["force"],
            reverse=True,
        ),
    )


def write_observation_artifacts(
    episode_dir,
    observation_index,
    rgb,
    depth,
):
    from PIL import Image

    rgb_relative = Path("observations") / "rgb" / f"{observation_index:06d}.png"
    depth_relative = Path("observations") / "depth" / f"{observation_index:06d}.npy"
    rgb_path = episode_dir / rgb_relative
    depth_path = episode_dir / depth_relative
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, depth)
    return rgb_relative.as_posix(), depth_relative.as_posix()


def usd_camera_calibration(stage, camera_path, resolution):
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(camera_path)
    if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
        raise PerceptionError(f"USD camera prim is unavailable: {camera_path}")
    camera = UsdGeom.Camera(prim)
    focal_length = float(camera.GetFocalLengthAttr().Get())
    horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
    vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
    width, height = int(resolution[0]), int(resolution[1])
    intrinsics = np.asarray(
        [
            [width * focal_length / horizontal_aperture, 0.0, width * 0.5],
            [0.0, height * focal_length / vertical_aperture, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    position = transform.ExtractTranslation()
    right = transform.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    down = transform.TransformDir(Gf.Vec3d(0.0, -1.0, 0.0))
    forward = transform.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, 0] = np.asarray(right, dtype=np.float64)
    camera_to_world[:3, 1] = np.asarray(down, dtype=np.float64)
    camera_to_world[:3, 2] = np.asarray(forward, dtype=np.float64)
    camera_to_world[:3, 3] = np.asarray(position, dtype=np.float64)
    return intrinsics, camera_to_world


def smoothstep(value):
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def lerp_pose(start_pose, end_pose, amount):
    alpha = smoothstep(amount)
    return {
        name: math.radians(start_pose[name] + (end_pose[name] - start_pose[name]) * alpha)
        for name in start_pose
    }


def arm_targets(frame, frame_count, base_positions, dof_names):
    targets = list(base_positions)
    ready_pose = {
        "shoulder_pan_joint": 8.0,
        "shoulder_lift_joint": -40.0,
        "elbow_joint": 62.0,
        "wrist_1_joint": -48.0,
        "wrist_2_joint": 50.0,
        "wrist_3_joint": 8.0,
    }
    approach_pose = {
        "shoulder_pan_joint": 17.5,
        "shoulder_lift_joint": -32.2,
        "elbow_joint": 67.2,
        "wrist_1_joint": -46.1,
        "wrist_2_joint": 48.4,
        "wrist_3_joint": 3.2,
    }
    lift_pose = {
        "shoulder_pan_joint": -1.8,
        "shoulder_lift_joint": -49.9,
        "elbow_joint": 53.3,
        "wrist_1_joint": -54.5,
        "wrist_2_joint": 46.5,
        "wrist_3_joint": 7.9,
    }
    settle_pose = {
        "shoulder_pan_joint": 0.4,
        "shoulder_lift_joint": -44.9,
        "elbow_joint": 60.4,
        "wrist_1_joint": -46.1,
        "wrist_2_joint": 55.2,
        "wrist_3_joint": 15.8,
    }
    t = frame / max(frame_count - 1, 1)
    if t < 0.30:
        pose = lerp_pose(ready_pose, approach_pose, t / 0.30)
    elif t < 0.72:
        pose = lerp_pose(approach_pose, lift_pose, (t - 0.30) / 0.42)
    else:
        pose = lerp_pose(lift_pose, settle_pose, (t - 0.72) / 0.28)
    for name, base in ready_pose.items():
        if name in dof_names:
            index = dof_names.index(name)
            targets[index] = pose[name]
    return targets


def interpolate_position(start, end, amount):
    alpha = smoothstep(amount)
    return [float(start[i] + (end[i] - start[i]) * alpha) for i in range(3)]


def cartesian_pick_and_place_target(
    frame,
    attach_frame,
    release_frame,
    pick_grasp_position,
    place_grasp_position,
    lift_height,
    grasp_retry_frames,
    lift_duration_frames=300,
    transfer_end_frame=None,
    place_hover_end_frame=None,
    place_descent_start_frame=None,
    place_descent_end_frame=None,
):
    ready = [0.72, 0.42, 0.72]
    pick_approach = [pick_grasp_position[0], pick_grasp_position[1], pick_grasp_position[2] + 0.22]
    pick_lift = [pick_grasp_position[0], pick_grasp_position[1], pick_grasp_position[2] + lift_height]
    place_lift = [place_grasp_position[0], place_grasp_position[1], place_grasp_position[2] + lift_height]
    place_hover = [place_grasp_position[0], place_grasp_position[1], place_grasp_position[2] + 0.08]
    retreat = [place_grasp_position[0], place_grasp_position[1], place_grasp_position[2] + 0.25]
    ready_end = max(1, attach_frame - 120)
    descend_start = max(ready_end + 1, attach_frame - 80)
    descend_end = max(descend_start + 1, attach_frame - 30)
    lift_start = attach_frame + grasp_retry_frames
    lift_end = lift_start + max(1, int(lift_duration_frames))
    transfer_end = (
        max(lift_end + 1, release_frame - 300)
        if transfer_end_frame is None
        else max(lift_end + 1, int(transfer_end_frame))
    )
    place_start = transfer_end + 1
    place_hover_end = (
        max(place_start + 1, release_frame - 160)
        if place_hover_end_frame is None
        else max(place_start + 1, int(place_hover_end_frame))
    )
    place_descent_start = (
        max(place_hover_end + 1, release_frame - 80)
        if place_descent_start_frame is None
        else max(place_hover_end + 1, int(place_descent_start_frame))
    )
    place_descent_end = (
        max(place_descent_start + 1, release_frame - 30)
        if place_descent_end_frame is None
        else max(place_descent_start + 1, int(place_descent_end_frame))
    )
    retreat_end = release_frame + 60

    if frame < ready_end:
        return interpolate_position(ready, pick_approach, frame / ready_end)
    if frame < descend_start:
        return list(pick_approach)
    if frame < descend_end:
        return interpolate_position(pick_approach, pick_grasp_position, (frame - descend_start) / (descend_end - descend_start))
    if frame < attach_frame:
        return list(pick_grasp_position)
    if frame < lift_start:
        return list(pick_grasp_position)
    if frame < lift_end:
        return interpolate_position(pick_grasp_position, pick_lift, (frame - lift_start) / (lift_end - lift_start))
    if frame < transfer_end:
        return interpolate_position(pick_lift, place_lift, (frame - lift_end) / (transfer_end - lift_end))
    if frame < place_start:
        return list(place_lift)
    if frame < place_hover_end:
        return interpolate_position(place_lift, place_hover, (frame - place_start) / (place_hover_end - place_start))
    if frame < place_descent_start:
        return list(place_hover)
    if frame < place_descent_end:
        return interpolate_position(
            place_hover,
            place_grasp_position,
            (frame - place_descent_start) / (place_descent_end - place_descent_start),
        )
    if frame < release_frame:
        return list(place_grasp_position)
    if frame < retreat_end:
        return interpolate_position(place_grasp_position, retreat, (frame - release_frame) / (retreat_end - release_frame))
    return retreat


def merge_action_positions(base_positions, action):
    merged = [float(value) for value in base_positions]
    if action is None or action.joint_positions is None:
        return merged
    positions = np.asarray(action.joint_positions, dtype=np.float32).tolist()
    indices = action.joint_indices
    if indices is None:
        for index, value in enumerate(positions[: len(merged)]):
            merged[index] = float(value)
        return merged
    for raw_index, value in zip(np.asarray(indices, dtype=np.int32).tolist(), positions):
        if 0 <= raw_index < len(merged):
            merged[raw_index] = float(value)
    return merged


def build_ur10e_rmpflow_controller(robot, physics_dt):
    import isaacsim.robot_motion.motion_generation as mg

    rmpflow_root = (
        "/isaac-sim/standalone_examples/deprecated/api/"
        "isaacsim.robot.manipulators/ur10e/rmpflow"
    )
    rmpflow = mg.lula.motion_policies.RmpFlow(
        robot_description_path=f"{rmpflow_root}/robot_descriptor.yaml",
        rmpflow_config_path=f"{rmpflow_root}/ur10e_rmpflow_common.yaml",
        urdf_path=f"{rmpflow_root}/ur10e.urdf",
        end_effector_frame_name="ee_link_robotiq_arg2f_base_link",
        maximum_substep_size=0.00334,
    )
    articulation_rmp = mg.ArticulationMotionPolicy(robot, rmpflow, physics_dt)
    controller = mg.MotionPolicyController(
        name="ur10e_rmpflow_pickup_controller",
        articulation_motion_policy=articulation_rmp,
    )
    return controller, rmpflow_root


def build_ur10e_lula_ik_solver(robot):
    import isaacsim.robot_motion.motion_generation as mg

    config_root = (
        "/isaac-sim/standalone_examples/deprecated/api/"
        "isaacsim.robot.manipulators/ur10e/rmpflow"
    )
    kinematics_solver = mg.LulaKinematicsSolver(
        robot_description_path=f"{config_root}/robot_descriptor.yaml",
        urdf_path=f"{config_root}/ur10e.urdf",
    )
    robot_position, robot_orientation = robot.get_world_pose()
    kinematics_solver.set_robot_base_pose(
        robot_position,
        robot_orientation,
    )
    articulation_solver = mg.ArticulationKinematicsSolver(
        robot,
        kinematics_solver,
        "ee_link_robotiq_arg2f_base_link",
    )
    return articulation_solver, config_root


def evaluate_success(task, metrics):
    criteria = task["success_criteria"]

    def at_most(metric_name, criterion_name):
        value = metrics.get(metric_name)
        return value is not None and float(value) <= float(criteria[criterion_name])

    checks = {
        "recorded_frames": metrics["recorded_frames"] >= int(criteria["min_recorded_frames"]),
        "robot_loaded": bool(metrics["robot_loaded"]),
        "robot_prim_count": metrics["robot_prim_count"] >= int(criteria["min_robot_prim_count"]),
        "articulation_controller_initialized": bool(metrics["articulation_controller_initialized"]),
        "articulation_dofs": metrics["articulation_dofs"] >= int(criteria["min_articulation_dofs"]),
        "controlled_joints": metrics["controlled_joints"] >= int(criteria["min_controlled_joints"]),
        "joint_motion": metrics["max_joint_motion_degrees"] >= float(criteria["min_joint_motion_degrees"]),
        "gripper_motion": metrics["gripper_motion_degrees"] >= float(criteria["min_gripper_motion_degrees"]),
        "preview_images": metrics["preview_images_written"] >= int(criteria["min_preview_images"]),
    }
    if criteria.get("require_robotiq_asset"):
        checks["robotiq_asset"] = metrics["robotiq_prim_count"] > 0
    if criteria.get("require_mimic_joint_prims"):
        checks["mimic_joint_prims"] = metrics["mimic_joint_prim_count"] > 0
    if criteria.get("require_real_grasp_body"):
        checks["real_grasp_body"] = bool(metrics["real_grasp_body_found"])
    if "min_lift_height" in criteria:
        checks["lift_height"] = metrics.get("object_lift_height", 0.0) >= float(criteria["min_lift_height"])
    if "min_object_attached_frames" in criteria:
        checks["object_attached_frames"] = metrics.get("object_attached_frames", 0) >= int(
            criteria["min_object_attached_frames"]
        )
    if "max_grasp_rigidity_error" in criteria:
        checks["grasp_rigidity"] = at_most("max_grasp_rigidity_error", "max_grasp_rigidity_error")
    if "max_grasp_attach_distance" in criteria:
        checks["grasp_attach_distance"] = at_most("grasp_attach_distance", "max_grasp_attach_distance")
    if criteria.get("require_grasp_created"):
        checks["grasp_created"] = bool(metrics.get("grasp_was_created"))
    if criteria.get("require_final_released"):
        checks["final_released"] = not bool(metrics.get("final_object_attached"))
    if criteria.get("require_final_inside_target_zone"):
        checks["final_inside_target_zone"] = bool(metrics.get("final_object_inside_target_zone"))
    if "min_release_settle_frames" in criteria:
        checks["release_settle_frames"] = metrics.get("release_settle_frames", 0) >= int(
            criteria["min_release_settle_frames"]
        )
    if criteria.get("require_place_release_gate_converged"):
        checks["place_release_gate_converged"] = (
            metrics.get("place_release_gate_open_frame") is not None
            and not bool(metrics.get("place_release_gate_timed_out"))
        )
    if criteria.get("require_place_descent_gate_converged"):
        checks["place_descent_gate_converged"] = (
            metrics.get("place_descent_gate_open_frame") is not None
        )
    if criteria.get("require_soft_landing"):
        checks["soft_landing"] = (
            metrics.get("soft_landing_support_frame") is not None
            and metrics.get("soft_release_complete_frame") is not None
            and metrics.get("soft_landing_state") == "released"
        )
    if criteria.get("require_grasp_approach_gate_converged"):
        checks["grasp_approach_gate_converged"] = (
            metrics.get("grasp_approach_gate_open_frame") is not None
        )
    if "max_final_target_xy_distance" in criteria:
        checks["final_target_xy_distance"] = at_most("final_target_xy_distance", "max_final_target_xy_distance")
    if "min_final_object_height" in criteria:
        checks["final_object_height"] = metrics.get("final_object_height", 0.0) >= float(
            criteria["min_final_object_height"]
        )
    if "max_final_object_height" in criteria:
        checks["max_final_object_height"] = at_most("final_object_height", "max_final_object_height")
    if "max_post_release_motion" in criteria:
        checks["post_release_stability"] = at_most("post_release_motion", "max_post_release_motion")
    if criteria.get("require_rgbd_perception"):
        checks["rgbd_perception"] = bool(metrics.get("rgbd_perception_succeeded"))
    if "max_perception_xy_error" in criteria:
        checks["perception_xy_error"] = at_most(
            "initial_object_perception_xy_error",
            "max_perception_xy_error",
        )
    if criteria.get("require_no_grasp_joint"):
        checks["no_grasp_joint"] = (
            metrics.get("grasp_constraint") == "contact_only"
            and not bool(metrics.get("temporary_grasp_joint_created"))
        )
    if "min_bilateral_contact_frames" in criteria:
        checks["bilateral_contact"] = metrics.get(
            "bilateral_contact_frames", 0
        ) >= int(criteria["min_bilateral_contact_frames"])
    if "min_transport_contact_frames" in criteria:
        checks["transport_contact"] = metrics.get(
            "max_continuous_transport_contact_frames",
            metrics.get("transport_contact_frames", 0),
        ) >= int(criteria["min_transport_contact_frames"])
    if criteria.get("require_no_transport_contact_loss"):
        checks["no_transport_contact_loss"] = (
            int(metrics.get("grasp_contact_loss_events", 0)) == 0
        )
    if criteria.get("require_grasp_proof_lift"):
        checks["grasp_proof_lift"] = (
            int(metrics.get("grasp_proof_successes", 0)) >= 1
            and metrics.get("grasp_proof_state") == "qualified"
        )
    if criteria.get("require_dataset_valid"):
        checks["dataset_valid"] = bool(metrics.get("dataset_valid"))
    if "min_dataset_observations" in criteria:
        checks["dataset_observations"] = metrics.get(
            "dataset_observation_count", 0
        ) >= int(criteria["min_dataset_observations"])
    return all(checks.values()), checks


def distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def add_vector(a, b):
    return [float(x) + float(y) for x, y in zip(a, b)]


def subtract_vector(a, b):
    return [float(x) - float(y) for x, y in zip(a, b)]


def pickup_phase(
    frame,
    attach_frame,
    release_frame,
    *,
    grasp_retry_frames=0,
    lift_duration_frames=300,
):
    if frame < attach_frame - 50:
        return "approach"
    if frame < attach_frame:
        return "grasp"
    lift_end = (
        attach_frame
        + int(grasp_retry_frames)
        + max(1, int(lift_duration_frames))
    )
    if frame < lift_end:
        return "lift"
    if frame < release_frame - 260:
        return "transfer"
    if frame < release_frame:
        return "place"
    if frame < release_frame + 45:
        return "release"
    return "settle"


def main():
    started_at = utc_now()
    base_task = parse_simple_yaml(TASK_PATH)
    episode_seed = int(os.environ.get("FARPOINT_EPISODE_SEED", "0"))
    position_plan_path = os.environ.get("FARPOINT_POSITION_PLAN") or None
    trial_id = os.environ.get("FARPOINT_TRIAL_ID") or None
    reserve_index = int(os.environ.get("FARPOINT_RESERVE_INDEX", "0"))
    variation_id = os.environ.get("FARPOINT_VARIATION_ID") or None
    variation = None
    position_trial = None
    position_plan = None
    if bool(position_plan_path) != bool(trial_id):
        raise ValueError("FARPOINT_POSITION_PLAN and FARPOINT_TRIAL_ID must be provided together")
    if position_plan_path:
        position_plan = load_position_plan(position_plan_path)
        base_task, position_trial = apply_position_trial(
            base_task,
            position_plan,
            trial_id,
            reserve_index=reserve_index,
        )
        variation = position_trial["variation"]
        variation_id = variation["variation_id"]
        episode_seed = int(position_trial["seed"])
    elif variation_id:
        variation_config_path = Path(
            os.environ.get(
                "FARPOINT_VARIATION_CONFIG",
                str(
                    PROJECT_ROOT
                    / "configs"
                    / "variations"
                    / "ur10e_robotiq_2f85_pickup.json"
                ),
            )
        )
        variation_config = load_variation_config(variation_config_path)
        variation = resolve_variation(variation_config, variation_id, episode_seed)
        base_task["scene"]["pick_object"]["position"][:2] = variation[
            "object_position_xy"
        ]
        base_task["randomization"]["enabled"] = False
        if variation.get("grasp_profile") == "cylinder_grip_v1":
            # Cylinders need a slightly higher, still physical, friction/force
            # envelope to survive the proof lift without a solver attachment.
            base_task["scene"]["pick_object"]["cylinder_radius_scale"] = 0.50
            base_task["pickup"]["static_friction"] = 2.5
            base_task["pickup"]["dynamic_friction"] = 2.0
            base_task["pickup"]["finger_contact_max_effort"] = 20.0
    benchmark_id = os.environ.get("FARPOINT_BENCHMARK_ID") or None
    benchmark_repeat = int(os.environ.get("FARPOINT_BENCHMARK_REPEAT", "0"))
    task, randomization = randomize_task(base_task, episode_seed)
    episode_suffix = f"_s{episode_seed:04d}" if benchmark_id else ""
    episode_id = f"episode_{started_at.strftime('%Y%m%d_%H%M%S')}{episode_suffix}"
    episode_dir = Path(task["output"]["root"]) / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    phase_path = episode_dir / "phase_events.jsonl"
    trajectory_path = episode_dir / "trajectory.jsonl"
    preview_dir = episode_dir / "preview"
    observations_path = episode_dir / "observations.jsonl"
    labels_path = episode_dir / "labels.jsonl"
    frame_limit = os.environ.get("FARPOINT_FRAME_LIMIT")
    frame_count = int(frame_limit) if frame_limit else int(task["frames"])
    record_every = int(task["record_every_n_frames"])
    camera_config = task.get("camera", {})
    perception_config = task.get("perception", {})
    dataset_config = task.get("dataset", {})
    pickup_config = task.get("pickup", {})
    rendering_dt = 1.0 / 60.0
    physics_substeps_per_frame = max(
        1,
        int(pickup_config.get("physics_substeps_per_frame", 1)),
    )
    physics_dt = rendering_dt / physics_substeps_per_frame
    contact_only = pickup_config.get("grasp_mode") == "contact_only"
    perception_enabled = bool(perception_config.get("enabled", False))
    dataset_enabled = bool(dataset_config.get("enabled", False))
    preview_enabled = bool(camera_config.get("enabled", False))
    preview_frames = set(camera_config.get("preview_frames", []))
    simulation_app = None
    try:
        append_phase(
            phase_path,
            "scene_script_start",
            episode_id=episode_id,
            episode_seed=episode_seed,
            variation_id=variation_id,
            benchmark_id=benchmark_id,
            benchmark_repeat=benchmark_repeat,
        )
        append_phase(
            phase_path,
            "task_randomized",
            episode_seed=episode_seed,
            pick_object_xy=randomization["pick_object_xy"],
            target_zone_xy=randomization["target_zone_xy"],
        )
        append_phase(phase_path, "simulation_app_start")
        simulation_app = SimulationApp({"headless": True, "multi_gpu": False})
        append_phase(phase_path, "simulation_app_ready")

        import omni.usd
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
        from isaacsim.core.experimental.objects import (
            Cube,
            Cylinder,
            DistantLight,
            GroundPlane,
        )
        from isaacsim.core.experimental.prims import GeomPrim
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import UsdGeom

        rep = None
        if preview_enabled:
            import omni.replicator.core as rep

        append_phase(phase_path, "scene_create_start")
        stage_utils.create_new_stage()
        world = World(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=1.0,
        )

        def step_control_frame():
            for substep in range(physics_substeps_per_frame):
                world.step(
                    render=substep == physics_substeps_per_frame - 1
                )

        stage = omni.usd.get_context().get_stage()

        if task["scene"].get("ground", {}).get("enabled", True):
            GroundPlane(task["scene"]["ground"]["path"], positions=[0, 0, 0])
        light = DistantLight(task["scene"]["lighting"]["path"])
        light.set_intensities(task["scene"]["lighting"]["intensity"])
        object_type = (
            variation.get("object_type")
            or variation.get("resolved", {}).get("object_shape")
            or "cube"
        ) if variation else "cube"
        for scene_key in ["table", "target_zone", "pick_object"]:
            shape = make_visual_shape(
                Cube,
                Cylinder,
                PreviewSurfaceMaterial,
                scene_key,
                task["scene"][scene_key],
                object_type if scene_key == "pick_object" else "cube",
            )
            if scene_key != "target_zone":
                GeomPrim(paths=shape.paths, apply_collision_apis=True)
        pick_object_config = task["scene"]["pick_object"]
        apply_physics_body(
            stage,
            pick_object_config["path"],
            mass=float(pick_object_config.get("mass", 0.12)),
            kinematic=False,
        )

        robot_config = task["robot"]
        robot_prim = stage.DefinePrim(robot_config["prim_path"], "Xform")
        robot_prim.GetReferences().AddReference(robot_config["asset_path"])
        selected_gripper, gripper_variants = select_variant(
            robot_prim, "Gripper", robot_config.get("gripper_variant", "Robotiq_2F_85")
        )
        selected_physics, physics_variants = select_variant(
            robot_prim, "Physics", robot_config.get("physics_variant", "PhysX")
        )
        UsdGeom.XformCommonAPI(robot_prim).SetTranslate(tuple(robot_config["position"]))
        UsdGeom.XformCommonAPI(robot_prim).SetRotate(tuple(robot_config["rotation_degrees"]))
        for _ in range(8):
            simulation_app.update()

        robot_prim_count = count_prim_subtree(stage, robot_config["prim_path"])
        arm_drive_records = configure_arm_articulation_drives(stage, robot_config["prim_path"], ARM_JOINT_NAMES)
        prim_paths = [str(prim.GetPath()) for prim in stage.Traverse() if str(prim.GetPath()).startswith(robot_config["prim_path"])]
        robotiq_paths = [path for path in prim_paths if "robotiq" in path.lower()]
        mimic_joint_paths = [
            path
            for path in prim_paths
            if prim_has_mimic_signal(stage.GetPrimAtPath(path))
        ]
        end_effector_prim_path = find_end_effector_prim_path(
            stage, robot_config["prim_path"], robot_config.get("end_effector_path_hint", "")
        )
        grasp_body_path = pickup_config.get("grasp_body_path") or end_effector_prim_path
        real_grasp_body_found = bool(grasp_body_path and stage.GetPrimAtPath(grasp_body_path))
        left_contact_body_paths = list(
            pickup_config.get("left_contact_body_paths")
            or [pickup_config.get("left_contact_body_path")]
        )
        right_contact_body_paths = list(
            pickup_config.get("right_contact_body_paths")
            or [pickup_config.get("right_contact_body_path")]
        )
        left_contact_body_paths = [
            path for path in left_contact_body_paths if path
        ]
        right_contact_body_paths = [
            path for path in right_contact_body_paths if path
        ]
        left_contact_body_path = (
            left_contact_body_paths[0]
            if left_contact_body_paths
            else None
        )
        right_contact_body_path = (
            right_contact_body_paths[0]
            if right_contact_body_paths
            else None
        )
        contact_material_bound_paths = []
        contact_sensor_paths = {
            "left_finger": [],
            "right_finger": [],
        }
        ContactSensor = None
        if contact_only:
            from isaacsim.sensors.experimental.physics import Contact, ContactSensor

            contact_material_bound_paths = configure_contact_material(
                stage,
                "/World/Materials/ContactGrip",
                [
                    pick_object_config["path"],
                    *left_contact_body_paths,
                    *right_contact_body_paths,
                ],
                pickup_config.get("static_friction", 1.5),
                pickup_config.get("dynamic_friction", 1.2),
                pickup_config.get("restitution", 0.0),
            )
            for side, body_paths in (
                ("left_finger", left_contact_body_paths),
                ("right_finger", right_contact_body_paths),
            ):
                for sensor_index, body_path in enumerate(body_paths):
                    if not stage.GetPrimAtPath(body_path):
                        continue
                    sensor_path = (
                        f"{body_path}/farpoint_contact_sensor_"
                        f"{sensor_index}"
                    )
                    Contact.create(
                        sensor_path,
                        min_threshold=0.0,
                        max_threshold=1000000.0,
                        radius=0.12,
                        translations=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
                    )
                    contact_sensor_paths[side].append(sensor_path)

        robot = world.scene.add(SingleArticulation(prim_path=robot_config["prim_path"], name=robot_config["name"]))
        initial_arm_pose = [-math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0.0]
        initial_robot_pose = np.asarray(initial_arm_pose + [0.0] * 6, dtype=np.float32)
        if hasattr(robot, "set_joints_default_state"):
            robot.set_joints_default_state(positions=initial_robot_pose)
        append_phase(phase_path, "world_reset_start")
        world.reset()
        append_phase(phase_path, "world_reset_end")
        controller = robot.get_articulation_controller()
        controller.switch_control_mode("position")
        dof_names = list(robot.dof_names)
        finger_joint_names = [name for name in dof_names if "finger" in name.lower()]
        primary_finger_joint = "finger_joint" if "finger_joint" in dof_names else (finger_joint_names[0] if finger_joint_names else None)
        primary_finger_index = robot.get_dof_index(primary_finger_joint) if primary_finger_joint else None
        actuated_finger_joint_names = (
            [primary_finger_joint] if primary_finger_joint else []
        )
        mimic_follower_joint_names = [
            name
            for name in finger_joint_names
            if name != primary_finger_joint
        ]
        controlled_joint_names = [
            name
            for name in [*ARM_JOINT_NAMES, *actuated_finger_joint_names]
            if name in dof_names
        ]
        controlled_joint_indices = [
            dof_names.index(name) for name in controlled_joint_names
        ]
        kps = None
        kds = None
        max_efforts = None
        finger_search_max_effort = float(
            pickup_config.get("finger_max_effort", 180.0)
        )
        if hasattr(controller, "set_gains"):
            kps = np.full(robot.num_dof, 1200.0, dtype=np.float32)
            kds = np.full(robot.num_dof, 80.0, dtype=np.float32)
            for joint_name in ARM_JOINT_NAMES:
                if joint_name in dof_names:
                    joint_index = dof_names.index(joint_name)
                    kps[joint_index] = 22000.0
                    kds[joint_index] = 2600.0
            if contact_only:
                for joint_name in mimic_follower_joint_names:
                    joint_index = dof_names.index(joint_name)
                    kps[joint_index] = 0.0
                    kds[joint_index] = 0.0
                for joint_name in actuated_finger_joint_names:
                    joint_index = dof_names.index(joint_name)
                    kps[joint_index] = float(
                        pickup_config.get("finger_stiffness", 600.0)
                    )
                    kds[joint_index] = float(
                        pickup_config.get("finger_damping", 45.0)
                    )
            controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
        if hasattr(controller, "set_max_efforts"):
            max_efforts = np.full(robot.num_dof, 1000000000.0, dtype=np.float32)
            if contact_only:
                for joint_name in mimic_follower_joint_names:
                    max_efforts[dof_names.index(joint_name)] = 0.0
                for joint_name in actuated_finger_joint_names:
                    max_efforts[dof_names.index(joint_name)] = float(
                        finger_search_max_effort
                    )
            controller.set_max_efforts(max_efforts)
        if hasattr(robot, "set_solver_position_iteration_count"):
            robot.set_solver_position_iteration_count(64)
        if hasattr(robot, "set_solver_velocity_iteration_count"):
            robot.set_solver_velocity_iteration_count(64)
        if hasattr(robot, "post_reset"):
            robot.post_reset()
        if hasattr(robot, "set_world_pose"):
            robot.set_world_pose(position=np.asarray(robot_config["position"], dtype=np.float32))
            for _ in range(2):
                step_control_frame()
        robot_world_position, robot_world_orientation = robot.get_world_pose()
        append_phase(
            phase_path,
            "robot_base_pose_applied",
            requested_position=robot_config["position"],
            actual_position=np.asarray(robot_world_position, dtype=np.float32).tolist(),
        )
        arm_drive_records = configure_arm_articulation_drives(stage, robot_config["prim_path"], ARM_JOINT_NAMES)
        if hasattr(controller, "set_gains"):
            controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
        if hasattr(controller, "set_max_efforts"):
            controller.set_max_efforts(max_efforts)
        if hasattr(robot, "set_joint_positions") and robot.num_dof == len(initial_robot_pose):
            robot.set_joint_positions(initial_robot_pose)
            for _ in range(4):
                step_control_frame()
        articulation_controller_initialized = bool(robot.handles_initialized)
        contact_sensors = {
            side: [ContactSensor(path) for path in paths]
            for side, paths in contact_sensor_paths.items()
        }
        for sensors in contact_sensors.values():
            for sensor in sensors:
                sensor.add_raw_contact_data_to_frame()

        base_positions = robot.get_joint_positions()
        base_positions = np.asarray(base_positions, dtype=np.float32).tolist()
        if len(base_positions) < robot.num_dof:
            base_positions.extend([0.0] * (robot.num_dof - len(base_positions)))
        joint_limits = None
        if hasattr(robot, "get_dof_limits"):
            raw_joint_limits = np.asarray(
                robot.get_dof_limits(),
                dtype=np.float32,
            ).squeeze()
            if (
                raw_joint_limits.ndim == 2
                and raw_joint_limits.shape == (robot.num_dof, 2)
            ):
                joint_limits = raw_joint_limits.tolist()

        rgb_annotator = None
        depth_annotator = None
        camera_intrinsics = None
        camera_to_world = None
        camera_prim_path = None
        preview_writer = None
        if preview_enabled:
            preview_dir.mkdir(parents=True, exist_ok=True)
            camera = rep.create.camera(position=tuple(camera_config["position"]), look_at=tuple(camera_config["target"]))
            render_product = rep.create.render_product(camera, tuple(camera_config["resolution"]))
            if perception_enabled or dataset_enabled:
                rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                depth_annotator = rep.AnnotatorRegistry.get_annotator(
                    "distance_to_image_plane"
                )
                rgb_annotator.attach([render_product])
                depth_annotator.attach([render_product])
                camera_paths = [
                    str(prim.GetPath())
                    for prim in stage.Traverse()
                    if prim.IsA(UsdGeom.Camera)
                ]
                camera_prim_path = camera_paths[-1] if camera_paths else None
                camera_intrinsics, camera_to_world = look_at_calibration(
                    camera_config["position"],
                    camera_config["target"],
                    camera_config["resolution"],
                )
            else:
                preview_writer = rep.WriterRegistry.get("BasicWriter")
                preview_writer.initialize(output_dir=str(preview_dir), rgb=True)

        app_utils.play()
        step_control_frame()
        initial_object_estimate = None
        initial_target_estimate = None
        latest_object_estimate = None
        latest_target_estimate = None
        latest_rgb = None
        latest_depth = None
        perception_failures = []
        perception_updates = 0
        if perception_enabled:
            for _ in range(int(perception_config.get("warmup_frames", 8))):
                step_control_frame()
            rep.orchestrator.step(
                rt_subframes=int(camera_config.get("rt_subframes", 1))
            )
            app_utils.play()
            latest_rgb, latest_depth = capture_rgbd(
                rgb_annotator,
                depth_annotator,
            )
            object_half_height_for_perception = (
                float(pick_object_config["scale"][2])
                * float(pick_object_config["size"])
                * 0.5
            )
            initial_object_estimate = estimate_dominant_color_pose(
                latest_rgb,
                latest_depth,
                camera_intrinsics,
                camera_to_world,
                perception_config.get("object_channel", "red"),
                min_pixels=perception_config.get("min_pixels", 20),
                min_channel=perception_config.get("min_channel", 80),
                min_dominance=perception_config.get("min_dominance", 30),
                surface_to_center_m=object_half_height_for_perception,
            )
            initial_target_estimate = estimate_dominant_color_pose(
                latest_rgb,
                latest_depth,
                camera_intrinsics,
                camera_to_world,
                perception_config.get("target_channel", "green"),
                min_pixels=perception_config.get("min_pixels", 20),
                min_channel=perception_config.get("min_channel", 80),
                min_dominance=perception_config.get("min_dominance", 30),
            )
            latest_object_estimate = initial_object_estimate
            latest_target_estimate = initial_target_estimate
            perception_updates = 1
            append_phase(
                phase_path,
                "rgbd_initial_pose_estimated",
                object_position=initial_object_estimate["position"],
                target_position=initial_target_estimate["position"],
                object_confidence=initial_object_estimate["confidence"],
                target_confidence=initial_target_estimate["confidence"],
            )
        append_phase(
            phase_path,
            "articulation_controller_ready",
            initialized=articulation_controller_initialized,
            dofs=robot.num_dof,
            controlled_joints=len(controlled_joint_names),
            gripper_joint=primary_finger_joint,
            arm_drive_mode=pickup_config.get("arm_motion_mode", "controller_position_targets"),
        )
        append_phase(
            phase_path,
            "arm_articulation_drives_configured",
            configured=sum(1 for row in arm_drive_records if row["configured"]),
            requested=len(arm_drive_records),
        )
        append_phase(
            phase_path,
            "scene_created",
            robot=robot_config["name"],
            robot_prim_count=robot_prim_count,
            robotiq_prim_count=len(robotiq_paths),
            mimic_joint_prim_count=len(mimic_joint_paths),
        )
        append_phase(
            phase_path,
            "physics_timing_configured",
            control_frequency_hz=round(1.0 / rendering_dt, 3),
            physics_frequency_hz=round(1.0 / physics_dt, 3),
            physics_substeps_per_control_frame=(
                physics_substeps_per_frame
            ),
        )

        joint_history = []
        target_history = []
        end_effector_history = []
        object_history = []
        attached_distance_history = []
        phase_names_seen = []
        grasp_mode = pickup_config.get("grasp_mode", "kinematic_contact_lock")
        arm_motion_mode = pickup_config.get("arm_motion_mode", "controller_position_targets")
        attach_frame = int(pickup_config.get("attach_frame", frame_count * 0.34))
        grasp_retry_frames = max(0, int(pickup_config.get("grasp_retry_frames", 0)))
        release_frame = int(pickup_config.get("release_frame", frame_count * 0.88))
        gripper_close_start_frame = pickup_config.get("gripper_close_start_frame")
        gripper_close_end_frame = pickup_config.get("gripper_close_end_frame")
        lift_height = float(pickup_config.get("lift_height", 0.18))
        max_grasp_attach_distance = float(pickup_config.get("max_grasp_attach_distance", 0.09))
        object_grasp_offset = list(pickup_config.get("object_grasp_offset", [0.0, 0.0, 0.0]))
        grasp_alignment_start_frame = int(
            pickup_config.get(
                "grasp_alignment_start_frame",
                max(0, attach_frame - 30),
            )
        )
        grasp_alignment_mode = pickup_config.get(
            "grasp_alignment_mode",
            "cartesian_tracking",
        )
        grasp_tracking_gain = pickup_config.get(
            "grasp_tracking_gain",
            0.25,
        )
        grasp_tracking_max_step = pickup_config.get(
            "grasp_tracking_max_step",
            0.01,
        )
        grasp_tracking_max_correction = float(
            pickup_config.get(
                "grasp_tracking_max_correction",
                0.12,
            )
        )
        grasp_tracking_max_xy_error = float(
            pickup_config.get(
                "grasp_tracking_max_xy_error",
                0.015,
            )
        )
        grasp_tracking_max_z_error = float(
            pickup_config.get(
                "grasp_tracking_max_z_error",
                0.015,
            )
        )
        grasp_approach_max_xy_error = float(
            pickup_config.get(
                "grasp_approach_max_xy_error",
                grasp_tracking_max_xy_error,
            )
        )
        grasp_tracking_max_finger_z_skew = float(
            pickup_config.get(
                "grasp_tracking_max_finger_z_skew",
                float("inf"),
            )
        )
        grasp_axis_target_yaw_degrees = float(
            pickup_config.get("grasp_axis_target_yaw_degrees", 90.0)
        )
        grasp_axis_max_error_degrees = float(
            pickup_config.get("grasp_axis_max_error_degrees", 10.0)
        )
        grasp_axis_calibration_max_error_degrees = float(
            pickup_config.get(
                "grasp_axis_calibration_max_error_degrees",
                grasp_axis_max_error_degrees,
            )
        )
        grasp_descent_max_xy_error = float(
            pickup_config.get(
                "grasp_descent_max_xy_error",
                grasp_approach_max_xy_error,
            )
        )
        grasp_descent_max_axis_yaw_error = float(
            pickup_config.get(
                "grasp_descent_max_axis_yaw_error_degrees",
                grasp_axis_max_error_degrees,
            )
        )
        grasp_axis_max_delta_degrees = float(
            pickup_config.get(
                "grasp_axis_max_delta_degrees_per_frame",
                float("inf"),
            )
        )
        grasp_aperture_bias_xy = [
            float(value)
            for value in pickup_config.get(
                "grasp_aperture_bias_xy",
                [0.0, 0.0],
            )
        ]
        if len(grasp_aperture_bias_xy) != 2:
            raise ValueError(
                "grasp_aperture_bias_xy must contain two values"
            )
        grasp_tracking_stable_frames = max(
            1,
            int(
                pickup_config.get(
                    "grasp_tracking_stable_frames",
                    20,
                )
            ),
        )
        freeze_visual_target_after_xy_gate = bool(
            pickup_config.get(
                "freeze_visual_target_after_xy_gate",
                True,
            )
        )
        grasp_visual_lock_source = pickup_config.get(
            "grasp_visual_lock_source",
            "xy_gate",
        )
        grasp_close_duration_frames = max(
            1,
            int(
                pickup_config.get(
                    "grasp_close_duration_frames",
                    180,
                )
            ),
        )
        gripper_max_position_error = float(
            pickup_config.get(
                "gripper_max_position_error_radians",
                float("inf"),
            )
        )
        finger_close_stiffness = float(
            pickup_config.get(
                "finger_close_stiffness",
                pickup_config.get("finger_stiffness", 600.0),
            )
        )
        finger_close_damping = float(
            pickup_config.get(
                "finger_close_damping",
                pickup_config.get("finger_damping", 45.0),
            )
        )
        finger_contact_max_effort = float(
            pickup_config.get(
                "finger_contact_max_effort",
                pickup_config.get("finger_max_effort", 180.0),
            )
        )
        unilateral_recenter_enabled = bool(
            pickup_config.get("unilateral_recenter_enabled", False)
        )
        unilateral_recenter_step = float(
            pickup_config.get("unilateral_recenter_step", 0.00025)
        )
        unilateral_recenter_max_correction = float(
            pickup_config.get("unilateral_recenter_max_correction", 0.02)
        )
        unilateral_recenter_persistence_frames = max(
            0,
            int(
                pickup_config.get(
                    "unilateral_recenter_persistence_frames",
                    0,
                )
            ),
        )
        unilateral_force_limit = float(
            pickup_config.get("unilateral_force_limit_newtons", 5.0)
        )
        unilateral_force_backoff = float(
            pickup_config.get("unilateral_force_backoff_radians", 0.0003)
        )
        bilateral_force_limit = float(
            pickup_config.get("bilateral_force_limit_newtons", float("inf"))
        )
        bilateral_force_backoff = float(
            pickup_config.get("bilateral_force_backoff_radians", 0.0005)
        )
        bilateral_hold_min_force = float(
            pickup_config.get("bilateral_hold_min_force_newtons", 1.0)
        )
        bilateral_hold_max_force = float(
            pickup_config.get(
                "bilateral_hold_max_force_newtons",
                bilateral_force_limit,
            )
        )
        ik_max_joint_step = float(
            pickup_config.get("ik_max_joint_step_radians", 0.004)
        )
        lula_ik_all_phases = bool(
            pickup_config.get("lula_ik_all_phases", False)
        )
        post_grasp_target_return_step = float(
            pickup_config.get("post_grasp_target_return_step", 0.0)
        )
        end_effector_orientation_degrees = pickup_config.get(
            "end_effector_orientation_degrees"
        )
        place_servo_frames = max(0, int(pickup_config.get("place_servo_frames", 0)))
        place_servo_gain = pickup_config.get("place_servo_gain", 0.0)
        place_servo_max_step = pickup_config.get(
            "place_servo_max_step",
            0.0,
        )
        place_servo_max_correction = float(pickup_config.get("place_servo_max_correction", 0.0))
        place_release_clearance = float(
            pickup_config.get("place_release_clearance", 0.01)
        )
        place_release_max_xy_error = float(
            pickup_config.get("place_release_max_xy_error", 0.03)
        )
        place_release_max_z_error = float(
            pickup_config.get("place_release_max_z_error", 0.04)
        )
        place_release_max_grasp_tracking_error = float(
            pickup_config.get(
                "place_release_max_grasp_tracking_error",
                0.04,
            )
        )
        place_require_grasp_tracking = bool(
            pickup_config.get(
                "place_require_grasp_tracking",
                True,
            )
        )
        place_hover_guard_clearance = float(
            pickup_config.get("place_hover_guard_clearance", 0.10)
        )
        place_release_stable_updates = max(
            1,
            int(pickup_config.get("place_release_stable_updates", 2)),
        )
        place_release_timeout_frames = max(
            1,
            int(pickup_config.get("place_release_timeout_frames", 600)),
        )
        place_descent_start_frame = int(
            pickup_config.get(
                "place_descent_start_frame",
                release_frame - 80,
            )
        )
        place_descent_gate_max_object_xy_error = max(
            0.0,
            float(
                pickup_config.get(
                    "place_descent_gate_max_object_xy_error",
                    0.02,
                )
            ),
        )
        place_descent_gate_max_grasp_xy_error = max(
            0.0,
            float(
                pickup_config.get(
                    "place_descent_gate_max_grasp_xy_error",
                    0.025,
                )
            ),
        )
        place_descent_gate_stable_updates = max(
            1,
            int(
                pickup_config.get(
                    "place_descent_gate_stable_updates",
                    3,
                )
            ),
        )
        soft_landing_enabled = bool(
            pickup_config.get("soft_landing_enabled", False)
        )
        soft_landing_clearance = max(
            0.0,
            float(pickup_config.get("soft_landing_clearance", 0.003)),
        )
        soft_landing_step = max(
            0.0,
            float(pickup_config.get("soft_landing_step_meters", 0.00015)),
        )
        soft_landing_force_threshold = max(
            0.0,
            float(
                pickup_config.get(
                    "soft_landing_force_threshold_newtons",
                    8.0,
                )
            ),
        )
        soft_landing_force_max_clearance = max(
            soft_landing_clearance,
            float(
                pickup_config.get(
                    "soft_landing_force_max_clearance",
                    0.03,
                )
            ),
        )
        soft_landing_hold_frames = max(
            1,
            int(pickup_config.get("soft_landing_hold_frames", 30)),
        )
        soft_release_frames = max(
            1,
            int(pickup_config.get("soft_release_frames", 90)),
        )
        transport_recenter_enabled = bool(
            pickup_config.get("transport_recenter_enabled", False)
        )
        transport_recenter_step = max(
            0.0,
            float(
                pickup_config.get(
                    "transport_recenter_step_meters",
                    0.0005,
                )
            ),
        )
        transport_recenter_max_correction = max(
            0.0,
            float(
                pickup_config.get(
                    "transport_recenter_max_correction",
                    0.015,
                )
            ),
        )
        grasp_joint_path = "/World/UR10eRobotiqGraspJoint"
        grasp_attached = False
        grasp_created = False
        grasp_was_created = False
        grasp_attach_distance = None
        grasp_attempt_count = 0
        grasp_success_frame = None
        grasp_center_distance_at_attach = None
        grasp_offset_world = None
        physical_grasp_contact_offset_world = None
        table_config = task["scene"]["table"]
        target_zone_config = task["scene"]["target_zone"]
        table_top = float(table_config["position"][2]) + float(table_config["scale"][2]) * float(table_config["size"]) * 0.5
        object_half_height = float(pick_object_config["scale"][2]) * float(pick_object_config["size"]) * 0.5
        supported_object_height = table_top + object_half_height
        grasp_height_offset = float(
            pickup_config.get("grasp_height_offset_meters", 0.0)
        )
        grasp_target_height = (
            supported_object_height + grasp_height_offset
        )
        ground_truth_pick_start_position = [
            float(pick_object_config["position"][0]),
            float(pick_object_config["position"][1]),
            supported_object_height,
        ]
        ground_truth_place_target_position = [
            float(target_zone_config["position"][0]),
            float(target_zone_config["position"][1]),
            supported_object_height,
        ]
        if perception_enabled:
            pick_start_position = list(initial_object_estimate["position"])
            pick_start_position[2] = supported_object_height
            place_target_position = list(initial_target_estimate["position"])
            place_target_position[2] = supported_object_height
            control_pose_source = "rgbd_estimate"
        else:
            pick_start_position = list(ground_truth_pick_start_position)
            place_target_position = list(ground_truth_place_target_position)
            control_pose_source = "task_ground_truth"
        pick_grasp_reference = list(pick_start_position)
        pick_grasp_reference[2] = grasp_target_height
        place_grasp_reference = list(place_target_position)
        place_grasp_reference[2] = grasp_target_height
        pick_grasp_position = subtract_vector(
            pick_grasp_reference,
            object_grasp_offset,
        )
        nominal_pick_grasp_position = list(pick_grasp_position)
        place_grasp_position = subtract_vector(
            place_grasp_reference,
            object_grasp_offset,
        )
        place_grasp_position[0] += float(pickup_config.get("place_grasp_x_compensation", 0.0))
        place_grasp_position[1] += float(pickup_config.get("place_grasp_y_compensation", 0.0))
        nominal_place_grasp_position = list(place_grasp_position)
        place_servo_correction = [0.0, 0.0]
        place_servo_updates = 0
        place_servo_last_error = None
        place_servo_last_error_xyz = None
        place_control_pose_source = "rgbd_estimate"
        place_control_object_estimate = None
        place_kinematic_estimate_start_frame = None
        place_release_gate_start_frame = None
        place_release_gate_open_frame = None
        place_release_gate_timed_out = False
        place_release_converged_updates = 0
        place_release_grasp_tracking_error = None
        place_hover_guard_activations = 0
        place_hover_guard_active = False
        place_descent_gate_start_frame = None
        place_descent_gate_open_frame = None
        place_descent_gate_converged_updates = 0
        soft_landing_state = "inactive"
        soft_landing_start_frame = None
        soft_landing_support_frame = None
        soft_landing_support_reason = None
        soft_landing_object_estimate = None
        soft_landing_command_position = None
        soft_landing_clearance_at_support = None
        soft_landing_peak_force_at_support = None
        soft_landing_force_baseline = None
        soft_release_start_frame = None
        soft_release_complete_frame = None
        soft_release_initial_target = None
        transport_recenter_offset_xy = [0.0, 0.0]
        transport_recenter_updates = 0
        transport_recenter_last_side = None
        pick_lift_position = [
            float(pick_start_position[0]),
            float(pick_start_position[1]),
            float(pick_start_position[2]) + lift_height,
        ]
        rmpflow_controller = None
        rmpflow_config_root = None
        lula_ik_solver = None
        lula_ik_config_root = None
        lula_ik_successes = 0
        lula_ik_failures = 0
        if arm_motion_mode in {
            "rmpflow_motion_policy",
            "hybrid_rmpflow_lula_ik",
        }:
            from isaacsim.core.utils.rotations import (
                euler_angles_to_quat,
                rot_matrix_to_quat,
            )

            rmpflow_controller, rmpflow_config_root = build_ur10e_rmpflow_controller(
                robot,
                physics_dt,
            )
            rmpflow_controller.reset()
            rmpflow_robot_position, rmpflow_robot_orientation = (
                robot.get_world_pose()
            )
            rmpflow_controller.get_motion_policy().set_robot_base_pose(
                robot_position=rmpflow_robot_position,
                robot_orientation=rmpflow_robot_orientation,
            )
            target_end_effector_orientation = (
                euler_angles_to_quat(
                    np.radians(
                        np.asarray(
                            end_effector_orientation_degrees,
                            dtype=np.float32,
                        )
                    )
                )
                if end_effector_orientation_degrees is not None
                else None
            )
            append_phase(phase_path, "rmpflow_controller_ready", config_root=rmpflow_config_root)
            append_phase(
                phase_path,
                "rmpflow_base_pose_configured",
                position=np.asarray(
                    rmpflow_robot_position,
                    dtype=np.float32,
                ).tolist(),
                orientation=np.asarray(
                    rmpflow_robot_orientation,
                    dtype=np.float32,
                ).tolist(),
                configured_after_reset=True,
            )
            append_phase(
                phase_path,
                "end_effector_orientation_target_configured",
                orientation_degrees=end_effector_orientation_degrees,
                orientation_quaternion=(
                    np.asarray(
                        target_end_effector_orientation,
                        dtype=np.float32,
                    ).tolist()
                    if target_end_effector_orientation is not None
                    else None
                ),
            )
        if arm_motion_mode == "hybrid_rmpflow_lula_ik":
            lula_ik_solver, lula_ik_config_root = (
                build_ur10e_lula_ik_solver(robot)
            )
            append_phase(
                phase_path,
                "lula_ik_solver_ready",
                config_root=lula_ik_config_root,
                end_effector_frame=(
                    "ee_link_robotiq_arg2f_base_link"
                ),
            )
        else:
            target_end_effector_orientation = None
        arm_joint_indices = [
            dof_names.index(joint_name)
            for joint_name in ARM_JOINT_NAMES
            if joint_name in dof_names
        ]
        grasp_body_position, grasp_body_orientation = prim_world_pose(
            stage,
            grasp_body_path,
        )
        control_frame_position = None
        control_frame_rotation = None
        if lula_ik_solver is not None:
            (
                control_frame_position,
                control_frame_rotation,
            ) = lula_ik_solver.compute_end_effector_pose()
        contact_force_history = []
        bilateral_contact_frames = 0
        raw_bilateral_contact_frames = 0
        debounced_bilateral_contact_frames = 0
        raw_bilateral_contact_streak = 0
        max_raw_bilateral_contact_streak = 0
        raw_bilateral_contact_event_frames = []
        max_recent_raw_bilateral_events = 0
        last_left_contact_frame = None
        last_right_contact_frame = None
        transport_contact_frames = 0
        direct_transport_contact_frames = 0
        inferred_transport_support_frames = 0
        continuous_transport_contact_frames = 0
        max_continuous_transport_contact_frames = 0
        max_transport_rigidity_error = 0.0
        left_contact_sensor_valid_frames = 0
        right_contact_sensor_valid_frames = 0
        contact_hold_frame = None
        contact_hold_position = None
        gripper_preload_reference_position = None
        bilateral_contact_latched = False
        bilateral_contact_validated = False
        grasp_validation_start_frame = None
        grasp_validation_support_frames = 0
        grasp_validation_terminal_stable_frames = 0
        grasp_validation_attempt_max_terminal_stable_frames = 0
        grasp_validation_max_terminal_stable_frames = 0
        grasp_validation_failures = 0
        grasp_proof_state = "inactive"
        grasp_proof_start_frame = None
        grasp_proof_lower_start_frame = None
        grasp_proof_start_position = None
        grasp_proof_target_position = None
        grasp_proof_start_object_position = None
        grasp_proof_contact_rebuild_wait_frame = None
        grasp_proof_max_object_lift = 0.0
        grasp_proof_contact_streak = 0
        grasp_proof_max_contact_streak = 0
        grasp_proof_max_rigidity_error = 0.0
        grasp_proof_successes = 0
        grasp_proof_failures = 0
        grasp_proof_complete_frame = None
        grasp_validated_lift_anchor = None
        grasp_validated_orientation = None
        grasp_validated_finger_position = None
        grasp_validated_finger_target = None
        contact_loss_streak = 0
        max_contact_loss_streak = 0
        grasp_contact_loss_events = 0
        lift_start_frame = attach_frame + grasp_retry_frames
        grasp_validation_frames = max(
            0,
            int(
                pickup_config.get(
                    "grasp_validation_frames",
                    0,
                )
            ),
        )
        grasp_validation_max_frames = max(
            grasp_validation_frames,
            int(
                pickup_config.get(
                    "grasp_validation_max_frames",
                    grasp_validation_frames,
                )
            ),
        )
        grasp_validation_required_terminal_frames = max(
            1,
            int(
                pickup_config.get(
                    "grasp_validation_terminal_stable_frames",
                    1,
                )
            ),
        )
        grasp_proof_enabled = bool(
            pickup_config.get("grasp_proof_lift_enabled", False)
        )
        grasp_proof_lift_height = max(
            0.001,
            float(
                pickup_config.get(
                    "grasp_proof_lift_height_meters",
                    0.02,
                )
            ),
        )
        grasp_proof_lift_frames = max(
            1,
            int(pickup_config.get("grasp_proof_lift_frames", 90)),
        )
        grasp_proof_hold_frames = max(
            0,
            int(pickup_config.get("grasp_proof_hold_frames", 30)),
        )
        grasp_proof_hold_max_frames = max(
            grasp_proof_hold_frames,
            int(
                pickup_config.get(
                    "grasp_proof_hold_max_frames",
                    grasp_proof_hold_frames,
                )
            ),
        )
        grasp_proof_lower_frames = max(
            1,
            int(pickup_config.get("grasp_proof_lower_frames", 90)),
        )
        grasp_proof_min_object_lift = max(
            0.0,
            float(
                pickup_config.get(
                    "grasp_proof_min_object_lift_meters",
                    grasp_proof_lift_height * 0.6,
                )
            ),
        )
        grasp_proof_required_contact_frames = max(
            1,
            int(
                pickup_config.get(
                    "grasp_proof_required_contact_frames",
                    20,
                )
            ),
        )
        grasp_proof_terminal_contact_frames = max(
            1,
            int(
                pickup_config.get(
                    "grasp_proof_terminal_contact_frames",
                    1,
                )
            ),
        )
        grasp_proof_max_allowed_rigidity_error = max(
            0.0,
            float(
                pickup_config.get(
                    "grasp_proof_max_rigidity_error_meters",
                    0.015,
                )
            ),
        )
        grasp_proof_min_force = max(
            0.0,
            float(
                pickup_config.get(
                    "grasp_proof_min_force_newtons",
                    pickup_config.get(
                        "grasp_validation_min_force_newtons",
                        bilateral_hold_min_force,
                    ),
                )
            ),
        )
        recovery_extension_frames = max(
            0,
            int(pickup_config.get("recovery_extension_frames", 0)),
        )
        required_release_settle_frames = max(
            1,
            int(
                task["success_criteria"].get(
                    "min_release_settle_frames",
                    45,
                )
            ),
        )
        simulation_frame_cap = frame_count + recovery_extension_frames
        frames_simulated = 0
        simulation_stop = None
        grasp_motion_start_frame = None
        actual_release_frame = None
        grasp_recovery_state = "initial"
        grasp_recovery_attempts = 0
        grasp_recovery_phase_start = None
        recovery_approach_position = None
        recovery_grasp_position = None
        grasp_approach_gate_open_frame = None
        grasp_close_control_height = None
        grasp_approach_converged_frames = 0
        grasp_approach_max_converged_frames = 0
        grasp_xy_gate_open_frame = None
        grasp_xy_aligned_frames = 0
        grasp_xy_max_aligned_frames = 0
        grasp_visual_handoff_frame = None
        grasp_visual_handoff_position = None
        grasp_visual_post_handoff_max_xy_drift = 0.0
        if (
            freeze_visual_target_after_xy_gate
            and grasp_visual_lock_source == "initial_rgbd"
            and initial_object_estimate is not None
        ):
            grasp_visual_handoff_position = list(
                initial_object_estimate["position"]
            )
            grasp_visual_handoff_position[2] = grasp_target_height
        grasp_hover_clearance = float(
            pickup_config.get("grasp_hover_clearance", 0.10)
        )
        grasp_descent_frames = max(
            1,
            int(pickup_config.get("grasp_descent_frames", 120)),
        )
        grasp_descent_step = float(
            pickup_config.get(
                "grasp_descent_step_meters",
                0.00025,
            )
        )
        grasp_tracking_error_xyz = None
        grasp_tracking_xy_error = None
        grasp_tracking_z_error = None
        grasp_tracking_servo_updates = 0
        grasp_aperture_center = None
        grasp_aperture_finger_z_skew = None
        grasp_aperture_axis_yaw_degrees = None
        grasp_aperture_axis_yaw_error_degrees = None
        grasp_aperture_axis_yaw_delta_degrees = None
        previous_grasp_aperture_axis_yaw_degrees = None
        grasp_descent_interlock_active = False
        grasp_descent_interlock_activations = 0
        grasp_descent_interlock_paused_frames = 0
        grasp_aperture_left_bounds = None
        grasp_aperture_right_bounds = None
        aperture_tool_offset_world = None
        aperture_tcp_calibration_frame = None
        unilateral_recenter_updates = 0
        unilateral_recenter_last_side = None
        unilateral_recenter_persistence_remaining = 0
        unilateral_force_backoff_updates = 0
        bilateral_force_backoff_updates = 0
        bilateral_hold_close_updates = 0
        gripper_target_envelope_updates = 0
        finger_close_drive_configured = False
        finger_contact_effort_configured = False
        raw_contact_pair_peak_forces = {}
        peak_primary_finger_effort = 0.0
        peak_mimic_follower_error = 0.0
        post_grasp_target_return_updates = 0
        pickup_visual_tracking_updates = 0
        pickup_visual_tracking_last_observed_nominal = None
        temporary_grasp_joint_created = False
        dataset_observation_count = 0
        preview_capture_index = 0
        observation_every = max(
            1,
            int(dataset_config.get("observation_every_n_frames", 10)),
        )
        perception_every = max(
            1,
            int(perception_config.get("update_every_n_frames", 20)),
        )
        observation_context = (
            observations_path.open("w", encoding="utf-8")
            if dataset_enabled
            else contextlib.nullcontext(None)
        )
        labels_context = (
            labels_path.open("w", encoding="utf-8")
            if dataset_enabled
            else contextlib.nullcontext(None)
        )
        measured = list(base_positions)

        def mark_physical_grasp_validated(
            frame,
            *,
            validation_elapsed,
            validation_source,
            left_force,
            right_force,
        ):
            nonlocal bilateral_contact_validated
            nonlocal grasp_recovery_state
            nonlocal grasp_validated_lift_anchor
            nonlocal target_end_effector_orientation
            nonlocal grasp_validated_orientation
            nonlocal grasp_validated_finger_position
            nonlocal grasp_validated_finger_target
            nonlocal grasp_motion_start_frame
            nonlocal grasp_was_created
            nonlocal grasp_created
            nonlocal grasp_success_frame
            nonlocal grasp_attempt_count
            nonlocal grasp_proof_state
            nonlocal grasp_proof_complete_frame

            bilateral_contact_validated = True
            grasp_recovery_state = "grasped"
            grasp_validated_finger_position = float(
                measured_finger_position
            )
            grasp_validated_finger_target = float(
                contact_hold_position
                if contact_hold_position is not None
                else measured_finger_position
            )
            grasp_proof_state = (
                "qualified"
                if validation_source == "proof_lift"
                else "disabled"
            )
            grasp_proof_complete_frame = (
                frame
                if validation_source == "proof_lift"
                else None
            )
            if (
                control_frame_position is not None
                or grasp_body_position is not None
            ):
                lift_anchor_position = (
                    control_frame_position
                    if control_frame_position is not None
                    else grasp_body_position
                )
                pick_grasp_position[:] = [
                    float(value) for value in lift_anchor_position
                ]
                nominal_pick_grasp_position[:] = list(
                    pick_grasp_position
                )
                grasp_validated_lift_anchor = list(
                    pick_grasp_position
                )
            if (
                pickup_config.get("lock_orientation_on_grasp", False)
                and (
                    control_frame_rotation is not None
                    or grasp_body_orientation is not None
                )
            ):
                target_end_effector_orientation = np.asarray(
                    (
                        rot_matrix_to_quat(control_frame_rotation)
                        if control_frame_rotation is not None
                        else grasp_body_orientation
                    ),
                    dtype=np.float32,
                )
                grasp_validated_orientation = (
                    target_end_effector_orientation.tolist()
                )
            elif pickup_config.get(
                "free_orientation_after_grasp",
                False,
            ):
                target_end_effector_orientation = None
                grasp_validated_orientation = None
            grasp_motion_start_frame = max(
                frame + 1,
                lift_start_frame,
            )
            if not grasp_was_created:
                grasp_was_created = True
                grasp_created = True
                grasp_success_frame = frame
                grasp_attempt_count += 1
            append_phase(
                phase_path,
                "physical_bilateral_grasp_validated",
                frame=frame,
                validation_source=validation_source,
                left_force_newtons=left_force,
                right_force_newtons=right_force,
                support_frames=grasp_validation_support_frames,
                terminal_stable_frames=(
                    grasp_validation_terminal_stable_frames
                ),
                validation_frames=validation_elapsed,
                grasp_motion_start_frame=grasp_motion_start_frame,
                lift_anchor=grasp_validated_lift_anchor,
                locked_orientation=grasp_validated_orientation,
                validated_finger_position=(
                    grasp_validated_finger_position
                ),
                validated_finger_target=(
                    grasp_validated_finger_target
                ),
                transport_hold_max_close_adjustment_radians=float(
                    pickup_config.get(
                        "transport_hold_max_close_adjustment_radians",
                        0.0,
                    )
                ),
                proof_object_lift_meters=grasp_proof_max_object_lift,
                proof_max_contact_streak=(
                    grasp_proof_max_contact_streak
                ),
                proof_max_rigidity_error_meters=(
                    grasp_proof_max_rigidity_error
                ),
            )

        with (
            trajectory_path.open("w", encoding="utf-8") as trajectory,
            observation_context as observations,
            labels_context as labels,
        ):
            for frame in range(simulation_frame_cap):
                frames_simulated = frame + 1
                if (
                    contact_only
                    and bilateral_contact_validated
                    and grasp_motion_start_frame is not None
                    and frame < grasp_motion_start_frame
                    and post_grasp_target_return_step > 0.0
                ):
                    for axis in range(2):
                        target_delta = (
                            float(nominal_pick_grasp_position[axis])
                            - float(pick_grasp_position[axis])
                        )
                        pick_grasp_position[axis] += max(
                            -post_grasp_target_return_step,
                            min(post_grasp_target_return_step, target_delta),
                        )
                    post_grasp_target_return_updates += 1
                if (
                    contact_only
                    and not bilateral_contact_latched
                    and grasp_recovery_state == "initial"
                    and frame >= grasp_alignment_start_frame
                    and grasp_body_position is not None
                ):
                    if grasp_alignment_mode == "aperture_center":
                        alignment_object_position = list(
                            (
                                grasp_visual_handoff_position
                                if grasp_visual_handoff_position is not None
                                else latest_object_estimate["position"]
                            )
                            if (
                                grasp_visual_handoff_position is not None
                                or latest_object_estimate is not None
                            )
                            else pick_start_position
                        )
                        alignment_object_position[2] = (
                            grasp_target_height
                        )
                        alignment_object_position[0] += (
                            grasp_aperture_bias_xy[0]
                        )
                        alignment_object_position[1] += (
                            grasp_aperture_bias_xy[1]
                        )
                        grasp_aperture_left_bounds = prim_world_bounds(
                            stage,
                            left_contact_body_path,
                        )
                        grasp_aperture_right_bounds = prim_world_bounds(
                            stage,
                            right_contact_body_path,
                        )
                        tracking_result = gripper_aperture_alignment(
                            grasp_aperture_left_bounds,
                            grasp_aperture_right_bounds,
                            alignment_object_position,
                        )
                        grasp_aperture_center = tracking_result[
                            "aperture_center"
                        ]
                        grasp_aperture_finger_z_skew = tracking_result[
                            "finger_z_skew"
                        ]
                        grasp_aperture_axis_yaw_degrees = tracking_result[
                            "finger_axis_yaw_degrees"
                        ]
                        if (
                            previous_grasp_aperture_axis_yaw_degrees
                            is not None
                        ):
                            grasp_aperture_axis_yaw_delta_degrees = (
                                undirected_axis_angle_error_degrees(
                                    grasp_aperture_axis_yaw_degrees,
                                    previous_grasp_aperture_axis_yaw_degrees,
                                )
                            )
                        previous_grasp_aperture_axis_yaw_degrees = (
                            grasp_aperture_axis_yaw_degrees
                        )
                        grasp_aperture_axis_yaw_error_degrees = (
                            undirected_axis_angle_error_degrees(
                                grasp_aperture_axis_yaw_degrees,
                                grasp_axis_target_yaw_degrees,
                            )
                        )
                        aperture_tool_offset_world = subtract_vector(
                            grasp_aperture_center,
                            (
                                control_frame_position
                                if control_frame_position is not None
                                else grasp_body_position
                            ),
                        )
                        if (
                            aperture_tcp_calibration_frame is None
                            and grasp_aperture_axis_yaw_error_degrees
                            <= grasp_axis_calibration_max_error_degrees
                        ):
                            aperture_tcp_calibration_frame = frame
                            calibrated_pick_grasp_position = (
                                subtract_vector(
                                    pick_grasp_reference,
                                    aperture_tool_offset_world,
                                )
                            )
                            pick_grasp_position[:] = (
                                calibrated_pick_grasp_position
                            )
                            nominal_pick_grasp_position[:] = (
                                calibrated_pick_grasp_position
                            )
                            calibrated_place_grasp_position = (
                                subtract_vector(
                                    place_grasp_reference,
                                    aperture_tool_offset_world,
                                )
                            )
                            calibrated_place_grasp_position[0] += float(
                                pickup_config.get(
                                    "place_grasp_x_compensation",
                                    0.0,
                                )
                            )
                            calibrated_place_grasp_position[1] += float(
                                pickup_config.get(
                                    "place_grasp_y_compensation",
                                    0.0,
                                )
                            )
                            place_grasp_position[:] = (
                                calibrated_place_grasp_position
                            )
                            nominal_place_grasp_position[:] = (
                                calibrated_place_grasp_position
                            )
                            append_phase(
                                phase_path,
                                "gripper_aperture_tcp_calibrated",
                                frame=frame,
                                finger_axis_yaw_degrees=(
                                    grasp_aperture_axis_yaw_degrees
                                ),
                                finger_axis_yaw_error_degrees=(
                                    grasp_aperture_axis_yaw_error_degrees
                                ),
                                aperture_tool_offset_world=(
                                    aperture_tool_offset_world
                                ),
                                pick_grasp_position=(
                                    pick_grasp_position
                                ),
                                place_grasp_position=(
                                    place_grasp_position
                                ),
                            )
                        if grasp_approach_gate_open_frame is None:
                            previous_grasp_target_height = float(
                                pick_grasp_position[2]
                            )
                            if pickup_config.get(
                                "grasp_tracking_mode",
                                "proportional",
                            ) == "integral":
                                integral_servo = (
                                    integral_visual_servo_grasp_target(
                                        grasp_aperture_center,
                                        alignment_object_position,
                                        pick_grasp_position,
                                        nominal_pick_grasp_position,
                                        gain=grasp_tracking_gain,
                                        max_step=grasp_tracking_max_step,
                                        max_correction=(
                                            grasp_tracking_max_correction
                                        ),
                                    )
                                )
                                aperture_servo = {
                                    "position": integral_servo[
                                        "grasp_position"
                                    ],
                                    "position_error": integral_servo[
                                        "object_error"
                                    ],
                                    "xy_error": integral_servo["xy_error"],
                                    "z_error": integral_servo["z_error"],
                                    "saturated_axes": integral_servo[
                                        "saturated_axes"
                                    ],
                                }
                            else:
                                aperture_servo = (
                                    cartesian_tracking_servo_target(
                                        grasp_aperture_center,
                                        alignment_object_position,
                                        pick_grasp_position,
                                        nominal_pick_grasp_position,
                                        gain=grasp_tracking_gain,
                                        max_step=grasp_tracking_max_step,
                                        max_correction=(
                                            grasp_tracking_max_correction
                                        ),
                                    )
                                )
                            aperture_servo_position = aperture_servo[
                                "position"
                            ]
                            if grasp_xy_gate_open_frame is None:
                                aperture_servo_position[2] = (
                                    pick_grasp_position[2]
                                )
                            descent_alignment_ready = (
                                aperture_servo["xy_error"]
                                <= grasp_descent_max_xy_error
                                and (
                                    grasp_aperture_axis_yaw_error_degrees
                                    is None
                                    or grasp_aperture_axis_yaw_error_degrees
                                    <= grasp_descent_max_axis_yaw_error
                                )
                                and (
                                    grasp_aperture_axis_yaw_delta_degrees
                                    is None
                                    or grasp_aperture_axis_yaw_delta_degrees
                                    <= grasp_axis_max_delta_degrees
                                )
                                and (
                                    grasp_aperture_finger_z_skew is None
                                    or grasp_aperture_finger_z_skew
                                    <= grasp_tracking_max_finger_z_skew
                                )
                            )
                            if (
                                grasp_xy_gate_open_frame is not None
                                and not descent_alignment_ready
                            ):
                                aperture_servo_position[2] = (
                                    previous_grasp_target_height
                                )
                                grasp_descent_interlock_paused_frames += 1
                                if not grasp_descent_interlock_active:
                                    grasp_descent_interlock_active = True
                                    grasp_descent_interlock_activations += 1
                                    append_phase(
                                        phase_path,
                                        "grasp_descent_interlock_paused",
                                        frame=frame,
                                        xy_error=aperture_servo[
                                            "xy_error"
                                        ],
                                        finger_axis_yaw_error_degrees=(
                                            grasp_aperture_axis_yaw_error_degrees
                                        ),
                                        finger_axis_yaw_delta_degrees=(
                                            grasp_aperture_axis_yaw_delta_degrees
                                        ),
                                        held_target_height=(
                                            previous_grasp_target_height
                                        ),
                                    )
                            elif grasp_descent_interlock_active:
                                grasp_descent_interlock_active = False
                                append_phase(
                                    phase_path,
                                    "grasp_descent_interlock_resumed",
                                    frame=frame,
                                    xy_error=aperture_servo["xy_error"],
                                    finger_axis_yaw_error_degrees=(
                                        grasp_aperture_axis_yaw_error_degrees
                                    ),
                                    finger_axis_yaw_delta_degrees=(
                                        grasp_aperture_axis_yaw_delta_degrees
                                    ),
                                )
                            pick_grasp_position[:] = (
                                aperture_servo_position
                            )
                    else:
                        tracking_result = cartesian_tracking_servo_target(
                            grasp_body_position,
                            nominal_pick_grasp_position,
                            pick_grasp_position,
                            nominal_pick_grasp_position,
                            gain=grasp_tracking_gain,
                            max_step=grasp_tracking_max_step,
                            max_correction=grasp_tracking_max_correction,
                        )
                        pick_grasp_position[:] = tracking_result["position"]
                    grasp_tracking_error_xyz = tracking_result[
                        "position_error"
                    ]
                    grasp_tracking_xy_error = tracking_result["xy_error"]
                    grasp_tracking_z_error = tracking_result["z_error"]
                    grasp_tracking_servo_updates += 1
                    if (
                        grasp_tracking_xy_error
                        <= grasp_tracking_max_xy_error
                        and (
                            grasp_aperture_finger_z_skew is None
                            or grasp_aperture_finger_z_skew
                            <= grasp_tracking_max_finger_z_skew
                        )
                        and (
                            grasp_aperture_axis_yaw_error_degrees is None
                            or grasp_aperture_axis_yaw_error_degrees
                            <= grasp_axis_max_error_degrees
                        )
                        and (
                            grasp_aperture_axis_yaw_delta_degrees is None
                            or grasp_aperture_axis_yaw_delta_degrees
                            <= grasp_axis_max_delta_degrees
                        )
                    ):
                        grasp_xy_aligned_frames += 1
                    else:
                        grasp_xy_aligned_frames = 0
                    grasp_xy_max_aligned_frames = max(
                        grasp_xy_max_aligned_frames,
                        grasp_xy_aligned_frames,
                    )
                    if (
                        grasp_xy_gate_open_frame is None
                        and grasp_xy_aligned_frames
                        >= grasp_tracking_stable_frames
                    ):
                        grasp_xy_gate_open_frame = frame + 1
                        if control_frame_position is not None:
                            pick_grasp_position[2] = float(
                                control_frame_position[2]
                            )
                        if (
                            freeze_visual_target_after_xy_gate
                            and grasp_visual_handoff_position is None
                        ):
                            grasp_visual_handoff_frame = frame
                            grasp_visual_handoff_position = list(
                                alignment_object_position
                            )
                        elif grasp_visual_handoff_position is not None:
                            grasp_visual_handoff_frame = frame
                        append_phase(
                            phase_path,
                            "grasp_xy_hover_gate_converged",
                            frame=frame,
                            descent_start_frame=(
                                grasp_xy_gate_open_frame
                            ),
                            xy_error=grasp_tracking_xy_error,
                            finger_axis_yaw_degrees=(
                                grasp_aperture_axis_yaw_degrees
                            ),
                            finger_axis_yaw_error_degrees=(
                                grasp_aperture_axis_yaw_error_degrees
                            ),
                            finger_axis_yaw_delta_degrees=(
                                grasp_aperture_axis_yaw_delta_degrees
                            ),
                            stable_frames=grasp_xy_aligned_frames,
                            descent_start_control_height=(
                                pick_grasp_position[2]
                            ),
                        )
                        if grasp_visual_handoff_position is not None:
                            append_phase(
                                phase_path,
                                "visual_to_tactile_handoff",
                                frame=frame,
                                frozen_object_position=(
                                    grasp_visual_handoff_position
                                ),
                                control_mode="frozen_rgbd_then_tactile",
                            )
                    if (
                        grasp_xy_gate_open_frame is not None
                        and grasp_tracking_xy_error
                        <= grasp_approach_max_xy_error
                        and grasp_tracking_z_error
                        <= grasp_tracking_max_z_error
                        and (
                            grasp_aperture_finger_z_skew is None
                            or grasp_aperture_finger_z_skew
                            <= grasp_tracking_max_finger_z_skew
                        )
                        and (
                            grasp_aperture_axis_yaw_error_degrees is None
                            or grasp_aperture_axis_yaw_error_degrees
                            <= grasp_axis_max_error_degrees
                        )
                        and (
                            grasp_aperture_axis_yaw_delta_degrees is None
                            or grasp_aperture_axis_yaw_delta_degrees
                            <= grasp_axis_max_delta_degrees
                        )
                    ):
                        grasp_approach_converged_frames += 1
                    else:
                        grasp_approach_converged_frames = 0
                    grasp_approach_max_converged_frames = max(
                        grasp_approach_max_converged_frames,
                        grasp_approach_converged_frames,
                    )
                    if (
                        grasp_approach_gate_open_frame is None
                        and grasp_approach_converged_frames
                        >= grasp_tracking_stable_frames
                    ):
                        grasp_approach_gate_open_frame = frame + 1
                        grasp_close_control_height = float(
                            pick_grasp_position[2]
                        )
                        if (
                            kps is not None
                            and kds is not None
                            and hasattr(controller, "set_gains")
                        ):
                            for joint_name in actuated_finger_joint_names:
                                joint_index = dof_names.index(joint_name)
                                kps[joint_index] = finger_close_stiffness
                                kds[joint_index] = finger_close_damping
                            controller.set_gains(
                                kps=kps,
                                kds=kds,
                                save_to_usd=False,
                            )
                            finger_close_drive_configured = True
                        append_phase(
                            phase_path,
                            "grasp_approach_gate_converged",
                            frame=frame,
                            close_start_frame=(
                                grasp_approach_gate_open_frame
                            ),
                            position_error=(
                                grasp_tracking_error_xyz
                            ),
                            finger_axis_yaw_degrees=(
                                grasp_aperture_axis_yaw_degrees
                            ),
                            finger_axis_yaw_error_degrees=(
                                grasp_aperture_axis_yaw_error_degrees
                            ),
                            stable_frames=(
                                grasp_approach_converged_frames
                            ),
                            locked_control_target=[
                                pick_grasp_position[0],
                                pick_grasp_position[1],
                                grasp_close_control_height,
                            ],
                            finger_close_stiffness=(
                                finger_close_stiffness
                            ),
                            finger_close_damping=finger_close_damping,
                        )
                if contact_only and not bilateral_contact_latched:
                    recovery_retreat_frames = int(
                        pickup_config.get(
                            "recovery_retreat_frames",
                            120,
                        )
                    )
                    recovery_descend_frames = int(
                        pickup_config.get(
                            "recovery_descend_frames",
                            120,
                        )
                    )
                    recovery_close_frames = int(
                        pickup_config.get(
                            "recovery_close_frames",
                            220,
                        )
                    )
                    max_grasp_retries = int(
                        pickup_config.get("max_grasp_retries", 0)
                    )
                    initial_grasp_timeout_frame = int(
                        pickup_config.get(
                            "initial_grasp_timeout_frame",
                            lift_start_frame + 180,
                        )
                    )
                    if grasp_approach_gate_open_frame is not None:
                        initial_grasp_timeout_frame = max(
                            initial_grasp_timeout_frame,
                            grasp_approach_gate_open_frame
                            + grasp_close_duration_frames
                            + grasp_validation_frames,
                        )
                    retry_due = (
                        grasp_recovery_state == "initial"
                        and frame >= initial_grasp_timeout_frame
                    ) or (
                        grasp_recovery_state == "close"
                        and grasp_recovery_phase_start is not None
                        and frame - grasp_recovery_phase_start
                        >= recovery_close_frames
                    )
                    if retry_due:
                        if grasp_recovery_attempts < max_grasp_retries:
                            grasp_recovery_attempts += 1
                            grasp_recovery_state = "retreat"
                            grasp_recovery_phase_start = frame
                            contact_hold_frame = None
                            contact_hold_position = None
                            gripper_preload_reference_position = None
                            grasp_validated_finger_position = None
                            grasp_validated_finger_target = None
                            raw_bilateral_contact_streak = 0
                            raw_bilateral_contact_event_frames.clear()
                            last_left_contact_frame = None
                            last_right_contact_frame = None
                            if (
                                max_efforts is not None
                                and hasattr(controller, "set_max_efforts")
                            ):
                                for joint_name in (
                                    actuated_finger_joint_names
                                ):
                                    max_efforts[
                                        dof_names.index(joint_name)
                                    ] = finger_search_max_effort
                                controller.set_max_efforts(max_efforts)
                                finger_contact_effort_configured = False
                                append_phase(
                                    phase_path,
                                    "gripper_search_effort_restored",
                                    frame=frame,
                                    search_max_effort=(
                                        finger_search_max_effort
                                    ),
                                    attempt=grasp_recovery_attempts,
                                )
                            estimated_pick = list(
                                latest_object_estimate["position"]
                                if latest_object_estimate
                                else pick_start_position
                            )
                            estimated_pick[2] = grasp_target_height
                            recovery_grasp_position = subtract_vector(
                                estimated_pick,
                                object_grasp_offset,
                            )
                            recovery_approach_position = [
                                recovery_grasp_position[0],
                                recovery_grasp_position[1],
                                recovery_grasp_position[2] + 0.25,
                            ]
                            append_phase(
                                phase_path,
                                "grasp_recovery_retreat_start",
                                frame=frame,
                                attempt=grasp_recovery_attempts,
                                estimated_object_position=estimated_pick,
                            )
                        else:
                            grasp_recovery_state = "exhausted"
                            grasp_recovery_phase_start = frame
                            contact_hold_position = None
                            gripper_preload_reference_position = None
                            append_phase(
                                phase_path,
                                "grasp_recovery_exhausted",
                                frame=frame,
                                attempts=grasp_recovery_attempts,
                            )
                    elif (
                        grasp_recovery_state == "retreat"
                        and frame - grasp_recovery_phase_start
                        >= recovery_retreat_frames
                    ):
                        estimated_pick = list(
                            latest_object_estimate["position"]
                            if latest_object_estimate
                            else pick_start_position
                        )
                        estimated_pick[2] = grasp_target_height
                        recovery_grasp_position = subtract_vector(
                            estimated_pick,
                            object_grasp_offset,
                        )
                        recovery_approach_position = [
                            recovery_grasp_position[0],
                            recovery_grasp_position[1],
                            recovery_grasp_position[2] + 0.25,
                        ]
                        grasp_recovery_state = "descend"
                        grasp_recovery_phase_start = frame
                        append_phase(
                            phase_path,
                            "grasp_recovery_descend_start",
                            frame=frame,
                            attempt=grasp_recovery_attempts,
                            estimated_object_position=estimated_pick,
                        )
                    elif (
                        grasp_recovery_state == "descend"
                        and frame - grasp_recovery_phase_start
                        >= recovery_descend_frames
                    ):
                        grasp_recovery_state = "close"
                        grasp_recovery_phase_start = frame
                        contact_hold_frame = None
                        contact_hold_position = None
                        gripper_preload_reference_position = None
                        append_phase(
                            phase_path,
                            "grasp_recovery_close_start",
                            frame=frame,
                            attempt=grasp_recovery_attempts,
                        )
                if contact_only and not bilateral_contact_validated:
                    raw_motion_frame = min(frame, lift_start_frame - 1)
                elif (
                    contact_only
                    and bilateral_contact_latched
                    and grasp_motion_start_frame is not None
                    and frame < grasp_motion_start_frame
                ):
                    raw_motion_frame = lift_start_frame - 1
                elif (
                    contact_only
                    and grasp_motion_start_frame is not None
                    and frame >= grasp_motion_start_frame
                ):
                    raw_motion_frame = (
                        lift_start_frame
                        + frame
                        - grasp_motion_start_frame
                    )
                else:
                    raw_motion_frame = frame
                if (
                    contact_only
                    and bilateral_contact_validated
                    and raw_motion_frame >= place_descent_start_frame
                ):
                    if place_descent_gate_start_frame is None:
                        place_descent_gate_start_frame = frame
                        append_phase(
                            phase_path,
                            "place_descent_gate_wait_start",
                            frame=frame,
                            motion_frame=raw_motion_frame,
                        )
                    if place_descent_gate_open_frame is None:
                        gated_motion_frame = (
                            place_descent_start_frame - 1
                        )
                    elif (
                        soft_landing_enabled
                        and soft_release_complete_frame is None
                    ):
                        gated_motion_frame = release_frame - 1
                    elif soft_landing_enabled:
                        gated_motion_frame = (
                            release_frame
                            + frame
                            - soft_release_complete_frame
                        )
                    else:
                        gated_motion_frame = (
                            place_descent_start_frame
                            + frame
                            - place_descent_gate_open_frame
                        )
                else:
                    gated_motion_frame = raw_motion_frame
                if (
                    contact_only
                    and bilateral_contact_validated
                    and gated_motion_frame >= release_frame
                ):
                    if place_release_gate_start_frame is None:
                        place_release_gate_start_frame = frame
                        append_phase(
                            phase_path,
                            "place_release_gate_wait_start",
                            frame=frame,
                            motion_frame=gated_motion_frame,
                        )
                    if (
                        place_release_gate_open_frame is None
                        or (
                            soft_landing_enabled
                            and soft_release_complete_frame is None
                        )
                    ):
                        motion_frame = release_frame - 1
                    else:
                        release_progress_frame = (
                            soft_release_complete_frame
                            if soft_landing_enabled
                            else place_release_gate_open_frame
                        )
                        motion_frame = (
                            release_frame
                            + frame
                            - release_progress_frame
                        )
                else:
                    motion_frame = gated_motion_frame
                task_phase = (
                    f"recovery_{grasp_recovery_state}"
                    if grasp_recovery_state
                    in {"retreat", "descend", "close", "exhausted"}
                    else (
                        f"grasp_proof_{grasp_proof_state}"
                        if grasp_proof_state
                        in {"lifting", "holding", "lowering"}
                        else (
                            "grasp_wait"
                            if (
                                contact_only
                                and not bilateral_contact_validated
                                and frame >= lift_start_frame
                            )
                            else pickup_phase(
                                motion_frame,
                                attach_frame,
                                release_frame,
                                grasp_retry_frames=grasp_retry_frames,
                                lift_duration_frames=pickup_config.get(
                                    "lift_duration_frames",
                                    300,
                                ),
                            )
                        )
                    )
                )
                if task_phase not in phase_names_seen:
                    phase_names_seen.append(task_phase)
                    append_phase(
                        phase_path,
                        f"pickup_{task_phase}_start",
                        frame=frame,
                        motion_frame=motion_frame,
                    )
                if rmpflow_controller is not None:
                    cartesian_target = cartesian_pick_and_place_target(
                        motion_frame,
                        attach_frame,
                        release_frame,
                        pick_grasp_position,
                        place_grasp_position,
                        lift_height,
                        grasp_retry_frames,
                        lift_duration_frames=pickup_config.get(
                            "lift_duration_frames",
                            300,
                        ),
                        transfer_end_frame=pickup_config.get(
                            "transfer_end_frame"
                        ),
                        place_hover_end_frame=pickup_config.get(
                            "place_hover_end_frame"
                        ),
                        place_descent_start_frame=pickup_config.get(
                            "place_descent_start_frame"
                        ),
                        place_descent_end_frame=pickup_config.get(
                            "place_descent_end_frame"
                        ),
                    )
                    cartesian_policy_target = rmpflow_world_target(
                        cartesian_target
                    )
                else:
                    cartesian_target = None
                    cartesian_policy_target = None
                if (
                    contact_only
                    and grasp_proof_state
                    in {"lifting", "holding", "lowering"}
                    and grasp_proof_start_position is not None
                    and grasp_proof_target_position is not None
                ):
                    if grasp_proof_state == "lifting":
                        proof_progress = (
                            frame - grasp_proof_start_frame
                        ) / grasp_proof_lift_frames
                        cartesian_target = interpolate_position(
                            grasp_proof_start_position,
                            grasp_proof_target_position,
                            proof_progress,
                        )
                    elif grasp_proof_state == "holding":
                        cartesian_target = list(
                            grasp_proof_target_position
                        )
                    else:
                        proof_lower_progress = (
                            frame - grasp_proof_lower_start_frame
                        ) / grasp_proof_lower_frames
                        cartesian_target = interpolate_position(
                            grasp_proof_target_position,
                            grasp_proof_start_position,
                            proof_lower_progress,
                        )
                    cartesian_policy_target = rmpflow_world_target(
                        cartesian_target
                    )
                if (
                    contact_only
                    and not bilateral_contact_latched
                    and grasp_recovery_state == "initial"
                    and frame >= grasp_alignment_start_frame
                ):
                    if grasp_xy_gate_open_frame is None:
                        aperture_target_height = (
                            grasp_target_height
                            + grasp_hover_clearance
                        )
                        control_target_height = (
                            aperture_target_height
                            - float(
                                (
                                    aperture_tool_offset_world
                                    or object_grasp_offset
                                )[2]
                            )
                        )
                    elif (
                        grasp_approach_gate_open_frame is not None
                        and grasp_close_control_height is not None
                    ):
                        control_target_height = (
                            grasp_close_control_height
                        )
                    else:
                        control_target_height = float(
                            pick_grasp_position[2]
                        )
                    cartesian_target = [
                        pick_grasp_position[0],
                        pick_grasp_position[1],
                        control_target_height,
                    ]
                    cartesian_policy_target = rmpflow_world_target(
                        cartesian_target
                    )
                if (
                    contact_only
                    and not bilateral_contact_latched
                    and grasp_recovery_state
                    in {"retreat", "descend", "close", "exhausted"}
                ):
                    if grasp_recovery_state in {"retreat", "exhausted"}:
                        cartesian_target = list(
                            recovery_approach_position
                            or [
                                pick_grasp_position[0],
                                pick_grasp_position[1],
                                pick_grasp_position[2] + 0.25,
                            ]
                        )
                    elif grasp_recovery_state == "descend":
                        recovery_progress = (
                            frame - grasp_recovery_phase_start
                        ) / max(recovery_descend_frames, 1)
                        cartesian_target = interpolate_position(
                            recovery_approach_position,
                            recovery_grasp_position,
                            recovery_progress,
                        )
                    else:
                        cartesian_target = list(
                            recovery_grasp_position
                        )
                    cartesian_policy_target = rmpflow_world_target(
                        cartesian_target
                    )
                if (
                    contact_only
                    and transport_recenter_enabled
                    and bilateral_contact_validated
                    and cartesian_target is not None
                    and task_phase
                    in {"lift", "transfer", "place"}
                ):
                    cartesian_target = list(cartesian_target)
                    cartesian_target[0] += (
                        transport_recenter_offset_xy[0]
                    )
                    cartesian_target[1] += (
                        transport_recenter_offset_xy[1]
                    )
                    cartesian_policy_target = rmpflow_world_target(
                        cartesian_target
                    )
                if (
                    contact_only
                    and soft_landing_enabled
                    and soft_landing_state
                    in {"lowering", "support_hold", "unloading"}
                    and soft_landing_command_position is not None
                ):
                    cartesian_target = list(
                        soft_landing_command_position
                    )
                    cartesian_target[0] = place_grasp_position[0]
                    cartesian_target[1] = place_grasp_position[1]
                    cartesian_target[0] += (
                        transport_recenter_offset_xy[0]
                    )
                    cartesian_target[1] += (
                        transport_recenter_offset_xy[1]
                    )
                    cartesian_policy_target = rmpflow_world_target(
                        cartesian_target
                    )
                finger_target = gripper_target(
                    motion_frame,
                    attach_frame,
                    release_frame,
                    close_position=pickup_config.get(
                        "close_position_radians",
                        47.0,
                    ),
                    close_start_frame=gripper_close_start_frame,
                    close_end_frame=gripper_close_end_frame,
                )
                if (
                    contact_only
                    and not bilateral_contact_latched
                    and grasp_recovery_state == "initial"
                ):
                    if grasp_approach_gate_open_frame is None:
                        finger_target = 0.0
                    else:
                        finger_target = float(
                            pickup_config.get(
                                "close_position_radians",
                                47.0,
                            )
                        ) * smoothstep(
                            (
                                frame
                                - grasp_approach_gate_open_frame
                            )
                            / grasp_close_duration_frames
                        )
                if (
                    contact_only
                    and not bilateral_contact_latched
                    and grasp_recovery_state
                    in {"retreat", "descend", "exhausted"}
                ):
                    finger_target = 0.0
                elif (
                    contact_only
                    and not bilateral_contact_latched
                    and grasp_recovery_state == "close"
                ):
                    recovery_close_progress = (
                        frame - grasp_recovery_phase_start
                    ) / max(recovery_close_frames, 1)
                    finger_target = float(
                        pickup_config.get(
                            "close_position_radians",
                            47.0,
                        )
                    ) * smoothstep(recovery_close_progress)
                if contact_hold_position is not None:
                    if motion_frame < release_frame:
                        finger_target = contact_hold_position
                    else:
                        finger_target = contact_hold_position * (
                            1.0
                            - smoothstep(
                                (motion_frame - release_frame) / 30.0
                            )
                        )
                if (
                    contact_only
                    and soft_landing_enabled
                    and soft_landing_state == "unloading"
                    and soft_release_start_frame is not None
                    and soft_release_initial_target is not None
                ):
                    soft_release_progress = (
                        frame - soft_release_start_frame
                    ) / soft_release_frames
                    finger_target = soft_release_initial_target * (
                        1.0 - smoothstep(soft_release_progress)
                    )
                elif (
                    contact_only
                    and soft_landing_enabled
                    and soft_landing_state == "released"
                ):
                    finger_target = 0.0
                if (
                    contact_only
                    and primary_finger_index is not None
                    and math.isfinite(gripper_max_position_error)
                ):
                    bounded_finger_target = bounded_position_target(
                        finger_target,
                        measured[primary_finger_index],
                        gripper_max_position_error,
                    )
                    gripper_target_envelope_updates += int(
                        not math.isclose(
                            bounded_finger_target,
                            finger_target,
                            abs_tol=1e-9,
                        )
                    )
                    finger_target = bounded_finger_target
                targets = None
                left_substep_contact_samples = []
                right_substep_contact_samples = []
                for substep in range(physics_substeps_per_frame):
                    if (
                        lula_ik_solver is not None
                        and (
                            bilateral_contact_validated
                            or grasp_proof_state
                            in {"lifting", "holding", "lowering"}
                            or lula_ik_all_phases
                        )
                    ):
                        ik_action, ik_success = (
                            lula_ik_solver.compute_inverse_kinematics(
                                np.asarray(
                                    cartesian_policy_target,
                                    dtype=np.float32,
                                ),
                                (
                                    np.asarray(
                                        target_end_effector_orientation,
                                        dtype=np.float32,
                                    )
                                    if target_end_effector_orientation
                                    is not None
                                    else None
                                ),
                                position_tolerance=0.002,
                                orientation_tolerance=0.05,
                            )
                        )
                        if ik_success:
                            targets = merge_action_positions(
                                robot.get_joint_positions(),
                                ik_action,
                            )
                            targets = rate_limit_revolute_joint_targets(
                                robot.get_joint_positions(),
                                targets,
                                arm_joint_indices,
                                max_step=ik_max_joint_step,
                            )
                            lula_ik_successes += 1
                        else:
                            targets = np.asarray(
                                robot.get_joint_positions(),
                                dtype=np.float32,
                            )
                            lula_ik_failures += 1
                    elif rmpflow_controller is not None:
                        rmpflow_arguments = {
                            "target_end_effector_position": np.asarray(
                                cartesian_policy_target,
                                dtype=np.float32,
                            )
                        }
                        if target_end_effector_orientation is not None:
                            rmpflow_arguments["target_end_effector_orientation"] = (
                                target_end_effector_orientation
                            )
                        rmp_action = rmpflow_controller.forward(**rmpflow_arguments)
                        targets = merge_action_positions(robot.get_joint_positions(), rmp_action)
                    else:
                        targets = arm_targets(frame, frame_count, base_positions, dof_names)
                    if primary_finger_index is not None:
                        targets[primary_finger_index] = finger_target
                    if arm_motion_mode == "kinematic_joint_targets" and hasattr(robot, "set_joint_positions"):
                        arm_only_targets = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                        for joint_name in ARM_JOINT_NAMES:
                            if joint_name in dof_names:
                                joint_index = dof_names.index(joint_name)
                                arm_only_targets[joint_index] = targets[joint_index]
                        robot.set_joint_positions(arm_only_targets)
                    controller.apply_action(
                        ArticulationAction(
                            joint_positions=np.asarray(
                                [
                                    targets[joint_index]
                                    for joint_index in controlled_joint_indices
                                ],
                                dtype=np.float32,
                            ),
                            joint_indices=np.asarray(
                                controlled_joint_indices,
                                dtype=np.int32,
                            ),
                        )
                    )
                    if (
                        arm_motion_mode == "kinematic_joint_targets"
                        and grasp_mode != "solver_fixed_joint"
                        and hasattr(robot, "set_joint_positions")
                    ):
                        robot.set_joint_positions(np.asarray(targets, dtype=np.float32))
                    world.step(
                        render=substep
                        == physics_substeps_per_frame - 1
                    )
                    left_substep_contact_samples.append(
                        read_contact_group(
                            contact_sensors.get("left_finger"),
                            required_body_path=pick_object_config[
                                "path"
                            ],
                            physics_dt=physics_dt,
                        )
                    )
                    right_substep_contact_samples.append(
                        read_contact_group(
                            contact_sensors.get("right_finger"),
                            required_body_path=pick_object_config[
                                "path"
                            ],
                            physics_dt=physics_dt,
                        )
                    )
                target_history.append(targets)
                measured = np.asarray(robot.get_joint_positions(), dtype=np.float32).tolist()
                joint_history.append(measured)
                end_effector_position = prim_world_position(stage, end_effector_prim_path)
                end_effector_history.append(end_effector_position)
                grasp_body_position, grasp_body_orientation = prim_world_pose(
                    stage,
                    grasp_body_path,
                )
                if lula_ik_solver is not None:
                    (
                        control_frame_position,
                        control_frame_rotation,
                    ) = lula_ik_solver.compute_end_effector_pose()
                left_finger_position = prim_world_position(
                    stage,
                    left_contact_body_path,
                )
                right_finger_position = prim_world_position(
                    stage,
                    right_contact_body_path,
                )
                left_finger_bounds = (
                    prim_world_bounds(stage, left_contact_body_path)
                    if contact_only and frame % 10 == 0
                    else None
                )
                right_finger_bounds = (
                    prim_world_bounds(stage, right_contact_body_path)
                    if contact_only and frame % 10 == 0
                    else None
                )
                constraint_target_position = None
                should_capture_observation = dataset_enabled and frame % observation_every == 0
                should_update_perception = (
                    perception_enabled and frame > 0 and frame % perception_every == 0
                )
                if should_capture_observation or should_update_perception:
                    try:
                        rep.orchestrator.step(
                            rt_subframes=(
                                int(camera_config.get("rt_subframes", 1))
                                if should_update_perception
                                else 1
                            )
                        )
                        app_utils.play()
                        latest_rgb, latest_depth = capture_rgbd(
                            rgb_annotator,
                            depth_annotator,
                        )
                        if should_update_perception:
                            latest_object_estimate = estimate_dominant_color_pose(
                                latest_rgb,
                                latest_depth,
                                camera_intrinsics,
                                camera_to_world,
                                perception_config.get("object_channel", "red"),
                                min_pixels=perception_config.get("min_pixels", 20),
                                min_channel=perception_config.get("min_channel", 80),
                                min_dominance=perception_config.get("min_dominance", 30),
                                surface_to_center_m=object_half_height,
                            )
                            latest_target_estimate = estimate_dominant_color_pose(
                                latest_rgb,
                                latest_depth,
                                camera_intrinsics,
                                camera_to_world,
                                perception_config.get("target_channel", "green"),
                                min_pixels=perception_config.get("min_pixels", 20),
                                min_channel=perception_config.get("min_channel", 80),
                                min_dominance=perception_config.get("min_dominance", 30),
                            )
                            if (
                                grasp_visual_handoff_frame is not None
                                and grasp_visual_handoff_position is not None
                            ):
                                post_handoff_xy_drift = math.hypot(
                                    float(
                                        latest_object_estimate["position"][0]
                                    )
                                    - float(
                                        grasp_visual_handoff_position[0]
                                    ),
                                    float(
                                        latest_object_estimate["position"][1]
                                    )
                                    - float(
                                        grasp_visual_handoff_position[1]
                                    ),
                                )
                                grasp_visual_post_handoff_max_xy_drift = max(
                                    grasp_visual_post_handoff_max_xy_drift,
                                    post_handoff_xy_drift,
                                )
                            if (
                                contact_only
                                and not bilateral_contact_latched
                                and grasp_recovery_state == "initial"
                                and grasp_approach_gate_open_frame is not None
                                and grasp_visual_handoff_position is None
                                and pickup_config.get(
                                    "pickup_visual_tracking_enabled",
                                    False,
                                )
                            ):
                                tracked_object_position = list(
                                    latest_object_estimate["position"]
                                )
                                tracked_object_position[2] = (
                                    grasp_target_height
                                )
                                tracking_update = track_observed_pick_target(
                                    tracked_object_position,
                                    (
                                        aperture_tool_offset_world
                                        or object_grasp_offset
                                    ),
                                    pick_grasp_position,
                                    nominal_pick_grasp_position,
                                    max_step=pickup_config.get(
                                        "pickup_visual_tracking_max_step",
                                        [0.01, 0.01, 0.01],
                                    ),
                                )
                                pick_grasp_position[:] = tracking_update[
                                    "position"
                                ]
                                nominal_pick_grasp_position[:] = (
                                    tracking_update["nominal_position"]
                                )
                                pickup_visual_tracking_last_observed_nominal = (
                                    tracking_update[
                                        "observed_nominal_position"
                                    ]
                                )
                                pickup_visual_tracking_updates += 1
                            perception_updates += 1
                    except PerceptionError as error:
                        perception_failures.append(
                            {"frame": frame, "error": str(error)}
                        )

                grasp_window_end = attach_frame + grasp_retry_frames
                if (
                    attach_frame <= frame <= grasp_window_end
                    and not grasp_created
                    and real_grasp_body_found
                    and not contact_only
                ):
                    grasp_attempt_count += 1
                    object_before_attach = prim_world_position(stage, pick_object_config["path"])
                    expected_object_position = add_vector(grasp_body_position, object_grasp_offset)
                    grasp_attach_distance = distance(object_before_attach, expected_object_position)
                    if grasp_attach_distance <= max_grasp_attach_distance:
                        if grasp_mode == "solver_fixed_joint":
                            create_fixed_grasp_joint(
                                stage,
                                grasp_joint_path,
                                grasp_body_path,
                                pick_object_config["path"],
                                anchor_world_position=object_before_attach,
                            )
                            temporary_grasp_joint_created = True
                        else:
                            set_rigid_body_kinematic(stage, pick_object_config["path"], True)
                        grasp_attached = True
                        grasp_created = True
                        grasp_was_created = True
                        grasp_success_frame = frame
                        grasp_offset_world = subtract_vector(object_before_attach, grasp_body_position)
                        grasp_center_distance_at_attach = distance(object_before_attach, grasp_body_position)
                        append_phase(
                            phase_path,
                            "real_robotiq_grasp_attach",
                            frame=frame,
                            grasp_body=grasp_body_path,
                            object=pick_object_config["path"],
                            mode=grasp_mode,
                            distance=grasp_attach_distance,
                        )
                    elif frame in {attach_frame, grasp_window_end}:
                        append_phase(
                            phase_path,
                            "real_robotiq_grasp_rejected",
                            frame=frame,
                            distance=grasp_attach_distance,
                            max_distance=max_grasp_attach_distance,
                        )
                place_servo_start = release_frame - place_servo_frames
                if (
                    (grasp_attached or contact_only)
                    and place_servo_frames > 0
                    and place_servo_start
                    <= motion_frame
                    < release_frame
                    and should_update_perception
                    and latest_object_estimate
                    and grasp_body_position
                    and soft_landing_state == "inactive"
                ):
                    place_control_object_estimate = list(
                        latest_object_estimate["position"]
                    )
                    place_control_pose_source = "rgbd_estimate"
                    if (
                        contact_only
                        and bilateral_contact_validated
                        and physical_grasp_contact_offset_world is not None
                    ):
                        place_control_object_estimate = add_vector(
                            grasp_body_position,
                            physical_grasp_contact_offset_world,
                        )
                        place_control_pose_source = (
                            "rgbd_kinematic_propagation"
                        )
                        if place_kinematic_estimate_start_frame is None:
                            place_kinematic_estimate_start_frame = frame
                            kinematic_release_grasp_height = (
                                supported_object_height
                                + place_release_clearance
                                - float(
                                    physical_grasp_contact_offset_world[
                                        2
                                    ]
                                )
                            )
                            place_grasp_position[2] = (
                                kinematic_release_grasp_height
                            )
                            nominal_place_grasp_position[2] = (
                                kinematic_release_grasp_height
                            )
                            append_phase(
                                phase_path,
                                "place_kinematic_estimate_start",
                                frame=frame,
                                object_to_gripper_offset=(
                                    physical_grasp_contact_offset_world
                                ),
                                release_grasp_height=(
                                    kinematic_release_grasp_height
                                ),
                            )
                    desired_object_position = list(place_target_position)
                    desired_object_position[2] += place_release_clearance
                    active_place_servo_gain = (
                        list(place_servo_gain)
                        if isinstance(
                            place_servo_gain,
                            (list, tuple),
                        )
                        else [float(place_servo_gain)] * 3
                    )
                    if place_descent_gate_open_frame is None:
                        active_place_servo_gain[2] = 0.0
                    servo_result = integral_visual_servo_grasp_target(
                        place_control_object_estimate,
                        desired_object_position,
                        place_grasp_position,
                        nominal_place_grasp_position,
                        gain=active_place_servo_gain,
                        max_step=place_servo_max_step,
                        max_correction=place_servo_max_correction,
                    )
                    place_grasp_position[:] = servo_result[
                        "grasp_position"
                    ]
                    place_servo_last_error_xyz = servo_result[
                        "object_error"
                    ]
                    place_servo_last_error = (
                        place_servo_last_error_xyz[:2]
                    )
                    place_servo_correction = [
                        place_grasp_position[axis]
                        - nominal_place_grasp_position[axis]
                        for axis in range(2)
                    ]
                    place_servo_updates += 1
                    place_release_grasp_xy_tracking_error = math.hypot(
                        float(grasp_body_position[0])
                        - float(place_grasp_position[0]),
                        float(grasp_body_position[1])
                        - float(place_grasp_position[1]),
                    )
                    hover_guard = apply_place_hover_guard(
                        place_grasp_position,
                        nominal_place_grasp_position,
                        object_xy_error=servo_result["xy_error"],
                        grasp_xy_tracking_error=(
                            place_release_grasp_xy_tracking_error
                        ),
                        max_object_xy_error=(
                            place_release_max_xy_error
                        ),
                        max_grasp_xy_tracking_error=(
                            place_release_max_grasp_tracking_error
                            if place_require_grasp_tracking
                            else math.inf
                        ),
                        hover_clearance=(
                            place_hover_guard_clearance
                        ),
                    )
                    place_grasp_position[:] = hover_guard[
                        "grasp_position"
                    ]
                    place_hover_guard_active = hover_guard["active"]
                    place_hover_guard_activations += int(
                        place_hover_guard_active
                    )
                    place_release_grasp_tracking_error = distance(
                        grasp_body_position,
                        place_grasp_position,
                    )
                    if (
                        place_descent_gate_start_frame is not None
                        and place_descent_gate_open_frame is None
                    ):
                        if (
                            servo_result["xy_error"]
                            <= place_descent_gate_max_object_xy_error
                            and place_release_grasp_xy_tracking_error
                            <= place_descent_gate_max_grasp_xy_error
                        ):
                            place_descent_gate_converged_updates += 1
                        else:
                            place_descent_gate_converged_updates = 0
                        if (
                            place_descent_gate_converged_updates
                            >= place_descent_gate_stable_updates
                        ):
                            place_descent_gate_open_frame = frame + 1
                            if soft_landing_enabled:
                                soft_landing_state = "lowering"
                                soft_landing_start_frame = frame + 1
                                soft_landing_command_position = [
                                    float(place_grasp_position[0]),
                                    float(place_grasp_position[1]),
                                    float(place_grasp_position[2])
                                    + 0.08,
                                ]
                            append_phase(
                                phase_path,
                                "place_descent_gate_converged",
                                frame=frame,
                                descent_frame=(
                                    place_descent_gate_open_frame
                                ),
                                object_xy_error=(
                                    servo_result["xy_error"]
                                ),
                                grasp_xy_tracking_error=(
                                    place_release_grasp_xy_tracking_error
                                ),
                                stable_updates=(
                                    place_descent_gate_converged_updates
                                ),
                            )
                            if soft_landing_enabled:
                                append_phase(
                                    phase_path,
                                    "soft_landing_start",
                                    frame=frame + 1,
                                    alignment_clearance=(
                                        place_release_clearance
                                    ),
                                    landing_clearance=(
                                        soft_landing_clearance
                                    ),
                                    descent_step=soft_landing_step,
                                    command_position=(
                                        soft_landing_command_position
                                    ),
                                )
                    if placement_converged(
                        place_servo_last_error_xyz,
                        place_release_grasp_tracking_error,
                        max_xy_error=place_release_max_xy_error,
                        max_z_error=place_release_max_z_error,
                        max_grasp_tracking_error=(
                            place_release_max_grasp_tracking_error
                            if place_require_grasp_tracking
                            else math.inf
                        ),
                    ):
                        place_release_converged_updates += 1
                    else:
                        place_release_converged_updates = 0
                    if (
                        place_release_gate_start_frame is not None
                        and place_release_gate_open_frame is None
                        and place_release_converged_updates
                        >= place_release_stable_updates
                    ):
                        place_release_gate_open_frame = frame + 1
                        if (
                            soft_landing_enabled
                            and soft_landing_state == "inactive"
                        ):
                            soft_landing_state = "lowering"
                            soft_landing_start_frame = frame + 1
                        append_phase(
                            phase_path,
                            "place_release_gate_converged",
                            frame=frame,
                            release_frame=place_release_gate_open_frame,
                            object_error=place_servo_last_error_xyz,
                            grasp_tracking_error=(
                                place_release_grasp_tracking_error
                            ),
                            stable_updates=(
                                place_release_converged_updates
                            ),
                        )
                        if (
                            soft_landing_enabled
                            and soft_landing_state == "lowering"
                            and soft_landing_start_frame == frame + 1
                        ):
                            append_phase(
                                phase_path,
                                "soft_landing_start",
                                frame=frame + 1,
                                alignment_clearance=(
                                    place_release_clearance
                                ),
                                landing_clearance=(
                                    soft_landing_clearance
                                ),
                                descent_step=soft_landing_step,
                            )
                if (
                    place_release_gate_start_frame is not None
                    and place_release_gate_open_frame is None
                    and not place_release_gate_timed_out
                    and frame - place_release_gate_start_frame
                    >= place_release_timeout_frames
                ):
                    place_release_gate_timed_out = True
                    append_phase(
                        phase_path,
                        "place_release_gate_timeout",
                        frame=frame,
                        object_error=place_servo_last_error_xyz,
                        grasp_tracking_error=(
                            place_release_grasp_tracking_error
                        ),
                    )
                if (
                    motion_frame == release_frame
                    and actual_release_frame is None
                ):
                    actual_release_frame = frame
                    if grasp_mode == "solver_fixed_joint" and grasp_created:
                        remove_grasp_joint(stage, grasp_joint_path)
                    elif grasp_created and not contact_only:
                        set_rigid_body_kinematic(stage, pick_object_config["path"], False)
                    grasp_attached = False
                    grasp_created = False
                    append_phase(
                        phase_path,
                        "real_robotiq_grasp_release",
                        frame=frame,
                        motion_frame=motion_frame,
                        mode=grasp_mode,
                    )

                if (
                    grasp_attached
                    and grasp_mode != "solver_fixed_joint"
                    and not contact_only
                ):
                    lift_amount = max(
                        0.0,
                        min(
                            1.0,
                            (motion_frame - attach_frame)
                            / max(release_frame - attach_frame, 1),
                        ),
                    )
                    desired_position = [
                        pick_start_position[0],
                        pick_start_position[1],
                        pick_start_position[2] + lift_height * lift_amount,
                    ]
                    constraint_target_position = desired_position
                    set_cube_transform(stage, pick_object_config["path"], desired_position, pick_object_config["scale"])
                (
                    left_contact_force,
                    left_contact_sensor_valid,
                    left_contact_link_forces,
                    left_raw_contact_pairs,
                ) = merge_contact_group_samples(
                    left_substep_contact_samples
                )
                (
                    right_contact_force,
                    right_contact_sensor_valid,
                    right_contact_link_forces,
                    right_raw_contact_pairs,
                ) = merge_contact_group_samples(
                    right_substep_contact_samples
                )
                raw_contact_pairs = {}
                for record in [
                    *left_raw_contact_pairs,
                    *right_raw_contact_pairs,
                ]:
                    key = (record["body0"], record["body1"])
                    existing = raw_contact_pairs.get(key)
                    if (
                        existing is None
                        or record["force"] > existing["force"]
                    ):
                        raw_contact_pairs[key] = dict(record)
                    raw_contact_pair_peak_forces[key] = max(
                        float(record["force"]),
                        raw_contact_pair_peak_forces.get(key, 0.0),
                    )
                raw_contact_pairs = sorted(
                    raw_contact_pairs.values(),
                    key=lambda record: record["force"],
                    reverse=True,
                )
                left_contact_sensor_valid_frames += int(left_contact_sensor_valid)
                right_contact_sensor_valid_frames += int(right_contact_sensor_valid)
                contact_threshold = float(
                    pickup_config.get(
                        "contact_force_threshold_newtons",
                        0.25,
                    )
                )
                any_finger_contact = (
                    contact_only
                    and max(left_contact_force, right_contact_force)
                    >= contact_threshold
                )
                measured_finger_position = (
                    float(measured[primary_finger_index])
                    if primary_finger_index is not None
                    else 0.0
                )
                bilateral_contact_threshold = float(
                    pickup_config.get(
                        "bilateral_contact_force_threshold_newtons",
                        0.25,
                    )
                )
                bilateral_min_finger_position = float(
                    pickup_config.get(
                        "bilateral_contact_min_finger_position_radians",
                        0.0,
                    )
                )
                raw_bilateral_contact = contact_only and bilateral_grasp_ready(
                    left_contact_force,
                    right_contact_force,
                    measured_finger_position,
                    min_force=bilateral_contact_threshold,
                    min_finger_position=bilateral_min_finger_position,
                )
                raw_bilateral_contact_streak = (
                    raw_bilateral_contact_streak + 1
                    if raw_bilateral_contact
                    else 0
                )
                max_raw_bilateral_contact_streak = max(
                    max_raw_bilateral_contact_streak,
                    raw_bilateral_contact_streak,
                )
                bilateral_confirmation_frames = max(
                    1,
                    int(
                        pickup_config.get(
                            "bilateral_contact_confirmation_frames",
                            3,
                        )
                    ),
                )
                bilateral_confirmation_window_frames = max(
                    bilateral_confirmation_frames,
                    int(
                        pickup_config.get(
                            "bilateral_contact_confirmation_window_frames",
                            12,
                        )
                    ),
                )
                if raw_bilateral_contact:
                    raw_bilateral_contact_event_frames.append(frame)
                recent_event_cutoff = (
                    frame - bilateral_confirmation_window_frames + 1
                )
                raw_bilateral_contact_event_frames = [
                    event_frame
                    for event_frame in raw_bilateral_contact_event_frames
                    if event_frame >= recent_event_cutoff
                ]
                max_recent_raw_bilateral_events = max(
                    max_recent_raw_bilateral_events,
                    len(raw_bilateral_contact_event_frames),
                )
                confirmed_raw_bilateral_contact = (
                    temporal_contact_confirmed(
                        raw_bilateral_contact_event_frames,
                        frame,
                        required_events=bilateral_confirmation_frames,
                        window_frames=(
                            bilateral_confirmation_window_frames
                        ),
                    )
                )
                if (
                    contact_only
                    and measured_finger_position
                    >= bilateral_min_finger_position
                ):
                    if left_contact_force >= bilateral_contact_threshold:
                        last_left_contact_frame = frame
                    if right_contact_force >= bilateral_contact_threshold:
                        last_right_contact_frame = frame
                bilateral_contact_memory_frames = max(
                    0,
                    int(
                        pickup_config.get(
                            "bilateral_contact_memory_frames",
                            0,
                        )
                    ),
                )
                debounced_bilateral_contact = (
                    contact_only
                    and bilateral_contact_latched
                    and last_left_contact_frame is not None
                    and last_right_contact_frame is not None
                    and frame - last_left_contact_frame
                    <= bilateral_contact_memory_frames
                    and frame - last_right_contact_frame
                    <= bilateral_contact_memory_frames
                )
                bilateral_contact = (
                    confirmed_raw_bilateral_contact
                    or debounced_bilateral_contact
                )
                raw_bilateral_contact_frames += int(raw_bilateral_contact)
                debounced_bilateral_contact_frames += int(
                    debounced_bilateral_contact
                    and not raw_bilateral_contact
                )
                tactile_window_open = tactile_search_active(
                    close_started=(
                        (
                            grasp_approach_gate_open_frame
                            is not None
                            and frame
                            >= int(grasp_approach_gate_open_frame)
                        )
                        if contact_only
                        else (
                            gripper_close_start_frame is None
                            or frame
                            >= int(gripper_close_start_frame)
                        )
                    ),
                    measured_finger_position=(
                        measured_finger_position
                    ),
                    activation_position=float(
                        pickup_config.get(
                            "tactile_activation_position_radians",
                            0.0,
                        )
                    ),
                    hold_active=contact_hold_position is not None,
                    recovery_state=grasp_recovery_state,
                )
                if (
                    contact_only
                    and tactile_window_open
                    and motion_frame < release_frame
                ):
                    if (
                        any_finger_contact
                        and contact_hold_position is None
                    ):
                        contact_hold_frame = frame
                        contact_hold_position = (
                            tactile_contact_hold_target(
                                measured_finger_position,
                                float(
                                    pickup_config.get(
                                        "unilateral_contact_preload_radians",
                                        0.005,
                                    )
                                ),
                                float(
                                    pickup_config.get(
                                        "close_position_radians",
                                        47.0,
                                    )
                                ),
                            )
                        )
                        append_phase(
                            phase_path,
                            "gripper_tactile_search_engaged",
                            frame=frame,
                            finger_position=measured_finger_position,
                            left_force_newtons=left_contact_force,
                            right_force_newtons=right_contact_force,
                            inherited_target_position=(
                                contact_hold_position
                            ),
                            search_max_effort=(
                                float(
                                    pickup_config.get(
                                        "finger_max_effort",
                                        180.0,
                                    )
                                )
                            ),
                        )
                    elif (
                        contact_hold_position is not None
                        and not bilateral_contact_latched
                    ):
                        if (
                            max(left_contact_force, right_contact_force)
                            > unilateral_force_limit
                            and not bilateral_contact
                        ):
                            contact_hold_position = max(
                                0.0,
                                contact_hold_position
                                - unilateral_force_backoff,
                            )
                            unilateral_force_backoff_updates += 1
                        else:
                            contact_hold_position = min(
                                float(
                                    pickup_config.get(
                                        "close_position_radians",
                                        47.0,
                                    )
                                ),
                                contact_hold_position
                                + float(
                                    pickup_config.get(
                                        "tactile_close_step_radians",
                                        0.0004,
                                    )
                                ),
                            )
                if (
                    unilateral_recenter_enabled
                    and contact_only
                    and grasp_recovery_state == "initial"
                    and grasp_approach_gate_open_frame is not None
                    and not bilateral_contact_validated
                    and not bilateral_contact_latched
                    and grasp_aperture_left_bounds is not None
                    and grasp_aperture_right_bounds is not None
                ):
                    recenter_force_threshold = float(
                        pickup_config.get(
                            (
                                "grasp_validation_min_force_newtons"
                                if bilateral_contact_latched
                                else "bilateral_contact_force_threshold_newtons"
                            ),
                            (
                                bilateral_hold_min_force
                                if bilateral_contact_latched
                                else 0.25
                            ),
                        )
                    )
                    left_recenter_active = (
                        left_contact_force >= recenter_force_threshold
                    )
                    right_recenter_active = (
                        right_contact_force >= recenter_force_threshold
                    )
                    if left_recenter_active != right_recenter_active:
                        unilateral_recenter_last_side = (
                            "left" if left_recenter_active else "right"
                        )
                        unilateral_recenter_persistence_remaining = (
                            unilateral_recenter_persistence_frames
                        )
                    elif left_recenter_active and right_recenter_active:
                        unilateral_recenter_persistence_remaining = 0
                    elif unilateral_recenter_persistence_remaining > 0:
                        unilateral_recenter_persistence_remaining -= 1
                        left_recenter_active = (
                            unilateral_recenter_last_side == "left"
                        )
                        right_recenter_active = (
                            unilateral_recenter_last_side == "right"
                        )
                    recenter_result = unilateral_contact_recenter_target(
                        pick_grasp_position,
                        nominal_pick_grasp_position,
                        grasp_aperture_left_bounds,
                        grasp_aperture_right_bounds,
                        (
                            recenter_force_threshold
                            if left_recenter_active
                            else 0.0
                        ),
                        (
                            recenter_force_threshold
                            if right_recenter_active
                            else 0.0
                        ),
                        min_force=recenter_force_threshold,
                        step=unilateral_recenter_step,
                        max_correction=(
                            unilateral_recenter_max_correction
                        ),
                    )
                    if recenter_result["active"]:
                        pick_grasp_position[:] = recenter_result["position"]
                        unilateral_recenter_updates += 1
                        if (
                            recenter_result["contact_side"]
                            != unilateral_recenter_last_side
                        ):
                            unilateral_recenter_last_side = recenter_result[
                                "contact_side"
                            ]
                            append_phase(
                                phase_path,
                                "unilateral_contact_recenter",
                                frame=frame,
                                contact_side=(
                                    unilateral_recenter_last_side
                                ),
                                axis_xy=recenter_result["axis_xy"],
                                target_position=pick_grasp_position,
                            )
                if (
                    transport_recenter_enabled
                    and contact_only
                    and bilateral_contact_validated
                    and task_phase in {"lift", "transfer", "place"}
                    and soft_landing_state == "inactive"
                    and left_finger_bounds is not None
                    and right_finger_bounds is not None
                ):
                    transport_recenter_result = (
                        unilateral_contact_recenter_target(
                            transport_recenter_offset_xy + [0.0],
                            [0.0, 0.0, 0.0],
                            left_finger_bounds,
                            right_finger_bounds,
                            left_contact_force,
                            right_contact_force,
                            min_force=bilateral_hold_min_force,
                            step=transport_recenter_step,
                            max_correction=(
                                transport_recenter_max_correction
                            ),
                            move_toward_contact=False,
                        )
                    )
                    if transport_recenter_result["active"]:
                        transport_recenter_offset_xy[:] = (
                            transport_recenter_result["position"][:2]
                        )
                        transport_recenter_updates += 1
                        if (
                            transport_recenter_result["contact_side"]
                            != transport_recenter_last_side
                        ):
                            transport_recenter_last_side = (
                                transport_recenter_result[
                                    "contact_side"
                                ]
                            )
                            append_phase(
                                phase_path,
                                "transport_contact_recenter",
                                frame=frame,
                                contact_side=(
                                    transport_recenter_last_side
                                ),
                                axis_xy=transport_recenter_result[
                                    "axis_xy"
                                ],
                                correction_xy=(
                                    transport_recenter_offset_xy
                                ),
                                left_force_newtons=(
                                    left_contact_force
                                ),
                                right_force_newtons=(
                                    right_contact_force
                                ),
                            )
                if (
                    contact_only
                    and soft_landing_enabled
                    and soft_landing_state
                    in {"lowering", "support_hold", "unloading"}
                    and (
                        soft_landing_start_frame is None
                        or frame >= soft_landing_start_frame
                    )
                    and grasp_body_position is not None
                    and physical_grasp_contact_offset_world is not None
                ):
                    soft_landing_object_estimate = add_vector(
                        grasp_body_position,
                        physical_grasp_contact_offset_world,
                    )
                    soft_landing_effective_command = (
                        list(soft_landing_command_position)
                        if soft_landing_command_position is not None
                        else None
                    )
                    if soft_landing_effective_command is not None:
                        soft_landing_effective_command[0] = (
                            float(place_grasp_position[0])
                            + transport_recenter_offset_xy[0]
                        )
                        soft_landing_effective_command[1] = (
                            float(place_grasp_position[1])
                            + transport_recenter_offset_xy[1]
                        )
                    estimated_clearance = (
                        float(soft_landing_object_estimate[2])
                        - supported_object_height
                    )
                    peak_finger_force = max(
                        left_contact_force,
                        right_contact_force,
                    )
                    if soft_landing_state == "lowering":
                        if soft_landing_force_baseline is None:
                            soft_landing_force_baseline = (
                                peak_finger_force
                            )
                        support_by_height = (
                            estimated_clearance
                            <= soft_landing_clearance
                        )
                        support_by_force = (
                            peak_finger_force
                            >= soft_landing_force_baseline
                            + soft_landing_force_threshold
                            and estimated_clearance
                            <= soft_landing_force_max_clearance
                        )
                        if support_by_height or support_by_force:
                            soft_landing_state = "support_hold"
                            soft_landing_support_frame = frame
                            soft_landing_support_reason = (
                                "finger_force"
                                if support_by_force
                                else "estimated_height"
                            )
                            soft_landing_clearance_at_support = (
                                estimated_clearance
                            )
                            soft_landing_peak_force_at_support = (
                                peak_finger_force
                            )
                            append_phase(
                                phase_path,
                                "soft_landing_support_detected",
                                frame=frame,
                                reason=soft_landing_support_reason,
                                estimated_object_position=(
                                    soft_landing_object_estimate
                                ),
                                estimated_clearance=(
                                    estimated_clearance
                                ),
                                peak_finger_force_newtons=(
                                    peak_finger_force
                                ),
                                force_baseline_newtons=(
                                    soft_landing_force_baseline
                                ),
                            )
                            if place_release_gate_start_frame is None:
                                place_release_gate_start_frame = (
                                    soft_landing_start_frame
                                )
                            if place_release_gate_open_frame is None:
                                place_release_gate_open_frame = frame
                                append_phase(
                                    phase_path,
                                    "place_release_gate_converged",
                                    frame=frame,
                                    release_frame=frame,
                                    object_error=[
                                        float(place_target_position[0])
                                        - float(
                                            soft_landing_object_estimate[
                                                0
                                            ]
                                        ),
                                        float(place_target_position[1])
                                        - float(
                                            soft_landing_object_estimate[
                                                1
                                            ]
                                        ),
                                        (
                                            supported_object_height
                                            + soft_landing_clearance
                                            - float(
                                                soft_landing_object_estimate[
                                                    2
                                                ]
                                            )
                                        ),
                                    ],
                                    grasp_tracking_error=(
                                        distance(
                                            control_frame_position,
                                            soft_landing_effective_command,
                                        )
                                        if (
                                            control_frame_position
                                            is not None
                                            and soft_landing_effective_command
                                            is not None
                                        )
                                        else None
                                    ),
                                    stable_updates=1,
                                    source="soft_landing_support",
                                )
                        else:
                            command_tracking_error = (
                                distance(
                                    control_frame_position,
                                    soft_landing_effective_command,
                                )
                                if (
                                    control_frame_position is not None
                                    and soft_landing_effective_command
                                    is not None
                                )
                                else 0.0
                            )
                            if (
                                soft_landing_command_position is not None
                                and command_tracking_error <= 0.015
                            ):
                                soft_landing_command_position[2] -= (
                                    soft_landing_step
                                )
                    elif (
                        soft_landing_state == "support_hold"
                        and soft_landing_support_frame is not None
                        and frame - soft_landing_support_frame
                        >= soft_landing_hold_frames
                    ):
                        soft_landing_state = "unloading"
                        soft_release_start_frame = frame + 1
                        soft_release_initial_target = max(
                            0.0,
                            float(
                                contact_hold_position
                                if contact_hold_position is not None
                                else measured_finger_position
                            ),
                        )
                        append_phase(
                            phase_path,
                            "soft_release_start",
                            frame=soft_release_start_frame,
                            initial_finger_target=(
                                soft_release_initial_target
                            ),
                            unload_frames=soft_release_frames,
                        )
                    elif (
                        soft_landing_state == "unloading"
                        and soft_release_start_frame is not None
                        and frame - soft_release_start_frame
                        >= soft_release_frames
                    ):
                        soft_landing_state = "released"
                        soft_release_complete_frame = frame + 1
                        contact_hold_position = None
                        append_phase(
                            phase_path,
                            "soft_release_complete",
                            frame=frame,
                            release_frame=soft_release_complete_frame,
                            estimated_object_position=(
                                soft_landing_object_estimate
                            ),
                            peak_finger_force_newtons=(
                                peak_finger_force
                            ),
                        )
                if (
                    contact_only
                    and bilateral_contact_latched
                    and contact_hold_position is not None
                    and motion_frame < release_frame
                    and soft_landing_state
                    not in {"support_hold", "unloading", "released"}
                ):
                    configured_close_position = float(
                        pickup_config.get(
                            "close_position_radians",
                            47.0,
                        )
                    )
                    transport_hold_max_close_adjustment = max(
                        0.0,
                        float(
                            pickup_config.get(
                                "transport_hold_max_close_adjustment_radians",
                                0.0,
                            )
                        ),
                    )
                    hold_max_position = configured_close_position
                    if (
                        bilateral_contact_validated
                        and grasp_validated_finger_target is not None
                    ):
                        hold_max_position = min(
                            configured_close_position,
                            grasp_validated_finger_target
                            + transport_hold_max_close_adjustment,
                        )
                    hold_update = force_controlled_gripper_target(
                        contact_hold_position,
                        measured_finger_position,
                        left_contact_force,
                        right_contact_force,
                        min_force=bilateral_hold_min_force,
                        max_force=bilateral_hold_max_force,
                        close_step=float(
                            pickup_config.get(
                                "bilateral_hold_close_step_radians",
                                0.0001,
                            )
                        ),
                        backoff_step=bilateral_force_backoff,
                        max_position=hold_max_position,
                        close_on_unilateral=bool(
                            pickup_config.get(
                                "transport_unilateral_close_enabled",
                                False,
                            )
                        ),
                        max_preload_error=pickup_config.get(
                            "gripper_force_max_preload_error_radians"
                        ),
                        preload_reference_position=(
                            grasp_validated_finger_target
                            if (
                                bilateral_contact_validated
                                and grasp_validated_finger_target
                                is not None
                            )
                            else gripper_preload_reference_position
                        ),
                    )
                    contact_hold_position = hold_update["position"]
                    if hold_update["action"] == "backoff":
                        bilateral_force_backoff_updates += 1
                    elif hold_update["action"] == "close":
                        bilateral_hold_close_updates += 1
                contact_force_history.append(
                    {
                        "frame": frame,
                        "left_finger": left_contact_force,
                        "right_finger": right_contact_force,
                        "left_sensor_valid": left_contact_sensor_valid,
                        "right_sensor_valid": right_contact_sensor_valid,
                        "left_link_forces": left_contact_link_forces,
                        "right_link_forces": right_contact_link_forces,
                        "bilateral": bilateral_contact,
                        "raw_bilateral": raw_bilateral_contact,
                    }
                )
                if bilateral_contact:
                    bilateral_contact_frames += 1
                    if (
                        not bilateral_contact_latched
                        and grasp_recovery_state in {"initial", "close"}
                    ):
                        if (
                            max_efforts is not None
                            and hasattr(
                                controller,
                                "set_max_efforts",
                            )
                            and not finger_contact_effort_configured
                        ):
                            for joint_name in (
                                actuated_finger_joint_names
                            ):
                                max_efforts[
                                    dof_names.index(joint_name)
                                ] = finger_contact_max_effort
                            controller.set_max_efforts(max_efforts)
                            finger_contact_effort_configured = True
                            append_phase(
                                phase_path,
                                "gripper_bilateral_effort_handoff",
                                frame=frame,
                                contact_max_effort=(
                                    finger_contact_max_effort
                                ),
                                left_force_newtons=(
                                    left_contact_force
                                ),
                                right_force_newtons=(
                                    right_contact_force
                                ),
                            )
                        bilateral_contact_latched = True
                        grasp_validation_start_frame = frame
                        grasp_validation_support_frames = 0
                        grasp_validation_terminal_stable_frames = min(
                            raw_bilateral_contact_streak,
                            grasp_validation_required_terminal_frames,
                        )
                        grasp_validation_support_frames = (
                            grasp_validation_terminal_stable_frames
                        )
                        grasp_validation_attempt_max_terminal_stable_frames = (
                            grasp_validation_terminal_stable_frames
                        )
                        grasp_proof_state = "inactive"
                        grasp_contact_object_position = (
                            prim_world_position(
                                stage,
                                pick_object_config["path"],
                            )
                        )
                        physical_grasp_contact_offset_world = (
                            subtract_vector(
                                grasp_contact_object_position,
                                grasp_body_position,
                            )
                            if (
                                grasp_contact_object_position
                                and grasp_body_position
                            )
                            else None
                        )
                        grasp_motion_start_frame = None
                        if gripper_preload_reference_position is None:
                            gripper_preload_reference_position = float(
                                measured_finger_position
                            )
                        contact_hold_position = max(
                            float(measured[primary_finger_index]),
                            float(contact_hold_position or 0.0),
                            float(
                                pickup_config.get(
                                    "grasp_validation_min_finger_target_radians",
                                    0.0,
                                )
                            ),
                        )
                        append_phase(
                            phase_path,
                            "gripper_contact_hold_engaged",
                            frame=frame,
                            grasp_motion_start_frame=None,
                            finger_position=contact_hold_position,
                            preload_reference_position=(
                                gripper_preload_reference_position
                            ),
                            left_force_newtons=left_contact_force,
                            right_force_newtons=right_contact_force,
                            object_to_gripper_offset=(
                                physical_grasp_contact_offset_world
                            ),
                            validation_frames=grasp_validation_frames,
                            minimum_validation_finger_target=float(
                                pickup_config.get(
                                    "grasp_validation_min_finger_target_radians",
                                    0.0,
                                )
                            ),
                        )
                        if grasp_validation_frames > 0:
                            append_phase(
                                phase_path,
                                "grasp_validation_start",
                                frame=frame,
                                validation_end_frame=frame
                                + grasp_validation_max_frames,
                            )
                validation_terminal_contact = (
                    confirmed_raw_bilateral_contact
                    and bilateral_grasp_ready(
                        left_contact_force,
                        right_contact_force,
                        measured_finger_position,
                        min_force=float(
                            pickup_config.get(
                                "grasp_validation_min_force_newtons",
                                bilateral_hold_min_force,
                            )
                        ),
                        min_finger_position=(
                            bilateral_min_finger_position
                        ),
                    )
                )
                if (
                    bilateral_contact_latched
                    and not bilateral_contact_validated
                    and grasp_validation_start_frame is not None
                    and grasp_proof_state == "inactive"
                ):
                    grasp_validation_support_frames += int(
                        bilateral_contact
                    )
                    if validation_terminal_contact:
                        grasp_validation_terminal_stable_frames += 1
                    else:
                        grasp_validation_terminal_stable_frames = 0
                    grasp_validation_max_terminal_stable_frames = max(
                        grasp_validation_max_terminal_stable_frames,
                        grasp_validation_terminal_stable_frames,
                    )
                    grasp_validation_attempt_max_terminal_stable_frames = max(
                        grasp_validation_attempt_max_terminal_stable_frames,
                        grasp_validation_terminal_stable_frames,
                    )
                    validation_elapsed = (
                        frame - grasp_validation_start_frame + 1
                    )
                    validation_min_support_frames = max(
                        1,
                        int(
                            pickup_config.get(
                                "grasp_validation_min_support_frames",
                                max(1, grasp_validation_frames // 2),
                            )
                        ),
                    )
                    validation_decision = grasp_validation_decision(
                        validation_elapsed,
                        grasp_validation_support_frames,
                        min_frames=grasp_validation_frames,
                        max_frames=grasp_validation_max_frames,
                        min_support_frames=(
                            validation_min_support_frames
                        ),
                        confirmed_raw_contact=validation_terminal_contact,
                        terminal_stable_frames=(
                            grasp_validation_attempt_max_terminal_stable_frames
                        ),
                        required_terminal_stable_frames=(
                            grasp_validation_required_terminal_frames
                        ),
                    )
                    if validation_decision == "validated":
                        if grasp_proof_enabled:
                            grasp_proof_state = "lifting"
                            grasp_proof_start_frame = frame
                            grasp_proof_lower_start_frame = None
                            grasp_proof_start_position = [
                                float(value)
                                for value in (
                                    control_frame_position
                                    if control_frame_position is not None
                                    else grasp_body_position
                                )
                            ]
                            grasp_proof_target_position = list(
                                grasp_proof_start_position
                            )
                            grasp_proof_target_position[2] += (
                                grasp_proof_lift_height
                            )
                            grasp_proof_start_object_position = (
                                prim_world_position(
                                    stage,
                                    pick_object_config["path"],
                                )
                            )
                            grasp_proof_max_object_lift = 0.0
                            # Validation and proof are contiguous parts of the
                            # same grasp attempt. Preserve the contact evidence
                            # already established immediately before lift.
                            grasp_proof_contact_streak = (
                                grasp_validation_terminal_stable_frames
                            )
                            grasp_proof_max_contact_streak = (
                                grasp_validation_terminal_stable_frames
                            )
                            grasp_proof_max_rigidity_error = 0.0
                            grasp_proof_contact_rebuild_wait_frame = None
                            append_phase(
                                phase_path,
                                "grasp_proof_lift_start",
                                frame=frame,
                                lift_height_meters=(
                                    grasp_proof_lift_height
                                ),
                                lift_frames=grasp_proof_lift_frames,
                                hold_frames=grasp_proof_hold_frames,
                                hold_max_frames=(
                                    grasp_proof_hold_max_frames
                                ),
                                required_contact_frames=(
                                    grasp_proof_required_contact_frames
                                ),
                                required_terminal_contact_frames=(
                                    grasp_proof_terminal_contact_frames
                                ),
                                inherited_validation_contact_frames=(
                                    grasp_validation_terminal_stable_frames
                                ),
                                start_position=(
                                    grasp_proof_start_position
                                ),
                                target_position=(
                                    grasp_proof_target_position
                                ),
                            )
                        else:
                            mark_physical_grasp_validated(
                                frame,
                                validation_elapsed=validation_elapsed,
                                validation_source="contact_stability",
                                left_force=left_contact_force,
                                right_force=right_contact_force,
                            )
                    elif validation_decision == "failed":
                        validation_failure_support_frames = (
                            grasp_validation_support_frames
                        )
                        validation_failure_terminal_stable_frames = (
                            grasp_validation_attempt_max_terminal_stable_frames
                        )
                        grasp_validation_failures += 1
                        bilateral_contact_latched = False
                        grasp_validation_start_frame = None
                        grasp_validation_support_frames = 0
                        grasp_validation_terminal_stable_frames = 0
                        grasp_validation_attempt_max_terminal_stable_frames = 0
                        append_phase(
                            phase_path,
                            "grasp_validation_failed",
                            frame=frame,
                            left_force_newtons=left_contact_force,
                            right_force_newtons=right_contact_force,
                            validation_frames=validation_elapsed,
                            support_frames=(
                                validation_failure_support_frames
                            ),
                            required_support_frames=(
                                validation_min_support_frames
                            ),
                            terminal_stable_frames=(
                                validation_failure_terminal_stable_frames
                            ),
                            required_terminal_stable_frames=(
                                grasp_validation_required_terminal_frames
                            ),
                            max_terminal_stable_frames=(
                                validation_failure_terminal_stable_frames
                            ),
                        )
                transport_pick_object_position = prim_world_position(
                    stage,
                    pick_object_config["path"],
                )
                transport_support = transport_grasp_support(
                    bilateral_contact,
                    transport_pick_object_position,
                    grasp_body_position,
                    physical_grasp_contact_offset_world,
                    max_rigidity_error=float(
                        pickup_config.get(
                            "transport_rigidity_max_error_meters",
                            0.015,
                        )
                    ),
                )
                if transport_support["rigidity_error"] is not None:
                    max_transport_rigidity_error = max(
                        max_transport_rigidity_error,
                        transport_support["rigidity_error"],
                    )
                if grasp_proof_state in {"lifting", "holding"}:
                    if (
                        transport_pick_object_position is not None
                        and grasp_proof_start_object_position is not None
                    ):
                        grasp_proof_max_object_lift = max(
                            grasp_proof_max_object_lift,
                            float(transport_pick_object_position[2])
                            - float(
                                grasp_proof_start_object_position[2]
                            ),
                        )
                    if transport_support["rigidity_error"] is not None:
                        grasp_proof_max_rigidity_error = max(
                            grasp_proof_max_rigidity_error,
                            float(transport_support["rigidity_error"]),
                        )
                    grasp_proof_contact = bilateral_grasp_ready(
                        left_contact_force,
                        right_contact_force,
                        measured_finger_position,
                        min_force=grasp_proof_min_force,
                        min_finger_position=(
                            bilateral_min_finger_position
                        ),
                    )
                    if grasp_proof_contact:
                        grasp_proof_contact_streak += 1
                    else:
                        grasp_proof_contact_streak = 0
                    grasp_proof_max_contact_streak = max(
                        grasp_proof_max_contact_streak,
                        grasp_proof_contact_streak,
                    )
                    if (
                        grasp_proof_state == "lifting"
                        and frame - grasp_proof_start_frame
                        >= grasp_proof_lift_frames
                    ):
                        grasp_proof_state = "holding"
                        grasp_proof_start_frame = frame
                        append_phase(
                            phase_path,
                            "grasp_proof_hold_start",
                            frame=frame,
                            object_lift_meters=(
                                grasp_proof_max_object_lift
                            ),
                            max_contact_streak=(
                                grasp_proof_max_contact_streak
                            ),
                        )
                    elif grasp_proof_state == "holding":
                        grasp_proof_hold_elapsed = (
                            frame - grasp_proof_start_frame
                        )
                        proof_lift_passed = (
                            grasp_proof_max_object_lift
                            >= grasp_proof_min_object_lift
                        )
                        proof_contact_history_passed = (
                            grasp_proof_max_contact_streak
                            >= grasp_proof_required_contact_frames
                        )
                        proof_terminal_contact_passed = (
                            grasp_proof_contact_streak
                            >= grasp_proof_terminal_contact_frames
                        )
                        proof_contact_passed = (
                            proof_contact_history_passed
                            and transport_support["present"]
                        )
                        proof_rigidity_passed = (
                            grasp_proof_max_rigidity_error
                            <= grasp_proof_max_allowed_rigidity_error
                        )
                        if (
                            grasp_proof_hold_elapsed
                            >= grasp_proof_hold_frames
                            and proof_lift_passed
                            and proof_contact_passed
                            and proof_rigidity_passed
                        ):
                            grasp_proof_successes += 1
                            append_phase(
                                phase_path,
                                "grasp_proof_lift_passed",
                                frame=frame,
                                object_lift_meters=(
                                    grasp_proof_max_object_lift
                                ),
                                terminal_contact_streak=(
                                    grasp_proof_contact_streak
                                ),
                                hold_frames=grasp_proof_hold_elapsed,
                                max_rigidity_error_meters=(
                                    grasp_proof_max_rigidity_error
                                ),
                            )
                            mark_physical_grasp_validated(
                                frame,
                                validation_elapsed=(
                                    frame
                                    - grasp_validation_start_frame
                                    + 1
                                ),
                                validation_source="proof_lift",
                                left_force=left_contact_force,
                                right_force=right_contact_force,
                            )
                        elif (
                            grasp_proof_hold_elapsed
                            >= grasp_proof_hold_max_frames
                        ):
                            grasp_proof_failures += 1
                            grasp_validation_failures += 1
                            grasp_proof_state = "lowering"
                            grasp_proof_lower_start_frame = frame
                            append_phase(
                                phase_path,
                                "grasp_proof_lift_failed",
                                frame=frame,
                                object_lift_meters=(
                                    grasp_proof_max_object_lift
                                ),
                                max_contact_streak=(
                                    grasp_proof_max_contact_streak
                                ),
                                terminal_contact_streak=(
                                    grasp_proof_contact_streak
                                ),
                                hold_frames=grasp_proof_hold_elapsed,
                                max_rigidity_error_meters=(
                                    grasp_proof_max_rigidity_error
                                ),
                                lift_passed=proof_lift_passed,
                                contact_passed=proof_contact_passed,
                                contact_history_passed=(
                                    proof_contact_history_passed
                                ),
                                terminal_contact_passed=(
                                    proof_terminal_contact_passed
                                ),
                                rigidity_passed=proof_rigidity_passed,
                            )
                        elif (
                            grasp_proof_hold_elapsed
                            >= grasp_proof_hold_frames
                            and grasp_proof_contact_rebuild_wait_frame
                            is None
                        ):
                            grasp_proof_contact_rebuild_wait_frame = (
                                frame
                            )
                            append_phase(
                                phase_path,
                                "grasp_proof_contact_rebuild_wait",
                                frame=frame,
                                max_wait_frame=(
                                    grasp_proof_start_frame
                                    + grasp_proof_hold_max_frames
                                ),
                                object_lift_meters=(
                                    grasp_proof_max_object_lift
                                ),
                                max_contact_streak=(
                                    grasp_proof_max_contact_streak
                                ),
                                terminal_contact_streak=(
                                    grasp_proof_contact_streak
                                ),
                            )
                elif (
                    grasp_proof_state == "lowering"
                    and frame - grasp_proof_lower_start_frame
                    >= grasp_proof_lower_frames
                ):
                    append_phase(
                        phase_path,
                        "grasp_proof_return_complete",
                        frame=frame,
                        return_position=grasp_proof_start_position,
                    )
                    bilateral_contact_latched = False
                    grasp_validation_start_frame = None
                    grasp_validation_support_frames = 0
                    grasp_validation_terminal_stable_frames = 0
                    grasp_validation_attempt_max_terminal_stable_frames = 0
                    grasp_proof_state = "inactive"
                    grasp_proof_start_frame = None
                    grasp_proof_lower_start_frame = None
                    grasp_proof_start_position = None
                    grasp_proof_target_position = None
                    grasp_proof_start_object_position = None
                    grasp_proof_contact_rebuild_wait_frame = None
                stable_transport_contact = (
                    contact_only
                    and bilateral_contact_validated
                    and transport_support["present"]
                    and motion_frame < release_frame
                )
                if (
                    stable_transport_contact
                    and task_phase in {"lift", "transfer", "place"}
                ):
                    transport_contact_frames += 1
                    direct_transport_contact_frames += int(
                        transport_support["source"] == "direct_contact"
                    )
                    inferred_transport_support_frames += int(
                        transport_support["source"] == "rigidity_inferred"
                    )
                    continuous_transport_contact_frames += 1
                    max_continuous_transport_contact_frames = max(
                        max_continuous_transport_contact_frames,
                        continuous_transport_contact_frames,
                    )
                elif task_phase in {"lift", "transfer", "place"}:
                    continuous_transport_contact_frames = 0
                contact_force_history[-1]["stable_transport"] = (
                    stable_transport_contact
                )
                contact_force_history[-1]["transport_support_source"] = (
                    transport_support["source"]
                )
                contact_force_history[-1]["transport_rigidity_error"] = (
                    transport_support["rigidity_error"]
                )
                contact_loss_streak = update_contact_loss_streak(
                    contact_loss_streak,
                    monitoring=(
                        contact_only
                        and bilateral_contact_validated
                        and motion_frame < release_frame
                        and task_phase in {"lift", "transfer"}
                    ),
                    contact_present=transport_support["present"],
                )
                max_contact_loss_streak = max(
                    max_contact_loss_streak,
                    contact_loss_streak,
                )
                contact_loss_limit = max(
                    1,
                    int(
                        pickup_config.get(
                            "grasp_contact_loss_frames",
                            12,
                        )
                    ),
                )
                if (
                    contact_only
                    and bilateral_contact_validated
                    and contact_loss_streak >= contact_loss_limit
                    and motion_frame < release_frame
                    and task_phase in {"lift", "transfer"}
                ):
                    grasp_contact_loss_events += 1
                    continuous_transport_contact_frames = 0
                    bilateral_contact_latched = False
                    bilateral_contact_validated = False
                    grasp_validation_start_frame = None
                    grasp_validation_support_frames = 0
                    grasp_validation_terminal_stable_frames = 0
                    grasp_validation_attempt_max_terminal_stable_frames = 0
                    grasp_proof_state = "inactive"
                    grasp_proof_start_frame = None
                    grasp_proof_lower_start_frame = None
                    grasp_proof_start_position = None
                    grasp_proof_target_position = None
                    grasp_proof_start_object_position = None
                    grasp_attached = False
                    grasp_created = False
                    grasp_motion_start_frame = None
                    grasp_recovery_state = "initial"
                    contact_hold_frame = None
                    contact_hold_position = None
                    gripper_preload_reference_position = None
                    grasp_validated_finger_position = None
                    grasp_validated_finger_target = None
                    contact_loss_streak = 0
                    append_phase(
                        phase_path,
                        "grasp_contact_lost",
                        frame=frame,
                        loss_frames=contact_loss_limit,
                        left_force_newtons=left_contact_force,
                        right_force_newtons=right_contact_force,
                    )
                if contact_only:
                    grasp_attached = stable_transport_contact
                    if motion_frame >= release_frame:
                        grasp_created = False
                if grasp_attached and grasp_mode == "solver_fixed_joint" and grasp_body_position and grasp_offset_world:
                    constraint_target_position = add_vector(grasp_body_position, grasp_offset_world)
                pick_object_position = prim_world_position(stage, pick_object_config["path"])
                attached_distance = (
                    abs(distance(pick_object_position, grasp_body_position) - grasp_center_distance_at_attach)
                    if pick_object_position
                    and grasp_body_position
                    and grasp_center_distance_at_attach is not None
                    and grasp_attached
                    else None
                )
                if attached_distance is not None:
                    attached_distance_history.append(attached_distance)
                object_history.append(
                    {
                        "frame": frame,
                        "motion_frame": motion_frame,
                        "position": pick_object_position,
                        "attached": grasp_attached,
                        "phase": task_phase,
                        "contact": stable_transport_contact
                        if contact_only
                        else task_phase in {"grasp", "lift", "transfer", "place"}
                        and grasp_attached,
                        "grasp_body_position": grasp_body_position,
                        "constraint_target_position": constraint_target_position,
                        "attached_distance": attached_distance,
                        "grasp_constraint": grasp_mode if grasp_created else None,
                    }
                )
                if preview_enabled and frame in preview_frames:
                    append_phase(phase_path, "preview_frame_capture_start", frame=frame)
                    if perception_enabled or dataset_enabled:
                        rep.orchestrator.step(
                            rt_subframes=int(
                                camera_config.get("rt_subframes", 1)
                            )
                        )
                        latest_rgb, latest_depth = capture_rgbd(
                            rgb_annotator,
                            depth_annotator,
                        )
                        from PIL import Image

                        Image.fromarray(latest_rgb).save(
                            preview_dir
                            / f"rgb_{preview_capture_index:04d}.png"
                        )
                        preview_capture_index += 1
                    else:
                        preview_writer.attach([render_product])
                        rep.orchestrator.step(rt_subframes=int(camera_config.get("rt_subframes", 1)))
                        preview_writer.detach()
                    app_utils.play()
                    append_phase(phase_path, "preview_frame_capture_end", frame=frame)
                if frame % record_every == 0:
                    measured_joint_efforts = None
                    if hasattr(robot, "get_measured_joint_efforts"):
                        measured_joint_efforts = np.asarray(
                            robot.get_measured_joint_efforts(),
                            dtype=np.float32,
                        ).reshape(-1).tolist()
                        if len(measured_joint_efforts) != robot.num_dof:
                            measured_joint_efforts = None
                    primary_finger_effort = (
                        measured_joint_efforts[primary_finger_index]
                        if measured_joint_efforts is not None
                        and primary_finger_index is not None
                        else None
                    )
                    if primary_finger_effort is not None:
                        peak_primary_finger_effort = max(
                            peak_primary_finger_effort,
                            abs(float(primary_finger_effort)),
                        )
                    mimic_follower_error = (
                        max(
                            (
                                abs(
                                    abs(float(measured[dof_names.index(name)]))
                                    - abs(
                                        float(
                                            measured[
                                                primary_finger_index
                                            ]
                                        )
                                    )
                                )
                                for name in mimic_follower_joint_names
                            ),
                            default=0.0,
                        )
                        if primary_finger_index is not None
                        else None
                    )
                    if mimic_follower_error is not None:
                        peak_mimic_follower_error = max(
                            peak_mimic_follower_error,
                            mimic_follower_error,
                        )
                    trajectory.write(
                        json.dumps(
                            {
                                "frame": frame,
                                "motion_frame": motion_frame,
                                "joint_names": dof_names,
                                "controlled_joint_names": controlled_joint_names,
                                "joint_position_targets": targets,
                                "joint_position_targets_degrees": radians_to_degrees(targets),
                                "joint_positions": measured,
                                "joint_positions_degrees": radians_to_degrees(measured),
                                "end_effector_position": end_effector_position,
                                "control_frame_position": (
                                    np.asarray(
                                        control_frame_position,
                                        dtype=np.float32,
                                    ).tolist()
                                    if control_frame_position is not None
                                    else None
                                ),
                                "cartesian_motion_target": cartesian_target,
                                "cartesian_policy_target": cartesian_policy_target,
                                "task_phase": task_phase,
                                "objects": {
                                    "end_effector_marker": {
                                        "position": end_effector_position,
                                        "orientation": [0, 0, 0],
                                        "linear_velocity": [0, 0, 0],
                                        "angular_velocity": [0, 0, 0],
                                    },
                                    "pick_object": {
                                        "position": pick_object_position,
                                        "orientation": [0, 0, 0],
                                        "linear_velocity": [0, 0, 0],
                                        "angular_velocity": [0, 0, 0],
                                        "attached": grasp_attached,
                                        "contact": bilateral_contact
                                        if contact_only
                                        else task_phase
                                        in {"grasp", "lift", "transfer", "place"}
                                        and grasp_attached,
                                        "grasp_constraint": grasp_mode if grasp_created else None,
                                    },
                                    "real_grasp_body": {
                                        "position": grasp_body_position,
                                        "orientation": grasp_body_orientation,
                                        "linear_velocity": [0, 0, 0],
                                        "angular_velocity": [0, 0, 0],
                                    },
                                    "left_inner_finger": {
                                        "position": left_finger_position,
                                        "world_bounds": left_finger_bounds,
                                        "orientation": [0, 0, 0],
                                        "linear_velocity": [0, 0, 0],
                                        "angular_velocity": [0, 0, 0],
                                    },
                                    "right_inner_finger": {
                                        "position": right_finger_position,
                                        "world_bounds": right_finger_bounds,
                                        "orientation": [0, 0, 0],
                                        "linear_velocity": [0, 0, 0],
                                        "angular_velocity": [0, 0, 0],
                                    },
                                },
                                "gripper": {
                                    "joint_name": primary_finger_joint,
                                    "target": finger_target,
                                    "position": measured[primary_finger_index] if primary_finger_index is not None else None,
                                    "measured_effort": (
                                        primary_finger_effort
                                    ),
                                    "joint_limits": (
                                        joint_limits[
                                            primary_finger_index
                                        ]
                                        if joint_limits is not None
                                        and primary_finger_index is not None
                                        else None
                                    ),
                                    "mimic_follower_error": (
                                        mimic_follower_error
                                    ),
                                    "finger_joint_names": finger_joint_names,
                                    "actuated_joint_names": (
                                        actuated_finger_joint_names
                                    ),
                                    "mimic_follower_joint_names": (
                                        mimic_follower_joint_names
                                    ),
                                    "contact_forces_newtons": {
                                        "left_finger": left_contact_force,
                                        "right_finger": right_contact_force,
                                        "left_links": left_contact_link_forces,
                                        "right_links": right_contact_link_forces,
                                    },
                                    "raw_contact_pairs": (
                                        raw_contact_pairs[:20]
                                    ),
                                },
                                "control_pose_source": control_pose_source,
                                "object_pose_estimate": (
                                    latest_object_estimate["position"]
                                    if latest_object_estimate
                                    else None
                                ),
                                "target_pose_estimate": (
                                    latest_target_estimate["position"]
                                    if latest_target_estimate
                                    else None
                                ),
                            },
                            default=json_default,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                if (
                    should_capture_observation
                    and latest_rgb is not None
                    and latest_depth is not None
                ):
                    rgb_relative, depth_relative = write_observation_artifacts(
                        episode_dir,
                        dataset_observation_count,
                        latest_rgb,
                        latest_depth,
                    )
                    joint_velocities = np.asarray(
                        robot.get_joint_velocities(),
                        dtype=np.float32,
                    ).tolist()
                    observations.write(
                        json.dumps(
                            {
                                "frame": frame,
                                "timestamp_seconds": round(frame / 60.0, 6),
                                "phase": task_phase,
                                "rgb_path": rgb_relative,
                                "depth_path": depth_relative,
                                "joint_names": dof_names,
                                "joint_positions": measured,
                                "joint_velocities": joint_velocities,
                                "action_joint_positions": targets,
                                "contact_forces_newtons": {
                                    "left_finger": left_contact_force,
                                    "right_finger": right_contact_force,
                                },
                                "object_pose_estimate": (
                                    latest_object_estimate["position"]
                                    if latest_object_estimate
                                    else None
                                ),
                                "target_pose_estimate": (
                                    latest_target_estimate["position"]
                                    if latest_target_estimate
                                    else None
                                ),
                                "control_pose_source": control_pose_source,
                            },
                            default=json_default,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    labels.write(
                        json.dumps(
                            {
                                "frame": frame,
                                "object_pose_ground_truth": pick_object_position,
                                "target_pose_ground_truth": ground_truth_place_target_position,
                                "bilateral_contact": bilateral_contact,
                                "raw_bilateral_contact": (
                                    raw_bilateral_contact
                                ),
                                "task_success_label_only": True,
                            },
                            default=json_default,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    dataset_observation_count += 1
                simulation_stop = simulation_stop_reason(
                    frame,
                    nominal_frame_count=frame_count,
                    extension_frames=recovery_extension_frames,
                    release_complete_frame=soft_release_complete_frame,
                    required_settle_frames=(
                        required_release_settle_frames
                    ),
                    grasp_validated=bilateral_contact_validated,
                )
                if simulation_stop is not None:
                    append_phase(
                        phase_path,
                        "simulation_stopped",
                        frame=frame,
                        reason=simulation_stop,
                        frames_simulated=frames_simulated,
                        nominal_frame_count=frame_count,
                        recovery_extension_frames=(
                            recovery_extension_frames
                        ),
                    )
                    break

        if preview_enabled and not (perception_enabled or dataset_enabled):
            rep.orchestrator.wait_until_complete()
        if rgb_annotator is not None:
            rgb_annotator.detach([render_product])
        if depth_annotator is not None:
            depth_annotator.detach([render_product])
        preview_images = sorted(path.name for path in preview_dir.glob("*.png"))
        joint_motion = joint_ranges_degrees(joint_history, dof_names)
        max_joint_motion = max(joint_motion.values()) if joint_motion else 0.0
        gripper_motion_degrees = joint_motion.get(primary_finger_joint, 0.0) if primary_finger_joint else 0.0
        final_positions = joint_history[-1] if joint_history else base_positions
        object_positions = [row["position"] for row in object_history if row.get("position")]
        attached_frames = [row for row in object_history if row.get("attached")]
        object_start_height = object_positions[0][2] if object_positions else None
        object_max_height = max((position[2] for position in object_positions), default=None)
        object_lift_height = (
            object_max_height - object_start_height
            if object_start_height is not None and object_max_height is not None
            else 0.0
        )
        final_pick_object_position = object_positions[-1] if object_positions else None
        final_object_row = object_history[-1] if object_history else {}
        final_grasp_body_position = final_object_row.get("grasp_body_position")
        final_object_attached = bool(final_object_row.get("attached"))
        final_gripper_object_distance = (
            distance(final_pick_object_position, final_grasp_body_position)
            if final_pick_object_position and final_grasp_body_position
            else None
        )
        target_zone_half_extents = [
            float(target_zone_config["scale"][0]) * float(target_zone_config["size"]) * 0.5,
            float(target_zone_config["scale"][1]) * float(target_zone_config["size"]) * 0.5,
        ]
        final_target_xy_distance = (
            math.hypot(
                float(final_pick_object_position[0])
                - float(ground_truth_place_target_position[0]),
                float(final_pick_object_position[1])
                - float(ground_truth_place_target_position[1]),
            )
            if final_pick_object_position
            else None
        )
        final_target_distance = (
            distance(
                final_pick_object_position,
                ground_truth_place_target_position,
            )
            if final_pick_object_position
            else None
        )
        final_object_inside_target_zone = bool(
            final_pick_object_position
            and abs(
                float(final_pick_object_position[0])
                - float(ground_truth_place_target_position[0])
            )
            <= target_zone_half_extents[0] - float(pick_object_config["scale"][0]) * float(pick_object_config["size"]) * 0.5
            and abs(
                float(final_pick_object_position[1])
                - float(ground_truth_place_target_position[1])
            )
            <= target_zone_half_extents[1] - float(pick_object_config["scale"][1]) * float(pick_object_config["size"]) * 0.5
        )
        release_monitor_start_frame = (
            soft_release_start_frame
            if soft_landing_enabled
            and soft_release_start_frame is not None
            else actual_release_frame
        )
        released_rows = [
            row
            for row in object_history
            if release_monitor_start_frame is not None
            and int(row.get("frame", -1))
            >= release_monitor_start_frame
        ]
        actual_released_rows = [
            row
            for row in object_history
            if actual_release_frame is not None
            and int(row.get("frame", -1)) >= actual_release_frame
            and not row.get("attached")
        ]
        release_settle_frames = len(actual_released_rows)
        stability_window_frames = max(
            1,
            int(
                task["success_criteria"].get(
                    "min_release_settle_frames",
                    45,
                )
            ),
        )
        stability_window = actual_released_rows[
            -stability_window_frames:
        ]
        stability_positions = [row["position"] for row in stability_window if row.get("position")]
        release_motion_positions = [
            row["position"]
            for row in released_rows
            if row.get("position")
        ]
        post_release_motion = (
            max(
                distance(position, release_motion_positions[0])
                for position in release_motion_positions
            )
            if release_motion_positions
            else None
        )
        post_release_settling_motion = (
            max(
                distance(position, stability_positions[-1])
                for position in stability_positions
            )
            if stability_positions
            else None
        )
        dataset_validation = (
            validate_episode_dataset(episode_dir)
            if dataset_enabled
            else {
                "valid": False,
                "observation_count": 0,
                "label_count": 0,
                "errors": ["dataset recording is disabled"],
            }
        )
        initial_object_perception_xy_error = (
            xy_error(
                initial_object_estimate["position"],
                ground_truth_pick_start_position,
            )
            if initial_object_estimate
            else None
        )
        initial_target_perception_xy_error = (
            xy_error(
                initial_target_estimate["position"],
                ground_truth_place_target_position,
            )
            if initial_target_estimate
            else None
        )

        metrics = {
            "success": False,
            "frames_requested": frame_count,
            "frames_simulated": frames_simulated,
            "recovery_extension_frames": recovery_extension_frames,
            "simulation_stop_reason": simulation_stop,
            "record_every_n_frames": record_every,
            "control_frequency_hz": round(1.0 / rendering_dt, 3),
            "physics_frequency_hz": round(1.0 / physics_dt, 3),
            "physics_substeps_per_control_frame": (
                physics_substeps_per_frame
            ),
            "recorded_frames": read_recorded_frames(trajectory_path),
            "preview_frames_requested": sorted(preview_frames),
            "preview_images_written": len(preview_images),
            "preview_images": preview_images,
            "elapsed_seconds": (utc_now() - started_at).total_seconds(),
            "object_count": 3,
            "final_object_positions": {
                "end_effector_marker": end_effector_history[-1] if end_effector_history else None,
                "pick_object": final_pick_object_position,
                "real_grasp_body": prim_world_position(stage, grasp_body_path),
            },
            "task_type": (
                "perception_driven_contact_pick_and_place_v3"
                if contact_only and perception_enabled
                else (
                    "randomized_ur10e_robotiq_pick_and_place_v2"
                    if randomization["enabled"]
                    else "real_ur10e_robotiq_pick_and_place_v1"
                )
            ),
            "pickup_mode": f"real_robotiq_2f85_{grasp_mode}_pick_and_place",
            "arm_motion_mode": arm_motion_mode,
            "end_effector_orientation_target_degrees": (
                end_effector_orientation_degrees
            ),
            "rmpflow_config_root": rmpflow_config_root,
            "lula_ik_config_root": lula_ik_config_root,
            "lula_ik_all_phases": lula_ik_all_phases,
            "lula_ik_successes": lula_ik_successes,
            "lula_ik_failures": lula_ik_failures,
            "rmpflow_target_coordinate_frame": (
                "usd_stage_world"
                if rmpflow_controller is not None
                else None
            ),
            "rmpflow_base_pose_configured_after_reset": (
                rmpflow_controller is not None
            ),
            "arm_drive_records": arm_drive_records,
            "arm_drive_configured_count": sum(1 for row in arm_drive_records if row["configured"]),
            "task_phases": phase_names_seen,
            "control_pose_source": control_pose_source,
            "pick_object_start_position": pick_start_position,
            "ground_truth_pick_object_start_position": ground_truth_pick_start_position,
            "pick_object_lift_target_position": pick_lift_position,
            "place_target_position": place_target_position,
            "ground_truth_place_target_position": ground_truth_place_target_position,
            "place_grasp_position": place_grasp_position,
            "nominal_place_grasp_position": nominal_place_grasp_position,
            "place_servo_frames": place_servo_frames,
            "place_servo_updates": place_servo_updates,
            "place_servo_correction_xy": [
                round(value, 6) for value in place_servo_correction
            ],
            "place_servo_last_error_xy": [
                round(value, 6) for value in place_servo_last_error
            ]
            if place_servo_last_error
            else None,
            "place_servo_last_error_xyz": [
                round(value, 6)
                for value in place_servo_last_error_xyz
            ]
            if place_servo_last_error_xyz
            else None,
            "place_control_pose_source": place_control_pose_source,
            "place_control_object_estimate": [
                round(value, 6)
                for value in place_control_object_estimate
            ]
            if place_control_object_estimate
            else None,
            "place_kinematic_estimate_start_frame": (
                place_kinematic_estimate_start_frame
            ),
            "place_release_gate_start_frame": (
                place_release_gate_start_frame
            ),
            "place_descent_gate_start_frame": (
                place_descent_gate_start_frame
            ),
            "place_descent_gate_open_frame": (
                place_descent_gate_open_frame
            ),
            "place_descent_gate_converged_updates": (
                place_descent_gate_converged_updates
            ),
            "place_release_gate_open_frame": (
                place_release_gate_open_frame
            ),
            "place_release_gate_timed_out": (
                place_release_gate_timed_out
            ),
            "place_release_converged_updates": (
                place_release_converged_updates
            ),
            "place_release_grasp_tracking_error": (
                round(place_release_grasp_tracking_error, 6)
                if place_release_grasp_tracking_error is not None
                else None
            ),
            "place_require_grasp_tracking": (
                place_require_grasp_tracking
            ),
            "place_hover_guard_clearance": (
                place_hover_guard_clearance
            ),
            "place_hover_guard_activations": (
                place_hover_guard_activations
            ),
            "place_hover_guard_active_at_end": (
                place_hover_guard_active
            ),
            "soft_landing_enabled": soft_landing_enabled,
            "soft_landing_state": soft_landing_state,
            "soft_landing_start_frame": soft_landing_start_frame,
            "soft_landing_support_frame": (
                soft_landing_support_frame
            ),
            "soft_landing_support_reason": (
                soft_landing_support_reason
            ),
            "soft_landing_object_estimate": (
                [
                    round(value, 6)
                    for value in soft_landing_object_estimate
                ]
                if soft_landing_object_estimate
                else None
            ),
            "soft_landing_command_position": (
                [
                    round(value, 6)
                    for value in soft_landing_command_position
                ]
                if soft_landing_command_position
                else None
            ),
            "soft_landing_clearance_at_support": (
                round(soft_landing_clearance_at_support, 6)
                if soft_landing_clearance_at_support is not None
                else None
            ),
            "soft_landing_peak_force_at_support": (
                round(soft_landing_peak_force_at_support, 6)
                if soft_landing_peak_force_at_support is not None
                else None
            ),
            "soft_landing_force_baseline": (
                round(soft_landing_force_baseline, 6)
                if soft_landing_force_baseline is not None
                else None
            ),
            "soft_release_start_frame": soft_release_start_frame,
            "soft_release_complete_frame": (
                soft_release_complete_frame
            ),
            "gripper_preload_reference_position": (
                gripper_preload_reference_position
            ),
            "transport_recenter_offset_xy": [
                round(value, 6)
                for value in transport_recenter_offset_xy
            ],
            "transport_recenter_updates": (
                transport_recenter_updates
            ),
            "transport_recenter_last_side": (
                transport_recenter_last_side
            ),
            "target_zone_half_extents": target_zone_half_extents,
            "final_pick_object_position": final_pick_object_position,
            "final_object_attached": final_object_attached,
            "final_object_inside_target_zone": final_object_inside_target_zone,
            "final_object_height": round(float(final_pick_object_position[2]), 4) if final_pick_object_position else None,
            "final_target_xy_distance": round(final_target_xy_distance, 5)
            if final_target_xy_distance is not None
            else None,
            "final_target_distance": round(final_target_distance, 5) if final_target_distance is not None else None,
            "release_settle_frames": release_settle_frames,
            "post_release_stability_window_frames": len(stability_positions),
            "post_release_motion": round(post_release_motion, 5) if post_release_motion is not None else None,
            "post_release_settling_motion": (
                round(post_release_settling_motion, 5)
                if post_release_settling_motion is not None
                else None
            ),
            "final_gripper_object_distance": round(final_gripper_object_distance, 5)
            if final_gripper_object_distance is not None
            else None,
            "object_lift_height": round(object_lift_height, 4),
            "object_attached_frames": len(attached_frames),
            "grasp_contact_frames": sum(1 for row in object_history if row.get("contact")),
            "grasp_constraint": grasp_mode,
            "grasp_attach_method": (
                "physical_bilateral_finger_contact"
                if contact_only
                else f"real_robotiq_{grasp_mode}"
            ),
            "grasp_joint_schema": None
            if contact_only
            else "UsdPhysics.Joint_locked_6dof",
            "grasp_joint_path": None if contact_only else grasp_joint_path,
            "temporary_grasp_joint_created": temporary_grasp_joint_created,
            "grasp_attach_frame": attach_frame,
            "grasp_retry_frames": grasp_retry_frames,
            "gripper_close_start_frame": gripper_close_start_frame,
            "gripper_close_end_frame": gripper_close_end_frame,
            "grasp_alignment_start_frame": (
                grasp_alignment_start_frame
            ),
            "grasp_approach_gate_open_frame": (
                grasp_approach_gate_open_frame
            ),
            "grasp_close_control_height": (
                round(grasp_close_control_height, 6)
                if grasp_close_control_height is not None
                else None
            ),
            "grasp_xy_gate_open_frame": grasp_xy_gate_open_frame,
            "freeze_visual_target_after_xy_gate": (
                freeze_visual_target_after_xy_gate
            ),
            "grasp_visual_lock_source": grasp_visual_lock_source,
            "grasp_visual_handoff_frame": grasp_visual_handoff_frame,
            "grasp_visual_handoff_position": (
                [
                    round(float(value), 6)
                    for value in grasp_visual_handoff_position
                ]
                if grasp_visual_handoff_position is not None
                else None
            ),
            "grasp_visual_post_handoff_max_xy_drift": round(
                grasp_visual_post_handoff_max_xy_drift,
                6,
            ),
            "grasp_xy_max_aligned_frames": (
                grasp_xy_max_aligned_frames
            ),
            "grasp_hover_clearance": grasp_hover_clearance,
            "grasp_height_offset_meters": grasp_height_offset,
            "grasp_target_height": grasp_target_height,
            "grasp_descent_frames": grasp_descent_frames,
            "grasp_descent_step_meters": grasp_descent_step,
            "aperture_tcp_calibration_frame": (
                aperture_tcp_calibration_frame
            ),
            "grasp_approach_converged_frames": (
                grasp_approach_max_converged_frames
            ),
            "grasp_tracking_stable_frames_required": (
                grasp_tracking_stable_frames
            ),
            "grasp_alignment_mode": grasp_alignment_mode,
            "aperture_tool_offset_world": (
                [
                    round(float(value), 6)
                    for value in aperture_tool_offset_world
                ]
                if aperture_tool_offset_world is not None
                else None
            ),
            "grasp_aperture_center": (
                [
                    round(float(value), 6)
                    for value in grasp_aperture_center
                ]
                if grasp_aperture_center is not None
                else None
            ),
            "grasp_aperture_finger_z_skew": (
                round(grasp_aperture_finger_z_skew, 6)
                if grasp_aperture_finger_z_skew is not None
                else None
            ),
            "grasp_aperture_axis_yaw_degrees": (
                round(grasp_aperture_axis_yaw_degrees, 3)
                if grasp_aperture_axis_yaw_degrees is not None
                else None
            ),
            "grasp_aperture_axis_yaw_error_degrees": (
                round(grasp_aperture_axis_yaw_error_degrees, 3)
                if grasp_aperture_axis_yaw_error_degrees is not None
                else None
            ),
            "grasp_aperture_axis_yaw_delta_degrees": (
                round(grasp_aperture_axis_yaw_delta_degrees, 4)
                if grasp_aperture_axis_yaw_delta_degrees is not None
                else None
            ),
            "grasp_axis_target_yaw_degrees": (
                grasp_axis_target_yaw_degrees
            ),
            "grasp_axis_max_error_degrees": (
                grasp_axis_max_error_degrees
            ),
            "grasp_axis_max_delta_degrees_per_frame": (
                grasp_axis_max_delta_degrees
                if math.isfinite(grasp_axis_max_delta_degrees)
                else None
            ),
            "grasp_descent_max_xy_error": (
                grasp_descent_max_xy_error
            ),
            "grasp_descent_max_axis_yaw_error_degrees": (
                grasp_descent_max_axis_yaw_error
            ),
            "grasp_descent_interlock_activations": (
                grasp_descent_interlock_activations
            ),
            "grasp_descent_interlock_paused_frames": (
                grasp_descent_interlock_paused_frames
            ),
            "grasp_tracking_max_finger_z_skew": (
                grasp_tracking_max_finger_z_skew
                if math.isfinite(grasp_tracking_max_finger_z_skew)
                else None
            ),
            "grasp_aperture_bias_xy": grasp_aperture_bias_xy,
            "grasp_tracking_servo_updates": (
                grasp_tracking_servo_updates
            ),
            "grasp_tracking_error_xyz": (
                [
                    round(float(value), 6)
                    for value in grasp_tracking_error_xyz
                ]
                if grasp_tracking_error_xyz is not None
                else None
            ),
            "grasp_tracking_xy_error": (
                round(grasp_tracking_xy_error, 6)
                if grasp_tracking_xy_error is not None
                else None
            ),
            "grasp_approach_max_xy_error": (
                grasp_approach_max_xy_error
            ),
            "grasp_tracking_z_error": (
                round(grasp_tracking_z_error, 6)
                if grasp_tracking_z_error is not None
                else None
            ),
            "grasp_tracking_servo_correction": [
                round(
                    float(pick_grasp_position[axis])
                    - float(nominal_pick_grasp_position[axis]),
                    6,
                )
                for axis in range(3)
            ],
            "unilateral_recenter_enabled": unilateral_recenter_enabled,
            "unilateral_recenter_updates": unilateral_recenter_updates,
            "unilateral_recenter_last_side": unilateral_recenter_last_side,
            "unilateral_recenter_persistence_remaining": (
                unilateral_recenter_persistence_remaining
            ),
            "unilateral_recenter_target_correction": [
                round(
                    float(pick_grasp_position[axis])
                    - float(nominal_pick_grasp_position[axis]),
                    6,
                )
                for axis in range(2)
            ],
            "unilateral_force_limit_newtons": unilateral_force_limit,
            "unilateral_force_backoff_updates": (
                unilateral_force_backoff_updates
            ),
            "bilateral_force_limit_newtons": (
                bilateral_force_limit
                if math.isfinite(bilateral_force_limit)
                else None
            ),
            "bilateral_force_backoff_updates": (
                bilateral_force_backoff_updates
            ),
            "bilateral_hold_close_updates": (
                bilateral_hold_close_updates
            ),
            "transport_hold_max_close_adjustment_radians": max(
                0.0,
                float(
                    pickup_config.get(
                        "transport_hold_max_close_adjustment_radians",
                        0.0,
                    )
                ),
            ),
            "gripper_max_position_error_radians": (
                gripper_max_position_error
                if math.isfinite(gripper_max_position_error)
                else None
            ),
            "gripper_target_envelope_updates": (
                gripper_target_envelope_updates
            ),
            "finger_close_drive_configured": (
                finger_close_drive_configured
            ),
            "finger_close_stiffness": finger_close_stiffness,
            "finger_close_damping": finger_close_damping,
            "finger_contact_effort_configured": (
                finger_contact_effort_configured
            ),
            "finger_contact_max_effort": (
                finger_contact_max_effort
            ),
            "post_grasp_target_return_updates": (
                post_grasp_target_return_updates
            ),
            "pickup_visual_tracking_updates": (
                pickup_visual_tracking_updates
            ),
            "pickup_visual_tracking_last_observed_nominal": (
                pickup_visual_tracking_last_observed_nominal
            ),
            "grasp_close_duration_frames": (
                grasp_close_duration_frames
            ),
            "gripper_contact_hold_frame": contact_hold_frame,
            "gripper_contact_hold_position": contact_hold_position,
            "bilateral_contact_latched": bilateral_contact_latched,
            "bilateral_contact_validated": (
                bilateral_contact_validated
            ),
            "grasp_validation_support_frames": (
                grasp_validation_support_frames
            ),
            "grasp_validation_terminal_stable_frames": (
                grasp_validation_terminal_stable_frames
            ),
            "grasp_validation_max_terminal_stable_frames": (
                grasp_validation_max_terminal_stable_frames
            ),
            "grasp_validation_required_terminal_stable_frames": (
                grasp_validation_required_terminal_frames
            ),
            "grasp_validation_failures": grasp_validation_failures,
            "grasp_validation_max_frames": (
                grasp_validation_max_frames
            ),
            "grasp_validated_lift_anchor": (
                grasp_validated_lift_anchor
            ),
            "grasp_validated_orientation": (
                grasp_validated_orientation
            ),
            "grasp_validated_finger_position": (
                grasp_validated_finger_position
            ),
            "grasp_validated_finger_target": (
                grasp_validated_finger_target
            ),
            "grasp_attempt_count": grasp_attempt_count,
            "grasp_success_frame": grasp_success_frame,
            "grasp_release_frame": release_frame,
            "actual_grasp_release_frame": actual_release_frame,
            "grasp_motion_start_frame": grasp_motion_start_frame,
            "grasp_validation_frames": grasp_validation_frames,
            "grasp_proof_enabled": grasp_proof_enabled,
            "grasp_proof_state": grasp_proof_state,
            "grasp_proof_successes": grasp_proof_successes,
            "grasp_proof_failures": grasp_proof_failures,
            "grasp_proof_complete_frame": grasp_proof_complete_frame,
            "grasp_proof_lift_height_meters": (
                grasp_proof_lift_height
            ),
            "grasp_proof_hold_frames": grasp_proof_hold_frames,
            "grasp_proof_hold_max_frames": grasp_proof_hold_max_frames,
            "grasp_proof_terminal_contact_frames": (
                grasp_proof_terminal_contact_frames
            ),
            "grasp_proof_min_force_newtons": grasp_proof_min_force,
            "grasp_proof_contact_rebuild_wait_frame": (
                grasp_proof_contact_rebuild_wait_frame
            ),
            "grasp_proof_max_object_lift_meters": (
                round(grasp_proof_max_object_lift, 6)
            ),
            "grasp_proof_max_contact_streak": (
                grasp_proof_max_contact_streak
            ),
            "grasp_proof_max_rigidity_error_meters": (
                round(grasp_proof_max_rigidity_error, 6)
            ),
            "grasp_recovery_attempts": grasp_recovery_attempts,
            "grasp_recovery_final_state": grasp_recovery_state,
            "grasp_contact_loss_events": grasp_contact_loss_events,
            "max_grasp_contact_loss_streak": max_contact_loss_streak,
            "physical_grasp_contact_offset_world": (
                physical_grasp_contact_offset_world
            ),
            "grasp_was_created": grasp_was_created,
            "grasp_attach_distance": round(grasp_attach_distance, 5)
            if grasp_attach_distance is not None
            else None,
            "real_grasp_body_path": grasp_body_path,
            "real_grasp_body_found": real_grasp_body_found,
            "contact_sensor_paths": contact_sensor_paths,
            "left_contact_sensor_valid_frames": left_contact_sensor_valid_frames,
            "right_contact_sensor_valid_frames": right_contact_sensor_valid_frames,
            "contact_material_bound_paths": contact_material_bound_paths,
            "bilateral_contact_frames": bilateral_contact_frames,
            "raw_bilateral_contact_frames": raw_bilateral_contact_frames,
            "raw_bilateral_contact_streak_max": (
                max_raw_bilateral_contact_streak
            ),
            "max_recent_raw_bilateral_events": (
                max_recent_raw_bilateral_events
            ),
            "bilateral_contact_confirmation_frames": (
                max(
                    1,
                    int(
                        pickup_config.get(
                            "bilateral_contact_confirmation_frames",
                            3,
                        )
                    ),
                )
            ),
            "bilateral_contact_confirmation_window_frames": (
                max(
                    1,
                    int(
                        pickup_config.get(
                            "bilateral_contact_confirmation_window_frames",
                            12,
                        )
                    ),
                )
            ),
            "debounced_bilateral_contact_frames": (
                debounced_bilateral_contact_frames
            ),
            "transport_contact_frames": transport_contact_frames,
            "direct_transport_contact_frames": (
                direct_transport_contact_frames
            ),
            "inferred_transport_support_frames": (
                inferred_transport_support_frames
            ),
            "max_transport_rigidity_error": round(
                max_transport_rigidity_error,
                6,
            ),
            "max_continuous_transport_contact_frames": (
                max_continuous_transport_contact_frames
            ),
            "peak_left_contact_force_newtons": round(
                max(
                    (
                        row["left_finger"]
                        for row in contact_force_history
                    ),
                    default=0.0,
                ),
                5,
            ),
            "peak_right_contact_force_newtons": round(
                max(
                    (
                        row["right_finger"]
                        for row in contact_force_history
                    ),
                    default=0.0,
                ),
                5,
            ),
            "rgbd_perception_succeeded": bool(
                initial_object_estimate and initial_target_estimate
            ),
            "initial_object_pose_estimate": (
                initial_object_estimate["position"]
                if initial_object_estimate
                else None
            ),
            "initial_target_pose_estimate": (
                initial_target_estimate["position"]
                if initial_target_estimate
                else None
            ),
            "initial_object_perception_xy_error": round(
                initial_object_perception_xy_error,
                6,
            )
            if initial_object_perception_xy_error is not None
            else None,
            "initial_target_perception_xy_error": round(
                initial_target_perception_xy_error,
                6,
            )
            if initial_target_perception_xy_error is not None
            else None,
            "perception_updates": perception_updates,
            "perception_failures": perception_failures,
            "dataset_format": dataset_config.get("format"),
            "dataset_valid": dataset_validation["valid"],
            "dataset_observation_count": dataset_validation[
                "observation_count"
            ],
            "dataset_label_count": dataset_validation["label_count"],
            "dataset_validation_errors": dataset_validation["errors"],
            "max_attached_gripper_object_distance": round(max(attached_distance_history), 5)
            if attached_distance_history
            else None,
            "max_grasp_rigidity_error": round(max(attached_distance_history), 5)
            if attached_distance_history
            else None,
            "mean_attached_gripper_object_distance": round(
                sum(attached_distance_history) / len(attached_distance_history), 5
            )
            if attached_distance_history
            else None,
            "robot_name": robot_config["name"],
            "robot_world_position": np.asarray(robot_world_position, dtype=np.float32).tolist(),
            "robot_world_orientation": np.asarray(robot_world_orientation, dtype=np.float32).tolist(),
            "robot_asset_path": robot_config["asset_path"],
            "robot_prim_path": robot_config["prim_path"],
            "robot_loaded": robot_prim_count > 0,
            "robot_prim_count": robot_prim_count,
            "selected_gripper_variant": selected_gripper,
            "available_gripper_variants": gripper_variants,
            "selected_physics_variant": selected_physics,
            "available_physics_variants": physics_variants,
            "robotiq_prim_count": len(robotiq_paths),
            "robotiq_prim_paths_sample": robotiq_paths[:40],
            "mimic_joint_prim_count": len(mimic_joint_paths),
            "mimic_joint_paths_sample": mimic_joint_paths[:40],
            "articulation_controller": True,
            "articulation_controller_initialized": articulation_controller_initialized,
            "articulation_dofs": robot.num_dof,
            "joint_names": dof_names,
            "controlled_joint_names": controlled_joint_names,
            "controlled_joints": len(controlled_joint_names),
            "finger_joint_names": finger_joint_names,
            "primary_finger_joint": primary_finger_joint,
            "primary_finger_joint_index": primary_finger_index,
            "actuated_finger_joint_names": (
                actuated_finger_joint_names
            ),
            "mimic_follower_joint_names": (
                mimic_follower_joint_names
            ),
            "primary_finger_joint_limits": (
                joint_limits[primary_finger_index]
                if joint_limits is not None
                and primary_finger_index is not None
                else None
            ),
            "peak_primary_finger_effort": round(
                peak_primary_finger_effort,
                6,
            ),
            "peak_mimic_follower_error_radians": round(
                peak_mimic_follower_error,
                6,
            ),
            "raw_contact_pair_peak_forces": [
                {
                    "body0": key[0],
                    "body1": key[1],
                    "peak_force_newtons": round(force, 6),
                }
                for key, force in sorted(
                    raw_contact_pair_peak_forces.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:30]
            ],
            "gripper_motion_degrees": gripper_motion_degrees,
            "joint_motion_degrees": joint_motion,
            "max_joint_motion_degrees": max_joint_motion,
            "debug_end_effector_prim_path": end_effector_prim_path,
            "initial_joint_positions": base_positions,
            "initial_joint_positions_degrees": radians_to_degrees(base_positions),
            "final_joint_positions": final_positions,
            "final_joint_positions_degrees": radians_to_degrees(final_positions),
            "final_end_effector_position": end_effector_history[-1] if end_effector_history else None,
        }
        success, checks = evaluate_success(task, metrics)
        metrics["success"] = success
        metrics["success_checks"] = checks
        metrics.update(classify_failure(checks))
        metrics["episode_seed"] = episode_seed
        metrics["variation_id"] = variation_id
        metrics["variation"] = variation
        metrics["trial_id"] = trial_id
        metrics["position_plan_id"] = position_trial["plan_id"] if position_trial else None
        metrics["position_plan_sha256"] = (
            position_trial["plan_sha256"] if position_trial else None
        )
        metrics["reserve_index"] = reserve_index if position_trial else None
        metrics["benchmark_id"] = benchmark_id
        metrics["benchmark_repeat"] = benchmark_repeat
        metrics["randomization"] = randomization

        joint_smoothness = measured_joint_smoothness(joint_history)
        metrics["joint_smoothness_score"] = (
            round(joint_smoothness, 8) if joint_smoothness is not None else None
        )

        metadata = {
            "episode_id": episode_id,
            "run_id": os.environ.get("FARPOINT_RUN_ID"),
            "episode_seed": episode_seed,
            "variation_id": variation_id,
            "variation": variation,
            "trial_id": trial_id,
            "split": position_trial["split"] if position_trial else None,
            "position_plan_id": position_trial["plan_id"] if position_trial else None,
            "position_plan_sha256": (
                position_trial["plan_sha256"] if position_trial else None
            ),
            "reserve_index": reserve_index if position_trial else None,
            "benchmark_id": benchmark_id,
            "benchmark_repeat": benchmark_repeat,
            "randomization": randomization,
            "task_schema_version": task["schema_version"],
            "task_name": task["name"],
            "language_instruction": task["language_instruction"],
            "success_criteria": task["success_criteria"],
            "simulator": "Isaac Sim",
            "image": "nvcr.io/nvidia/isaac-sim:6.0.0",
            "python": sys.version,
            "platform": platform.platform(),
            "started_at": started_at.isoformat(),
            "finished_at": utc_now().isoformat(),
            "preview_enabled": preview_enabled,
            "preview_resolution": camera_config.get("resolution"),
        }
        if position_trial:
            git_commit = os.environ.get("FARPOINT_GIT_COMMIT", "")
            image_name = os.environ.get(
                "FARPOINT_SIMULATOR_IMAGE",
                "nvcr.io/nvidia/isaac-sim:6.0.0",
            )
            image_digest = os.environ.get("FARPOINT_SIMULATOR_IMAGE_DIGEST", "")
            configured_sha = os.environ.get("FARPOINT_CONFIG_SHA256", "")
            if not (
                len(git_commit) == 40
                and all(character in "0123456789abcdef" for character in git_commit)
            ):
                raise ValueError("FARPOINT_GIT_COMMIT must be a full lowercase Git SHA")
            if not (
                image_digest.startswith("sha256:")
                and len(image_digest) == len("sha256:") + 64
            ):
                raise ValueError("FARPOINT_SIMULATOR_IMAGE_DIGEST must be an immutable digest")
            if configured_sha != position_plan["config_sha256"]:
                raise ValueError("FARPOINT_CONFIG_SHA256 does not match the position plan")
            if camera_intrinsics is None or camera_to_world is None:
                raise ValueError("release metadata requires measured camera calibration")
            frozen = position_plan["frozen_factors"]
            camera_rotation = np.asarray(camera_to_world, dtype=np.float64)[:3, :3]
            camera_position = np.asarray(camera_to_world, dtype=np.float64)[:3, 3].tolist()
            camera_quaternion = rotation_matrix_quaternion_xyzw(camera_rotation)
            camera_yaw = math.degrees(
                math.atan2(float(camera_rotation[1, 0]), float(camera_rotation[0, 0]))
            )
            calibration_payload = {
                "intrinsics": np.asarray(camera_intrinsics, dtype=np.float64).tolist(),
                "extrinsics": np.asarray(camera_to_world, dtype=np.float64).tolist(),
                "resolution": camera_config["resolution"],
            }
            object_position = list(ground_truth_pick_start_position)
            target_position = list(ground_truth_place_target_position)
            identity_orientation = [0.0, 0.0, 0.0, 1.0]
            recording_fps = (1.0 / rendering_dt) / int(
                dataset_config["observation_every_n_frames"]
            )
            metadata.update(
                {
                    "provenance": {
                        "git_commit": git_commit,
                        "config_sha256": position_plan["config_sha256"],
                        "simulator": "Isaac Sim 6.0.0",
                        "physics_engine": "PhysX",
                        "simulator_image": image_name,
                        "simulator_image_digest": image_digest,
                        "robot_asset_id": "ur10e_robotiq_2f85_usd_v1",
                        "robot_asset_path": robot_config["asset_path"],
                        "episode_seed": episode_seed,
                        "derived_seed": position_trial["seed"],
                    },
                    "task": {
                        "task_id": task["name"],
                        "instruction": task["language_instruction"],
                        "object_shape": frozen["object_shape"],
                        "success_criteria_id": "contact_pick_place_v1",
                    },
                    "embodiment": {
                        "robot": robot_config["name"],
                        "gripper": "robotiq_2f85",
                        "arm_dof": len(ARM_JOINT_NAMES),
                        "gripper_dof": max(1, robot.num_dof - len(ARM_JOINT_NAMES)),
                        "controller": frozen["controller_profile_id"],
                        "control_mode": "articulation_position_drive",
                        "grasp_mode": pickup_config["grasp_mode"],
                    },
                    "scene": {
                        "coordinate_frame": "world",
                        "object": {
                            "shape": frozen["object_shape"],
                            "dimensions_m": frozen["object_dimensions_m"],
                            "mass_kg": float(pick_object_config["mass"]),
                            "material_id": frozen["material_id"],
                            "initial_pose": {
                                "position_m": object_position,
                                "orientation_xyzw": identity_orientation,
                                "yaw_degrees": frozen["object_yaw_degrees"],
                                "coordinate_frame": "world",
                            },
                        },
                        "target": {
                            "target_id": target_zone_config["path"],
                            "pose": {
                                "position_m": target_position,
                                "orientation_xyzw": identity_orientation,
                                "yaw_degrees": 0.0,
                                "coordinate_frame": "world",
                            },
                        },
                        "camera": {
                            "profile_id": frozen["camera_profile_id"],
                            "calibration_id": "front_rgbd_" + sha256_json(calibration_payload)[:16],
                            "intrinsics": {
                                "fx": float(camera_intrinsics[0, 0]),
                                "fy": float(camera_intrinsics[1, 1]),
                                "cx": float(camera_intrinsics[0, 2]),
                                "cy": float(camera_intrinsics[1, 2]),
                            },
                            "extrinsics": {
                                "position_m": camera_position,
                                "orientation_xyzw": camera_quaternion,
                                "yaw_degrees": camera_yaw,
                                "coordinate_frame": "world",
                            },
                        },
                        "lighting_profile_id": frozen["lighting_profile_id"],
                        "appearance_profile_id": frozen["appearance_profile_id"],
                    },
                    "recording": {
                        "fps": recording_fps,
                        "cameras": ["observation.images.front"],
                        "image_width": int(camera_config["resolution"][0]),
                        "image_height": int(camera_config["resolution"][1]),
                        "frame_count": int(dataset_validation["observation_count"]),
                    },
                    "outcome": {
                        "success": success,
                        "dataset_valid": bool(dataset_validation["valid"]),
                        "failure_category": metrics.get("failure_category"),
                        "failure_reason": metrics.get("failure_reason"),
                        "quality": {
                            "final_xy_error_m": metrics.get("final_target_xy_distance"),
                            "perception_error_m": metrics.get(
                                "initial_object_perception_xy_error"
                            ),
                            "bilateral_contact_frames": metrics.get(
                                "bilateral_contact_frames"
                            ),
                            "lift_height_m": metrics.get("object_lift_height"),
                            "settling_error": metrics.get(
                                "post_release_settling_motion"
                            ),
                            "joint_smoothness_score": metrics.get(
                                "joint_smoothness_score"
                            ),
                        },
                    },
                }
            )
            try:
                validate_simulator_metadata_v2(metadata, metrics)
            except ValueError as error:
                success = False
                metrics["success"] = False
                metrics["failure_category"] = "evaluation"
                metrics["failure_reason"] = f"episode_metadata_v2_invalid:{error}"
                metrics.setdefault("failed_checks", []).append("episode_metadata_v2")
                metadata["outcome"].update(
                    {
                        "success": False,
                        "failure_category": "evaluation",
                        "failure_reason": metrics["failure_reason"],
                    }
                )
        write_json(episode_dir / "metadata.json", metadata)
        write_json(episode_dir / "metrics.json", metrics)
        append_phase(phase_path, "episode_written", recorded_frames=metrics["recorded_frames"], success=success)
        print(f"episode written: {episode_dir}", flush=True)
        print(
            f"SMOKE_TEST_RESULT: {'PASS' if success else 'FAIL'} {episode_id} "
            f"real UR10e Robotiq articulation with {robot.num_dof} DOFs",
            flush=True,
        )
        return 0 if success else 1
    except Exception as exc:
        episode_dir.mkdir(parents=True, exist_ok=True)
        append_phase(phase_path, "scene_script_error", error_type=type(exc).__name__, error=str(exc))
        write_json(episode_dir / "metrics.json", {"success": False, "error_type": type(exc).__name__, "error": str(exc)})
        print(f"SMOKE_TEST_RESULT: FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        if simulation_app is not None:
            append_phase(phase_path, "simulation_app_close_start")
            simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    sys.exit(main())
