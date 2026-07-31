import ast
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
TASK_PATH = PROJECT_ROOT / "examples" / "isaac_robot_arm_scene" / "task.yaml"
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
DEBUG_COLORS = {
    "x": [1.0, 0.08, 0.04],
    "y": [0.08, 0.70, 0.18],
    "z": [0.10, 0.32, 1.0],
    "marker": [0.00, 0.85, 0.95],
    "target": [1.0, 0.72, 0.05],
    "gripper": [0.08, 0.36, 0.95],
    "gripper_tip": [0.02, 0.16, 0.42],
}


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

        try:
            current[key] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            current[key] = value

    return data


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_key(data, key, path):
    if key not in data:
        raise ValueError(f"missing required task field: {path}.{key}")
    return data[key]


def require_vector(value, length, path):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{path} must be a list of {length} values")
    return value


def validate_task(task):
    if task.get("schema_version") != "task.v1":
        raise ValueError("task schema_version must be 'task.v1'")
    for key in [
        "name",
        "language_instruction",
        "frames",
        "record_every_n_frames",
        "scene",
        "robot",
        "camera",
        "success_criteria",
        "output",
    ]:
        require_key(task, key, "task")
    if int(task["frames"]) <= 0:
        raise ValueError("task.frames must be positive")
    if int(task["record_every_n_frames"]) <= 0:
        raise ValueError("task.record_every_n_frames must be positive")

    for scene_key in ["table", "target_zone"]:
        config = require_key(task["scene"], scene_key, "task.scene")
        for key in ["path", "color", "position", "scale", "size"]:
            require_key(config, key, f"task.scene.{scene_key}")
        require_vector(config["color"], 3, f"task.scene.{scene_key}.color")
        require_vector(config["position"], 3, f"task.scene.{scene_key}.position")
        require_vector(config["scale"], 3, f"task.scene.{scene_key}.scale")

    for optional_scene_key in ["robot_pedestal", "pick_object"]:
        config = task["scene"].get(optional_scene_key)
        if not config:
            continue
        for key in ["path", "color", "position", "scale", "size"]:
            require_key(config, key, f"task.scene.{optional_scene_key}")
        require_vector(config["color"], 3, f"task.scene.{optional_scene_key}.color")
        require_vector(config["position"], 3, f"task.scene.{optional_scene_key}.position")
        require_vector(config["scale"], 3, f"task.scene.{optional_scene_key}.scale")

    lighting = require_key(task["scene"], "lighting", "task.scene")
    if lighting.get("type") != "distant":
        raise ValueError("task.scene.lighting.type must be 'distant'")

    robot = task["robot"]
    for key in ["name", "prim_path", "asset_path", "position", "rotation_degrees"]:
        require_key(robot, key, "task.robot")
    require_vector(robot["position"], 3, "task.robot.position")
    require_vector(robot["rotation_degrees"], 3, "task.robot.rotation_degrees")

    camera = task["camera"]
    if camera.get("enabled", False):
        require_vector(require_key(camera, "position", "task.camera"), 3, "task.camera.position")
        require_vector(require_key(camera, "target", "task.camera"), 3, "task.camera.target")
        require_vector(require_key(camera, "resolution", "task.camera"), 2, "task.camera.resolution")
        require_key(camera, "preview_frames", "task.camera")
        require_key(camera, "rt_subframes", "task.camera")

    criteria = task["success_criteria"]
    require_key(criteria, "min_recorded_frames", "task.success_criteria")
    require_key(criteria, "min_robot_prim_count", "task.success_criteria")
    require_key(criteria, "min_articulation_dofs", "task.success_criteria")
    require_key(criteria, "min_controlled_joints", "task.success_criteria")
    require_key(criteria, "min_joint_motion_degrees", "task.success_criteria")
    require_key(criteria, "min_preview_images", "task.success_criteria")
    require_key(task["output"], "root", "task.output")


def evaluate_success(task, metrics):
    criteria = task["success_criteria"]
    checks = {
        "recorded_frames": metrics["recorded_frames"] >= int(criteria["min_recorded_frames"]),
        "robot_loaded": bool(metrics["robot_loaded"]),
        "robot_prim_count": metrics["robot_prim_count"] >= int(criteria["min_robot_prim_count"]),
        "articulation_controller_initialized": bool(metrics["articulation_controller_initialized"]),
        "articulation_dofs": metrics["articulation_dofs"] >= int(criteria["min_articulation_dofs"]),
        "controlled_joints": metrics["controlled_joints"] >= int(criteria["min_controlled_joints"]),
        "joint_motion": metrics["max_joint_motion_degrees"] >= float(criteria["min_joint_motion_degrees"]),
        "preview_images": metrics["preview_images_written"] >= int(criteria["min_preview_images"]),
    }
    if "max_pick_place_distance" in criteria:
        checks["pick_place_distance"] = metrics.get("pick_place_distance", float("inf")) <= float(
            criteria["max_pick_place_distance"]
        )
    if "min_grasp_contact_frames" in criteria:
        checks["grasp_contact_frames"] = metrics.get("grasp_contact_frames", 0) >= int(
            criteria["min_grasp_contact_frames"]
        )
    if "min_object_attached_frames" in criteria:
        checks["object_attached_frames"] = metrics.get("object_attached_frames", 0) >= int(
            criteria["min_object_attached_frames"]
        )
    if "max_gripper_object_distance" in criteria:
        checks["gripper_object_distance"] = metrics.get("max_attached_gripper_object_distance", float("inf")) <= float(
            criteria["max_gripper_object_distance"]
        )
    return all(checks.values()), checks


def utc_now():
    return datetime.now(timezone.utc)


def append_phase(path, phase, **fields):
    payload = {"time": utc_now().isoformat(), "phase": phase, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def make_visual_cube(Cube, PreviewSurfaceMaterial, path, config):
    material = PreviewSurfaceMaterial(f"/World/Materials/{Path(path).name}")
    material.set_input_values("diffuseColor", config["color"])
    shape = Cube(
        paths=config["path"],
        positions=config["position"],
        sizes=config["size"],
        scales=config["scale"],
    )
    shape.apply_visual_materials(material)
    return shape


def set_debug_cube_transform(stage, path, center, scale, rotation=(0.0, 0.0, 0.0)):
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(tuple(center))
    xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble).Set(tuple(rotation))
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(tuple(scale))


def make_debug_cube(stage, path, color, center, scale, rotation=(0.0, 0.0, 0.0)):
    from pxr import UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([tuple(color)])
    set_debug_cube_transform(stage, path, center, scale, rotation)


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
    return prim


def create_fixed_grasp_joint(stage, joint_path, body0_path, body1_path):
    from pxr import Sdf, UsdPhysics

    if stage.GetPrimAtPath(joint_path):
        stage.RemovePrim(joint_path)
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateLocalPos0Attr().Set((0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set((0.0, 0.0, 0.0))
    return joint


def remove_grasp_joint(stage, joint_path):
    if stage.GetPrimAtPath(joint_path):
        stage.RemovePrim(joint_path)


def make_grasp_proxy(stage, end_effector_parent):
    root = "/World/GraspProxy"
    visual_root = f"{end_effector_parent}/MountedGripper"
    make_debug_cube(stage, f"{root}/Anchor", DEBUG_COLORS["gripper_tip"], [0.96, 0.25, 0.47], [0.018, 0.018, 0.018])
    make_debug_cube(stage, f"{root}/MountLink", DEBUG_COLORS["gripper_tip"], [0.96, 0.25, 0.47], [0.025, 0.025, 0.18])
    make_debug_cube(stage, f"{root}/ContactPalm", DEBUG_COLORS["gripper"], [0.96, 0.25, 0.53], [0.14, 0.04, 0.028])
    make_debug_cube(stage, f"{root}/ContactLeftFinger", DEBUG_COLORS["gripper"], [0.96, 0.16, 0.48], [0.035, 0.025, 0.12])
    make_debug_cube(stage, f"{root}/ContactRightFinger", DEBUG_COLORS["gripper"], [0.96, 0.34, 0.48], [0.035, 0.025, 0.12])
    make_debug_cube(stage, f"{root}/ContactLeftPad", DEBUG_COLORS["gripper_tip"], [0.96, 0.16, 0.43], [0.05, 0.038, 0.04])
    make_debug_cube(stage, f"{root}/ContactRightPad", DEBUG_COLORS["gripper_tip"], [0.96, 0.34, 0.43], [0.05, 0.038, 0.04])
    make_debug_cube(stage, f"{visual_root}/MountBracket", DEBUG_COLORS["gripper_tip"], [0.0, 0.0, 0.33], [0.055, 0.055, 0.22])
    return {
        "root": root,
        "visual_root": visual_root,
        "mount_parent": end_effector_parent,
        "anchor": f"{root}/Anchor",
        "mount_link": f"{root}/MountLink",
        "contact_palm": f"{root}/ContactPalm",
        "contact_left_finger": f"{root}/ContactLeftFinger",
        "contact_right_finger": f"{root}/ContactRightFinger",
        "contact_left_pad": f"{root}/ContactLeftPad",
        "contact_right_pad": f"{root}/ContactRightPad",
        "mount_bracket": f"{visual_root}/MountBracket",
        "parts": 8,
    }


def update_grasp_proxy(stage, proxy, gripper_position, jaw_width, anchor_position=None):
    anchor = list(anchor_position or gripper_position)
    half_width = float(jaw_width) * 0.5
    set_debug_cube_transform(stage, proxy["anchor"], anchor, [0.018, 0.018, 0.018])
    link_center = [
        float(gripper_position[0]),
        float(gripper_position[1]),
        (float(gripper_position[2]) + float(anchor[2])) * 0.5,
    ]
    link_length = max(abs(float(anchor[2]) - float(gripper_position[2])), 0.08)
    set_debug_cube_transform(stage, proxy["mount_link"], link_center, [0.025, 0.025, link_length * 0.5])
    set_debug_cube_transform(stage, proxy["contact_palm"], [anchor[0], anchor[1], anchor[2] + 0.075], [0.14, 0.04, 0.028])
    set_debug_cube_transform(
        stage,
        proxy["contact_left_finger"],
        [anchor[0], anchor[1] - half_width, anchor[2] + 0.015],
        [0.035, 0.025, 0.12],
    )
    set_debug_cube_transform(
        stage,
        proxy["contact_right_finger"],
        [anchor[0], anchor[1] + half_width, anchor[2] + 0.015],
        [0.035, 0.025, 0.12],
    )
    set_debug_cube_transform(
        stage,
        proxy["contact_left_pad"],
        [anchor[0], anchor[1] - half_width, anchor[2] - 0.045],
        [0.05, 0.038, 0.04],
    )
    set_debug_cube_transform(
        stage,
        proxy["contact_right_pad"],
        [anchor[0], anchor[1] + half_width, anchor[2] - 0.045],
        [0.05, 0.038, 0.04],
    )
    set_debug_cube_transform(stage, proxy["mount_bracket"], [0.0, 0.0, 0.33], [0.055, 0.055, 0.22])


def target_marker_center(target_config):
    center = list(target_config["position"])
    marker_height = 0.10
    zone_half_height = float(target_config["scale"][2]) * float(target_config.get("size", 1.0)) * 0.5
    center[2] = float(target_config["position"][2]) + zone_half_height + marker_height * 0.5
    return center


def pick_place_target_center(target_config, object_config):
    center = list(target_config["position"])
    object_half_height = float(object_config["scale"][2]) * float(object_config.get("size", 1.0)) * 0.5
    zone_half_height = float(target_config["scale"][2]) * float(target_config.get("size", 1.0)) * 0.5
    center[2] = float(target_config["position"][2]) + zone_half_height + object_half_height
    return center


def lerp_vector(start, end, amount):
    return [float(a) + (float(b) - float(a)) * amount for a, b in zip(start, end)]


def smoothstep(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def manipulation_phase(frame, frame_count):
    t = frame / max(frame_count - 1, 1)
    if t < 0.18:
        return "approach"
    if t < 0.30:
        return "pick"
    if t < 0.68:
        return "transport"
    if t < 0.98:
        return "place"
    return "retreat"


def manipulation_phase_changed(frame, frame_count):
    if frame == 0:
        return True
    return manipulation_phase(frame, frame_count) != manipulation_phase(frame - 1, frame_count)


def pick_object_pose(frame, frame_count, start_position, target_position, end_effector_position):
    phase = manipulation_phase(frame, frame_count)
    t = frame / max(frame_count - 1, 1)
    open_width = 0.18
    closed_width = 0.08

    def arm_anchor(height):
        return [float(end_effector_position[0]), float(end_effector_position[1]), height]

    start_anchor = arm_anchor(start_position[2])

    def with_gripper(anchor_position, object_position, attached, jaw_width, contact=False):
        return {
            "object_position": list(object_position),
            "gripper_position": list(end_effector_position),
            "anchor_position": list(anchor_position),
            "jaw_width": jaw_width,
            "attached": attached,
            "contact": contact,
            "phase": phase,
            "attach_method": "full_contact_fixed_joint",
        }

    if phase == "approach":
        return with_gripper(start_anchor, start_anchor, False, open_width)
    if phase == "pick":
        close = smoothstep((t - 0.18) / 0.06)
        attached = close > 0.55
        jaw_width = open_width + (closed_width - open_width) * close
        anchor = start_anchor
        return with_gripper(anchor, anchor, attached, jaw_width, contact=close > 0.35)
    if phase == "transport":
        anchor = arm_anchor(max(start_position[2], target_position[2]) + 0.18)
        return with_gripper(anchor, anchor, True, closed_width, contact=True)
    if phase == "place":
        lower = smoothstep((t - 0.68) / 0.30)
        place_height = target_position[2] + (0.18 * (1.0 - lower))
        anchor = arm_anchor(place_height)
        jaw_width = closed_width + (open_width - closed_width) * max(0.0, (lower - 0.75) / 0.25)
        return with_gripper(anchor, anchor, lower < 0.95, jaw_width, contact=lower < 0.95)
    anchor = arm_anchor(target_position[2])
    return with_gripper(anchor, anchor, False, open_width)


def distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def path_length(points):
    valid_points = [point for point in points if point]
    if len(valid_points) < 2:
        return 0.0
    return sum(distance(valid_points[index - 1], valid_points[index]) for index in range(1, len(valid_points)))


def joint_smoothness_score(joint_history):
    if len(joint_history) < 3:
        return None
    deltas = []
    for index in range(2, len(joint_history)):
        previous = joint_history[index - 2]
        current = joint_history[index - 1]
        next_position = joint_history[index]
        if not previous or not current or not next_position:
            continue
        acceleration = [
            float(next_position[joint_index]) - (2 * float(current[joint_index])) + float(previous[joint_index])
            for joint_index in range(min(len(previous), len(current), len(next_position)))
        ]
        deltas.append(math.sqrt(sum(value * value for value in acceleration)))
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def evaluate_task_quality(object_history, end_effector_history, joint_history, target_position, criteria):
    object_positions = [row["position"] for row in object_history if row.get("position")]
    attached_frames = [row for row in object_history if row.get("attached")]
    attached_gripper_distances = [
        distance(row["position"], row["anchor_position"])
        for row in attached_frames
        if row.get("position") and row.get("anchor_position")
    ]
    phase_frame_counts = {}
    for row in object_history:
        phase = row.get("phase", "unknown")
        phase_frame_counts[phase] = phase_frame_counts.get(phase, 0) + 1

    final_object_position = object_positions[-1] if object_positions else None
    settling_error = distance(final_object_position, target_position) if final_object_position and target_position else None
    start_height = object_positions[0][2] if object_positions else None
    max_height = max((position[2] for position in object_positions), default=None)
    lift_height = (max_height - start_height) if max_height is not None and start_height is not None else None
    max_allowed_error = float(criteria.get("max_pick_place_distance", 0.08))

    return {
        "schema_version": "task_evaluation.v1",
        "success": bool(settling_error is not None and settling_error <= max_allowed_error),
        "pick_success": bool(attached_frames),
        "place_success": bool(
            final_object_position
            and target_position
            and settling_error is not None
            and settling_error <= max_allowed_error
            and not object_history[-1].get("attached")
        ),
        "settling_error_m": round(settling_error, 4) if settling_error is not None else None,
        "object_lift_height_m": round(lift_height, 4) if lift_height is not None else None,
        "object_path_length_m": round(path_length(object_positions), 4),
        "end_effector_path_length_m": round(path_length(end_effector_history), 4),
        "joint_smoothness_score": round(joint_smoothness_score(joint_history), 6)
        if joint_smoothness_score(joint_history) is not None
        else None,
        "frames_with_object_attached": len(attached_frames),
        "frames_with_grasp_contact": sum(1 for row in object_history if row.get("contact")),
        "max_attached_gripper_object_distance_m": round(max(attached_gripper_distances), 4)
        if attached_gripper_distances
        else None,
        "mean_attached_gripper_object_distance_m": round(
            sum(attached_gripper_distances) / len(attached_gripper_distances), 4
        )
        if attached_gripper_distances
        else None,
        "grasp_attach_method": object_history[0].get("attach_method") if object_history else None,
        "phase_frame_counts": phase_frame_counts,
        "max_allowed_settling_error_m": max_allowed_error,
    }


def make_debug_overlay(stage, target_config):
    root = "/World/DebugOverlay"
    make_debug_cube(stage, f"{root}/EndEffectorMarker", DEBUG_COLORS["marker"], [-0.15, 0.0, 1.0], [0.08, 0.08, 0.08])
    make_debug_cube(stage, f"{root}/TargetMarker", DEBUG_COLORS["target"], target_marker_center(target_config), [0.10, 0.10, 0.10])
    make_debug_cube(stage, f"{root}/AxisX", DEBUG_COLORS["x"], [0.0, 0.0, 1.0], [0.24, 0.018, 0.018])
    make_debug_cube(stage, f"{root}/AxisY", DEBUG_COLORS["y"], [0.0, 0.0, 1.0], [0.018, 0.24, 0.018])
    make_debug_cube(stage, f"{root}/AxisZ", DEBUG_COLORS["z"], [0.0, 0.0, 1.0], [0.018, 0.018, 0.24])
    for index in range(len(JOINT_NAMES)):
        make_debug_cube(
            stage,
            f"{root}/JointBar{index}",
            [0.12 + 0.12 * index, 0.78 - 0.07 * index, 0.94],
            [0.75 + index * 0.07, -0.72, 0.70],
            [0.04, 0.04, 0.04],
        )
    return {
        "root": root,
        "end_effector_marker": f"{root}/EndEffectorMarker",
        "target_marker": f"{root}/TargetMarker",
        "axis_paths": {
            "x": f"{root}/AxisX",
            "y": f"{root}/AxisY",
            "z": f"{root}/AxisZ",
        },
        "joint_bars": [f"{root}/JointBar{index}" for index in range(len(JOINT_NAMES))],
        "parts": 2 + 3 + len(JOINT_NAMES),
    }


def find_end_effector_prim_path(stage, robot_root):
    keywords = ["tool0", "tool", "flange", "wrist_3", "ee", "tcp"]
    candidates = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(robot_root):
            continue
        lower = path.lower()
        if any(keyword in lower for keyword in keywords):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (0 if "tool" in item.lower() or "flange" in item.lower() else 1, -len(item)))
    return candidates[0]


def prim_world_position(stage, path):
    from pxr import UsdGeom

    if not path:
        return None
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    translation = transform.ExtractTranslation()
    return [float(translation[0]), float(translation[1]), float(translation[2])]


def fallback_end_effector_position(joint_positions):
    base = [-0.55, 0.0, 0.72]
    lengths = [0.55, 0.46, 0.28]
    angles = [
        float(joint_positions[1]) if len(joint_positions) > 1 else 0.0,
        (float(joint_positions[1]) + float(joint_positions[2])) if len(joint_positions) > 2 else 0.0,
        (float(joint_positions[1]) + float(joint_positions[2]) + float(joint_positions[3]))
        if len(joint_positions) > 3
        else 0.0,
    ]
    point = list(base)
    yaw = float(joint_positions[0]) if joint_positions else 0.0
    for length, angle in zip(lengths, angles):
        reach = length * math.cos(angle)
        point[0] += reach * math.cos(yaw)
        point[1] += reach * math.sin(yaw)
        point[2] += length * math.sin(angle)
    return point


def update_debug_overlay(stage, overlay, end_effector_position, joint_positions):
    marker = list(end_effector_position)
    set_debug_cube_transform(stage, overlay["end_effector_marker"], marker, [0.08, 0.08, 0.08])
    set_debug_cube_transform(stage, overlay["axis_paths"]["x"], [marker[0] + 0.14, marker[1], marker[2]], [0.28, 0.018, 0.018])
    set_debug_cube_transform(stage, overlay["axis_paths"]["y"], [marker[0], marker[1] + 0.14, marker[2]], [0.018, 0.28, 0.018])
    set_debug_cube_transform(stage, overlay["axis_paths"]["z"], [marker[0], marker[1], marker[2] + 0.14], [0.018, 0.018, 0.28])

    for index, path in enumerate(overlay["joint_bars"]):
        value = abs(math.degrees(float(joint_positions[index]))) if index < len(joint_positions) else 0.0
        height = 0.06 + min(value / 180.0, 1.0) * 0.42
        x = 0.72 + index * 0.08
        y = -0.72
        z = 0.47 + height / 2.0
        set_debug_cube_transform(stage, path, [x, y, z], [0.045, 0.045, height])


def count_prim_subtree(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0
    return sum(1 for prim in stage.Traverse() if str(prim.GetPath()).startswith(root_path))


def read_recorded_frames(path):
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def joint_targets(frame, frame_count, base_positions):
    t = frame / max(frame_count - 1, 1)
    phase = 2.0 * math.pi * t
    ready_pose = [
        math.radians(8.0),
        math.radians(-36.0),
        math.radians(58.0),
        math.radians(-42.0),
        math.radians(54.0),
        math.radians(8.0),
    ]
    offsets = [
        math.radians(35.0) * math.sin(phase),
        math.radians(16.0) * math.sin(phase + 0.45),
        math.radians(20.0) * math.sin(phase + 0.95),
        math.radians(18.0) * math.sin(phase + 1.45),
        math.radians(30.0) * math.sin(phase + 2.10),
        math.radians(40.0) * math.sin(phase + 2.70),
    ]
    targets = list(base_positions)
    for index, offset in enumerate(offsets):
        targets[index] = ready_pose[index] + offset
    return targets


def apply_initial_joint_pose(robot, initial_positions):
    if hasattr(robot, "set_joint_positions"):
        robot.set_joint_positions(np.array(initial_positions, dtype=np.float32))


def vector_to_list(value, fallback):
    if value is None:
        return list(fallback)
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def radians_to_degrees(values):
    return [round(math.degrees(float(value)), 3) for value in values]


def joint_motion_ranges_degrees(history, joint_names):
    if not history:
        return {name: 0.0 for name in joint_names}
    ranges = {}
    for index, name in enumerate(joint_names):
        values = [row[index] for row in history]
        ranges[name] = round(math.degrees(max(values) - min(values)), 3)
    return ranges


def main():
    started_at = utc_now()
    task = parse_simple_yaml(TASK_PATH)
    validate_task(task)
    episode_id = f"episode_{started_at.strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(task["output"]["root"])
    episode_dir = output_root / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    simulation_app = None
    phase_path = episode_dir / "phase_events.jsonl"
    trajectory_path = episode_dir / "trajectory.jsonl"
    frame_count = int(task["frames"])
    record_every = int(task["record_every_n_frames"])
    camera_config = task.get("camera", {})
    preview_dir = episode_dir / "preview"
    preview_enabled = bool(camera_config.get("enabled", False))
    preview_frames = set(camera_config.get("preview_frames", []))

    try:
        append_phase(phase_path, "scene_script_start", episode_id=episode_id)
        append_phase(phase_path, "simulation_app_start")
        simulation_app = SimulationApp({"headless": True})
        append_phase(phase_path, "simulation_app_ready")

        import omni.replicator.core as rep
        import omni.usd
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
        from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
        from isaacsim.core.experimental.prims import GeomPrim
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import UsdGeom

        append_phase(phase_path, "scene_create_start")
        stage_utils.create_new_stage()
        world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
        stage = omni.usd.get_context().get_stage()

        ground_config = task["scene"].get("ground", {})
        if ground_config.get("enabled", True):
            GroundPlane(ground_config["path"], positions=[0, 0, 0])

        light_config = task["scene"]["lighting"]
        light = DistantLight(light_config["path"])
        light.set_intensities(light_config["intensity"])

        pedestal_config = task["scene"].get("robot_pedestal")
        if pedestal_config:
            pedestal_shape = make_visual_cube(Cube, PreviewSurfaceMaterial, "RobotPedestal", pedestal_config)
            GeomPrim(paths=pedestal_shape.paths, apply_collision_apis=True)

        table_shape = make_visual_cube(Cube, PreviewSurfaceMaterial, "WorkTable", task["scene"]["table"])
        GeomPrim(paths=table_shape.paths, apply_collision_apis=True)
        target_shape = make_visual_cube(Cube, PreviewSurfaceMaterial, "TargetZone", task["scene"]["target_zone"])
        GeomPrim(paths=target_shape.paths, apply_collision_apis=True)
        pick_object_config = task["scene"].get("pick_object")
        if pick_object_config:
            pick_object_shape = make_visual_cube(Cube, PreviewSurfaceMaterial, "PickObject", pick_object_config)
            GeomPrim(paths=pick_object_shape.paths, apply_collision_apis=True)
            apply_physics_body(stage, pick_object_config["path"], mass=0.25, kinematic=False)
        debug_overlay = make_debug_overlay(stage, task["scene"]["target_zone"])
        grasp_proxy = None

        robot_config = task["robot"]
        robot_asset_path = robot_config["asset_path"]
        if not Path(robot_asset_path).exists():
            raise FileNotFoundError(f"robot asset not found: {robot_asset_path}")
        robot_prim = stage.DefinePrim(robot_config["prim_path"], "Xform")
        robot_prim.GetReferences().AddReference(robot_asset_path)
        xform = UsdGeom.XformCommonAPI(robot_prim)
        xform.SetTranslate(tuple(robot_config["position"]))
        xform.SetRotate(tuple(robot_config["rotation_degrees"]))

        for _ in range(3):
            simulation_app.update()
        robot_prim_count = count_prim_subtree(stage, robot_config["prim_path"])
        robot = world.scene.add(
            SingleArticulation(
                prim_path=robot_config["prim_path"],
                name=robot_config["name"],
                reset_xform_properties=False,
            )
        )
        append_phase(phase_path, "world_reset_start")
        world.reset()
        append_phase(phase_path, "world_reset_end")

        controller = robot.get_articulation_controller()
        controller.switch_control_mode("position")
        dof_names = list(robot.dof_names)
        joint_indices = [robot.get_dof_index(name) for name in JOINT_NAMES if name in dof_names]
        controlled_joint_names = [name for name in JOINT_NAMES if name in dof_names]
        end_effector_prim_path = find_end_effector_prim_path(stage, robot_config["prim_path"])
        grasp_proxy = make_grasp_proxy(stage, end_effector_prim_path or robot_config["prim_path"])
        apply_physics_body(stage, grasp_proxy["anchor"], mass=1.0, kinematic=True)
        base_positions = vector_to_list(robot.get_joint_positions(), [0.0] * robot.num_dof)
        base_positions = base_positions[: robot.num_dof]
        if len(base_positions) < robot.num_dof:
            base_positions.extend([0.0] * (robot.num_dof - len(base_positions)))
        initial_targets = joint_targets(0, frame_count, base_positions)
        apply_initial_joint_pose(robot, initial_targets)
        world.step(render=True)
        base_positions = vector_to_list(robot.get_joint_positions(), initial_targets)[: robot.num_dof]
        append_phase(
            phase_path,
            "articulation_controller_ready",
            dofs=robot.num_dof,
            controlled_joints=len(joint_indices),
        )
        append_phase(
            phase_path,
            "debug_overlay_ready",
            parts=debug_overlay["parts"],
            end_effector_prim_path=end_effector_prim_path or "fallback",
        )
        append_phase(
            phase_path,
            "scene_created",
            robot=robot_config["name"],
            robot_prim_count=robot_prim_count,
            articulation_dofs=robot.num_dof,
            debug_overlay_parts=debug_overlay["parts"],
        )

        if preview_enabled:
            append_phase(phase_path, "preview_writer_setup_start")
            preview_dir.mkdir(parents=True, exist_ok=True)
            camera = rep.create.camera(
                position=tuple(camera_config["position"]),
                look_at=tuple(camera_config["target"]),
            )
            render_product = rep.create.render_product(camera, tuple(camera_config["resolution"]))
            preview_writer = rep.WriterRegistry.get("BasicWriter")
            preview_writer.initialize(output_dir=str(preview_dir), rgb=True)
            preview_writer.attach([render_product])
            append_phase(phase_path, "preview_writer_ready")

        app_utils.play()
        world.step(render=True)
        append_phase(phase_path, "physics_recording_start", frames=frame_count)
        robot_position = list(robot_config["position"])
        joint_history = []
        target_history = []
        end_effector_history = []
        object_history = []
        phase_names_seen = []
        grasp_attached_previous = False
        grasp_contact_recorded = False
        grasp_joint_created = False
        grasp_joint_path = "/World/GraspProxy/FixedGraspJoint"
        pick_start_position = list(pick_object_config["position"]) if pick_object_config else None
        pick_target_position = (
            pick_place_target_center(task["scene"]["target_zone"], pick_object_config)
            if pick_object_config
            else None
        )
        with trajectory_path.open("w", encoding="utf-8") as trajectory:
            for frame in range(frame_count):
                task_phase = manipulation_phase(frame, frame_count)
                if task_phase not in phase_names_seen:
                    phase_names_seen.append(task_phase)
                if manipulation_phase_changed(frame, frame_count):
                    append_phase(phase_path, f"manipulation_{task_phase}_start", frame=frame)
                target = joint_targets(frame, frame_count, base_positions)
                target_history.append(target)
                controller.apply_action(
                    ArticulationAction(
                        joint_positions=np.array(target, dtype=np.float32),
                        joint_indices=np.array(list(range(robot.num_dof)), dtype=np.int32),
                    )
                )
                world.step(render=True)
                measured = vector_to_list(robot.get_joint_positions(), target)[: robot.num_dof]
                joint_history.append(measured)
                end_effector_position = prim_world_position(stage, end_effector_prim_path)
                end_effector_source = "asset_prim" if end_effector_position else "fallback_fk"
                if not end_effector_position:
                    end_effector_position = fallback_end_effector_position(measured)
                end_effector_history.append(end_effector_position)
                update_debug_overlay(stage, debug_overlay, end_effector_position, measured)
                pick_position = None
                pick_attached = False
                grasp_state = None
                if pick_object_config:
                    grasp_state = pick_object_pose(
                        frame,
                        frame_count,
                        pick_start_position,
                        pick_target_position,
                        end_effector_position,
                    )
                    pick_position = grasp_state["object_position"]
                    pick_attached = grasp_state["attached"]
                    update_grasp_proxy(
                        stage,
                        grasp_proxy,
                        grasp_state["gripper_position"],
                        grasp_state["jaw_width"],
                        anchor_position=grasp_state["anchor_position"],
                    )
                    if grasp_state["contact"] and not grasp_contact_recorded:
                        append_phase(phase_path, "grasp_proxy_contact", frame=frame, method=grasp_state["attach_method"])
                        grasp_contact_recorded = True
                    if pick_attached and not grasp_attached_previous:
                        set_debug_cube_transform(
                            stage,
                            pick_object_config["path"],
                            grasp_state["anchor_position"],
                            pick_object_config["scale"],
                        )
                        world.step(render=True)
                        create_fixed_grasp_joint(
                            stage,
                            grasp_joint_path,
                            grasp_proxy["anchor"],
                            pick_object_config["path"],
                        )
                        append_phase(phase_path, "grasp_proxy_attach", frame=frame, method=grasp_state["attach_method"])
                        grasp_joint_created = True
                    if grasp_attached_previous and not pick_attached:
                        remove_grasp_joint(stage, grasp_joint_path)
                        append_phase(phase_path, "grasp_proxy_release", frame=frame, method=grasp_state["attach_method"])
                        grasp_joint_created = False
                    grasp_attached_previous = pick_attached
                    if not pick_attached and not grasp_joint_created and task_phase in {"approach", "pick"}:
                        set_debug_cube_transform(
                            stage,
                            pick_object_config["path"],
                            pick_position,
                            pick_object_config["scale"],
                        )
                world.step(render=True)
                if pick_object_config and grasp_state:
                    actual_pick_position = prim_world_position(stage, pick_object_config["path"]) or pick_position
                    object_history.append(
                        {
                            "position": actual_pick_position,
                            "attached": pick_attached,
                            "phase": task_phase,
                            "contact": grasp_state["contact"],
                            "gripper_position": grasp_state["gripper_position"],
                            "anchor_position": grasp_state["anchor_position"],
                            "jaw_width": grasp_state["jaw_width"],
                            "attach_method": grasp_state["attach_method"],
                            "grasp_constraint": "fixed_joint" if grasp_joint_created else None,
                        }
                    )
                    pick_position = actual_pick_position

                if preview_enabled and frame in preview_frames:
                    append_phase(phase_path, "preview_frame_capture_start", frame=frame)
                    rep.orchestrator.step(rt_subframes=int(camera_config.get("rt_subframes", 1)))
                    app_utils.play()
                    append_phase(phase_path, "preview_frame_capture_end", frame=frame)
                if frame % record_every != 0:
                    continue
                trajectory.write(
                    json.dumps(
                        {
                            "frame": frame,
                            "joint_names": dof_names,
                            "controlled_joint_names": controlled_joint_names,
                            "joint_position_targets": target,
                            "joint_position_targets_degrees": radians_to_degrees(target),
                            "joint_positions": measured,
                            "joint_positions_degrees": radians_to_degrees(measured),
                            "end_effector_position": end_effector_position,
                            "end_effector_source": end_effector_source,
                            "task_phase": task_phase,
                            "objects": {
                                "robot_base": {
                                    "position": robot_position,
                                    "orientation": [0, 0, 0],
                                    "linear_velocity": [0, 0, 0],
                                    "angular_velocity": [0, 0, 0],
                                },
                                "end_effector_marker": {
                                    "position": end_effector_position,
                                    "orientation": [0, 0, 0],
                                    "linear_velocity": [0, 0, 0],
                                    "angular_velocity": [0, 0, 0],
                                },
                                **(
                                    {
                                        "pick_object": {
                                            "position": pick_position,
                                            "orientation": [0, 0, 0],
                                            "linear_velocity": [0, 0, 0],
                                            "angular_velocity": [0, 0, 0],
                                            "attached": pick_attached,
                                            "contact": grasp_state["contact"] if grasp_state else False,
                                            "attach_method": grasp_state["attach_method"] if grasp_state else None,
                                            "grasp_constraint": "fixed_joint" if grasp_joint_created else None,
                                            "phase": task_phase,
                                        }
                                    }
                                    if pick_position
                                    else {}
                                ),
                                **(
                                    {
                                        "grasp_proxy": {
                                            "position": grasp_state["gripper_position"],
                                            "anchor_position": grasp_state["anchor_position"],
                                            "orientation": [0, 0, 0],
                                            "linear_velocity": [0, 0, 0],
                                            "angular_velocity": [0, 0, 0],
                                            "jaw_width": grasp_state["jaw_width"],
                                            "contact": grasp_state["contact"],
                                            "attached": pick_attached,
                                            "attach_method": grasp_state["attach_method"],
                                            "grasp_constraint": "fixed_joint" if grasp_joint_created else None,
                                        }
                                    }
                                    if grasp_state
                                    else {}
                                ),
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        append_phase(phase_path, "physics_recording_end", frames=frame_count)

        finished_at = utc_now()
        elapsed_seconds = (finished_at - started_at).total_seconds()
        preview_images = sorted(path.name for path in preview_dir.glob("*.png"))
        joint_motion_degrees = joint_motion_ranges_degrees(joint_history, dof_names)
        max_joint_motion_degrees = max(joint_motion_degrees.values()) if joint_motion_degrees else 0.0
        final_positions = joint_history[-1] if joint_history else base_positions
        final_end_effector_position = end_effector_history[-1] if end_effector_history else None
        final_pick_object_position = object_history[-1]["position"] if object_history else None
        release_target_position = (
            object_history[-1]["anchor_position"]
            if object_history and object_history[-1].get("anchor_position")
            else pick_target_position
        )
        pick_place_distance = (
            distance(final_pick_object_position, release_target_position)
            if final_pick_object_position and release_target_position
            else None
        )
        task_evaluation = evaluate_task_quality(
            object_history,
            end_effector_history,
            joint_history,
            release_target_position,
            task["success_criteria"],
        )
        attached_gripper_distances = [
            distance(row["position"], row["anchor_position"])
            for row in object_history
            if row.get("attached") and row.get("position") and row.get("anchor_position")
        ]
        metadata = {
            "episode_id": episode_id,
            "run_id": os.environ.get("FARPOINT_RUN_ID"),
            "task_schema_version": task["schema_version"],
            "task_name": task["name"],
            "language_instruction": task["language_instruction"],
            "success_criteria": task["success_criteria"],
            "simulator": "Isaac Sim",
            "image": "nvcr.io/nvidia/isaac-sim:6.0.0",
            "python": sys.version,
            "platform": platform.platform(),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "preview_enabled": preview_enabled,
            "preview_resolution": camera_config.get("resolution"),
        }
        metrics = {
            "success": False,
            "frames_requested": frame_count,
            "record_every_n_frames": record_every,
            "recorded_frames": read_recorded_frames(trajectory_path),
            "preview_frames_requested": sorted(preview_frames),
            "preview_images_written": len(preview_images),
            "preview_images": preview_images,
            "elapsed_seconds": elapsed_seconds,
            "object_count": 4 if pick_object_config else 2,
            "final_object_positions": {
                "robot_base": robot_position,
                "end_effector_marker": final_end_effector_position,
                **({"grasp_proxy": object_history[-1]["gripper_position"]} if object_history else {}),
                **({"pick_object": final_pick_object_position} if final_pick_object_position else {}),
            },
            "task_type": "physics_pickup_v1",
            "pickup_mode": "dynamic_end_effector_contact_grasp",
            "grasp_proxy": True,
            "gripper_mount_parent": grasp_proxy["mount_parent"],
            "grasp_constraint": "fixed_joint",
            "grasp_attach_method": task_evaluation.get("grasp_attach_method"),
            "grasp_contact_frames": task_evaluation.get("frames_with_grasp_contact"),
            "max_attached_gripper_object_distance": round(max(attached_gripper_distances), 4)
            if attached_gripper_distances
            else None,
            "mean_attached_gripper_object_distance": round(
                sum(attached_gripper_distances) / len(attached_gripper_distances), 4
            )
            if attached_gripper_distances
            else None,
            "grasp_proxy_parts": grasp_proxy["parts"],
            "task_phases": phase_names_seen,
            "pick_object_start_position": pick_start_position,
            "pick_object_target_position": release_target_position,
            "final_pick_object_position": final_pick_object_position,
            "pick_place_distance": round(pick_place_distance, 4) if pick_place_distance is not None else None,
            "object_attached_frames": sum(1 for row in object_history if row["attached"]),
            "task_evaluation": task_evaluation,
            "robot_name": robot_config["name"],
            "robot_asset_path": robot_asset_path,
            "robot_prim_path": robot_config["prim_path"],
            "robot_loaded": robot_prim_count > 0,
            "robot_prim_count": robot_prim_count,
            "kinematic_proxy": False,
            "articulation_controller": True,
            "articulation_controller_initialized": robot.handles_initialized,
            "articulation_dofs": robot.num_dof,
            "joint_names": dof_names,
            "controlled_joint_names": controlled_joint_names,
            "controlled_joints": len(joint_indices),
            "debug_visualization": True,
            "debug_overlay_parts": debug_overlay["parts"] + grasp_proxy["parts"],
            "debug_end_effector_prim_path": end_effector_prim_path,
            "joint_motion_degrees": joint_motion_degrees,
            "max_joint_motion_degrees": max_joint_motion_degrees,
            "initial_joint_positions": base_positions,
            "initial_joint_positions_degrees": radians_to_degrees(base_positions),
            "final_joint_positions": final_positions,
            "final_joint_positions_degrees": radians_to_degrees(final_positions),
            "final_end_effector_position": final_end_effector_position,
        }
        success, success_checks = evaluate_success(task, metrics)
        metrics["success"] = success
        metrics["success_checks"] = success_checks

        write_json(episode_dir / "metadata.json", metadata)
        write_json(episode_dir / "metrics.json", metrics)
        append_phase(
            phase_path,
            "episode_written",
            recorded_frames=metrics["recorded_frames"],
            preview_images_written=metrics["preview_images_written"],
            robot_prim_count=robot_prim_count,
            articulation_dofs=robot.num_dof,
        )
        print(f"episode written: {episode_dir}", flush=True)
        print(
            f"SMOKE_TEST_RESULT: {'PASS' if success else 'FAIL'} {episode_id} "
            f"controlled {robot_config['name']} with {robot.num_dof} DOFs",
            flush=True,
        )
        return 0 if success else 1
    except Exception as exc:
        episode_dir.mkdir(parents=True, exist_ok=True)
        append_phase(phase_path, "scene_script_error", error_type=type(exc).__name__, error=str(exc))
        write_json(
            episode_dir / "metrics.json",
            {"success": False, "error_type": type(exc).__name__, "error": str(exc)},
        )
        print(f"SMOKE_TEST_RESULT: FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        if simulation_app is not None:
            append_phase(phase_path, "simulation_app_close_start")
            simulation_app.close()
            append_phase(phase_path, "simulation_app_close_end")
        append_phase(phase_path, "scene_script_end")


if __name__ == "__main__":
    sys.exit(main())
