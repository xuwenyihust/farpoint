import ast
from pathlib import Path


FORBIDDEN_CONTROL_CALLS = {
    "create_fixed_grasp_joint",
    "set_cube_transform",
    "set_rigid_body_kinematic",
}


def audit_contact_only_source(scene_path):
    scene_path = Path(scene_path)
    tree = ast.parse(scene_path.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in FORBIDDEN_CONTROL_CALLS:
            violations.append(
                {
                    "line": node.lineno,
                    "call": name,
                }
            )
    return {
        "valid": not violations,
        "scene_path": str(scene_path),
        "forbidden_calls": violations,
    }


def audit_episode_runtime(metrics):
    checks = {
        "contact_only_mode": metrics.get("grasp_constraint") == "contact_only",
        "rgbd_control_source": metrics.get("control_pose_source") == "rgbd_estimate",
        "no_temporary_grasp_joint": not bool(
            metrics.get("temporary_grasp_joint_created")
        ),
        "no_grasp_joint_path": metrics.get("grasp_joint_path") in (None, ""),
        "dataset_valid": bool(metrics.get("dataset_valid")),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
    }
