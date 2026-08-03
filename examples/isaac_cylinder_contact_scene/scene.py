"""Run the shared UR10e scene with a cylinder-safe pre-grasp hover."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path("/workspace/project")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.cylinder_control import hold_pregrasp_hover  # noqa: E402


os.environ["FARPOINT_TASK_PATH"] = str(
    PROJECT_ROOT / "examples" / "isaac_perception_contact_scene" / "task.yaml"
)

shared_scene_path = (
    PROJECT_ROOT / "examples" / "isaac_ur10e_robotiq_scene" / "scene.py"
)
spec = importlib.util.spec_from_file_location(
    "farpoint_isaac_ur10e_robotiq_scene",
    shared_scene_path,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load shared Isaac scene: {shared_scene_path}")
shared_scene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared_scene)

shared_cartesian_target = shared_scene.cartesian_pick_and_place_target
physical_frame = 0
pregrasp_hover_height = None


def cylinder_cartesian_pick_and_place_target(
    frame,
    attach_frame,
    release_frame,
    pick_grasp_position,
    place_grasp_position,
    lift_height,
    grasp_retry_frames,
    **kwargs,
):
    global physical_frame, pregrasp_hover_height

    target = shared_cartesian_target(
        frame,
        attach_frame,
        release_frame,
        pick_grasp_position,
        place_grasp_position,
        lift_height,
        grasp_retry_frames,
        **kwargs,
    )
    if pregrasp_hover_height is None:
        pregrasp_hover_height = float(pick_grasp_position[2]) + 0.22
    target = hold_pregrasp_hover(
        target,
        physical_frame=physical_frame,
        release_frame=int(attach_frame) + int(grasp_retry_frames) + 140,
        hover_height=pregrasp_hover_height,
    )
    physical_frame += 1
    return target


shared_scene.cartesian_pick_and_place_target = (
    cylinder_cartesian_pick_and_place_target
)

if __name__ == "__main__":
    raise SystemExit(shared_scene.main())
