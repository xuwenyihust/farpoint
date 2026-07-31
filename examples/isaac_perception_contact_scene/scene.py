import os
import runpy
from pathlib import Path


PROJECT_ROOT = Path("/workspace/project")
os.environ["FARPOINT_TASK_PATH"] = str(
    PROJECT_ROOT / "examples" / "isaac_perception_contact_scene" / "task.yaml"
)
runpy.run_path(
    str(PROJECT_ROOT / "examples" / "isaac_ur10e_robotiq_scene" / "scene.py"),
    run_name="__main__",
)
