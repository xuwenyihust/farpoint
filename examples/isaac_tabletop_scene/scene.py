import ast
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from isaacsim import SimulationApp


PROJECT_ROOT = Path("/workspace/project")
TASK_PATH = PROJECT_ROOT / "examples" / "isaac_tabletop_scene" / "task.yaml"


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
        "objects",
        "camera",
        "success_criteria",
        "output",
    ]:
        require_key(task, key, "task")

    if int(task["frames"]) <= 0:
        raise ValueError("task.frames must be positive")
    if int(task["record_every_n_frames"]) <= 0:
        raise ValueError("task.record_every_n_frames must be positive")

    table = require_key(task["scene"], "table", "task.scene")
    for key in ["path", "color", "position", "scale", "size"]:
        require_key(table, key, "task.scene.table")
    require_vector(table["color"], 3, "task.scene.table.color")
    require_vector(table["position"], 3, "task.scene.table.position")
    require_vector(table["scale"], 3, "task.scene.table.scale")

    lighting = require_key(task["scene"], "lighting", "task.scene")
    if lighting.get("type") != "distant":
        raise ValueError("task.scene.lighting.type must be 'distant'")
    require_key(lighting, "path", "task.scene.lighting")
    require_key(lighting, "intensity", "task.scene.lighting")

    if not task["objects"]:
        raise ValueError("task.objects must contain at least one object")
    for name, config in task["objects"].items():
        if config.get("type") != "dynamic_cube":
            raise ValueError(f"task.objects.{name}.type must be 'dynamic_cube'")
        for key in ["path", "color", "position", "scale", "size"]:
            require_key(config, key, f"task.objects.{name}")
        require_vector(config["color"], 3, f"task.objects.{name}.color")
        require_vector(config["position"], 3, f"task.objects.{name}.position")
        require_vector(config["scale"], 3, f"task.objects.{name}.scale")

    camera = task["camera"]
    if camera.get("enabled", False):
        require_vector(require_key(camera, "position", "task.camera"), 3, "task.camera.position")
        require_vector(require_key(camera, "target", "task.camera"), 3, "task.camera.target")
        require_vector(require_key(camera, "resolution", "task.camera"), 2, "task.camera.resolution")
        require_key(camera, "preview_frames", "task.camera")
        require_key(camera, "rt_subframes", "task.camera")

    criteria = task["success_criteria"]
    require_key(criteria, "min_recorded_frames", "task.success_criteria")
    require_key(criteria, "all_objects_min_z", "task.success_criteria")
    require_key(criteria, "all_objects_max_abs_xy", "task.success_criteria")
    require_key(task["output"], "root", "task.output")


def evaluate_success(task, metrics):
    criteria = task["success_criteria"]
    checks = {}
    checks["recorded_frames"] = metrics["recorded_frames"] >= int(criteria["min_recorded_frames"])

    min_z = float(criteria["all_objects_min_z"])
    max_abs_xy = float(criteria["all_objects_max_abs_xy"])
    positions = metrics["final_object_positions"]
    checks["all_objects_above_table"] = all(position[2] >= min_z for position in positions.values())
    checks["all_objects_on_tabletop"] = all(
        abs(position[0]) <= max_abs_xy and abs(position[1]) <= max_abs_xy
        for position in positions.values()
    )

    return all(checks.values()), checks


def utc_now():
    return datetime.now(timezone.utc)


def append_phase(path, phase, **fields):
    payload = {
        "time": utc_now().isoformat(),
        "phase": phase,
        **fields,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def to_plain(value):
    if hasattr(value, "numpy"):
        return to_plain(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    try:
        return list(value)
    except TypeError:
        return value


def first_item(value):
    plain = to_plain(value)
    if isinstance(plain, list) and plain:
        return plain[0]
    return plain


def read_recorded_frames(path):
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


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
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
        from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
        from isaacsim.core.experimental.prims import GeomPrim, RigidPrim

        append_phase(phase_path, "scene_create_start")
        stage_utils.create_new_stage()

        ground_config = task["scene"].get("ground", {})
        if ground_config.get("enabled", True):
            GroundPlane(ground_config["path"], positions=[0, 0, 0])

        light_config = task["scene"]["lighting"]
        light = DistantLight(light_config["path"])
        light.set_intensities(light_config["intensity"])

        table_config = task["scene"]["table"]
        table_material = PreviewSurfaceMaterial("/World/Materials/Table")
        table_material.set_input_values("diffuseColor", table_config["color"])
        table_shape = Cube(
            paths=table_config["path"],
            positions=table_config["position"],
            sizes=table_config["size"],
            scales=table_config["scale"],
        )
        table_shape.apply_visual_materials(table_material)
        GeomPrim(paths=table_shape.paths, apply_collision_apis=True)

        rigid_objects = {}
        for name, object_config in task["objects"].items():
            material = PreviewSurfaceMaterial(f"/World/Materials/{name}")
            material.set_input_values("diffuseColor", object_config["color"])
            shape = Cube(
                paths=object_config["path"],
                positions=object_config["position"],
                sizes=object_config["size"],
                scales=object_config["scale"],
            )
            shape.apply_visual_materials(material)
            rigid = RigidPrim(paths=shape.paths)
            GeomPrim(paths=shape.paths, apply_collision_apis=True)
            rigid_objects[name] = rigid

        append_phase(phase_path, "scene_created", objects=len(rigid_objects))

        if preview_enabled:
            append_phase(phase_path, "preview_writer_setup_start")
            preview_dir.mkdir(parents=True, exist_ok=True)
            camera = rep.create.camera(
                position=tuple(camera_config["position"]),
                look_at=tuple(camera_config["target"]),
            )
            render_product = rep.create.render_product(
                camera,
                tuple(camera_config["resolution"]),
            )
            preview_writer = rep.WriterRegistry.get("BasicWriter")
            preview_writer.initialize(output_dir=str(preview_dir), rgb=True)
            preview_writer.attach([render_product])
            append_phase(phase_path, "preview_writer_ready")

        app_utils.play()
        simulation_app.update()
        for rigid in rigid_objects.values():
            rigid.set_velocities(linear_velocities=[0, 0, 0], angular_velocities=[0, 0, 0])

        append_phase(phase_path, "physics_recording_start", frames=frame_count)
        with trajectory_path.open("w", encoding="utf-8") as trajectory:
            for frame in range(frame_count):
                simulation_app.update()
                if preview_enabled and frame in preview_frames:
                    append_phase(phase_path, "preview_frame_capture_start", frame=frame)
                    rep.orchestrator.step(rt_subframes=int(camera_config.get("rt_subframes", 1)))
                    app_utils.play()
                    append_phase(phase_path, "preview_frame_capture_end", frame=frame)
                if frame % record_every != 0:
                    continue

                objects = {}
                for name, rigid in rigid_objects.items():
                    positions, orientations = rigid.get_world_poses()
                    linear_velocities, angular_velocities = rigid.get_velocities()
                    objects[name] = {
                        "position": first_item(positions),
                        "orientation": first_item(orientations),
                        "linear_velocity": first_item(linear_velocities),
                        "angular_velocity": first_item(angular_velocities),
                    }
                trajectory.write(json.dumps({"frame": frame, "objects": objects}, sort_keys=True) + "\n")
        append_phase(phase_path, "physics_recording_end", frames=frame_count)

        final_object_positions = {}
        final_object_orientations = {}
        for name, rigid in rigid_objects.items():
            positions, orientations = rigid.get_world_poses()
            final_object_positions[name] = first_item(positions)
            final_object_orientations[name] = first_item(orientations)

        finished_at = utc_now()
        elapsed_seconds = (finished_at - started_at).total_seconds()
        preview_images = sorted(path.name for path in preview_dir.glob("*.png"))

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
            "object_count": len(rigid_objects),
            "final_object_positions": final_object_positions,
            "final_object_orientations": final_object_orientations,
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
            objects=metrics["object_count"],
        )

        print(f"episode written: {episode_dir}", flush=True)
        print(
            f"SMOKE_TEST_RESULT: {'PASS' if success else 'FAIL'} {episode_id} "
            f"recorded {metrics['recorded_frames']} frames with {metrics['object_count']} objects",
            flush=True,
        )
        return 0 if success else 1
    except Exception as exc:
        episode_dir.mkdir(parents=True, exist_ok=True)
        append_phase(phase_path, "scene_script_error", error_type=type(exc).__name__, error=str(exc))
        write_json(
            episode_dir / "metrics.json",
            {
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
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
