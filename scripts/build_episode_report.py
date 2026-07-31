#!/usr/bin/env python3
import argparse
import csv
import html
import json
import os
import re
from datetime import datetime
from pathlib import Path


BENCHMARK_ALIASES = {
    "robotsim_v1_release_candidate": "farpoint_v1_release_candidate",
}


def display_benchmark_id(value):
    if value is None:
        return None
    value = str(value)
    return BENCHMARK_ALIASES.get(value, value)


REPORT_NAV_CSS = """
    .app-nav {
      display: flex; align-items: center; justify-content: space-between; gap: 16px;
      padding: 10px 18px; background: #17252b; color: #ecf4f5; border-bottom: 1px solid #35464c;
      position: sticky; top: 0; z-index: 10;
    }
    .app-nav__crumbs, .app-nav__links { display: flex; align-items: center; gap: 8px; min-width: 0; }
    .app-nav a { color: #dcebed; text-decoration: none; font-size: 12px; font-weight: 650; white-space: nowrap; }
    .app-nav a:hover { color: white; text-decoration: underline; }
    .app-nav__current { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #9fb1b7; }
    .app-nav__sep { color: #70858b; }
    .app-nav__back { border: 1px solid #5a7077; border-radius: 4px; padding: 5px 8px; }
    @media (max-width: 680px) {
      .app-nav { align-items: flex-start; flex-direction: column; gap: 7px; overflow-x: auto; }
      .app-nav__links { min-width: max-content; }
    }
"""


def report_navigation(section, current):
    return f'''<nav class="app-nav" aria-label="Farpoint Data navigation">
      <div class="app-nav__crumbs">
        <a href="/">Farpoint Data</a><span class="app-nav__sep">/</span>
        <a href="/?view={html.escape(section)}">{html.escape(section.title())}</a><span class="app-nav__sep">/</span>
        <span class="app-nav__current" aria-current="page">{html.escape(current)}</span>
      </div>
      <div class="app-nav__links">
        <a href="/">Home</a><a href="/?view=episodes">Episodes</a><a href="/?view=benchmarks">Benchmarks</a>
        <a class="app-nav__back" href="/?view={html.escape(section)}" onclick="if (history.length > 1) {{ history.back(); return false; }}">Back</a>
      </div>
    </nav>'''


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_time(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def relpath(path, start):
    return Path(os.path.relpath(Path(path).resolve(), Path(start).resolve())).as_posix()


def read_trajectory(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path):
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path):
    if not path or not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def float_or_none(value):
    if value is None:
        return None
    value = str(value).strip().replace("%", "")
    if not value or value in {"[N/A]", "N/A"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_memory_mib(value):
    if not value:
        return None
    match = re.match(r"\s*([0-9.]+)\s*([KMGT]?i?B)\s*", value)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2)
    factors = {"B": 1 / (1024 * 1024), "KiB": 1 / 1024, "MiB": 1, "GiB": 1024, "TiB": 1024 * 1024}
    return amount * factors.get(unit, 1)


def find_matching_resource_summary(episode_dir, metadata):
    resources_dir = episode_dir.parent / "_resources"
    started_at = parse_time(metadata.get("started_at"))
    finished_at = parse_time(metadata.get("finished_at"))
    candidates = []

    for summary_path in sorted(resources_dir.glob("*_summary.json")):
        try:
            summary = read_json(summary_path)
        except json.JSONDecodeError:
            continue
        summary_start = parse_time(summary.get("started_at"))
        summary_end = parse_time(summary.get("finished_at"))
        if not summary_start or not started_at:
            continue

        if summary_end and finished_at and summary_start <= started_at <= summary_end:
            return summary_path

        delta = abs((summary_start - started_at).total_seconds())
        candidates.append((delta, summary_path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1] if candidates[0][0] < 300 else None


def find_log_for_summary(summary_path, episode_dir):
    if not summary_path:
        return None
    match = re.search(r"(.+)_(\d{8}_\d{6})_summary\.json$", summary_path.name)
    if not match:
        return None
    example_name = match.group(1)
    run_id = match.group(2)
    log_path = episode_dir.parent / "_logs" / f"{example_name}_{run_id}.log"
    return log_path if log_path.exists() else None


def run_id_from_summary(summary_path):
    if not summary_path:
        return None
    match = re.search(r"_(\d{8}_\d{6})_summary\.json$", summary_path.name)
    return match.group(1) if match else None


def example_name_from_summary(summary_path):
    if not summary_path:
        return None
    match = re.search(r"(.+)_(\d{8}_\d{6})_summary\.json$", summary_path.name)
    return match.group(1) if match else None


def find_runner_phase_files(summary_path, episode_dir):
    run_id = run_id_from_summary(summary_path)
    example_name = example_name_from_summary(summary_path)
    if not run_id or not example_name:
        return []
    phase_dir = episode_dir.parent / "_phases"
    return [
        path
        for path in [
            phase_dir / f"{example_name}_{run_id}_local_phase_events.jsonl",
            phase_dir / f"{example_name}_{run_id}_runner_phase_events.jsonl",
        ]
        if path.exists()
    ]


def read_phase_events(episode_dir, summary_path):
    events = []
    scene_phase_path = episode_dir / "phase_events.jsonl"
    for row in read_jsonl(scene_phase_path):
        row["_source"] = "scene"
        events.append(row)

    for phase_path in find_runner_phase_files(summary_path, episode_dir):
        source = "local runner" if "local" in phase_path.name else "remote runner"
        for row in read_jsonl(phase_path):
            row["_source"] = source
            events.append(row)

    return sorted(events, key=lambda row: row.get("time", ""))


def warning_lines(log_path, limit=30):
    if not log_path or not log_path.exists():
        return []
    lines = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        lower = line.lower()
        if "warning" in lower or "error" in lower or "fail" in lower:
            lines.append(line)
    return lines[-limit:]


def compact_resource_series(rows):
    series = {
        "labels": [],
        "gpuUtil": [],
        "gpuPower": [],
        "gpuTemp": [],
        "hostMem": [],
        "dockerMem": [],
    }
    for index, row in enumerate(rows):
        series["labels"].append(str(index))
        series["gpuUtil"].append(float_or_none(row.get("gpu_util_percent")))
        series["gpuPower"].append(float_or_none(row.get("gpu_power_w")))
        series["gpuTemp"].append(float_or_none(row.get("gpu_temp_c")))
        series["hostMem"].append(float_or_none(row.get("mem_used_mib")))
        docker_memory = row.get("docker_mem_usage", "").split("/", 1)[0]
        series["dockerMem"].append(parse_memory_mib(docker_memory))
    return series


def html_json(value):
    return html.escape(json.dumps(value, ensure_ascii=False), quote=False)


def format_number(value, suffix="", decimals=0):
    if value is None:
        return "Not matched"
    return f"{value:.{decimals}f}{suffix}"


def format_memory_mib(value):
    if value is None:
        return "Not matched"
    if abs(value) >= 1024:
        return f"{value / 1024:.1f} GiB"
    return f"{value:.0f} MiB"


def build_joint_series(trajectory):
    labels = [row.get("frame") for row in trajectory]
    datasets = []
    if trajectory and "joint_positions_degrees" in trajectory[0]:
        joint_names = trajectory[0].get("joint_names") or [
            f"joint_{index}"
            for index in range(len(trajectory[0].get("joint_positions_degrees", [])))
        ]
        colors = ["#0f766e", "#2563eb", "#b42318", "#b45309", "#7c3aed", "#0f5f8c"]
        for index, name in enumerate(joint_names):
            datasets.append(
                {
                    "label": f"{name} deg",
                    "values": [
                        row.get("joint_positions_degrees", [None] * len(joint_names))[index]
                        if index < len(row.get("joint_positions_degrees", []))
                        else None
                        for row in trajectory
                    ],
                    "color": colors[index % len(colors)],
                }
            )
        return {"labels": labels, "datasets": datasets}
    return {"labels": labels, "datasets": []}


def build_object_position_series(trajectory, object_name="pick_object"):
    labels = [row.get("frame") for row in trajectory]
    datasets = []
    axes = [("x", 0, "#0f766e"), ("y", 1, "#2563eb"), ("z", 2, "#b45309")]
    if trajectory and any(object_name in row.get("objects", {}) for row in trajectory):
        for axis, index, color in axes:
            datasets.append(
                {
                    "label": f"{object_name} {axis}",
                    "values": [
                        row.get("objects", {}).get(object_name, {}).get("position", [None, None, None])[index]
                        for row in trajectory
                    ],
                    "color": color,
                }
            )
        return {"labels": labels, "datasets": datasets}
    return {"labels": labels, "datasets": []}


def build_cube_series(trajectory):
    labels = [row.get("frame") for row in trajectory]
    if not trajectory or "cube_position" not in trajectory[0]:
        return {"labels": labels, "datasets": []}
    return {
        "labels": labels,
        "datasets": [
            {"label": "cube z", "values": [row["cube_position"][2] for row in trajectory], "color": "#0f766e"},
            {"label": "vertical velocity", "values": [row["cube_linear_velocity"][2] for row in trajectory], "color": "#b42318"},
        ],
    }


def build_frame_metadata(trajectory, preview_count, requested_frames=None):
    if not preview_count:
        return []
    if not trajectory:
        return [{"frame": index, "task_phase": "Not recorded"} for index in range(preview_count)]
    requested_frames = list(requested_frames or [])
    metadata = []
    for index in range(preview_count):
        if index < len(requested_frames):
            requested_frame = int(requested_frames[index])
            frame_fallback = requested_frame
            row = min(
                trajectory,
                key=lambda candidate: abs(int(candidate.get("frame", 0)) - requested_frame),
            )
        else:
            trajectory_index = round((index / max(1, preview_count - 1)) * (len(trajectory) - 1))
            frame_fallback = trajectory_index
            row = trajectory[trajectory_index]
        pick_object = row.get("objects", {}).get("pick_object", {})
        metadata.append(
            {
                "frame": row.get("frame", frame_fallback),
                "task_phase": row.get("task_phase", "Not recorded"),
                "pick_object_position": pick_object.get("position"),
                "pick_object_attached": pick_object.get("attached"),
                "pick_object_contact": pick_object.get("contact"),
                "grasp_proxy_position": row.get("objects", {}).get("grasp_proxy", {}).get("position"),
                "grasp_proxy_jaw_width": row.get("objects", {}).get("grasp_proxy", {}).get("jaw_width"),
            }
        )
    return metadata


def object_summary_html(metrics):
    if metrics.get("final_object_positions"):
        rows = "".join(
            f"<dt>{html.escape(name)}</dt><dd>{html.escape(json.dumps(position))}</dd>"
            for name, position in sorted(metrics["final_object_positions"].items())
        )
        return f"<dt>Objects</dt><dd>{html.escape(str(metrics.get('object_count', len(metrics['final_object_positions']))))}</dd>{rows}"
    return f"<dt>Final Cube Position</dt><dd>{html.escape(json.dumps(metrics.get('final_cube_position')))}</dd>"


def vector_distance(a, b):
    if not a or not b:
        return None
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5


def vector_path_length(points):
    valid_points = [point for point in points if point]
    if len(valid_points) < 2:
        return 0.0
    return sum(vector_distance(valid_points[index - 1], valid_points[index]) or 0.0 for index in range(1, len(valid_points)))


def joint_smoothness_score(trajectory):
    joint_rows = [row.get("joint_positions") for row in trajectory if row.get("joint_positions")]
    if len(joint_rows) < 3:
        return None
    scores = []
    for index in range(2, len(joint_rows)):
        previous = joint_rows[index - 2]
        current = joint_rows[index - 1]
        next_position = joint_rows[index]
        acceleration = [
            float(next_position[joint_index]) - (2 * float(current[joint_index])) + float(previous[joint_index])
            for joint_index in range(min(len(previous), len(current), len(next_position)))
        ]
        scores.append(sum(value * value for value in acceleration) ** 0.5)
    if not scores:
        return None
    return sum(scores) / len(scores)


def task_evaluation_from_trajectory(trajectory, metrics, success_criteria=None):
    existing = metrics.get("task_evaluation")
    if existing:
        return existing

    object_rows = [
        row.get("objects", {}).get("pick_object", {})
        for row in trajectory
        if row.get("objects", {}).get("pick_object")
    ]
    object_positions = [row.get("position") for row in object_rows if row.get("position")]
    end_effector_positions = [row.get("end_effector_position") for row in trajectory if row.get("end_effector_position")]
    attached_frames = [row for row in object_rows if row.get("attached")]
    phase_frame_counts = {}
    for trajectory_row in trajectory:
        object_row = trajectory_row.get("objects", {}).get("pick_object", {})
        phase = trajectory_row.get("task_phase") or object_row.get("phase", "unknown")
        phase_frame_counts[phase] = phase_frame_counts.get(phase, 0) + 1

    target = metrics.get("place_target_position") or metrics.get("pick_object_target_position")
    final_position = object_positions[-1] if object_positions else metrics.get("final_pick_object_position")
    settling_error = metrics.get("pick_place_distance")
    if settling_error is None:
        settling_error = metrics.get("final_target_xy_distance")
    if settling_error is None:
        settling_error = vector_distance(final_position, target)
    start_height = object_positions[0][2] if object_positions else None
    max_height = max((position[2] for position in object_positions), default=None)
    lift_height = (max_height - start_height) if max_height is not None and start_height is not None else None
    success_criteria = success_criteria or {}
    max_allowed_error = metrics.get("max_allowed_settling_error_m")
    if max_allowed_error is None:
        max_allowed_error = success_criteria.get("max_final_target_xy_distance", 0.08)

    if "final_object_inside_target_zone" in metrics:
        place_success = bool(
            final_position
            and metrics.get("final_object_inside_target_zone")
            and not metrics.get("final_object_attached", False)
            and settling_error is not None
            and float(settling_error) <= float(max_allowed_error)
        )
    else:
        place_success = bool(
            final_position
            and target
            and settling_error is not None
            and float(settling_error) <= float(max_allowed_error)
            and object_rows
            and not object_rows[-1].get("attached")
        )

    return {
        "schema_version": "task_evaluation.v1",
        "success": bool(metrics.get("success")),
        "pick_success": bool(attached_frames),
        "place_success": place_success,
        "settling_error_m": round(float(settling_error), 4) if settling_error is not None else None,
        "object_lift_height_m": round(lift_height, 4) if lift_height is not None else None,
        "object_path_length_m": round(vector_path_length(object_positions), 4),
        "end_effector_path_length_m": round(vector_path_length(end_effector_positions), 4),
        "joint_smoothness_score": round(joint_smoothness_score(trajectory), 6)
        if joint_smoothness_score(trajectory) is not None
        else None,
        "frames_with_object_attached": len(attached_frames),
        "frames_with_grasp_contact": sum(1 for row in object_rows if row.get("contact")),
        "grasp_attach_method": metrics.get("grasp_attach_method")
        or next((row.get("attach_method") for row in object_rows if row.get("attach_method")), None),
        "grasp_constraint": metrics.get("grasp_constraint")
        or next((row.get("grasp_constraint") for row in object_rows if row.get("grasp_constraint")), None),
        "max_attached_gripper_object_distance_m": metrics.get("max_grasp_rigidity_error")
        or metrics.get("max_attached_gripper_object_distance"),
        "mean_attached_gripper_object_distance_m": metrics.get("mean_attached_gripper_object_distance"),
        "phase_frame_counts": phase_frame_counts,
        "max_allowed_settling_error_m": max_allowed_error,
    }


def task_evaluation_html(evaluation):
    if not evaluation:
        return '<p class="subtle">No task evaluation was recorded for this episode.</p>'

    labels = [
        ("Overall Success", "PASS" if evaluation.get("success") else "FAIL"),
        ("Pick Success", "PASS" if evaluation.get("pick_success") else "FAIL"),
        ("Place Success", "PASS" if evaluation.get("place_success") else "FAIL"),
        ("Settling Error", f"{evaluation.get('settling_error_m'):.4f} m" if evaluation.get("settling_error_m") is not None else "Not recorded"),
        ("Allowed Error", f"{evaluation.get('max_allowed_settling_error_m'):.3f} m" if evaluation.get("max_allowed_settling_error_m") is not None else "Not recorded"),
        ("Object Lift Height", f"{evaluation.get('object_lift_height_m'):.4f} m" if evaluation.get("object_lift_height_m") is not None else "Not recorded"),
        ("Object Path Length", f"{evaluation.get('object_path_length_m'):.4f} m" if evaluation.get("object_path_length_m") is not None else "Not recorded"),
        ("End Effector Path Length", f"{evaluation.get('end_effector_path_length_m'):.4f} m" if evaluation.get("end_effector_path_length_m") is not None else "Not recorded"),
        ("Joint Smoothness Score", f"{evaluation.get('joint_smoothness_score'):.6f}" if evaluation.get("joint_smoothness_score") is not None else "Not recorded"),
        ("Attached Frames", str(evaluation.get("frames_with_object_attached", "Not recorded"))),
        ("Grasp Contact Frames", str(evaluation.get("frames_with_grasp_contact", "Not recorded"))),
        ("Grasp Attach Method", str(evaluation.get("grasp_attach_method", "Not recorded"))),
        ("Grasp Constraint", str(evaluation.get("grasp_constraint", "Not recorded"))),
        (
            "Max Gripper/Object Distance",
            f"{evaluation.get('max_attached_gripper_object_distance_m'):.4f} m"
            if evaluation.get("max_attached_gripper_object_distance_m") is not None
            else "Not recorded",
        ),
        (
            "Mean Gripper/Object Distance",
            f"{evaluation.get('mean_attached_gripper_object_distance_m'):.4f} m"
            if evaluation.get("mean_attached_gripper_object_distance_m") is not None
            else "Not recorded",
        ),
    ]
    rows = "\n".join(
        f"<dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd>"
        for label, value in labels
    )
    phase_counts = evaluation.get("phase_frame_counts") or {}
    phase_rows = "".join(
        f"<span>{html.escape(str(phase))}: {html.escape(str(count))}</span>"
        for phase, count in phase_counts.items()
    )
    return f"<dl>{rows}</dl><div class=\"phase-counts\">{phase_rows}</div>"


def phase_rows_html(events):
    if not events:
        return '<tr><td colspan="4" class="subtle">No phase events found for this episode.</td></tr>'

    first_time = parse_time(events[0].get("time"))
    rows = []
    for event in events:
        event_time = parse_time(event.get("time"))
        offset = ""
        if first_time and event_time:
            offset = f"+{(event_time - first_time).total_seconds():.2f}s"
        detail_fields = {
            key: value
            for key, value in event.items()
            if key not in {"time", "phase", "_source"}
        }
        details = json.dumps(detail_fields, sort_keys=True) if detail_fields else ""
        rows.append(
            "<tr>"
            f"<td>{html.escape(offset)}</td>"
            f"<td>{html.escape(event.get('_source', ''))}</td>"
            f"<td>{html.escape(event.get('phase', ''))}</td>"
            f"<td>{html.escape(details)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def event_time(events, source, phase):
    for event in events:
        if event.get("_source") == source and event.get("phase") == phase:
            return parse_time(event.get("time"))
    return None


def time_from_resource_row(row):
    return parse_time(row.get("timestamp"))


def phase_swimlane_html(events, resource_rows):
    timed_events = [
        (parse_time(event.get("time")), event)
        for event in events
        if parse_time(event.get("time")) is not None
    ]
    resource_times = [
        time_from_resource_row(row)
        for row in resource_rows
        if time_from_resource_row(row) is not None
    ]
    all_times = [time for time, _ in timed_events] + resource_times
    if not all_times:
        return '<p class="subtle">No phase events found for this episode.</p>'

    start = min(all_times)
    end = max(all_times)
    total_seconds = max((end - start).total_seconds(), 1.0)

    def pct(time):
        return max(0.0, min(100.0, ((time - start).total_seconds() / total_seconds) * 100))

    def offset_seconds(time):
        return max(0.0, (time - start).total_seconds())

    def tooltip_attrs(lane, label, start_time, end_time=None):
        start_offset = offset_seconds(start_time)
        if end_time:
            end_offset = offset_seconds(end_time)
            duration = max(0.0, (end_time - start_time).total_seconds())
            tooltip = f"{lane}: {label} | {start_offset:.1f}s to {end_offset:.1f}s | {duration:.1f}s"
        else:
            tooltip = f"{lane}: {label} | {start_offset:.1f}s"
        escaped = html.escape(tooltip, quote=True)
        return f'title="{escaped}" aria-label="{escaped}" data-tooltip="{escaped}"'

    def add_bar(lanes, lane, label, source, start_phase, end_phase, css_class=""):
        start_time = event_time(events, source, start_phase)
        end_time = event_time(events, source, end_phase)
        if not start_time or not end_time or end_time < start_time:
            return
        left = pct(start_time)
        width = max(0.6, pct(end_time) - left)
        attrs = tooltip_attrs(lane, label, start_time, end_time)
        lanes.setdefault(lane, []).append(
            f'<div class="phase-bar {css_class}" style="left:{left:.3f}%;width:{width:.3f}%;" {attrs}>'
            f"<span>{html.escape(label)}</span></div>"
        )

    def add_cross_bar(lanes, lane, label, start_source, start_phase, end_source, end_phase, css_class=""):
        start_time = event_time(events, start_source, start_phase)
        end_time = event_time(events, end_source, end_phase)
        if not start_time or not end_time or end_time < start_time:
            return
        left = pct(start_time)
        width = max(0.6, pct(end_time) - left)
        attrs = tooltip_attrs(lane, label, start_time, end_time)
        lanes.setdefault(lane, []).append(
            f'<div class="phase-bar {css_class}" style="left:{left:.3f}%;width:{width:.3f}%;" {attrs}>'
            f"<span>{html.escape(label)}</span></div>"
        )

    def add_tick(lanes, lane, label, source, phase, css_class=""):
        tick_time = event_time(events, source, phase)
        if not tick_time:
            return
        left = pct(tick_time)
        attrs = tooltip_attrs(lane, label, tick_time)
        lanes.setdefault(lane, []).append(
            f'<div class="phase-tick {css_class}" style="left:{left:.3f}%;" {attrs}>'
            f"<span>{html.escape(label)}</span></div>"
        )

    lanes = {}
    add_bar(lanes, "local runner", "remote prepare", "local runner", "remote_prepare_start", "remote_prepare_end")
    add_bar(lanes, "local runner", "project sync", "local runner", "project_sync_start", "project_sync_end")
    add_bar(lanes, "local runner", "remote example", "local runner", "remote_example_start", "remote_example_end", "long")
    add_bar(lanes, "local runner", "output sync", "local runner", "output_sync_start", "output_sync_end")

    add_cross_bar(lanes, "remote runner", "resource monitor", "remote runner", "resource_monitor_start", "remote runner", "resource_monitor_stop_start", "long")
    add_bar(lanes, "remote runner", "docker run", "remote runner", "docker_run_start", "docker_run_end", "long")
    add_bar(lanes, "remote runner", "resource summary", "remote runner", "resource_summary_start", "resource_summary_end")

    add_bar(lanes, "scene", "app startup", "scene", "simulation_app_start", "simulation_app_ready", "long")
    add_bar(lanes, "scene", "scene create", "scene", "scene_create_start", "scene_created")
    add_bar(lanes, "scene", "preview setup", "scene", "preview_writer_setup_start", "preview_writer_ready")
    add_bar(lanes, "scene", "physics + preview", "scene", "physics_recording_start", "physics_recording_end", "long")
    add_cross_bar(lanes, "scene", "app close / drain", "scene", "simulation_app_close_start", "remote runner", "docker_run_end", "drain")
    add_tick(lanes, "scene", "episode written", "scene", "episode_written", "important")
    add_tick(lanes, "grasp proxy", "contact", "scene", "grasp_proxy_contact", "important")
    add_tick(lanes, "grasp proxy", "attach", "scene", "grasp_proxy_attach", "important")
    add_tick(lanes, "grasp proxy", "release", "scene", "grasp_proxy_release", "important")

    manipulation_events = [
        event
        for event in events
        if event.get("_source") == "scene"
        and event.get("phase", "").startswith("manipulation_")
        and event.get("phase", "").endswith("_start")
        and parse_time(event.get("time")) is not None
    ]
    manipulation_events.sort(key=lambda event: parse_time(event.get("time")))
    physics_end = event_time(events, "scene", "physics_recording_end")
    for index, event in enumerate(manipulation_events):
        start_time = parse_time(event.get("time"))
        end_time = (
            parse_time(manipulation_events[index + 1].get("time"))
            if index + 1 < len(manipulation_events)
            else physics_end
        )
        if not start_time or not end_time or end_time < start_time:
            continue
        phase_name = event.get("phase", "").removeprefix("manipulation_").removesuffix("_start")
        left = pct(start_time)
        width = max(0.9, pct(end_time) - left)
        attrs = tooltip_attrs("manipulation", phase_name, start_time, end_time)
        lanes.setdefault("manipulation", []).append(
            f'<div class="phase-bar manipulation" style="left:{left:.3f}%;width:{width:.3f}%;" {attrs}>'
            f"<span>{html.escape(phase_name)}</span></div>"
        )

    gpu_items = []
    for index, row in enumerate(resource_rows):
        util = float_or_none(row.get("gpu_util_percent"))
        row_time = time_from_resource_row(row)
        if util is None or util < 80 or row_time is None:
            continue
        next_time = None
        for next_row in resource_rows[index + 1 :]:
            next_time = time_from_resource_row(next_row)
            if next_time:
                break
        previous_time = None
        for previous_row in reversed(resource_rows[:index]):
            previous_time = time_from_resource_row(previous_row)
            if previous_time:
                break
        if next_time and next_time > row_time:
            spike_end = next_time
        elif previous_time and row_time > previous_time:
            spike_end = row_time + (row_time - previous_time)
        else:
            spike_end = row_time
        left = pct(row_time)
        width = max(0.8, pct(spike_end) - left)
        label = f"GPU util {util:.0f}%"
        attrs = tooltip_attrs("GPU spikes", label, row_time, spike_end)
        gpu_items.append(
            f'<div class="phase-bar gpu-spike" style="left:{left:.3f}%;width:{width:.3f}%;" {attrs}>'
            f"<span>{util:.0f}%</span></div>"
        )
    lanes["GPU spikes"] = gpu_items

    axis_labels = [
        ("0s", 0),
        (f"{total_seconds / 2:.0f}s", 50),
        (f"{total_seconds:.0f}s", 100),
    ]
    axis_html = "".join(
        f'<span style="left:{position}%">{html.escape(label)}</span>' for label, position in axis_labels
    )
    lane_order = ["local runner", "remote runner", "scene", "manipulation", "grasp proxy", "GPU spikes"]
    lane_html = []
    for lane in lane_order:
        items = "".join(lanes.get(lane, [])) or '<span class="swimlane-empty">No events</span>'
        lane_html.append(
            '<div class="swimlane-row">'
            f'<div class="swimlane-label">{html.escape(lane)}</div>'
            f'<div class="swimlane-track">{items}</div>'
            "</div>"
        )

    return (
        '<div class="swimlane">'
        f'<div class="swimlane-axis"><div></div><div class="swimlane-axis-track">{axis_html}</div></div>'
        + "\n".join(lane_html)
        + "</div>"
    )


def build_report(episode_dir, output_dir, resource_summary_path=None):
    project_root = episode_dir.parents[2]
    metadata = read_json(episode_dir / "metadata.json")
    metrics = read_json(episode_dir / "metrics.json")
    trajectory = read_trajectory(episode_dir / "trajectory.jsonl")

    if resource_summary_path is None:
        resource_summary_path = find_matching_resource_summary(episode_dir, metadata)
    if resource_summary_path:
        resource_summary_path = Path(resource_summary_path)
        resource_summary = read_json(resource_summary_path)
        resource_csv = project_root / resource_summary["source_csv"]
        if not resource_csv.exists():
            resource_csv = resource_summary_path.parent / Path(resource_summary["source_csv"]).name
    else:
        resource_summary = None
        resource_csv = None

    resource_rows = read_csv(resource_csv) if resource_csv else []
    log_path = find_log_for_summary(resource_summary_path, episode_dir)
    phase_events = read_phase_events(episode_dir, resource_summary_path)
    warnings = warning_lines(log_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "index.html"

    preview_images = sorted((episode_dir / "preview").glob("*.png"))
    preview_rel = [relpath(path, output_dir) for path in preview_images]
    image_for_hero = preview_rel[min(len(preview_rel) // 2, len(preview_rel) - 1)] if preview_rel else ""
    preview_summary_rel = preview_rel[::max(1, len(preview_rel) // 6)][:6]
    frame_metadata = build_frame_metadata(
        trajectory,
        len(preview_rel),
        metrics.get("preview_frames_requested"),
    )

    joint_series = build_joint_series(trajectory)
    object_series = build_object_position_series(trajectory)
    fallback_series = build_cube_series(trajectory)
    task_evaluation = task_evaluation_from_trajectory(
        trajectory,
        metrics,
        metadata.get("success_criteria"),
    )
    task_evaluation_section = task_evaluation_html(task_evaluation)
    object_summary = object_summary_html(metrics)
    resource_series = compact_resource_series(resource_rows)
    workload_memory_mib = None
    host_memory_pressure_percent = None
    gpu_memory_status = "Not matched"
    if resource_summary:
        workload_memory_mib = resource_summary["container"].get("peak_memory_used_mib")
        host_used_mib = resource_summary["host"].get("peak_memory_used_mib")
        host_total_mib = resource_summary["host"].get("memory_total_mib")
        if host_used_mib is not None and host_total_mib:
            host_memory_pressure_percent = (host_used_mib / host_total_mib) * 100
        gpu_memory_status = (
            "available"
            if resource_summary["gpu"].get("memory_accounting_available")
            else "not exposed on GB10"
        )

    cards = [
        ("Status", "PASS" if metrics.get("success") else "FAIL"),
        ("Frames", f"{metrics.get('recorded_frames')} / {metrics.get('frames_requested')}"),
        ("Runtime", f"{metrics.get('elapsed_seconds', 0):.1f}s"),
        ("Preview Frames", str(metrics.get("preview_images_written", len(preview_images)))),
    ]
    if metrics.get("articulation_controller"):
        cards.extend(
            [
                ("Controller", "Articulation"),
                ("DOFs", str(metrics.get("articulation_dofs", "Not recorded"))),
                ("Controlled Joints", str(metrics.get("controlled_joints", "Not recorded"))),
            ]
        )
    if metrics.get("task_type"):
        task_type = str(metrics.get("task_type"))
        task_type_display = {
            "real_ur10e_robotiq_pick_and_place_v1": "UR10e + Robotiq Pick-and-Place v1",
            "randomized_ur10e_robotiq_pick_and_place_v2": "Randomized UR10e + Robotiq Pick-and-Place v2",
        }.get(task_type, task_type.replace("_", " "))
        cards.append(("Task Type", task_type_display))
    if metadata.get("episode_seed") is not None:
        cards.append(("Episode Seed", str(metadata["episode_seed"])))
    benchmark_id = display_benchmark_id(metadata.get("benchmark_id"))
    if benchmark_id:
        cards.append(("Benchmark", benchmark_id))
    if metrics.get("pick_place_distance") is not None:
        cards.append(("Pick Place Error", f"{metrics.get('pick_place_distance'):.3f} m"))
    elif metrics.get("final_target_xy_distance") is not None:
        cards.append(("Target Error", f"{metrics.get('final_target_xy_distance'):.3f} m"))
    if task_evaluation:
        if task_evaluation.get("object_lift_height_m") is not None:
            cards.append(("Object Lift", f"{task_evaluation.get('object_lift_height_m'):.3f} m"))
        if task_evaluation.get("object_path_length_m") is not None:
            cards.append(("Object Path", f"{task_evaluation.get('object_path_length_m'):.3f} m"))
        if task_evaluation.get("joint_smoothness_score") is not None:
            cards.append(("Smoothness", f"{task_evaluation.get('joint_smoothness_score'):.4f}"))
    if metrics.get("debug_visualization"):
        cards.append(("Debug Overlay", f"{metrics.get('debug_overlay_parts', 0)} parts"))
    if resource_summary:
        cards.extend(
            [
                ("Peak GPU", f"{resource_summary['gpu'].get('peak_util_percent')}%"),
                ("Peak Power", f"{resource_summary['gpu'].get('peak_power_w')} W"),
                ("Max Temp", f"{resource_summary['gpu'].get('max_temperature_c')} C"),
                ("Peak Workload Memory", format_memory_mib(workload_memory_mib)),
                ("Peak Host Memory Pressure", format_number(host_memory_pressure_percent, "%", 1)),
            ]
        )

    card_html = "\n".join(
        f'<section class="metric{" metric-long" if len(value) > 18 else ""}"><span>{html.escape(label)}</span>'
        f'<strong>{html.escape(value)}</strong></section>'
        for label, value in cards
    )
    warning_html = "\n".join(
        f"<li>{html.escape(line)}</li>" for line in warnings
    ) or "<li>No warnings were found in the matched run log.</li>"
    phases_html = phase_rows_html(phase_events)
    swimlane_html = phase_swimlane_html(phase_events, resource_rows)
    success_checks = metrics.get("success_checks", {})
    success_checks_text = ", ".join(
        f"{key}: {'pass' if value else 'fail'}" for key, value in success_checks.items()
    ) or "Not recorded"
    controller_summary = ""
    if metrics.get("articulation_controller"):
        controller_summary = (
            f"<dt>Controller</dt><dd>ArticulationController</dd>"
            f"<dt>Articulation DOFs</dt><dd>{html.escape(str(metrics.get('articulation_dofs', 'Not recorded')))}</dd>"
            f"<dt>Controlled Joints</dt><dd>{html.escape(', '.join(metrics.get('controlled_joint_names', [])))}</dd>"
        )
    debug_summary = ""
    if metrics.get("debug_visualization"):
        debug_summary = (
            f"<dt>Debug Overlay</dt><dd>{html.escape(str(metrics.get('debug_overlay_parts', 0)))} visual helpers</dd>"
            f"<dt>End Effector Source</dt><dd>{html.escape(str(metrics.get('debug_end_effector_prim_path') or 'fallback_fk'))}</dd>"
        )
    manipulation_summary = ""
    if metrics.get("task_type"):
        place_target = metrics.get("place_target_position") or metrics.get("pick_object_target_position")
        place_error = metrics.get("final_target_xy_distance")
        if place_error is None:
            place_error = metrics.get("pick_place_distance")
        manipulation_summary = (
            f"<dt>Task Type</dt><dd>{html.escape(str(metrics.get('task_type')))}</dd>"
            f"<dt>Pickup Mode</dt><dd>{html.escape(str(metrics.get('pickup_mode', 'Not recorded')))}</dd>"
            f"<dt>Grasp Method</dt><dd>{html.escape(str(metrics.get('grasp_attach_method', 'Not recorded')))}</dd>"
            f"<dt>Grasp Constraint</dt><dd>{html.escape(str(metrics.get('grasp_constraint', 'Not recorded')))}</dd>"
            f"<dt>Gripper Mount Parent</dt><dd>{html.escape(str(metrics.get('gripper_mount_parent', 'Not recorded')))}</dd>"
            f"<dt>Grasp Contact Frames</dt><dd>{html.escape(str(metrics.get('grasp_contact_frames', 'Not recorded')))}</dd>"
            f"<dt>Max Gripper/Object Distance</dt><dd>{html.escape(str(metrics.get('max_attached_gripper_object_distance', 'Not recorded')))} m</dd>"
            f"<dt>Task Phases</dt><dd>{html.escape(', '.join(metrics.get('task_phases', [])))}</dd>"
            f"<dt>Pick Object Start</dt><dd>{html.escape(json.dumps(metrics.get('pick_object_start_position')))}</dd>"
            f"<dt>Place Target</dt><dd>{html.escape(json.dumps(place_target))}</dd>"
            f"<dt>Final Target XY Error</dt><dd>{html.escape(str(place_error))} m</dd>"
            f"<dt>Attached Frames</dt><dd>{html.escape(str(metrics.get('object_attached_frames', 'Not recorded')))}</dd>"
        )
    benchmark_summary = ""
    if metadata.get("episode_seed") is not None:
        benchmark_id = display_benchmark_id(metadata.get("benchmark_id"))
        failure_reason = metrics.get("failure_reason") or "None"
        failure_category = metrics.get("failure_category") or "None"
        benchmark_summary = (
            f"<dt>Episode Seed</dt><dd>{html.escape(str(metadata.get('episode_seed')))}</dd>"
            f"<dt>Benchmark</dt><dd>{html.escape(str(benchmark_id or 'Standalone'))}</dd>"
            f"<dt>Benchmark Run</dt><dd>{html.escape(str(int(metadata.get('benchmark_repeat', 0)) + 1))}</dd>"
            f"<dt>Failure Stage</dt><dd>{html.escape(str(failure_category))}</dd>"
            f"<dt>Failure Reason</dt><dd>{html.escape(str(failure_reason).replace('_', ' '))}</dd>"
        )

    resource_note = ""
    if resource_summary and not resource_summary["gpu"].get("memory_accounting_available"):
        resource_note = (
            resource_summary["gpu"].get("memory_note")
            or "GPU memory accounting is unavailable on this platform."
        )
        resource_note += " Memory load is reported through unified host memory and Docker/cgroup workload memory."

    report_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(metadata["episode_id"])} | Farpoint</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #5b6873;
      --line: #d9e2e8;
      --panel: #ffffff;
      --bg: #f4f7f9;
      --accent: #0f766e;
      --accent-2: #b42318;
    }}
{REPORT_NAV_CSS}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 46vw);
      gap: 28px;
      align-items: center;
      padding: 32px;
      background: #0f1f24;
      color: #f8fbfc;
    }}
    h1, h2 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 34px; line-height: 1.1; }}
    h2 {{ font-size: 20px; margin-bottom: 14px; }}
    .subtle {{ color: var(--muted); }}
    header .subtle {{ color: #b9c7cf; }}
    .hero-image {{
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      border: 1px solid rgba(255,255,255,.18);
      background: #0a1418;
    }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .metric, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric {{ min-width: 0; padding: 14px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 4px; overflow-wrap: anywhere; font-size: 20px; line-height: 1.2; }}
    .metric-long strong {{ font-size: 15px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .panel {{ padding: 16px; overflow: hidden; }}
    .panel.wide {{ grid-column: 1 / -1; }}
    canvas {{ width: 100%; height: 260px; display: block; }}
    dl {{ display: grid; grid-template-columns: 170px minmax(0, 1fr); gap: 8px 14px; margin: 0; }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; word-break: break-word; }}
    .preview-strip {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }}
    .preview-strip img {{ width: 100%; aspect-ratio: 16 / 9; object-fit: cover; border: 1px solid var(--line); }}
    .frame-player {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 16px;
      align-items: start;
      margin-bottom: 16px;
    }}
    .player-stage {{
      background: #0a1418;
      border: 1px solid var(--line);
      aspect-ratio: 16 / 9;
      display: grid;
      place-items: center;
      overflow: hidden;
    }}
    .player-stage img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }}
    .player-controls {{
      display: grid;
      gap: 12px;
    }}
    .player-controls button {{
      border: 1px solid #0f766e;
      background: #0f766e;
      color: #fff;
      border-radius: 8px;
      padding: 10px 14px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    .player-controls input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .player-readout {{
      color: var(--muted);
      font-size: 13px;
    }}
    .phase-pill {{
      display: inline-flex;
      align-items: center;
      width: fit-content;
      min-height: 28px;
      padding: 4px 10px;
      border-radius: 999px;
      background: #e6f4f1;
      color: #0f766e;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .04em;
      font-size: 12px;
    }}
    .warning-list {{ max-height: 280px; overflow: auto; padding-left: 18px; }}
    .phase-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .phase-table th, .phase-table td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .phase-table th {{ color: var(--muted); font-size: 12px; }}
    .phase-table td:last-child {{ word-break: break-word; }}
    .phase-details {{
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdfe;
    }}
    .phase-details summary {{
      cursor: pointer;
      padding: 10px 12px;
      font-weight: 700;
      color: #32414a;
      user-select: none;
    }}
    .phase-details summary::marker {{ color: var(--accent); }}
    .phase-table-wrap {{
      max-height: 320px;
      overflow: auto;
      border-top: 1px solid var(--line);
    }}
    .phase-table th {{
      position: sticky;
      top: 0;
      background: #eef4f6;
      z-index: 1;
    }}
    .phase-counts {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }}
    .phase-counts span {{
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 3px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #f7fafb;
      color: #32414a;
      font-size: 12px;
      font-weight: 700;
    }}
    .swimlane {{
      --timeline-width: 960px;
      display: grid;
      gap: 10px;
      margin-bottom: 18px;
      overflow-x: auto;
      padding-bottom: 4px;
    }}
    .swimlane-axis,
    .swimlane-row {{
      display: grid;
      grid-template-columns: 130px minmax(var(--timeline-width), 1fr);
      gap: 12px;
      align-items: center;
    }}
    .timeline-toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin: 2px 0 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    .timeline-zoom {{
      display: grid;
      grid-template-columns: auto 220px auto;
      gap: 8px;
      align-items: center;
      min-width: 330px;
    }}
    .timeline-zoom label {{
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: .04em;
    }}
    .timeline-zoom input {{
      width: 100%;
      accent-color: var(--accent);
    }}
    #timelineZoomValue {{
      min-width: 46px;
      text-align: right;
      color: #32414a;
      font-weight: 700;
    }}
    .swimlane-axis-track,
    .swimlane-track {{
      position: relative;
      min-height: 42px;
      border-left: 1px solid var(--line);
      border-right: 1px solid var(--line);
      background:
        linear-gradient(to right, rgba(23,32,38,.07) 1px, transparent 1px) 0 0 / 25% 100%,
        #f7fafb;
    }}
    .swimlane-axis-track {{
      min-height: 22px;
      background: transparent;
      border-color: transparent;
    }}
    .swimlane-axis-track span {{
      position: absolute;
      top: 0;
      transform: translateX(-50%);
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .swimlane-label {{
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: .04em;
    }}
    .phase-bar {{
      position: absolute;
      top: 9px;
      height: 24px;
      min-width: 7px;
      border-radius: 6px;
      background: #0f766e;
      color: #fff;
      overflow: visible;
      white-space: nowrap;
      box-shadow: 0 1px 2px rgba(15, 31, 36, .16);
      z-index: 2;
    }}
    .phase-bar:hover,
    .phase-bar:focus {{
      z-index: 8;
    }}
    .phase-bar::after,
    .phase-tick::after {{
      content: attr(data-tooltip);
      position: absolute;
      left: 50%;
      bottom: calc(100% + 8px);
      transform: translateX(-50%);
      width: max-content;
      max-width: min(360px, 72vw);
      padding: 7px 9px;
      border-radius: 6px;
      background: #172026;
      color: #fff;
      font-size: 12px;
      font-weight: 700;
      line-height: 1.35;
      white-space: normal;
      box-shadow: 0 8px 22px rgba(15, 31, 36, .22);
      opacity: 0;
      pointer-events: none;
      visibility: hidden;
      transition: opacity .12s ease, visibility .12s ease;
      z-index: 20;
    }}
    .phase-bar::before,
    .phase-tick::before {{
      content: "";
      position: absolute;
      left: 50%;
      bottom: calc(100% + 3px);
      transform: translateX(-50%);
      border: 5px solid transparent;
      border-top-color: #172026;
      opacity: 0;
      pointer-events: none;
      visibility: hidden;
      transition: opacity .12s ease, visibility .12s ease;
      z-index: 21;
    }}
    .phase-bar:hover::after,
    .phase-bar:hover::before,
    .phase-bar:focus::after,
    .phase-bar:focus::before,
    .phase-tick:hover::after,
    .phase-tick:hover::before,
    .phase-tick:focus::after,
    .phase-tick:focus::before {{
      opacity: 1;
      visibility: visible;
    }}
    .phase-bar.long {{ background: #2563eb; }}
    .phase-bar.drain {{ background: #b45309; }}
    .phase-bar.gpu-spike {{ background: #b42318; }}
    .phase-bar.manipulation {{ background: #7c3aed; }}
    .phase-bar span {{
      display: block;
      height: 24px;
      padding: 3px 7px;
      border-radius: 6px;
      font-size: 12px;
      line-height: 18px;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .phase-tick {{
      position: absolute;
      top: 5px;
      width: 2px;
      height: 32px;
      background: #172026;
    }}
    .phase-tick span {{
      position: absolute;
      left: 6px;
      top: 2px;
      max-width: 140px;
      color: #172026;
      background: rgba(255,255,255,.88);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 2px 5px;
      font-size: 12px;
      white-space: nowrap;
    }}
    .phase-tick.important {{ background: #0f766e; }}
    .swimlane-empty {{
      position: absolute;
      left: 10px;
      top: 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    .note {{ margin-top: 10px; color: var(--accent-2); }}
    @media (max-width: 860px) {{
      header {{ grid-template-columns: 1fr; padding: 24px; }}
      h1 {{ font-size: 28px; }}
      .grid {{ grid-template-columns: 1fr; }}
      .frame-player {{ grid-template-columns: 1fr; }}
      dl {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  {report_navigation("episodes", metadata["episode_id"])}
  <header>
    <div>
      <p class="subtle">Farpoint Episode Report</p>
      <h1>{html.escape(metadata["episode_id"])}</h1>
      <p>{html.escape(metadata.get("language_instruction", ""))}</p>
      <p class="subtle">{html.escape(metadata.get("simulator", ""))} · {html.escape(metadata.get("image", ""))}</p>
    </div>
    {'<img class="hero-image" src="' + html.escape(image_for_hero) + '" alt="Simulation preview">' if image_for_hero else '<div class="hero-image"></div>'}
  </header>
  <main>
    <div class="metrics">{card_html}</div>
    <div class="grid">
      <section class="panel">
        <h2>Episode</h2>
        <dl>
          <dt>Task</dt><dd>{html.escape(metadata.get("task_name", ""))}</dd>
          <dt>Task Schema</dt><dd>{html.escape(metadata.get("task_schema_version", "Not recorded"))}</dd>
          <dt>Started</dt><dd>{html.escape(metadata.get("started_at", ""))}</dd>
          <dt>Finished</dt><dd>{html.escape(metadata.get("finished_at", ""))}</dd>
          {object_summary}
          {benchmark_summary}
          {manipulation_summary}
          {controller_summary}
          {debug_summary}
          <dt>Success Checks</dt><dd>{html.escape(success_checks_text)}</dd>
          <dt>Preview Resolution</dt><dd>{html.escape(json.dumps(metadata.get("preview_resolution")))}</dd>
        </dl>
      </section>
      <section class="panel">
        <h2>Resource Summary</h2>
        <dl>
          <dt>GPU</dt><dd>{html.escape((resource_summary or {}).get("gpu", {}).get("name") or "Not matched")}</dd>
          <dt>Samples</dt><dd>{html.escape(str((resource_summary or {}).get("sample_count", "Not matched")))}</dd>
          <dt>Peak Workload Memory</dt><dd>{html.escape(format_memory_mib(workload_memory_mib))}</dd>
          <dt>Peak Host Memory Pressure</dt><dd>{html.escape(format_number(host_memory_pressure_percent, "%", 1))}</dd>
          <dt>nvidia-smi Memory</dt><dd>{html.escape(gpu_memory_status)}</dd>
          <dt>Resource CSV</dt><dd>{html.escape(str(resource_csv) if resource_csv else "Not matched")}</dd>
          <dt>Run Log</dt><dd>{html.escape(str(log_path) if log_path else "Not matched")}</dd>
        </dl>
        {f'<p class="note">{html.escape(resource_note)}</p>' if resource_note else ''}
      </section>
      <section class="panel wide">
        <h2>Task Evaluation</h2>
        {task_evaluation_section}
      </section>
      <section class="panel wide">
        <h2>Frame Playback</h2>
        <div class="frame-player">
          <div class="player-stage">
            {'<img id="playerFrame" src="' + html.escape(preview_rel[0]) + '" alt="Simulation playback frame">' if preview_rel else '<div class="subtle">No preview frames found.</div>'}
          </div>
          <div class="player-controls">
            <button id="playButton" type="button">Play</button>
            <input id="frameSlider" type="range" min="0" max="{max(0, len(preview_rel) - 1)}" value="0" step="1">
            <div class="player-readout">
              <strong id="frameCounter">Frame 1 / {len(preview_rel)}</strong><br>
              <span id="phaseReadout" class="phase-pill">Not recorded</span><br>
              <span id="objectReadout">{len(preview_rel)} PNG frames from the Isaac Sim preview camera.</span>
            </div>
          </div>
        </div>
        <h2>Preview Frames</h2>
        <div class="preview-strip">
          {''.join(f'<img src="{html.escape(path)}" alt="Preview frame">' for path in preview_summary_rel)}
        </div>
      </section>
      <section class="panel">
        <h2>Joint Motion</h2>
        <canvas id="jointChart" width="900" height="320"></canvas>
      </section>
      <section class="panel">
        <h2>Pick Object Position</h2>
        <canvas id="objectChart" width="900" height="320"></canvas>
      </section>
      <section class="panel">
        <h2>GPU Load</h2>
        <canvas id="gpuChart" width="900" height="320"></canvas>
      </section>
      <section class="panel">
        <h2>Unified Memory Pressure</h2>
        <canvas id="memoryChart" width="900" height="320"></canvas>
      </section>
      <section class="panel wide">
        <h2>Phase Timeline</h2>
        <div class="timeline-toolbar">
          <span>Drag to stretch short phases, then scroll horizontally.</span>
          <div class="timeline-zoom">
            <label for="timelineZoom">Zoom</label>
            <input id="timelineZoom" type="range" min="720" max="2400" value="960" step="120">
            <span id="timelineZoomValue">100%</span>
          </div>
        </div>
        {swimlane_html}
        <details class="phase-details">
          <summary>Show detailed phase events ({len(phase_events)})</summary>
          <div class="phase-table-wrap">
            <table class="phase-table">
              <thead>
                <tr>
                  <th>Offset</th>
                  <th>Source</th>
                  <th>Phase</th>
                  <th>Details</th>
                </tr>
              </thead>
              <tbody>
                {phases_html}
              </tbody>
            </table>
          </div>
        </details>
      </section>
      <section class="panel">
        <h2>Warnings</h2>
        <ul class="warning-list">{warning_html}</ul>
      </section>
    </div>
  </main>
  <script>
    const joints = {html_json(joint_series)};
    const objectMotion = {html_json(object_series)};
    const fallbackMotion = {html_json(fallback_series)};
    const resources = {html_json(resource_series)};
    const previewFrames = {html_json(preview_rel)};
    const frameMetadata = {html_json(frame_metadata)};

    function setupFramePlayer() {{
      const image = document.getElementById('playerFrame');
      const button = document.getElementById('playButton');
      const slider = document.getElementById('frameSlider');
      const counter = document.getElementById('frameCounter');
      const phaseReadout = document.getElementById('phaseReadout');
      const objectReadout = document.getElementById('objectReadout');
      if (!image || !button || !slider || !counter || !previewFrames.length) return;

      let frame = 0;
      let timer = null;
      const frameMs = 1000 / 12;

      function render(nextFrame) {{
        frame = Math.max(0, Math.min(previewFrames.length - 1, nextFrame));
        image.src = previewFrames[frame];
        slider.value = String(frame);
        counter.textContent = `Frame ${{frame + 1}} / ${{previewFrames.length}}`;
        const metadata = frameMetadata[frame] || {{}};
        if (phaseReadout) phaseReadout.textContent = metadata.task_phase || 'Not recorded';
        if (objectReadout) {{
          const pos = metadata.pick_object_position;
          const attached = metadata.pick_object_attached === true ? 'attached' : 'released';
          const contact = metadata.pick_object_contact === true ? 'contact' : 'no contact';
          const jaw = typeof metadata.grasp_proxy_jaw_width === 'number'
            ? ` · jaw ${{metadata.grasp_proxy_jaw_width.toFixed(2)}} m`
            : '';
          objectReadout.textContent = Array.isArray(pos)
            ? `pick_object ${{attached}} · ${{contact}}${{jaw}} · [${{pos.map(v => Number(v).toFixed(2)).join(', ')}}]`
            : `${{previewFrames.length}} PNG frames from the Isaac Sim preview camera.`;
        }}
      }}

      function stop(ended = false) {{
        if (timer !== null) {{
          window.clearInterval(timer);
          timer = null;
        }}
        button.textContent = ended ? 'Replay' : 'Play';
      }}

      function play() {{
        if (timer !== null) {{
          stop();
          return;
        }}
        if (frame >= previewFrames.length - 1) render(0);
        button.textContent = 'Pause';
        timer = window.setInterval(() => {{
          const nextFrame = frame + 1;
          if (nextFrame >= previewFrames.length) {{
            render(previewFrames.length - 1);
            stop(true);
            return;
          }}
          render(nextFrame);
        }}, frameMs);
      }}

      button.addEventListener('click', play);
      slider.addEventListener('input', () => {{
        stop();
        render(Number(slider.value));
      }});
      previewFrames.forEach(src => {{
        const preload = new Image();
        preload.src = src;
      }});
      render(0);
    }}

    function setupTimelineZoom() {{
      const swimlane = document.querySelector('.swimlane');
      const slider = document.getElementById('timelineZoom');
      const value = document.getElementById('timelineZoomValue');
      if (!swimlane || !slider || !value) return;

      const baseWidth = 960;
      const storageKey = 'farpointTimelineWidth';
      const saved = Number(window.localStorage.getItem(storageKey));
      if (Number.isFinite(saved) && saved >= Number(slider.min) && saved <= Number(slider.max)) {{
        slider.value = String(saved);
      }}

      function renderTimelineZoom() {{
        const width = Number(slider.value);
        swimlane.style.setProperty('--timeline-width', `${{width}}px`);
        value.textContent = `${{Math.round((width / baseWidth) * 100)}}%`;
        window.localStorage.setItem(storageKey, String(width));
      }}

      slider.addEventListener('input', renderTimelineZoom);
      renderTimelineZoom();
    }}

    function drawChart(id, datasets, options = {{}}) {{
      const canvas = document.getElementById(id);
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      const pad = {{ left: 52, right: 18, top: 18, bottom: 34 }};
      ctx.clearRect(0, 0, width, height);
      ctx.strokeStyle = '#d9e2e8';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, height - pad.bottom);
      ctx.lineTo(width - pad.right, height - pad.bottom);
      ctx.stroke();

      const values = datasets.flatMap(ds => ds.values).filter(v => typeof v === 'number' && Number.isFinite(v));
      if (!values.length) {{
        ctx.fillStyle = '#5b6873';
        ctx.fillText('No data', pad.left + 12, pad.top + 24);
        return;
      }}
      let min = options.min ?? Math.min(...values);
      let max = options.max ?? Math.max(...values);
      if (min === max) {{ min -= 1; max += 1; }}
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      ctx.fillStyle = '#5b6873';
      ctx.fillText(max.toFixed(1), 8, pad.top + 4);
      ctx.fillText(min.toFixed(1), 8, height - pad.bottom);

      datasets.forEach(ds => {{
        ctx.strokeStyle = ds.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started = false;
        ds.values.forEach((value, index) => {{
          if (typeof value !== 'number' || !Number.isFinite(value)) return;
          const x = pad.left + (index / Math.max(1, ds.values.length - 1)) * plotW;
          const y = pad.top + (1 - ((value - min) / (max - min))) * plotH;
          if (!started) {{ ctx.moveTo(x, y); started = true; }}
          else ctx.lineTo(x, y);
        }});
        ctx.stroke();
      }});

      let legendX = pad.left;
      datasets.forEach(ds => {{
        ctx.fillStyle = ds.color;
        ctx.fillRect(legendX, height - 18, 10, 10);
        ctx.fillStyle = '#172026';
        ctx.fillText(ds.label, legendX + 14, height - 9);
        legendX += ctx.measureText(ds.label).width + 46;
      }});
    }}

    drawChart('jointChart', joints.datasets);
    drawChart('objectChart', objectMotion.datasets.length ? objectMotion.datasets : fallbackMotion.datasets);
    drawChart('gpuChart', [
      {{ label: 'gpu util %', values: resources.gpuUtil, color: '#0f766e' }},
      {{ label: 'power W', values: resources.gpuPower, color: '#7c3aed' }},
      {{ label: 'temp C', values: resources.gpuTemp, color: '#b45309' }}
    ], {{ min: 0 }});
    drawChart('memoryChart', [
      {{ label: 'host MiB', values: resources.hostMem, color: '#0f766e' }},
      {{ label: 'container MiB', values: resources.dockerMem, color: '#2563eb' }}
    ], {{ min: 0 }});
    setupFramePlayer();
    setupTimelineZoom();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Build a static Farpoint episode report.")
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resource-summary", type=Path)
    args = parser.parse_args()

    episode_dir = args.episode_dir.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = episode_dir.parents[1] / "reports" / episode_dir.name

    report_path = build_report(episode_dir, output_dir.resolve(), args.resource_summary)
    print(f"Episode report written: {report_path}")


if __name__ == "__main__":
    main()
