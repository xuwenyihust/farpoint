#!/usr/bin/env python3
import argparse
import html
import sys
from pathlib import Path

from build_episode_report import (
    build_report,
    find_log_for_summary,
    find_matching_resource_summary,
    format_memory_mib,
    format_number,
    read_json,
    read_trajectory,
    relpath,
    task_evaluation_from_trajectory,
    warning_lines,
)


def episode_dirs(episodes_root):
    return sorted(
        path
        for path in episodes_root.glob("episode_*")
        if is_complete_episode(path)
    )


def is_complete_episode(path):
    if not path.is_dir():
        return False
    required_files = [
        path / "metadata.json",
        path / "metrics.json",
        path / "trajectory.jsonl",
        path / "phase_events.jsonl",
    ]
    if not all(file_path.exists() for file_path in required_files):
        return False
    return any((path / "preview").glob("*.png"))


def middle_preview(episode_dir, reports_root):
    images = sorted((episode_dir / "preview").glob("*.png"))
    if not images:
        return ""
    image = images[min(len(images) // 2, len(images) - 1)]
    return relpath(image, reports_root)


def number_or_none(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_run_row(episode_dir, reports_root, rebuild_reports):
    metadata = read_json(episode_dir / "metadata.json")
    metrics = read_json(episode_dir / "metrics.json")
    trajectory = read_trajectory(episode_dir / "trajectory.jsonl")
    task_evaluation = task_evaluation_from_trajectory(trajectory, metrics)
    summary_path = find_matching_resource_summary(episode_dir, metadata)
    summary = read_json(summary_path) if summary_path else None
    log_path = find_log_for_summary(summary_path, episode_dir)
    warnings = warning_lines(log_path, limit=1000)

    report_dir = reports_root / episode_dir.name
    if rebuild_reports:
        build_report(episode_dir, report_dir, summary_path)

    workload_memory_mib = None
    host_memory_pressure_percent = None
    peak_gpu = None
    if summary:
        workload_memory_mib = summary["container"].get("peak_memory_used_mib")
        host_used_mib = summary["host"].get("peak_memory_used_mib")
        host_total_mib = summary["host"].get("memory_total_mib")
        if host_used_mib is not None and host_total_mib:
            host_memory_pressure_percent = (host_used_mib / host_total_mib) * 100
        peak_gpu = summary["gpu"].get("peak_util_percent")

    return {
        "episode_id": metadata.get("episode_id", episode_dir.name),
        "task_name": metadata.get("task_name", ""),
        "started_at": metadata.get("started_at", ""),
        "success": bool(metrics.get("success")),
        "controller": "Articulation" if metrics.get("articulation_controller") else "Kinematic proxy" if metrics.get("kinematic_proxy") else "Unknown",
        "runtime": metrics.get("elapsed_seconds"),
        "frames": f"{metrics.get('recorded_frames')} / {metrics.get('frames_requested')}",
        "preview": middle_preview(episode_dir, reports_root),
        "report": relpath(report_dir / "index.html", reports_root),
        "peak_gpu": peak_gpu,
        "workload_memory": workload_memory_mib,
        "host_pressure": host_memory_pressure_percent,
        "warning_count": len(warnings),
        "settling_error": number_or_none(task_evaluation.get("settling_error_m")),
        "object_lift": number_or_none(task_evaluation.get("object_lift_height_m")),
        "object_path": number_or_none(task_evaluation.get("object_path_length_m")),
        "end_effector_path": number_or_none(task_evaluation.get("end_effector_path_length_m")),
        "smoothness": number_or_none(task_evaluation.get("joint_smoothness_score")),
    }


def sort_rows(rows):
    return sorted(rows, key=lambda row: row["started_at"], reverse=True)


def td(value, class_name=""):
    attr = f' class="{class_name}"' if class_name else ""
    return f"<td{attr}>{html.escape(str(value))}</td>"


def raw_td(value, class_name=""):
    attr = f' class="{class_name}"' if class_name else ""
    return f"<td{attr}>{value}</td>"


def benchmark_cards(episodes_root, reports_root):
    benchmarks_root = episodes_root.parent / "benchmarks"
    cards = []
    for summary_path in sorted(
        benchmarks_root.glob("*/summary.json"),
        key=lambda path: path.parent.name,
        reverse=True,
    ):
        try:
            summary = read_json(summary_path)
        except (OSError, ValueError):
            continue
        benchmark_id = summary.get("benchmark_id", summary_path.parent.name)
        report_path = reports_root / "benchmarks" / benchmark_id / "index.html"
        if not report_path.exists():
            continue
        status = "PASS" if summary.get("accepted") else "IN PROGRESS / FAIL"
        status_class = "pass" if summary.get("accepted") else "fail"
        cards.append(
            '<a class="benchmark-card" '
            f'href="benchmarks/{html.escape(benchmark_id)}/index.html">'
            f'<span class="{status_class}">{status}</span>'
            f"<strong>{html.escape(benchmark_id)}</strong>"
            f'<small>{int(summary.get("passed_trials", 0))} / '
            f'{int(summary.get("completed_trials", 0))} passed '
            f'({float(summary.get("success_rate", 0.0)):.1%})</small>'
            f'<small>P95 target error: '
            f'{format_number(summary.get("p95_target_xy_error"), " m", 4)}</small>'
            "</a>"
        )
    return "\n".join(cards)


def build_index(episodes_root, reports_root, rebuild_reports=True):
    reports_root.mkdir(parents=True, exist_ok=True)
    rows = [
        load_run_row(episode_dir, reports_root, rebuild_reports)
        for episode_dir in episode_dirs(episodes_root)
    ]
    rows = sort_rows(rows)

    passed = sum(1 for row in rows if row["success"])
    failed = len(rows) - passed
    peak_gpu = max((row["peak_gpu"] for row in rows if row["peak_gpu"] is not None), default=None)
    peak_memory = max(
        (row["workload_memory"] for row in rows if row["workload_memory"] is not None),
        default=None,
    )

    table_rows = []
    for row in rows:
        status = "PASS" if row["success"] else "FAIL"
        status_class = "pass" if row["success"] else "fail"
        task_name = row["task_name"] or "unknown"
        controller = row["controller"] or "Unknown"
        preview = (
            f'<img class="thumb" src="{html.escape(row["preview"])}" alt="Preview">'
            if row["preview"]
            else '<div class="thumb missing">No preview</div>'
        )
        table_rows.append(
            f'<tr data-task="{html.escape(task_name)}" '
            f'data-status="{html.escape(status)}" '
            f'data-controller="{html.escape(controller)}" '
            f'data-started="{html.escape(str(row["started_at"]))}" '
            f'data-runtime="{html.escape(str(row["runtime"] if row["runtime"] is not None else ""))}" '
            f'data-settling-error="{html.escape(str(row["settling_error"] if row["settling_error"] is not None else ""))}" '
            f'data-object-lift="{html.escape(str(row["object_lift"] if row["object_lift"] is not None else ""))}" '
            f'data-object-path="{html.escape(str(row["object_path"] if row["object_path"] is not None else ""))}" '
            f'data-end-effector-path="{html.escape(str(row["end_effector_path"] if row["end_effector_path"] is not None else ""))}" '
            f'data-smoothness="{html.escape(str(row["smoothness"] if row["smoothness"] is not None else ""))}" '
            f'data-peak-gpu="{html.escape(str(row["peak_gpu"] if row["peak_gpu"] is not None else ""))}" '
            f'data-workload-memory="{html.escape(str(row["workload_memory"] if row["workload_memory"] is not None else ""))}">'
            + raw_td(preview, "preview-cell")
            + raw_td(
                f'<a href="{html.escape(row["report"])}">{html.escape(row["episode_id"])}</a>',
                "episode-cell",
            )
            + td(status, status_class)
            + td(task_name)
            + td(controller)
            + td(f"{row['runtime']:.1f}s" if row["runtime"] is not None else "Not matched")
            + td(row["frames"])
            + td(f"{row['settling_error']:.4f} m" if row["settling_error"] is not None else "Not matched")
            + td(f"{row['object_lift']:.4f} m" if row["object_lift"] is not None else "Not matched")
            + td(f"{row['object_path']:.4f} m" if row["object_path"] is not None else "Not matched")
            + td(f"{row['end_effector_path']:.4f} m" if row["end_effector_path"] is not None else "Not matched")
            + td(f"{row['smoothness']:.6f}" if row["smoothness"] is not None else "Not matched")
            + td(format_number(row["peak_gpu"], "%", 0))
            + td(format_memory_mib(row["workload_memory"]))
            + td(format_number(row["host_pressure"], "%", 1))
            + td(row["warning_count"])
            + "</tr>"
        )

    table_html = "\n".join(table_rows)
    task_names = sorted({row["task_name"] or "unknown" for row in rows})
    task_options = "\n".join(
        f'<option value="{html.escape(name)}">{html.escape(name)}</option>' for name in task_names
    )
    controller_names = sorted({row["controller"] or "Unknown" for row in rows})
    controller_options = "\n".join(
        f'<option value="{html.escape(name)}">{html.escape(name)}</option>' for name in controller_names
    )
    best_settling_error = min(
        (row["settling_error"] for row in rows if row["settling_error"] is not None),
        default=None,
    )
    best_smoothness = min(
        (row["smoothness"] for row in rows if row["smoothness"] is not None),
        default=None,
    )
    benchmarks_html = benchmark_cards(episodes_root, reports_root)
    benchmarks_section = (
        f'<section class="benchmarks"><h2>Benchmark Batches</h2>'
        f'<div class="benchmark-grid">{benchmarks_html}</div></section>'
        if benchmarks_html
        else ""
    )

    index_path = reports_root / "index.html"
    index_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Farpoint Episode Index</title>
  <style>
    :root {{
      --ink: #172026;
      --muted: #5b6873;
      --line: #d9e2e8;
      --panel: #ffffff;
      --bg: #f4f7f9;
      --pass: #0f766e;
      --fail: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 28px 32px;
      background: #0f1f24;
      color: #f8fbfc;
    }}
    h1 {{ margin: 0; font-size: 34px; line-height: 1.1; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .subtle {{ color: var(--muted); }}
    header .subtle {{ color: #b9c7cf; margin-bottom: 8px; }}
    main {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .metric, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric {{ padding: 14px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 22px; }}
    .benchmarks {{ margin-bottom: 20px; }}
    .benchmark-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; }}
    .benchmark-card {{
      display: grid;
      gap: 5px;
      padding: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--ink);
      text-decoration: none;
    }}
    .benchmark-card:hover {{ border-color: #8aa8b7; text-decoration: none; }}
    .benchmark-card small {{ color: var(--muted); }}
    .panel {{ overflow: auto; }}
    .toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin: 0 0 12px;
      flex-wrap: wrap;
    }}
    .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .filter-control {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    select {{
      min-width: 160px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    table {{ width: 100%; border-collapse: collapse; min-width: 1720px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; }}
    th {{ position: sticky; top: 0; background: #eef4f6; color: #32414a; font-size: 12px; }}
    a {{ color: #0f5f8c; text-decoration: none; font-weight: 650; }}
    a:hover {{ text-decoration: underline; }}
    .thumb {{
      width: 136px;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      border: 1px solid var(--line);
      background: #e9eef2;
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 12px;
    }}
    .preview-cell {{ width: 160px; }}
    .episode-cell {{ min-width: 230px; }}
    .pass {{ color: var(--pass); font-weight: 700; }}
    .fail {{ color: var(--fail); font-weight: 700; }}
    @media (max-width: 760px) {{
      header {{ padding: 24px; }}
      h1 {{ font-size: 28px; }}
      main {{ padding: 16px; }}
    }}
  </style>
</head>
<body>
  <header>
    <p class="subtle">Farpoint</p>
    <h1>Episode Index</h1>
  </header>
  <main>
    <section class="metrics">
      <div class="metric"><span>Total Episodes</span><strong>{len(rows)}</strong></div>
      <div class="metric"><span>Passed</span><strong>{passed}</strong></div>
      <div class="metric"><span>Failed</span><strong>{failed}</strong></div>
      <div class="metric"><span>Best Settling Error</span><strong>{html.escape(f'{best_settling_error:.4f} m' if best_settling_error is not None else 'Not matched')}</strong></div>
      <div class="metric"><span>Best Smoothness</span><strong>{html.escape(f'{best_smoothness:.6f}' if best_smoothness is not None else 'Not matched')}</strong></div>
      <div class="metric"><span>Peak GPU</span><strong>{html.escape(format_number(peak_gpu, "%", 0))}</strong></div>
      <div class="metric"><span>Peak Workload Memory</span><strong>{html.escape(format_memory_mib(peak_memory))}</strong></div>
    </section>
    {benchmarks_section}
    <div class="toolbar">
      <div class="filters">
        <div class="filter-control">
          <label for="taskFilter">Task</label>
          <select id="taskFilter">
            <option value="all">All tasks</option>
            {task_options}
          </select>
        </div>
        <div class="filter-control">
          <label for="statusFilter">Status</label>
          <select id="statusFilter">
            <option value="all">All statuses</option>
            <option value="PASS">PASS</option>
            <option value="FAIL">FAIL</option>
          </select>
        </div>
        <div class="filter-control">
          <label for="controllerFilter">Controller</label>
          <select id="controllerFilter">
            <option value="all">All controllers</option>
            {controller_options}
          </select>
        </div>
        <div class="filter-control">
          <label for="sortSelect">Sort</label>
          <select id="sortSelect">
            <option value="started_desc">Newest first</option>
            <option value="settling_error_asc">Lowest settling error</option>
            <option value="smoothness_asc">Lowest smoothness</option>
            <option value="gpu_desc">Highest GPU load</option>
            <option value="runtime_desc">Longest runtime</option>
            <option value="memory_desc">Highest workload memory</option>
          </select>
        </div>
      </div>
      <div class="subtle" id="visibleCount">{len(rows)} episodes shown</div>
    </div>
    <section class="panel">
      <table>
        <thead>
          <tr>
            <th>Preview</th>
            <th>Episode</th>
            <th>Status</th>
            <th>Task</th>
            <th>Controller</th>
            <th>Runtime</th>
            <th>Frames</th>
            <th>Settling Error</th>
            <th>Object Lift</th>
            <th>Object Path</th>
            <th>End Effector Path</th>
            <th>Smoothness</th>
            <th>Peak GPU</th>
            <th>Peak Workload Memory</th>
            <th>Peak Host Memory Pressure</th>
            <th>Warnings</th>
          </tr>
        </thead>
        <tbody>
          {table_html}
        </tbody>
      </table>
    </section>
  </main>
  <script>
    const taskFilter = document.getElementById('taskFilter');
    const statusFilter = document.getElementById('statusFilter');
    const controllerFilter = document.getElementById('controllerFilter');
    const sortSelect = document.getElementById('sortSelect');
    const visibleCount = document.getElementById('visibleCount');
    const tbody = document.querySelector('tbody');
    const rows = Array.from(document.querySelectorAll('tbody tr'));

    function numeric(row, name, fallback) {{
      const raw = row.dataset[name];
      if (raw === undefined || raw === '') return fallback;
      const value = Number(raw);
      return Number.isFinite(value) ? value : fallback;
    }}

    function applyDashboardState() {{
      const selectedTask = taskFilter.value;
      const selectedStatus = statusFilter.value;
      const selectedController = controllerFilter.value;
      const sortMode = sortSelect.value;

      const sortedRows = [...rows].sort((a, b) => {{
        if (sortMode === 'settling_error_asc') return numeric(a, 'settlingError', Infinity) - numeric(b, 'settlingError', Infinity);
        if (sortMode === 'smoothness_asc') return numeric(a, 'smoothness', Infinity) - numeric(b, 'smoothness', Infinity);
        if (sortMode === 'gpu_desc') return numeric(b, 'peakGpu', -Infinity) - numeric(a, 'peakGpu', -Infinity);
        if (sortMode === 'runtime_desc') return numeric(b, 'runtime', -Infinity) - numeric(a, 'runtime', -Infinity);
        if (sortMode === 'memory_desc') return numeric(b, 'workloadMemory', -Infinity) - numeric(a, 'workloadMemory', -Infinity);
        return String(b.dataset.started || '').localeCompare(String(a.dataset.started || ''));
      }});

      sortedRows.forEach(row => tbody.appendChild(row));
      let shown = 0;
      sortedRows.forEach(row => {{
        const visible =
          (selectedTask === 'all' || row.dataset.task === selectedTask)
          && (selectedStatus === 'all' || row.dataset.status === selectedStatus)
          && (selectedController === 'all' || row.dataset.controller === selectedController);
        row.style.display = visible ? '' : 'none';
        if (visible) shown += 1;
      }});
      visibleCount.textContent = `${{shown}} episode${{shown === 1 ? '' : 's'}} shown`;
    }}

    [taskFilter, statusFilter, controllerFilter, sortSelect].forEach(control => {{
      control.addEventListener('change', applyDashboardState);
    }});
    applyDashboardState();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return index_path


def main():
    parser = argparse.ArgumentParser(description="Build a static Farpoint episode index.")
    parser.add_argument("--episodes-root", type=Path, default=Path("outputs/episodes"))
    parser.add_argument("--reports-root", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--no-rebuild-reports", action="store_true")
    args = parser.parse_args()

    index_path = build_index(
        args.episodes_root.resolve(),
        args.reports_root.resolve(),
        rebuild_reports=not args.no_rebuild_reports,
    )
    print(f"Episode index written: {index_path}")


if __name__ == "__main__":
    sys.exit(main())
