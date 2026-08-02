#!/usr/bin/env python3
import argparse
import html
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from build_episode_index import load_run_row
from build_episode_report import format_memory_mib, format_number


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPISODES_ROOT = PROJECT_ROOT / "outputs" / "episodes"
REPORTS_ROOT = PROJECT_ROOT / "outputs" / "reports"

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


def report_navigation(current):
    return f'''<nav class="app-nav" aria-label="Farpoint Data navigation">
      <div class="app-nav__crumbs">
        <a href="/">Farpoint Data</a><span class="app-nav__sep">/</span>
        <a href="/?view=benchmarks">Benchmarks</a><span class="app-nav__sep">/</span>
        <span class="app-nav__current" aria-current="page">{html.escape(current)}</span>
      </div>
      <div class="app-nav__links">
        <a href="/">Home</a><a href="/?view=episodes">Episodes</a><a href="/?view=benchmarks">Benchmarks</a>
        <a class="app-nav__back" href="/?view=benchmarks" onclick="if (history.length > 1) {{ history.back(); return false; }}">Back</a>
      </div>
    </nav>'''


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_manifest(manifest):
    """Map legacy position-pilot fields onto the benchmark report contract."""
    normalized = dict(manifest)
    normalized["benchmark_id"] = normalized.get("benchmark_id") or normalized.get(
        "pilot_id"
    )
    normalized["task_name"] = normalized.get("task_name") or normalized.get(
        "task_id", "unknown_task"
    )
    if normalized.get("schema_version") in {
        "farpoint.collection.v1",
        "farpoint.collection-run.v1",
    }:
        normalized["report_kind"] = "collection"
        normalized["benchmark_id"] = normalized.get("collection_id")
        attempts = normalized.get("attempts", [])
        normalized["trials"] = [
            {
                **attempt,
                "success": bool(attempt.get("outcome_success")),
                "split": attempt.get("dataset_split") or attempt.get("source_split"),
            }
            for attempt in attempts
        ]
        nested = normalized.get("acceptance") or {}
        normalized["planned_trials"] = int(
            nested.get("maximum_task_attempts", normalized.get("task_attempts", len(attempts)))
        )
        normalized["completed_trials"] = len(attempts)
        normalized["passed_trials"] = sum(
            attempt.get("outcome_success") is True for attempt in attempts
        )
        normalized["success_rate"] = (
            normalized["passed_trials"] / len(attempts) if attempts else 0.0
        )
        normalized["accepted"] = nested.get("accepted") is True
        normalized["acceptance"] = {
            **nested,
            "min_success_rate": nested.get("required_task_yield", 0.75),
            "max_final_target_xy_distance": 0.05,
            "min_object_lift_height": 0.15,
            "min_release_settle_frames": 120,
            "max_perception_xy_error": 0.02,
            "min_bilateral_contact_frames": 20,
            "min_transport_contact_frames": 120,
            "require_contact_only": True,
            "require_dataset": True,
        }
    acceptance = dict(normalized.get("acceptance") or {})
    if normalized.get("schema_version") == "farpoint.benchmark.v2":
        trials = normalized.get("trials", [])
        nested_acceptance = normalized.get("acceptance") or {}
        normalized["planned_trials"] = len(trials)
        normalized["completed_trials"] = len(trials)
        normalized["passed_trials"] = int(nested_acceptance.get("observed_successes", 0))
        normalized["success_rate"] = float(
            nested_acceptance.get("observed_success_rate", 0.0)
        )
        normalized["accepted"] = nested_acceptance.get("accepted") is True
        acceptance.update(
            {
                "min_success_rate": nested_acceptance.get("required_success_rate", 0.90),
                "max_final_target_xy_distance": 0.05,
                "min_object_lift_height": 0.15,
                "min_release_settle_frames": 120,
                "max_perception_xy_error": 0.02,
                "min_bilateral_contact_frames": 20,
                "min_transport_contact_frames": 120,
                "require_contact_only": True,
                "require_dataset": True,
            }
        )
    aliases = {
        "max_final_target_xy_distance": "max_final_target_xy_error_m",
        "min_object_lift_height": "min_lift_height_m",
        "min_release_settle_frames": "min_settle_frames",
        "max_perception_xy_error": "max_perception_xy_error_m",
        "require_contact_only": "contact_only",
    }
    for canonical, legacy in aliases.items():
        if canonical not in acceptance and legacy in acceptance:
            acceptance[canonical] = acceptance[legacy]
    if "min_success_rate" not in acceptance:
        planned = max(1, int(normalized.get("planned_trials") or 0))
        required = int(acceptance.get("required_successes", planned))
        acceptance["min_success_rate"] = required / planned
    normalized["acceptance"] = acceptance
    return normalized


def finite_numbers(values):
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def percentile(values, percentile_value):
    ordered = sorted(finite_numbers(values))
    if not ordered:
        return None
    index = (len(ordered) - 1) * percentile_value
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def relative_path(path, start):
    return Path(os.path.relpath(Path(path).resolve(), Path(start).resolve()))


def load_trial(trial, report_dir):
    episode_id = trial.get("episode_id")
    episode_dir = EPISODES_ROOT / episode_id if episode_id else None
    row = {}
    metadata = {}
    metrics = {}
    if episode_dir and episode_dir.is_dir():
        metadata_path = episode_dir / "metadata.json"
        metrics_path = episode_dir / "metrics.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
        if metrics_path.exists():
            metrics = read_json(metrics_path)
        if metadata and metrics and (episode_dir / "trajectory.jsonl").exists():
            row = load_run_row(episode_dir, REPORTS_ROOT, rebuild_reports=True)

    preview_images = sorted((episode_dir / "preview").glob("*.png")) if episode_dir else []
    preview = preview_images[len(preview_images) // 2] if preview_images else None
    resolved_position = (
        (metadata.get("variation") or {}).get("resolved", {}).get("object_position_m")
        or []
    )
    planned_position = trial.get("object_position_xy_m") or []
    pick_object_xy = (metadata.get("randomization") or {}).get("pick_object_xy")
    if not pick_object_xy and len(resolved_position) >= 2:
        pick_object_xy = resolved_position[:2]
    if not pick_object_xy and len(planned_position) >= 2:
        pick_object_xy = planned_position[:2]
    return {
        **trial,
        "success": bool(trial["success"] if "success" in trial else metrics.get("success")),
        "failure_category": metrics.get("failure_category") or trial.get("failure_category"),
        "failure_reason": metrics.get("failure_reason") or trial.get("failure_reason"),
        "failed_checks": metrics.get("failed_checks", []),
        "pick_object_xy": pick_object_xy,
        "target_zone_xy": (metadata.get("randomization") or {}).get("target_zone_xy"),
        "final_target_xy_distance": metrics.get(
            "final_target_xy_distance", trial.get("final_target_xy_distance")
        ),
        "object_lift_height": metrics.get("object_lift_height", trial.get("object_lift_height")),
        "release_settle_frames": metrics.get(
            "release_settle_frames", trial.get("release_settle_frames")
        ),
        "post_release_motion": metrics.get("post_release_motion", trial.get("post_release_motion")),
        "elapsed_seconds": metrics.get("elapsed_seconds", trial.get("elapsed_seconds")),
        "initial_object_perception_xy_error": metrics.get(
            "initial_object_perception_xy_error",
            trial.get("initial_object_perception_xy_error"),
        ),
        "bilateral_contact_frames": metrics.get(
            "bilateral_contact_frames",
            trial.get("bilateral_contact_frames"),
        ),
        "transport_contact_frames": metrics.get(
            "transport_contact_frames",
            trial.get("transport_contact_frames"),
        ),
        "temporary_grasp_joint_created": metrics.get(
            "temporary_grasp_joint_created",
            trial.get("temporary_grasp_joint_created"),
        ),
        "grasp_constraint": metrics.get("grasp_constraint"),
        "dataset_valid": (
            trial["dataset_valid"]
            if "dataset_valid" in trial
            else metrics.get("dataset_valid")
        ),
        "dataset_observation_count": metrics.get(
            "dataset_observation_count",
            trial.get("dataset_observation_count"),
        ),
        "peak_gpu": row.get("peak_gpu"),
        "workload_memory": row.get("workload_memory"),
        "host_pressure": row.get("host_pressure"),
        "warning_count": row.get("warning_count"),
        "report_href": (
            relative_path(REPORTS_ROOT / episode_id / "index.html", report_dir).as_posix()
            if episode_id
            else None
        ),
        "preview_href": relative_path(preview, report_dir).as_posix() if preview else None,
    }


def reproducibility_summary(trials):
    groups = defaultdict(list)
    for trial in trials:
        if trial.get("seed") is None:
            continue
        groups[int(trial["seed"])].append(trial)
    repeated = {seed: rows for seed, rows in groups.items() if len(rows) > 1}
    if not repeated:
        return {
            "evaluated": False,
            "consistent": None,
            "repeated_seed_count": 0,
            "details": [],
        }

    details = []
    for seed, rows in sorted(repeated.items()):
        success_consistent = len({bool(row["success"]) for row in rows}) == 1
        errors = finite_numbers(row.get("final_target_xy_distance") for row in rows)
        max_error_delta = max(errors) - min(errors) if len(errors) > 1 else 0.0
        consistent = success_consistent and max_error_delta <= 0.01
        details.append(
            {
                "seed": seed,
                "runs": len(rows),
                "success_consistent": success_consistent,
                "max_target_error_delta": round(max_error_delta, 6),
                "consistent": consistent,
            }
        )
    return {
        "evaluated": True,
        "consistent": all(row["consistent"] for row in details),
        "repeated_seed_count": len(details),
        "details": details,
    }


def summarize(manifest, trials, reproducibility_trials=None):
    manifest = normalize_manifest(manifest)
    reproducibility_trials = reproducibility_trials or []
    passed = sum(1 for trial in trials if trial["success"])
    completed = len(trials)
    success_rate = passed / completed if completed else 0.0
    target_errors = finite_numbers(trial.get("final_target_xy_distance") for trial in trials)
    perception_errors = finite_numbers(
        trial.get("initial_object_perception_xy_error") for trial in trials
    )
    runtimes = finite_numbers(trial.get("elapsed_seconds") for trial in trials)
    peak_gpu_values = finite_numbers(trial.get("peak_gpu") for trial in trials)
    workload_memory_values = finite_numbers(trial.get("workload_memory") for trial in trials)
    host_pressure_values = finite_numbers(trial.get("host_pressure") for trial in trials)
    positions = [
        [float(position[0]), float(position[1])]
        for position in (trial.get("pick_object_xy") for trial in trials)
        if isinstance(position, (list, tuple)) and len(position) >= 2
    ]
    x_values = [position[0] for position in positions]
    y_values = [position[1] for position in positions]
    workspace_coverage = {
        "x_range_m": [min(x_values), max(x_values)] if x_values else None,
        "y_range_m": [min(y_values), max(y_values)] if y_values else None,
        "x_span_m": max(x_values) - min(x_values) if x_values else None,
        "y_span_m": max(y_values) - min(y_values) if y_values else None,
    }
    failure_counts = Counter(
        trial.get("failure_category") or "unclassified"
        for trial in trials
        if not trial["success"]
    )
    acceptance = manifest["acceptance"]
    threshold_checks = []
    for trial in trials:
        if not trial["success"]:
            continue
        threshold_checks.append(
            trial.get("final_target_xy_distance") is not None
            and float(trial["final_target_xy_distance"])
            <= float(acceptance["max_final_target_xy_distance"])
            and trial.get("object_lift_height") is not None
            and float(trial["object_lift_height"])
            >= float(acceptance["min_object_lift_height"])
            and trial.get("release_settle_frames") is not None
            and int(trial["release_settle_frames"])
            >= int(acceptance["min_release_settle_frames"])
        )
    failures_classified = all(
        trial.get("failure_category") and trial.get("failure_reason")
        for trial in trials
        if not trial["success"]
    )
    reproducibility = reproducibility_summary(trials + reproducibility_trials)
    acceptance_checks = {
        "planned_trials_completed": completed == int(manifest["planned_trials"]),
        "minimum_success_rate": success_rate >= float(acceptance["min_success_rate"]),
        "successful_trials_meet_task_thresholds": bool(threshold_checks)
        and all(threshold_checks),
        "failures_are_classified": failures_classified,
        "reproducibility": reproducibility["consistent"]
        if reproducibility["evaluated"]
        else None,
    }
    if manifest.get("report_kind") == "collection":
        nested = manifest.get("acceptance") or {}
        selected_per_cell = nested.get("selected_per_cell") or {}
        acceptance_checks["planned_trials_completed"] = (
            manifest.get("execution_status") == "FINISHED"
        )
        acceptance_checks["selected_episode_target"] = (
            nested.get("observed_selected_episodes")
            == nested.get("required_selected_episodes")
        )
        acceptance_checks["grid_cell_coverage"] = (
            nested.get("observed_covered_cells") == nested.get("required_cells")
        )
        acceptance_checks["balanced_cell_quota"] = bool(selected_per_cell) and set(
            selected_per_cell.values()
        ) == {nested.get("required_selected_per_cell")}
        acceptance_checks["dataset_split_counts"] = (
            nested.get("observed_splits") == nested.get("required_splits")
        )
        acceptance_checks["attempt_budget"] = (
            nested.get("observed_task_attempts", completed)
            <= nested.get("maximum_task_attempts", completed)
        )
    if acceptance.get("max_perception_xy_error") is not None:
        acceptance_checks["perception_accuracy"] = bool(trials) and all(
            trial.get("initial_object_perception_xy_error") is not None
            and float(trial["initial_object_perception_xy_error"])
            <= float(acceptance["max_perception_xy_error"])
            for trial in trials
            if trial["success"]
        )
    if acceptance.get("min_bilateral_contact_frames") is not None:
        acceptance_checks["bilateral_contact"] = bool(trials) and all(
            int(trial.get("bilateral_contact_frames") or 0)
            >= int(acceptance["min_bilateral_contact_frames"])
            for trial in trials
            if trial["success"]
        )
    if acceptance.get("min_transport_contact_frames") is not None:
        acceptance_checks["transport_contact"] = bool(trials) and all(
            int(trial.get("transport_contact_frames") or 0)
            >= int(acceptance["min_transport_contact_frames"])
            for trial in trials
            if trial["success"]
        )
    if acceptance.get("require_contact_only"):
        acceptance_checks["contact_only"] = bool(trials) and all(
            trial.get("grasp_constraint") == "contact_only"
            and not bool(trial.get("temporary_grasp_joint_created"))
            for trial in trials
        )
    if acceptance.get("require_dataset"):
        acceptance_checks["dataset_valid"] = bool(trials) and all(
            bool(trial.get("dataset_valid"))
            and int(trial.get("dataset_observation_count") or 0) > 0
            for trial in trials
            if trial["success"]
        )
    enforce_all_episode_checks = (
        manifest.get("schema_version") != "farpoint.benchmark.v2"
        and manifest.get("mode") != "formal"
    )
    if enforce_all_episode_checks and any(trial.get("checks") for trial in trials):
        acceptance_checks["pilot_episode_checks"] = bool(trials) and all(
            checks
            and all(bool(value) for value in checks.values())
            for checks in (trial.get("checks") for trial in trials)
        )
    if acceptance.get("min_selected_x_span_m") is not None:
        acceptance_checks["workspace_x_span"] = (
            workspace_coverage["x_span_m"] is not None
            and workspace_coverage["x_span_m"]
            >= float(acceptance["min_selected_x_span_m"])
        )
    if acceptance.get("min_selected_y_span_m") is not None:
        acceptance_checks["workspace_y_span"] = (
            workspace_coverage["y_span_m"] is not None
            and workspace_coverage["y_span_m"]
            >= float(acceptance["min_selected_y_span_m"])
        )
    required_checks = [
        acceptance_checks["planned_trials_completed"],
        acceptance_checks["minimum_success_rate"],
        acceptance_checks["successful_trials_meet_task_thresholds"],
        acceptance_checks["failures_are_classified"],
    ]
    if reproducibility_trials:
        required_checks.append(bool(acceptance_checks["reproducibility"]))
    required_checks.extend(
        bool(value)
        for key, value in acceptance_checks.items()
        if key
        not in {
            "planned_trials_completed",
            "minimum_success_rate",
            "successful_trials_meet_task_thresholds",
            "failures_are_classified",
            "reproducibility",
        }
    )
    return {
        "schema_version": "benchmark-summary.v1",
        "benchmark_id": manifest["benchmark_id"],
        "task_name": manifest["task_name"],
        "planned_trials": manifest["planned_trials"],
        "completed_trials": completed,
        "passed_trials": passed,
        "failed_trials": completed - passed,
        "success_rate": success_rate,
        "mean_target_xy_error": statistics.fmean(target_errors) if target_errors else None,
        "p95_target_xy_error": percentile(target_errors, 0.95),
        "mean_perception_xy_error": statistics.fmean(perception_errors)
        if perception_errors
        else None,
        "p95_perception_xy_error": percentile(perception_errors, 0.95),
        "median_runtime_seconds": statistics.median(runtimes) if runtimes else None,
        "peak_gpu_percent": max(peak_gpu_values) if peak_gpu_values else None,
        "peak_workload_memory_mib": max(workload_memory_values)
        if workload_memory_values
        else None,
        "peak_host_memory_pressure_percent": max(host_pressure_values)
        if host_pressure_values
        else None,
        "workspace_coverage": workspace_coverage,
        "failure_counts": dict(sorted(failure_counts.items())),
        "acceptance": acceptance,
        "acceptance_checks": acceptance_checks,
        "accepted": all(required_checks),
        "reproducibility": reproducibility,
        "reproducibility_trials": reproducibility_trials,
        "trials": trials,
        "report_kind": manifest.get("report_kind", "benchmark"),
        "execution_status": manifest.get("execution_status"),
        "collection": {
            "selected_episodes": (manifest.get("acceptance") or {}).get(
                "observed_selected_episodes", 0
            ),
            "required_selected_episodes": (manifest.get("acceptance") or {}).get(
                "required_selected_episodes", 0
            ),
            "covered_cells": (manifest.get("acceptance") or {}).get(
                "observed_covered_cells", 0
            ),
            "required_cells": (manifest.get("acceptance") or {}).get(
                "required_cells", 0
            ),
            "selected_per_cell": (manifest.get("acceptance") or {}).get(
                "selected_per_cell", {}
            ),
            "imported_attempts": sum(
                trial.get("origin") == "imported" for trial in trials
            ),
            "new_attempts": sum(trial.get("origin") == "new" for trial in trials),
        },
    }


def fmt_metric(value, suffix="", digits=3):
    if value is None:
        return "Unavailable"
    return f"{float(value):.{digits}f}{suffix}"


def build_report(manifest_path):
    raw_manifest = read_json(manifest_path)
    if raw_manifest.get("schema_version") == "farpoint.benchmark.v2":
        run_state_path = manifest_path.parent / "run-state.json"
        if run_state_path.is_file():
            run_state = read_json(run_state_path)
            details = {
                trial.get("trial_id"): trial for trial in run_state.get("trials", [])
            }
            raw_manifest["trials"] = [
                {**details.get(trial.get("trial_id"), {}), **trial}
                for trial in raw_manifest.get("trials", [])
            ]
    manifest = normalize_manifest(raw_manifest)
    benchmark_id = manifest["benchmark_id"]
    report_dir = REPORTS_ROOT / "benchmarks" / benchmark_id
    report_dir.mkdir(parents=True, exist_ok=True)
    trials = [load_trial(trial, report_dir) for trial in manifest.get("trials", [])]
    reproducibility_trials = [
        load_trial(trial, report_dir)
        for trial in manifest.get("reproducibility_trials", [])
    ]
    summary = summarize(manifest, trials, reproducibility_trials)
    write_json(manifest_path.parent / "summary.json", summary)
    write_json(report_dir / "summary.json", summary)

    failure_options = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name.replace("_", " ").title())}</option>'
        for name in sorted(summary["failure_counts"])
    )
    table_rows = []
    for trial in sorted(
        trials,
        key=lambda row: (
            row.get("seed") is None,
            int(row.get("seed") or 0),
            int(row.get("repeat", 0)),
            str(row.get("trial_id") or ""),
        ),
    ):
        status = "PASS" if trial["success"] else "FAIL"
        status_class = "pass" if trial["success"] else "fail"
        preview = (
            f'<img class="thumb" src="{html.escape(trial["preview_href"])}" alt="Episode preview">'
            if trial.get("preview_href")
            else '<div class="thumb missing">No preview</div>'
        )
        episode = (
            f'<a href="{html.escape(trial["report_href"])}">{html.escape(trial["episode_id"])}</a>'
            if trial.get("report_href")
            else "Output missing"
        )
        category = trial.get("failure_category") or "none"
        reason = (trial.get("failure_reason") or "Completed successfully").replace("_", " ")
        seed = str(int(trial["seed"])) if trial.get("seed") is not None else "Unavailable"
        position = trial.get("pick_object_xy")
        position_text = (
            f"{float(position[0]):.3f}, {float(position[1]):.3f}"
            if isinstance(position, (list, tuple)) and len(position) >= 2
            else "Unavailable"
        )
        table_rows.append(
            f'<tr data-status="{status}" data-failure="{html.escape(category)}">'
            f'<td>{preview}</td>'
            f"<td>{seed}</td>"
            f"<td>{position_text}</td>"
            f'<td>{int(trial.get("repeat", 0)) + 1}</td>'
            f'<td>{episode}</td>'
            f'<td class="{status_class}">{status}</td>'
            f'<td>{html.escape(str(trial.get("origin") or "native").title())}</td>'
            f'<td>{"YES" if trial.get("selected_for_dataset") else "NO"}</td>'
            f'<td>{html.escape(str(trial.get("cell_id") or "Unavailable"))}</td>'
            f'<td>{html.escape(category.replace("_", " ").title())}</td>'
            f'<td class="reason">{html.escape(reason)}</td>'
            f'<td>{fmt_metric(trial.get("final_target_xy_distance"), " m", 4)}</td>'
            f'<td>{fmt_metric(trial.get("initial_object_perception_xy_error"), " m", 4)}</td>'
            f'<td>{fmt_metric(trial.get("object_lift_height"), " m", 4)}</td>'
            f'<td>{trial.get("bilateral_contact_frames") if trial.get("bilateral_contact_frames") is not None else "Unavailable"}</td>'
            f'<td>{"PASS" if trial.get("dataset_valid") else "FAIL"}</td>'
            f'<td>{trial.get("release_settle_frames") if trial.get("release_settle_frames") is not None else "Unavailable"}</td>'
            f'<td>{fmt_metric(trial.get("elapsed_seconds"), " s", 1)}</td>'
            f'<td>{format_number(trial.get("peak_gpu"), "%", 0)}</td>'
            f'<td>{format_memory_mib(trial.get("workload_memory"))}</td>'
            f'<td>{format_number(trial.get("host_pressure"), "%", 1)}</td>'
            "</tr>"
        )

    failure_bars = []
    max_failure_count = max(summary["failure_counts"].values(), default=1)
    for category, count in summary["failure_counts"].items():
        width = count / max_failure_count * 100
        failure_bars.append(
            '<div class="failure-row">'
            f'<span>{html.escape(category.replace("_", " ").title())}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>'
            f"<strong>{count}</strong>"
            "</div>"
        )
    if not failure_bars:
        failure_bars.append('<p class="subtle">No failures recorded.</p>')

    acceptance_rows = []
    labels = {
        "planned_trials_completed": "All planned trials completed",
        "minimum_success_rate": "Success rate meets threshold",
        "successful_trials_meet_task_thresholds": "Successful trials meet task thresholds",
        "failures_are_classified": "Every failure has a reason",
        "reproducibility": "Repeated seeds are consistent",
        "perception_accuracy": "RGB-D pose error meets threshold",
        "bilateral_contact": "Successful grasps have bilateral contact",
        "transport_contact": "Contact persists through transport",
        "contact_only": "No temporary grasp joint is used",
        "dataset_valid": "Every trial exports a valid dataset",
        "pilot_episode_checks": "Every pilot episode passes all artifact and quality checks",
        "workspace_x_span": "Selected positions cover the required X span",
        "workspace_y_span": "Selected positions cover the required Y span",
        "selected_episode_target": "Selected episode target is complete",
        "grid_cell_coverage": "All 25 grid cells are covered",
        "balanced_cell_quota": "Every cell contains exactly two selected episodes",
        "dataset_split_counts": "Dataset split counts are 34/8/8",
        "attempt_budget": "Task attempts remain within the resource budget",
    }
    for key, value in summary["acceptance_checks"].items():
        state = "NOT RUN" if value is None else ("PASS" if value else "FAIL")
        state_class = "pending" if value is None else ("pass" if value else "fail")
        acceptance_rows.append(
            f"<tr><td>{html.escape(labels.get(key, key.replace('_', ' ').title()))}</td>"
            f'<td class="{state_class}">{state}</td></tr>'
        )
    reproducibility_rows = []
    for detail in summary["reproducibility"]["details"]:
        state = "PASS" if detail["consistent"] else "FAIL"
        state_class = "pass" if detail["consistent"] else "fail"
        reproducibility_rows.append(
            f"<tr><td>{detail['seed']}</td><td>{detail['runs']}</td>"
            f"<td>{detail['max_target_error_delta']:.4f} m</td>"
            f'<td class="{state_class}">{state}</td></tr>'
        )
    reproducibility_html = (
        "<table><thead><tr><th>Seed</th><th>Runs</th><th>Max Error Delta</th>"
        f"<th>Consistency</th></tr></thead><tbody>{''.join(reproducibility_rows)}</tbody></table>"
        if reproducibility_rows
        else '<p class="subtle">No repeated seeds have been evaluated.</p>'
    )

    progress = min(100.0, summary["completed_trials"] / max(summary["planned_trials"], 1) * 100)
    success_width = min(100.0, summary["success_rate"] * 100)
    if summary["reproducibility"]["evaluated"]:
        reproducibility_state = (
            "PASS" if summary["reproducibility"]["consistent"] else "FAIL"
        )
        reproducibility_class = (
            "pass" if summary["reproducibility"]["consistent"] else "fail"
        )
    else:
        reproducibility_state = "NOT RUN"
        reproducibility_class = "pending"
    coverage = summary["workspace_coverage"]
    is_collection = summary["report_kind"] == "collection"
    report_label = "Collection" if is_collection else "Benchmark"
    coverage_metrics = ""
    if coverage["x_span_m"] is not None:
        coverage_metrics = (
            '<div class="metric"><span>Position X Span</span>'
            f'<strong>{coverage["x_span_m"]:.3f} m</strong></div>'
            '<div class="metric"><span>Position Y Span</span>'
            f'<strong>{coverage["y_span_m"]:.3f} m</strong></div>'
        )
    collection_metrics = ""
    collection_grid = ""
    if is_collection:
        collection = summary["collection"]
        collection_metrics = (
            '<div class="metric"><span>Selected Episodes</span>'
            f'<strong>{collection["selected_episodes"]} / {collection["required_selected_episodes"]}</strong></div>'
            '<div class="metric"><span>Covered Cells</span>'
            f'<strong>{collection["covered_cells"]} / {collection["required_cells"]}</strong></div>'
            '<div class="metric"><span>Imported / New Attempts</span>'
            f'<strong>{collection["imported_attempts"]} / {collection["new_attempts"]}</strong></div>'
        )
        cells = []
        for row in range(5):
            for column in range(5):
                cell_id = f"r{row:02d}_c{column:02d}"
                count = int(collection["selected_per_cell"].get(cell_id, 0))
                state = "complete" if count == 2 else ("partial" if count else "empty")
                cells.append(
                    f'<div class="cell {state}"><span>{cell_id}</span><strong>{count} / 2</strong></div>'
                )
        collection_grid = (
            '<section class="panel"><h2>Grid Cell Coverage</h2>'
            f'<div class="cell-grid">{"".join(cells)}</div></section>'
        )
    report_path = report_dir / "index.html"
    report_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(benchmark_id)} | Farpoint</title>
  <style>
    :root {{
      --ink: #172026; --muted: #5b6873; --line: #d7e0e5; --panel: #fff;
      --bg: #f3f6f7; --dark: #10242a; --pass: #087f5b; --fail: #c0392b;
      --accent: #1379a5; --warn: #a15c00;
    }}
{REPORT_NAV_CSS}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 system-ui, sans-serif; }}
    header {{ background: var(--dark); color: #f7fbfc; padding: 28px 32px; }}
    header p {{ color: #bdd0d6; margin: 7px 0 0; }}
    h1 {{ margin: 0; font-size: 32px; letter-spacing: 0; }}
    h2 {{ font-size: 18px; margin: 0 0 14px; }}
    main {{ max-width: 1540px; margin: 0 auto; padding: 22px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-bottom: 16px; }}
    .metric, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
    .metric {{ padding: 13px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 21px; }}
    .panel {{ padding: 17px; margin-bottom: 16px; }}
    .two-column {{ display: grid; grid-template-columns: minmax(300px, 1fr) minmax(300px, 1fr); gap: 16px; }}
    .progress-label {{ display: flex; justify-content: space-between; margin-bottom: 6px; }}
    .progress {{ height: 12px; background: #e6ecef; overflow: hidden; border-radius: 4px; margin-bottom: 15px; }}
    .progress > div {{ height: 100%; background: var(--accent); }}
    .progress.success > div {{ background: var(--pass); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; }}
    th {{ background: #edf3f5; color: #35454d; font-size: 12px; position: sticky; top: 0; }}
    .table-wrap {{ overflow: auto; padding: 0; }}
    .table-wrap table {{ min-width: 1540px; }}
    .thumb {{ width: 112px; aspect-ratio: 16 / 9; object-fit: cover; display: block; border: 1px solid var(--line); }}
    .thumb.missing {{ display: grid; place-items: center; background: #edf1f3; color: var(--muted); font-size: 11px; }}
    .pass {{ color: var(--pass); font-weight: 750; }}
    .fail {{ color: var(--fail); font-weight: 750; }}
    .pending {{ color: var(--warn); font-weight: 750; }}
    .subtle {{ color: var(--muted); }}
    .failure-row {{ display: grid; grid-template-columns: 120px 1fr 30px; align-items: center; gap: 10px; margin: 9px 0; }}
    .bar-track {{ height: 12px; background: #e9eef1; border-radius: 3px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: var(--fail); }}
    .toolbar {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }}
    select {{ border: 1px solid var(--line); border-radius: 6px; padding: 7px 9px; background: #fff; }}
    a {{ color: #096692; text-decoration: none; font-weight: 650; }}
    a:hover {{ text-decoration: underline; }}
    .reason {{ min-width: 240px; }}
    .cell-grid {{ display: grid; grid-template-columns: repeat(5, minmax(90px, 1fr)); gap: 8px; }}
    .cell {{ border: 1px solid var(--line); padding: 10px; min-height: 58px; background: #f7fafb; }}
    .cell span {{ display: block; color: var(--muted); font-size: 11px; }}
    .cell strong {{ display: block; margin-top: 4px; }}
    .cell.complete {{ border-color: #65b89a; background: #eef9f5; }}
    .cell.partial {{ border-color: #d7a04c; background: #fff8eb; }}
    @media (max-width: 760px) {{
      header {{ padding: 22px 18px; }} main {{ padding: 14px; }}
      h1 {{ font-size: 24px; }} .two-column {{ grid-template-columns: 1fr; }}
      .cell-grid {{ grid-template-columns: repeat(5, minmax(72px, 1fr)); overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  {report_navigation(benchmark_id)}
  <header>
    <h1>{html.escape(benchmark_id)}</h1>
    <p>{html.escape(summary["task_name"].replace("_", " ").title())}</p>
  </header>
  <main>
    <section class="metrics">
      <div class="metric"><span>{report_label} Status</span><strong class="{"pass" if summary["accepted"] else "fail"}">{"PASS" if summary["accepted"] else "IN PROGRESS / FAIL"}</strong></div>
      <div class="metric"><span>{"Task Yield" if is_collection else "Success Rate"}</span><strong>{summary["success_rate"]:.1%}</strong></div>
      <div class="metric"><span>{"Task Attempts" if is_collection else "Trials"}</span><strong>{summary["completed_trials"]} / {summary["planned_trials"]}</strong></div>
      {collection_metrics}
      <div class="metric"><span>Mean Target Error</span><strong>{fmt_metric(summary["mean_target_xy_error"], " m", 4)}</strong></div>
      <div class="metric"><span>P95 Target Error</span><strong>{fmt_metric(summary["p95_target_xy_error"], " m", 4)}</strong></div>
      <div class="metric"><span>Mean Perception Error</span><strong>{fmt_metric(summary["mean_perception_xy_error"], " m", 4)}</strong></div>
      <div class="metric"><span>Median Runtime</span><strong>{fmt_metric(summary["median_runtime_seconds"], " s", 1)}</strong></div>
      <div class="metric"><span>Peak GPU Load</span><strong>{format_number(summary["peak_gpu_percent"], "%", 0)}</strong></div>
      <div class="metric"><span>Peak Workload Memory</span><strong>{format_memory_mib(summary["peak_workload_memory_mib"])}</strong></div>
      {coverage_metrics}
      <div class="metric"><span>Reproducibility</span><strong class="{reproducibility_class}">{reproducibility_state}</strong></div>
    </section>

    <section class="panel">
      <h2>{report_label} Progress</h2>
      <div class="progress-label"><span>Completed</span><strong>{summary["completed_trials"]} / {summary["planned_trials"]}</strong></div>
      <div class="progress"><div style="width:{progress:.1f}%"></div></div>
      <div class="progress-label"><span>Success rate</span><strong>{summary["success_rate"]:.1%} (target {summary["acceptance"]["min_success_rate"]:.0%})</strong></div>
      <div class="progress success"><div style="width:{success_width:.1f}%"></div></div>
    </section>

    {collection_grid}

    <section class="two-column">
      <div class="panel">
        <h2>Acceptance Checks</h2>
        <table>{"".join(acceptance_rows)}</table>
      </div>
      <div class="panel">
        <h2>Failure Taxonomy</h2>
        {"".join(failure_bars)}
      </div>
    </section>

    <section class="panel">
      <h2>Repeated-Seed Reproducibility</h2>
      {reproducibility_html}
    </section>

    <section class="panel table-wrap">
      <div class="toolbar">
        <strong>Trials</strong>
        <label>Status
          <select id="statusFilter">
            <option value="all">All</option><option value="PASS">PASS</option><option value="FAIL">FAIL</option>
          </select>
        </label>
        <label>Failure
          <select id="failureFilter"><option value="all">All</option>{failure_options}</select>
        </label>
        <span class="subtle" id="visibleCount">{len(trials)} trials shown</span>
      </div>
      <table>
        <thead><tr>
          <th>Preview</th><th>Seed</th><th>Object XY (m)</th><th>Run</th><th>Episode</th><th>Status</th>
          <th>Origin</th><th>Selected</th><th>Cell</th>
          <th>Failure Stage</th><th>Reason</th><th>Target Error</th><th>Perception Error</th><th>Lift</th>
          <th>Bilateral Contact</th><th>Dataset</th>
          <th>Settle Frames</th><th>Runtime</th><th>Peak GPU</th><th>Workload Memory</th><th>Host Pressure</th>
        </tr></thead>
        <tbody id="trialRows">{"".join(table_rows)}</tbody>
      </table>
    </section>
  </main>
  <script>
    const statusFilter = document.getElementById('statusFilter');
    const failureFilter = document.getElementById('failureFilter');
    const rows = [...document.querySelectorAll('#trialRows tr')];
    function applyFilters() {{
      let shown = 0;
      rows.forEach(row => {{
        const visible =
          (statusFilter.value === 'all' || row.dataset.status === statusFilter.value) &&
          (failureFilter.value === 'all' || row.dataset.failure === failureFilter.value);
        row.hidden = !visible;
        if (visible) shown += 1;
      }});
      document.getElementById('visibleCount').textContent = `${{shown}} trial${{shown === 1 ? '' : 's'}} shown`;
    }}
    statusFilter.addEventListener('change', applyFilters);
    failureFilter.addEventListener('change', applyFilters);
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(f"Benchmark report written: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Build a static randomized benchmark report.")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    build_report(args.manifest.resolve())


if __name__ == "__main__":
    main()
