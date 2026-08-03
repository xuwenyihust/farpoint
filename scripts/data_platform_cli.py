#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.registry import EpisodeRegistry  # noqa: E402
from farpoint.retention import RetentionManager  # noqa: E402


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_reports(registry):
    built_episodes = 0
    built_benchmarks = 0
    for row in registry.list_episodes(limit=100000):
        artifact = row.get("artifact_path")
        if row["status"] not in {"PASS", "FAIL"} or not artifact:
            continue
        artifact_path = Path(artifact)
        output = registry.layout.reports / row["episode_id"] / "index.html"
        newest_input = max(
            (
                path.stat().st_mtime
                for path in (
                    artifact_path / "metadata.json",
                    artifact_path / "metrics.json",
                    artifact_path / "trajectory.jsonl",
                )
                if path.exists()
            ),
            default=0,
        )
        if output.exists() and output.stat().st_mtime >= newest_input:
            continue
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "build_episode_report.py"),
                str(artifact_path),
                "--output-dir",
                str(output.parent),
            ],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if completed.returncode == 0:
            built_episodes += 1
    for row in registry.list_benchmarks():
        manifest = Path(row["manifest_path"])
        output = registry.layout.reports / "benchmarks" / row["benchmark_id"] / "index.html"
        newest_input = max(
            manifest.stat().st_mtime,
            registry.layout.display_names.stat().st_mtime
            if registry.layout.display_names.exists()
            else 0,
        )
        if output.exists() and output.stat().st_mtime >= newest_input:
            continue
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_benchmark_report.py"),
            str(manifest),
        ]
        if row.get("display_name"):
            command.extend(["--display-name", row["display_name"]])
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
        )
        if completed.returncode == 0:
            built_benchmarks += 1
    registry.scan()
    return {"episode_reports": built_episodes, "benchmark_reports": built_benchmarks}


def main():
    parser = argparse.ArgumentParser(description="Manage the Farpoint remote data platform.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=PROJECT_ROOT / "outputs",
    )
    parser.add_argument(
        "--incomplete-timeout-seconds",
        type=int,
        default=int(os.environ.get("FARPOINT_INCOMPLETE_TIMEOUT_SECONDS", "1800")),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("scan")
    subparsers.add_parser("rebuild")
    subparsers.add_parser("build-reports")

    run_start = subparsers.add_parser("run-start")
    run_start.add_argument("--run-id", required=True)
    run_start.add_argument("--task-name", required=True)
    run_start.add_argument("--task-type")
    run_start.add_argument("--seed", type=int)
    run_start.add_argument("--benchmark-id")
    run_start.add_argument("--benchmark-repeat", type=int, default=0)

    run_finish = subparsers.add_parser("run-finish")
    run_finish.add_argument("--run-id", required=True)
    run_finish.add_argument("--status", choices=["PASS", "FAIL", "INCOMPLETE"], required=True)
    run_finish.add_argument("--return-code", type=int)
    run_finish.add_argument("--failure-reason")

    retention = subparsers.add_parser("retention-preview")
    retention.add_argument("--minimum-age-hours", type=float)

    quarantine = subparsers.add_parser("quarantine")
    quarantine.add_argument("episode_ids", nargs="+")
    quarantine.add_argument("--reason", default="manual")

    restore = subparsers.add_parser("restore")
    restore.add_argument("quarantine_id")

    subparsers.add_parser("quarantine-list")
    purge = subparsers.add_parser("purge-expired")
    purge.add_argument("--execute", action="store_true")

    pin = subparsers.add_parser("pin")
    pin.add_argument("episode_id")
    pin.add_argument("--reason", required=True)
    unpin = subparsers.add_parser("unpin")
    unpin.add_argument("episode_id")

    args = parser.parse_args()
    registry = EpisodeRegistry(
        args.outputs_root,
        incomplete_timeout_seconds=args.incomplete_timeout_seconds,
    )
    manager = RetentionManager(registry)

    if args.command == "scan":
        result = registry.scan()
    elif args.command == "rebuild":
        result = registry.rebuild()
    elif args.command == "build-reports":
        registry.scan()
        result = build_reports(registry)
    elif args.command == "run-start":
        path = registry.layout.runs / f"{args.run_id}.json"
        result = {
            "schema_version": "run-state.v1",
            "run_id": args.run_id,
            "task_name": args.task_name,
            "task_type": args.task_type,
            "seed": args.seed,
            "benchmark_id": args.benchmark_id,
            "benchmark_repeat": args.benchmark_repeat,
            "status": "RUNNING",
            "started_at": utc_now(),
            "updated_at": utc_now(),
        }
        write_json(path, result)
        registry.scan()
    elif args.command == "run-finish":
        path = registry.layout.runs / f"{args.run_id}.json"
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            result = {"run_id": args.run_id, "started_at": utc_now()}
        result.update(
            {
                "status": args.status,
                "return_code": args.return_code,
                "failure_reason": args.failure_reason,
                "finished_at": utc_now(),
                "updated_at": utc_now(),
            }
        )
        write_json(path, result)
        registry.scan()
    elif args.command == "retention-preview":
        policy = manager.load_policy()
        if args.minimum_age_hours is not None:
            policy["minimum_age_hours"] = args.minimum_age_hours
        result = manager.preview(policy)
    elif args.command == "quarantine":
        result = manager.quarantine(args.episode_ids, reason=args.reason)
    elif args.command == "restore":
        result = manager.restore(args.quarantine_id)
    elif args.command == "quarantine-list":
        result = manager.list_quarantine()
    elif args.command == "purge-expired":
        result = manager.purge_expired(execute=args.execute)
    elif args.command == "pin":
        result = manager.pin(args.episode_id, args.reason)
    elif args.command == "unpin":
        result = manager.unpin(args.episode_id)
    else:
        parser.error(f"unsupported command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
