#!/usr/bin/env python3
"""Initialize or inspect the frozen P0 SO-101 oracle gate workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.object_variation import load_variation_config  # noqa: E402
from farpoint.so101_gate_workflow import (  # noqa: E402
    build_so101_gate_workflow,
    evaluate_so101_gate_workflow,
    write_so101_gate_workflow,
)
from farpoint.so101_watchdog import load_watchdog_policy  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init")
    initialize.add_argument("root", type=Path)
    initialize.add_argument("--workflow-id", required=True)
    initialize.add_argument("--git-commit", required=True)
    initialize.add_argument(
        "--workflow-config",
        type=Path,
        default=PROJECT_ROOT / "configs/workflows/so101_oracle_gates_p0.json",
    )
    initialize.add_argument(
        "--variation-config",
        type=Path,
        default=PROJECT_ROOT / "configs/variations/so101_cube_pick_place_v1.json",
    )
    initialize.add_argument(
        "--watchdog-policy",
        type=Path,
        default=PROJECT_ROOT / "configs/workflows/so101_watchdog_p0.json",
    )
    status = subparsers.add_parser("status")
    status.add_argument("workflow", type=Path)
    status.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "init":
        workflow_config = json.loads(
            args.workflow_config.read_text(encoding="utf-8")
        )
        watchdog_policy = load_watchdog_policy(args.watchdog_policy)
        workflow, plans = build_so101_gate_workflow(
            workflow_config,
            load_variation_config(args.variation_config),
            watchdog_policy,
            workflow_id=args.workflow_id,
            git_commit=args.git_commit,
        )
        workflow_path = write_so101_gate_workflow(
            args.root, workflow, plans, watchdog_policy
        )
        report = evaluate_so101_gate_workflow(workflow_path)
    else:
        workflow_path = args.workflow
        report = evaluate_so101_gate_workflow(workflow_path)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if report["status"] in {"INVALID", "BLOCKED"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
