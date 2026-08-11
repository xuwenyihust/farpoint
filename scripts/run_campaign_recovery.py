#!/usr/bin/env python3
"""Evaluate a self-healing campaign or freeze its next immutable segment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.campaign_recovery import (  # noqa: E402
    build_replacement_requests,
    create_continuation_segment,
    evaluate_self_healing_campaign,
    validate_replacement_plan,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _write(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--campaign", required=True, type=Path)
    evaluate.add_argument("--evidence-index", required=True, type=Path)
    evaluate.add_argument("--policy", required=True, type=Path)
    evaluate.add_argument("--live-status", required=True, type=Path)
    evaluate.add_argument("--disk-path", required=True, type=Path)
    evaluate.add_argument("--integrity-errors", type=Path)
    evaluate.add_argument("--output", required=True, type=Path)

    freeze = commands.add_parser("freeze-continuation")
    freeze.add_argument("--campaign", required=True, type=Path)
    freeze.add_argument("--parent-segment", required=True, type=Path)
    freeze.add_argument("--parent-plan", required=True, type=Path)
    freeze.add_argument("--parent-manifest", required=True, type=Path)
    freeze.add_argument("--continuation-plan", required=True, type=Path)
    freeze.add_argument("--segment-id", required=True)
    freeze.add_argument("--git-commit", required=True)
    freeze.add_argument("--oracle-profile", action="append", required=True)
    freeze.add_argument("--segment-output", required=True, type=Path)
    freeze.add_argument("--replacement-output", required=True, type=Path)
    return parser


def _evaluate(args: argparse.Namespace) -> int:
    index = _read(args.evidence_index)
    base = args.evidence_index.parent
    evidence = []
    for row in index.get("segments") or []:
        evidence.append(
            {
                "segment": _read(_resolve(base, row["segment"])),
                "plan": _read(_resolve(base, row["plan"])),
                "manifest": _read(_resolve(base, row["manifest"])),
            }
        )
    integrity_errors = []
    if args.integrity_errors is not None:
        integrity_errors = _read(args.integrity_errors).get("errors") or []
    report = evaluate_self_healing_campaign(
        _read(args.campaign),
        evidence,
        _read(args.policy),
        live_status=_read(args.live_status),
        free_disk_bytes=shutil.disk_usage(args.disk_path).free,
        integrity_errors=integrity_errors,
    )
    _write(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return {"INVALID": 1, "PAUSE": 2}.get(report["decision"], 0)


def _freeze(args: argparse.Namespace) -> int:
    campaign = _read(args.campaign)
    parent_plan = _read(args.parent_plan)
    parent_manifest = _read(args.parent_manifest)
    continuation_plan = _read(args.continuation_plan)
    requests = build_replacement_requests(campaign, parent_plan, parent_manifest)
    if not requests:
        raise ValueError("parent manifest has no deferred variations to replace")
    validate_replacement_plan(requests, continuation_plan)
    segment = create_continuation_segment(
        campaign,
        _read(args.parent_segment),
        parent_manifest,
        segment_id=args.segment_id,
        git_commit=args.git_commit,
        plan_sha256=continuation_plan["plan_sha256"],
        oracle_profile_allowlist=args.oracle_profile,
    )
    _write(args.replacement_output, requests)
    _write(args.segment_output, segment)
    print(json.dumps(segment, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = _parser().parse_args()
    return _evaluate(args) if args.command == "evaluate" else _freeze(args)


if __name__ == "__main__":
    raise SystemExit(main())
