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
    build_campaign_export_selection,
    build_replacement_requests,
    create_continuation_segment,
    evaluate_self_healing_campaign,
    validate_replacement_plan,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("evidence paths must be relative and cannot contain '..'")
    resolved_base = base.resolve()
    resolved = (resolved_base / path).resolve()
    if resolved != resolved_base and resolved_base not in resolved.parents:
        raise ValueError("evidence path escapes its index root")
    return resolved


def _write(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _resolve_episodes_root(base: Path, row: dict, manifest_path: Path) -> Path:
    """Resolve episode evidence, including the legacy segment-000 index alias."""
    value = row.get("episodes_root") or str(Path(row["manifest"]).parent / "episodes")
    resolved = _resolve(base, value)
    if resolved.exists() or Path(value) != Path("episodes"):
        return resolved
    segment_root = (manifest_path.parent / "episodes").resolve()
    if segment_root.exists():
        return segment_root
    return resolved


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
    evaluate.add_argument("--quality-exclusions", type=Path)
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

    export = commands.add_parser("export-selection")
    export.add_argument("--campaign", required=True, type=Path)
    export.add_argument("--evidence-index", required=True, type=Path)
    export.add_argument("--dataset-id", required=True)
    export.add_argument("--quality-exclusions", type=Path)
    export.add_argument("--output", required=True, type=Path)
    return parser


def _load_evidence(index_path: Path) -> list[dict]:
    index = _read(index_path)
    base = index_path.parent
    evidence = []
    for row in index.get("segments") or []:
        manifest_path = _resolve(base, row["manifest"])
        evidence.append(
            {
                "segment": _read(_resolve(base, row["segment"])),
                "plan": _read(_resolve(base, row["plan"])),
                "manifest": _read(manifest_path),
                "episodes_root": str(_resolve_episodes_root(base, row, manifest_path)),
            }
        )
    return evidence


def _evaluate(args: argparse.Namespace) -> int:
    evidence = _load_evidence(args.evidence_index)
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
        quality_exclusions=(
            _read(args.quality_exclusions)
            if args.quality_exclusions is not None
            else None
        ),
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


def _export(args: argparse.Namespace) -> int:
    selection = build_campaign_export_selection(
        _read(args.campaign),
        _load_evidence(args.evidence_index),
        dataset_id=args.dataset_id,
        quality_exclusions=(
            _read(args.quality_exclusions)
            if args.quality_exclusions is not None
            else None
        ),
    )
    _write(args.output, selection)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "freeze-continuation":
        return _freeze(args)
    return _export(args)


if __name__ == "__main__":
    raise SystemExit(main())
