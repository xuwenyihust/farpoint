#!/usr/bin/env python3
"""Validate fixed-yaw source collections and create one candidate selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.so101_yaw_recovery import (  # noqa: E402
    build_yaw_completion_report,
    build_yaw_completion_selection,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-plan", required=True, type=Path)
    parser.add_argument("--source-plan", action="append", required=True, type=Path)
    parser.add_argument("--source-manifest", action="append", required=True, type=Path)
    parser.add_argument("--source-episodes-root", action="append", required=True, type=Path)
    parser.add_argument("--recovery-plan", required=True, type=Path)
    parser.add_argument("--recovery-manifest", required=True, type=Path)
    parser.add_argument("--recovery-episodes-root", required=True, type=Path)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--selection-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    args = parser.parse_args()
    counts = {
        len(args.source_plan),
        len(args.source_manifest),
        len(args.source_episodes_root),
    }
    if len(counts) != 1:
        parser.error("historical source argument counts must match")
    historical = [
        (_read(plan), _read(manifest), root)
        for plan, manifest, root in zip(
            args.source_plan, args.source_manifest, args.source_episodes_root
        )
    ]
    reference = _read(args.reference_plan)
    recovery_plan = _read(args.recovery_plan)
    recovery_manifest = _read(args.recovery_manifest)
    report = build_yaw_completion_report(
        reference,
        historical,
        recovery_plan,
        recovery_manifest,
        recovery_episodes_root=args.recovery_episodes_root,
    )
    _write_new(args.report_output, report)
    if report["status"] != "PASS":
        print("SO101_YAW_COMPLETION_INVALID " + ";".join(report["evidence_errors"]))
        return 2
    manifest, selection, _ = build_yaw_completion_selection(
        reference,
        historical,
        recovery_plan,
        recovery_manifest,
        recovery_episodes_root=args.recovery_episodes_root,
        collection_id=args.collection_id,
    )
    _write_new(args.manifest_output, manifest)
    _write_new(args.selection_output, selection)
    print(
        f"SO101_YAW_COMPLETION_OK episodes={len(selection['episodes'])} "
        f"collection={manifest['collection_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
