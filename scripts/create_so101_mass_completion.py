#!/usr/bin/env python3
"""Validate parent plus continuation and create a 50-variation selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.so101_mass_continuation import (  # noqa: E402
    build_mass_completion_report,
    build_mass_completion_selection,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", required=True, type=Path)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--parent-episodes-root", required=True, type=Path)
    parser.add_argument("--continuation-plan", required=True, type=Path)
    parser.add_argument("--continuation-manifest", required=True, type=Path)
    parser.add_argument("--continuation-episodes-root", required=True, type=Path)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--selection-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    args = parser.parse_args()
    parent_plan = _read(args.parent_plan)
    parent_manifest = _read(args.parent_manifest)
    continuation_plan = _read(args.continuation_plan)
    continuation_manifest = _read(args.continuation_manifest)
    report = build_mass_completion_report(
        parent_plan,
        parent_manifest,
        continuation_plan,
        continuation_manifest,
        parent_episodes_root=args.parent_episodes_root,
        continuation_episodes_root=args.continuation_episodes_root,
    )
    _write_new(args.report_output, report)
    if report["status"] != "PASS":
        print(
            "SO101_MASS_COMPLETION_INVALID "
            + ";".join(report["evidence_errors"])
        )
        return 2
    manifest, selection, _report = build_mass_completion_selection(
        parent_plan,
        parent_manifest,
        continuation_plan,
        continuation_manifest,
        parent_episodes_root=args.parent_episodes_root,
        continuation_episodes_root=args.continuation_episodes_root,
        collection_id=args.collection_id,
    )
    _write_new(args.manifest_output, manifest)
    _write_new(args.selection_output, selection)
    print(
        f"SO101_MASS_COMPLETION_OK episodes={len(selection['episodes'])} "
        f"collection={manifest['collection_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
