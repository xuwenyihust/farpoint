#!/usr/bin/env python3
"""Create immutable exclusions for selected episodes that fail a later quality gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.campaign_recovery import create_campaign_quality_exclusions  # noqa: E402


def _read(path: Path) -> dict | list:
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


def _load_evidence(index_path: Path) -> list[dict]:
    index = _read(index_path)
    if not isinstance(index, dict):
        raise ValueError("evidence index must be an object")
    base = index_path.parent
    return [
        {
            "segment": _read(_resolve(base, row["segment"])),
            "plan": _read(_resolve(base, row["plan"])),
            "manifest": _read(_resolve(base, row["manifest"])),
        }
        for row in index.get("segments") or []
    ]


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--evidence-index", required=True, type=Path)
    parser.add_argument("--exclusions", required=True, type=Path)
    parser.add_argument("--exclusion-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    campaign = _read(args.campaign)
    requested = _read(args.exclusions)
    if not isinstance(campaign, dict):
        raise ValueError("campaign must be an object")
    if isinstance(requested, dict):
        requested = requested.get("exclusions") or []
    if not isinstance(requested, list):
        raise ValueError("exclusions input must be a list")
    artifact = create_campaign_quality_exclusions(
        campaign,
        _load_evidence(args.evidence_index),
        requested,
        exclusion_id=args.exclusion_id,
    )
    _write_new(args.output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
