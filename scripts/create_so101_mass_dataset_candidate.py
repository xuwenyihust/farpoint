#!/usr/bin/env python3
"""Combine v0.0.0 balanced50 with a passing mirrored 0.03 kg collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.so101_mass_candidate import build_mass_dataset_candidate  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--candidate-plan", required=True, type=Path)
    parser.add_argument("--baseline-episodes-root", required=True, type=Path)
    parser.add_argument("--candidate-episodes-root", required=True, type=Path)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--selection-output", required=True, type=Path)
    args = parser.parse_args()
    manifest, selection = build_mass_dataset_candidate(
        _read(args.baseline_manifest),
        _read(args.candidate_manifest),
        _read(args.candidate_plan),
        collection_id=args.collection_id,
        baseline_episodes_root=args.baseline_episodes_root,
        candidate_episodes_root=args.candidate_episodes_root,
    )
    _write_new(args.manifest_output, manifest)
    _write_new(args.selection_output, selection)
    print(
        f"SO101_MASS_CANDIDATE_OK episodes={len(selection['episodes'])} "
        f"collection={manifest['collection_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
