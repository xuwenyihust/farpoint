#!/usr/bin/env python3
"""Create a balanced SO-101 collection candidate and LeRobot export selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.so101_balanced_selection import (  # noqa: E402
    build_artifacts,
    select_balanced_attempts,
    validate_balance,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variation-plan", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    manifest = read_json(args.manifest)
    plan = read_json(args.variation_plan)
    selected, stats = select_balanced_attempts(
        manifest, plan, target_count=args.target_count, seed=args.seed
    )
    errors = validate_balance(stats, target_count=args.target_count)
    if errors:
        raise ValueError("balanced selection failed: " + "; ".join(errors))
    candidate, selection = build_artifacts(
        manifest,
        plan,
        selected,
        stats,
        collection_id=args.collection_id,
        dataset_id=args.dataset_id,
        episodes_root=args.episodes_root,
        git_commit=args.git_commit,
    )
    write_json(args.output_dir / "manifest.json", candidate)
    write_json(args.output_dir / "export-selection.json", selection)
    write_json(
        args.output_dir / "selection-validation.json",
        {
            "schema_version": "farpoint.collection-selection-validation.v1",
            "valid": True,
            "collection_id": args.collection_id,
            "selected_trial_ids": [row["trial_id"] for row in selected],
            "balance": stats,
            "errors": [],
        },
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
