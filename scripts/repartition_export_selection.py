#!/usr/bin/env python3
"""Create a deterministic dataset-split view of an immutable selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.export_selection import repartition_export_selection  # noqa: E402


def _split_target(value: str) -> tuple[str, int]:
    try:
        split, raw_count = value.split("=", 1)
        return split, int(raw_count)
    except ValueError as error:
        raise argparse.ArgumentTypeError("split target must be NAME=COUNT") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--selection-policy", required=True)
    parser.add_argument("--split", action="append", type=_split_target, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    split_counts = dict(args.split)
    if len(split_counts) != len(args.split):
        raise ValueError("split targets must not repeat a split name")
    selection = repartition_export_selection(
        args.source,
        dataset_id=args.dataset_id,
        selection_policy=args.selection_policy,
        split_counts=split_counts,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
