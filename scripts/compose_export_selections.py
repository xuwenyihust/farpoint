#!/usr/bin/env python3
"""Compose multiple immutable Farpoint export selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.export_selection import compose_export_selections  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--selection-policy", required=True)
    parser.add_argument(
        "--selection",
        nargs=2,
        action="append",
        metavar=("ROLE", "PATH"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selection = compose_export_selections(
        [(role, Path(path)) for role, path in args.selection],
        dataset_id=args.dataset_id,
        selection_policy=args.selection_policy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
