#!/usr/bin/env python3
"""Validate v0.2.0 selection balance and bind LeRobot/replay evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.v020_validation import build_v020_candidate_validation  # noqa: E402


def _read(path: Path | None):
    return None if path is None else json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, action="append", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--lerobot-validation", type=Path)
    parser.add_argument("--loader-replays", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_v020_candidate_validation(
        [_read(path) for path in args.plan], _read(args.selection), candidate_root=args.candidate_root,
        lerobot_validation=_read(args.lerobot_validation),
        loader_replays=_read(args.loader_replays),
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite validation report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": report["errors"]}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
