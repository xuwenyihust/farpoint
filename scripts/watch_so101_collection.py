#!/usr/bin/env python3
"""Evaluate an SO-101 collection once and emit a machine-readable stop decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.so101_watchdog import (  # noqa: E402
    evaluate_so101_collection,
    load_watchdog_policy,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--episodes-root", required=True, type=Path)
    parser.add_argument(
        "--policy",
        type=Path,
        default=PROJECT_ROOT / "configs/workflows/so101_watchdog_p0.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_so101_collection(
        json.loads(args.plan.read_text(encoding="utf-8")),
        json.loads(args.manifest.read_text(encoding="utf-8")),
        load_watchdog_policy(args.policy),
        episodes_root=args.episodes_root,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")
    return {"CONTINUE": 0, "COMPLETE": 0, "STOP": 2, "INVALID": 3}[
        report["decision"]
    ]


if __name__ == "__main__":
    raise SystemExit(main())
