#!/usr/bin/env python3
"""Create a frozen plan for variations missing from an aborted mass run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.so101_mass_continuation import (  # noqa: E402
    build_mass_continuation_plan,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", required=True, type=Path)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--continuation-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = build_mass_continuation_plan(
        _read(args.parent_plan),
        _read(args.parent_manifest),
        continuation_id=args.continuation_id,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing plan: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"SO101_MASS_CONTINUATION_PLAN_OK missing={len(plan['trials'])} "
        f"attempt_budget={plan['collection']['maximum_attempts']} "
        f"plan={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
