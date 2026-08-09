#!/usr/bin/env python3
"""Create a frozen recovery plan for missing fixed-yaw variations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.so101_yaw_recovery import build_yaw_recovery_plan  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-plan", required=True, type=Path)
    parser.add_argument("--source-plan", action="append", required=True, type=Path)
    parser.add_argument("--source-manifest", action="append", required=True, type=Path)
    parser.add_argument("--recovery-id", required=True)
    parser.add_argument("--maximum-attempts", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.source_plan) != len(args.source_manifest):
        parser.error("--source-plan and --source-manifest counts must match")
    plan = build_yaw_recovery_plan(
        _read(args.reference_plan),
        [
            (_read(plan), _read(manifest))
            for plan, manifest in zip(args.source_plan, args.source_manifest)
        ],
        recovery_id=args.recovery_id,
        maximum_attempts=args.maximum_attempts,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing plan: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"SO101_YAW_RECOVERY_PLAN_OK missing={len(plan['trials'])} "
        f"attempt_budget={plan['collection']['maximum_attempts']} "
        f"plan={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
