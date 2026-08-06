#!/usr/bin/env python3
"""Create the frozen 10-success SO-101 code-review pilot plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.object_variation import load_variation_config  # noqa: E402
from farpoint.so101_pilot import build_so101_pilot_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pilot-id", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/variations/so101_cube_pick_place_v1.json",
    )
    args = parser.parse_args()
    plan = build_so101_pilot_plan(
        load_variation_config(args.config), pilot_id=args.pilot_id
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {plan['pilot']['required_successes']}-success pilot with "
        f"{plan['pilot']['maximum_attempts']} frozen attempts to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
