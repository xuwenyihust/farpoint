#!/usr/bin/env python3
"""Create or verify the immutable v0.0.1 yaw-aware cube plan."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.yaw_plan import generate_yaw_plan, load_yaw_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/variations/farpoint_v0_0_1_cube_yaw.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "configs/plans/farpoint_v0_0_1_cube_yaw_aware.json",
    )
    args = parser.parse_args()
    plan = generate_yaw_plan(load_yaw_config(args.config))
    payload = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != payload:
        raise SystemExit("refusing to overwrite a different yaw plan")
    args.output.write_text(payload, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
