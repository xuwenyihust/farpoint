#!/usr/bin/env python3
"""Create a frozen fixed-cube repeatability gate plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.object_variation import load_variation_config  # noqa: E402
from farpoint.so101_gate import build_fixed_cube_gate_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gate-id", required=True)
    parser.add_argument("--edge-m", required=True, type=float, choices=(0.03, 0.04))
    parser.add_argument("--position-xy-m", nargs=2, required=True, type=float)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/variations/so101_cube_pick_place_v1.json",
    )
    args = parser.parse_args()
    plan = build_fixed_cube_gate_plan(
        load_variation_config(args.config),
        gate_id=args.gate_id,
        edge_m=args.edge_m,
        position_xy_m=tuple(args.position_xy_m),
        repetitions=args.repetitions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(plan['trials'])} frozen gate trials to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
