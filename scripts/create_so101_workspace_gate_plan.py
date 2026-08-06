#!/usr/bin/env python3
"""Create the frozen two-size by five-position SO-101 workspace gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.object_variation import load_variation_config  # noqa: E402
from farpoint.so101_gate import build_cube_workspace_matrix_plan  # noqa: E402


DEFAULT_POSITIONS = (
    (0.15, -0.11),
    (0.25, -0.11),
    (0.20, -0.07),
    (0.15, -0.03),
    (0.25, -0.03),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gate-id", required=True)
    parser.add_argument("--minimum-success-rate", type=float, default=0.90)
    parser.add_argument(
        "--positions-xy-m",
        nargs=10,
        type=float,
        metavar=("X0", "Y0", "X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/variations/so101_cube_pick_place_v1.json",
    )
    args = parser.parse_args()
    flat = args.positions_xy_m
    positions = (
        list(DEFAULT_POSITIONS)
        if flat is None
        else [(flat[index], flat[index + 1]) for index in range(0, 10, 2)]
    )
    plan = build_cube_workspace_matrix_plan(
        load_variation_config(args.config),
        gate_id=args.gate_id,
        positions_xy_m=positions,
        minimum_success_rate=args.minimum_success_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(plan['trials'])} frozen workspace cells to {args.output}; "
        f"requires {plan['gate']['required_successes']} successes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
