#!/usr/bin/env python3
"""Create a frozen paired 0.04/0.03 kg SO-101 cube feasibility plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.object_variation import load_variation_config  # noqa: E402
from farpoint.so101_mass_feasibility import (  # noqa: E402
    build_cube_mass_feasibility_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--baseline-mass-kg", type=float, default=0.04)
    parser.add_argument("--candidate-mass-kg", type=float, default=0.03)
    parser.add_argument("--edge-m", type=float, default=0.03)
    parser.add_argument("--position-xy-m", nargs=2, type=float, default=(0.20, -0.095))
    parser.add_argument("--repetitions-per-mass", type=int, default=5)
    parser.add_argument("--minimum-successes-per-mass", type=int, default=4)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/variations/so101_cube_pick_place_v1.json",
    )
    args = parser.parse_args()
    plan = build_cube_mass_feasibility_plan(
        load_variation_config(args.config),
        profile_id=args.profile_id,
        baseline_mass_kg=args.baseline_mass_kg,
        candidate_mass_kg=args.candidate_mass_kg,
        edge_m=args.edge_m,
        position_xy_m=tuple(args.position_xy_m),
        repetitions_per_mass=args.repetitions_per_mass,
        minimum_successes_per_mass=args.minimum_successes_per_mass,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(plan['trials'])} paired mass trials to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
