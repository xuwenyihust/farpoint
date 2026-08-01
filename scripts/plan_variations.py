#!/usr/bin/env python3
"""Print a deterministic Farpoint variation plan without simulation."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.variation import load_variation_config, plan_variations  # noqa: E402


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "variations" / "ur10e_robotiq_2f85_pickup.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        help="Seed to include; may be repeated (default: 0, 1).",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else [0, 1]
    config = load_variation_config(args.config)
    plan = plan_variations(config, seeds)
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    print(f"schema_version: {config['schema_version']}")
    print(f"profiles: {len(config['profiles'])}, seeds: {seeds}, episodes: {len(plan)}")
    for item in plan:
        position = ", ".join(f"{value:.6f}" for value in item["object_position_xy"])
        print(
            f"{item['variation_id']:<26} seed={item['seed']:<4} "
            f"object={item['object_type']:<8} bin={item['object_position_bin']:<6} "
            f"xy=({position}) derived_seed={item['derived_seed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
