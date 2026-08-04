#!/usr/bin/env python3
"""Create the deterministic 100-trial SO-101 cube pick-and-place plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.contracts import validate_contract  # noqa: E402
from farpoint.object_variation import generate_variation_plan, load_variation_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/variations/so101_cube_pick_place_v1.json",
    )
    args = parser.parse_args()
    plan = generate_variation_plan(load_variation_config(args.config))
    errors = validate_contract(plan)
    if errors:
        parser.error("invalid generated plan: " + "; ".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(plan['trials'])} SO-101 trials to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
