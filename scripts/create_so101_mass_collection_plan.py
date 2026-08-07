#!/usr/bin/env python3
"""Create the frozen balanced50-mirrored SO-101 mass collection plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.object_variation import load_variation_config  # noqa: E402
from farpoint.so101_mass_collection import (  # noqa: E402
    build_mirrored_mass_collection_plan,
    load_mass_collection_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--collection-config",
        type=Path,
        default=PROJECT_ROOT / "configs/collections/so101_cube_mass_003_v0_0_1.json",
    )
    parser.add_argument(
        "--variation-config",
        type=Path,
        default=PROJECT_ROOT / "configs/variations/so101_cube_pick_place_v1.json",
    )
    args = parser.parse_args()
    plan = build_mirrored_mass_collection_plan(
        load_variation_config(args.variation_config),
        load_mass_collection_config(args.collection_config),
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing plan: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(plan['trials'])} mirrored 0.03 kg trials to {args.output}; "
        f"success target={plan['collection']['required_successes']} "
        f"attempt budget={plan['collection']['maximum_attempts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
