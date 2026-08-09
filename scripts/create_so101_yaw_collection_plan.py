#!/usr/bin/env python3
"""Create a frozen SO-101 fixed-yaw, 30 mm collection plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.object_variation import load_variation_config  # noqa: E402
from farpoint.so101_yaw_collection import (  # noqa: E402
    build_yaw_collection_plan,
    load_yaw_collection_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--collection-config", type=Path, default=PROJECT_ROOT / "configs/collections/so101_cube_yaw0_30mm_v0_0_2.json")
    parser.add_argument("--variation-config", type=Path, default=PROJECT_ROOT / "configs/variations/so101_cube_pick_place_v1.json")
    args = parser.parse_args()
    plan = build_yaw_collection_plan(load_variation_config(args.variation_config), load_yaw_collection_config(args.collection_config))
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing plan: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    profile = plan["collection"]
    print(
        f"Wrote {len(plan['trials'])} yaw variations to {args.output}; "
        f"success target={profile['required_successes']} "
        f"attempt budget={profile['maximum_attempts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
