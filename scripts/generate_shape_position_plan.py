#!/usr/bin/env python3
"""Generate and validate an immutable Farpoint shape-position plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.contracts import validate_contract  # noqa: E402
from farpoint.shape_position import generate_shape_position_plan, load_shape_position_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan = generate_shape_position_plan(load_shape_position_config(args.config))
    errors = validate_contract(plan)
    if errors:
        raise ValueError("invalid generated plan: " + "; ".join(errors))
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PLAN_OK trials={len(plan['trials'])} sha256={plan['plan_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
