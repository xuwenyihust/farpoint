#!/usr/bin/env python3
"""Create or verify an immutable Farpoint cube position trial manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.position_plan import generate_position_plan, load_position_config  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "variations" / "farpoint_v1_3_cube_position.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "configs" / "plans" / "farpoint_v1_3_cube_position_baseline.json"


def write_immutable(path: Path, manifest: dict) -> str:
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(
                f"refusing to overwrite immutable plan {path}; use a new plan_id/output path"
            )
        return "verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return "created"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()
    manifest = generate_position_plan(load_position_config(args.config))
    if args.stdout:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    action = write_immutable(args.output, manifest)
    counts = {split: sum(item["split"] == split for item in manifest["trials"]) for split in ("train", "validation", "test")}
    print(
        f"POSITION_PLAN {action} plan_id={manifest['plan_id']} "
        f"trials={len(manifest['trials'])} splits={counts} sha256={manifest['plan_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
