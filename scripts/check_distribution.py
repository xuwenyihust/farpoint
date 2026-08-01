#!/usr/bin/env python3
"""Verify that a built Farpoint wheel contains its runtime contract resources."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


REQUIRED_MEMBERS = {
    "farpoint/contracts.py",
    "farpoint/schemas/__init__.py",
    "farpoint/schemas/farpoint_dataset_v2.schema.json",
    "farpoint/schemas/farpoint_episode_v2.schema.json",
    "farpoint/schemas/farpoint_variation_v2.schema.json",
    "farpoint/schemas/farpoint_benchmark_v2.schema.json",
}


def check_distribution(directory: Path) -> list[str]:
    wheels = sorted(directory.glob("farpoint-*.whl"))
    if len(wheels) != 1:
        return [f"expected one Farpoint wheel in {directory}, found {len(wheels)}"]
    with zipfile.ZipFile(wheels[0]) as archive:
        members = set(archive.namelist())
    return [f"wheel is missing runtime resource: {name}" for name in sorted(REQUIRED_MEMBERS - members)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel_directory", type=Path)
    args = parser.parse_args()
    errors = check_distribution(args.wheel_directory)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("DISTRIBUTION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
