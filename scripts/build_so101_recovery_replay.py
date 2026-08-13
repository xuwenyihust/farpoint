#!/usr/bin/env python3
"""Build a frozen state-restored expert replay for recovery demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from farpoint.recovery_replay import write_recovery_replay_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-count", type=int, default=3)
    parser.add_argument("--suite-id", required=True)
    args = parser.parse_args()
    result = write_recovery_replay_bundle(
        args.selection,
        args.template,
        args.runtime,
        args.output_root,
        scene_count=args.scene_count,
        suite_id=args.suite_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
