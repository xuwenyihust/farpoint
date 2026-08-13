#!/usr/bin/env python3
"""Build a frozen state-restored expert replay for recovery demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.recovery_replay import write_recovery_replay_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-count", type=int, default=3)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--action-safety-reference-suite-id", required=True)
    parser.add_argument("--action-safety-reference-report-sha256", required=True)
    parser.add_argument("--reference-minimum-delta-limited-actions", type=int, required=True)
    parser.add_argument("--reference-maximum-delta-limited-actions", type=int, required=True)
    parser.add_argument("--allowed-maximum-delta-limited-actions", type=int, required=True)
    args = parser.parse_args()
    result = write_recovery_replay_bundle(
        args.selection,
        args.template,
        args.runtime,
        args.output_root,
        scene_count=args.scene_count,
        suite_id=args.suite_id,
        action_safety_calibration={
            "reference_suite_id": args.action_safety_reference_suite_id,
            "reference_report_sha256": args.action_safety_reference_report_sha256,
            "reference_minimum_delta_limited_actions_per_episode": (
                args.reference_minimum_delta_limited_actions
            ),
            "reference_maximum_delta_limited_actions_per_episode": (
                args.reference_maximum_delta_limited_actions
            ),
            "allowed_maximum_delta_limited_actions_per_episode": (
                args.allowed_maximum_delta_limited_actions
            ),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
