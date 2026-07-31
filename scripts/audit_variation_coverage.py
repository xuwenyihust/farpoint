#!/usr/bin/env python3
"""Audit a Farpoint variation pilot without starting Isaac Sim."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farpoint.variation import load_variation_config
from farpoint.variation_coverage import audit_variation_coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--min-passes", type=int, default=10)
    parser.add_argument("--task-name")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    config = load_variation_config(args.config)
    variation_ids = [profile["variation_id"] for profile in config["profiles"]]
    report = audit_variation_coverage(
        args.episode_root,
        variation_ids,
        expected_seeds=args.seeds,
        min_passes=args.min_passes,
        task_name=args.task_name,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Variation pilot: {report['passing_episode_count']}/{report['episode_count']} passing")
    print(f"Profiles with a passing seed: {len(variation_ids) - len(report['missing_passing_profiles'])}/{len(variation_ids)}")
    for profile in report["profiles"]:
        status = "PASS" if profile["has_passing_seed"] else "FAIL"
        print(f"{status:4} {profile['variation_id']}: {profile['passing_episodes']} passing, missing seeds={profile['missing_seeds']}")
    print(f"GATE: {'PASS' if report['gate_passed'] else 'FAIL'}")
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
