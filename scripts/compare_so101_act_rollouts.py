#!/usr/bin/env python3
"""Build a hash-bound comparison of two paired SO-101 rollout reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from farpoint.policy_rollout import compare_paired_rollout_reports
from farpoint.policy_training import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"comparison output already exists: {args.output}")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    comparison = {
        **compare_paired_rollout_reports(baseline, candidate),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "baseline": {
                "path": str(args.baseline.resolve()),
                "sha256": file_sha256(args.baseline),
                "suite_id": baseline.get("suite_id"),
                "checkpoint": baseline.get("checkpoint"),
            },
            "candidate": {
                "path": str(args.candidate.resolve()),
                "sha256": file_sha256(args.candidate),
                "suite_id": candidate.get("suite_id"),
                "checkpoint": candidate.get("checkpoint"),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "scene_count": comparison["scene_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
