#!/usr/bin/env python3
"""Generate a precomputed Farpoint dataset quality report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.dataset_quality import generate_quality_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--resolved-dataset-commit", required=True)
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--tag-validation", type=Path)
    parser.add_argument("--viewer-validation", type=Path)
    parser.add_argument("--visual-episodes", type=int, default=12)
    parser.add_argument("--visual-sample-stride", type=int, default=30)
    parser.add_argument("--idle-delta-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = generate_quality_report(
        args.dataset_root,
        args.output,
        dataset_repo=args.dataset_repo,
        dataset_tag=args.dataset_tag,
        resolved_dataset_commit=args.resolved_dataset_commit,
        generator_commit=args.generator_commit,
        tag_validation_path=args.tag_validation,
        viewer_validation_path=args.viewer_validation,
        visual_episode_count=args.visual_episodes,
        visual_sample_stride=args.visual_sample_stride,
        idle_delta_threshold=args.idle_delta_threshold,
    )
    print(
        json.dumps(
            {
                "status": report["integrity"]["status"],
                "report_sha256": report["report_sha256"],
                "episodes": report["overview"]["episodes"],
                "frames": report["overview"]["frames"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0 if report["integrity"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
