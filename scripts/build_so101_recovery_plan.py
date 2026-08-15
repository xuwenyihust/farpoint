#!/usr/bin/env python3
"""Build a frozen SO-101 recovery plan and ACT handoff runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.recovery_plan import (  # noqa: E402
    build_recovery_plan,
    initialize_recovery_campaign,
)
from farpoint.recovery_runtime import load_recovery_runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--scene-count", type=int, choices=(6, 16, 20, 80), default=20)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_plan.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    plan, runtime = build_recovery_plan(
        source,
        config,
        campaign_id=args.campaign_id,
        scene_count=args.scene_count,
    )
    initialized = initialize_recovery_campaign(
        args.output_root,
        plan,
        runtime,
        git_commit=args.git_commit,
    )
    runtime_path = args.output_root / "segments/segment-000/recovery-runtime.json"
    load_recovery_runtime(runtime_path)
    print(
        json.dumps(
            {
                "campaign_id": args.campaign_id,
                "scene_count": len(plan["trials"]),
                "plan_sha256": plan["plan_sha256"],
                "runtime_id": runtime["runtime_id"],
                "segment_sha256": initialized["segment"]["segment_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
