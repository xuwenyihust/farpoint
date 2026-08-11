#!/usr/bin/env python3
"""Authorize and freeze the SO-101 v0.1.0 formal 200-scene campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.v010_formal import (  # noqa: E402
    build_v010_formal_plan,
    initialize_v010_formal_campaign,
    load_v010_formal_config,
    validate_pilot_authorization,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    config, base = load_v010_formal_config(
        args.config, project_root=PROJECT_ROOT
    )
    authorization = validate_pilot_authorization(
        config,
        report_path=args.pilot_report,
        manifest_path=args.pilot_manifest,
    )
    plan = build_v010_formal_plan(
        config,
        base,
        authorization,
        campaign_id=args.campaign_id,
    )
    initialized = initialize_v010_formal_campaign(
        args.output_root, plan, git_commit=args.git_commit
    )
    result = {
        "campaign_id": initialized["campaign"]["campaign_id"],
        "campaign_sha256": initialized["campaign"]["campaign_sha256"],
        "segment_id": initialized["segment"]["segment_id"],
        "plan_sha256": initialized["plan"]["plan_sha256"],
        "trial_count": len(initialized["plan"]["trials"]),
        "holdout_scene_count": len(
            initialized["plan"]["rollout_holdout"]["scenes"]
        ),
        "formal_collection_authorized": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
