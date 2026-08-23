#!/usr/bin/env python3
"""Create immutable SO-101 v0.2.0 pad, combined, or formal plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.v020_plan import (  # noqa: E402
    build_v020_plan,
    initialize_v020_campaign,
    load_v020_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pad-pilot", "combined-pilot", "formal"))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/variations/so101_v020_nominal300.json")
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pad", choices=("90x90", "100x90"), default="90x90")
    parser.add_argument("--initialize-campaign-root", type=Path)
    parser.add_argument("--git-commit")
    parser.add_argument("--pilot-authorization", type=Path)
    args = parser.parse_args()
    config = load_v020_config(args.config, project_root=PROJECT_ROOT)
    dimensions = [0.09, 0.09, 0.01] if args.pad == "90x90" else [0.10, 0.09, 0.01]
    authorization = (
        json.loads(args.pilot_authorization.read_text(encoding="utf-8"))
        if args.pilot_authorization is not None
        else None
    )
    plan = build_v020_plan(
        config,
        project_root=PROJECT_ROOT,
        plan_id=args.plan_id,
        mode=args.mode,
        pad_dimensions_m=dimensions,
        pilot_authorization=authorization,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable plan: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.initialize_campaign_root is not None:
        if not args.git_commit or len(args.git_commit) != 40:
            raise ValueError("campaign initialization requires --git-commit with an exact commit")
        initialize_v020_campaign(args.initialize_campaign_root, plan, git_commit=args.git_commit)
    print(json.dumps({"mode": args.mode, "plan_id": args.plan_id, "plan_sha256": plan["plan_sha256"], "trial_count": len(plan["trials"]), "coverage": plan["coverage"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
