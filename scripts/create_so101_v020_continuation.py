#!/usr/bin/env python3
"""Build an immutable v0.2.0 continuation from all prior segment evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.campaign_recovery import build_continuation_requests  # noqa: E402
from farpoint.v020_plan import (  # noqa: E402
    build_v020_continuation_plan,
    initialize_v020_campaign,
    load_v020_config,
    remaining_v020_attempt_budget,
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/variations/so101_v020_nominal300.json")
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--evidence-index", type=Path, required=True)
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--attempt-budget-extension", type=Path)
    args = parser.parse_args()
    index = _read(args.evidence_index)
    base = args.evidence_index.parent
    evidence = []
    plans = []
    total_attempts = 0
    parent_manifest_path = None
    for row in index.get("segments") or []:
        segment = _read(base / row["segment"])
        plan = _read(base / row["plan"])
        manifest_path = base / row["manifest"]
        manifest = _read(manifest_path)
        evidence.append({"segment": segment, "plan": plan, "manifest": manifest})
        plans.append(plan)
        total_attempts += len(manifest.get("attempts") or [])
        parent_manifest_path = manifest_path
    if not evidence or parent_manifest_path is None:
        raise ValueError("continuation requires prior segment evidence")
    campaign = _read(args.campaign_root / "campaign.json")
    budget_extension = (
        _read(args.attempt_budget_extension)
        if args.attempt_budget_extension is not None
        else None
    )
    requests = build_continuation_requests(campaign, evidence)
    remaining = remaining_v020_attempt_budget(
        campaign,
        total_attempts,
        attempt_budget_extension=budget_extension,
    )
    plan = build_v020_continuation_plan(
        load_v020_config(args.config, project_root=PROJECT_ROOT),
        project_root=PROJECT_ROOT,
        source_plan=plans,
        requests=requests,
        segment_id=args.segment_id,
        parent_manifest_sha256=hashlib.sha256(parent_manifest_path.read_bytes()).hexdigest(),
        remaining_global_attempts=remaining,
        attempt_budget_extension=budget_extension,
    )
    if args.output_plan.exists():
        raise FileExistsError(f"refusing to overwrite continuation: {args.output_plan}")
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    initialize_v020_campaign(
        args.campaign_root,
        plan,
        git_commit=args.git_commit,
        segment_id=args.segment_id,
        parent_manifest_sha256=plan["continuation"]["parent_manifest_sha256"],
    )
    print(json.dumps({"plan_sha256": plan["plan_sha256"], "requests": len(requests), "remaining_global_attempts": remaining}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
