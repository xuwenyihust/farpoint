#!/usr/bin/env python3
"""Freeze a v0.1.0 same-quota replacement plan and continuation segment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.campaign_recovery import (  # noqa: E402
    build_replacement_requests,
    create_continuation_segment,
    validate_replacement_plan,
)
from farpoint.v010_formal import (  # noqa: E402
    build_v010_replacement_plan,
    load_v010_formal_config,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _replace(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--parent-segment", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--oracle-profile", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    config, base = load_v010_formal_config(
        args.config, project_root=PROJECT_ROOT
    )
    campaign = _read(args.campaign)
    parent_plan = _read(args.parent_plan)
    parent_manifest = _read(args.parent_manifest)
    requests = build_replacement_requests(campaign, parent_plan, parent_manifest)
    plan = build_v010_replacement_plan(
        config,
        base,
        parent_plan["pilot_authorization"],
        campaign,
        requests,
        segment_id=args.segment_id,
    )
    validate_replacement_plan(requests, plan)
    segment = create_continuation_segment(
        campaign,
        _read(args.parent_segment),
        parent_manifest,
        segment_id=args.segment_id,
        git_commit=args.git_commit,
        plan_sha256=plan["plan_sha256"],
        oracle_profile_allowlist=args.oracle_profile,
    )
    destination = args.output_root / "segments" / args.segment_id
    requests_path = destination / "replacement-requests.json"
    plan_path = destination / "plan.json"
    segment_path = destination / "segment.json"
    existing = [path for path in (requests_path, plan_path, segment_path) if path.exists()]
    if existing:
        raise FileExistsError(existing[0])
    index_path = args.output_root / "evidence-index.json"
    evidence_index = _read(index_path)
    if evidence_index.get("campaign_id") != campaign["campaign_id"]:
        raise ValueError("evidence index campaign identity mismatch")
    evidence_entry = {
        "segment": f"segments/{args.segment_id}/segment.json",
        "plan": f"segments/{args.segment_id}/plan.json",
        "manifest": f"segments/{args.segment_id}/manifest.json",
    }
    if evidence_entry in (evidence_index.get("segments") or []):
        raise ValueError("continuation is already present in evidence index")
    _write(requests_path, requests)
    _write(plan_path, plan)
    _write(segment_path, segment)
    evidence_index.setdefault("segments", []).append(evidence_entry)
    _replace(index_path, evidence_index)
    result = {
        "campaign_id": campaign["campaign_id"],
        "segment_id": segment["segment_id"],
        "segment_index": segment["segment_index"],
        "plan_sha256": plan["plan_sha256"],
        "parent_manifest_sha256": segment["parent_manifest_sha256"],
        "replacement_count": len(requests),
        "evidence_index_entry": evidence_entry,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
