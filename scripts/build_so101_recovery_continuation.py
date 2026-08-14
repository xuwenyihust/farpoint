#!/usr/bin/env python3
"""Freeze a carryover continuation for a live-policy recovery campaign."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.campaign import canonical_sha256  # noqa: E402
from farpoint.campaign_recovery import (  # noqa: E402
    build_continuation_requests,
    create_continuation_segment,
)
from farpoint.recovery_plan import build_recovery_continuation_plan  # noqa: E402
from farpoint.recovery_runtime import load_recovery_runtime  # noqa: E402
from farpoint.v010_formal import (  # noqa: E402
    load_v010_formal_config,
    materialize_v010_recovery_replacement_trial,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("evidence paths must be relative and cannot contain '..'")
    resolved_base = base.resolve()
    resolved = (resolved_base / path).resolve()
    if resolved != resolved_base and resolved_base not in resolved.parents:
        raise ValueError("evidence path escapes its index root")
    return resolved


def _write_new(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _replace(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_evidence(index_path: Path) -> list[dict]:
    index = _read(index_path)
    base = index_path.parent
    return [
        {
            "segment": _read(_resolve(base, row["segment"])),
            "plan": _read(_resolve(base, row["plan"])),
            "manifest": _read(_resolve(base, row["manifest"])),
        }
        for row in index.get("segments") or []
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-formal-config", type=Path)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--parent-segment", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--quality-exclusions", type=Path)
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--oracle-profile", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    config = _read(args.config)
    campaign = _read(args.campaign)
    parent_segment = _read(args.parent_segment)
    parent_plan = _read(args.parent_plan)
    parent_manifest = _read(args.parent_manifest)
    index_path = args.output_root / "evidence-index.json"
    evidence_index = _read(index_path)
    evidence = _load_evidence(index_path)
    if not evidence:
        raise ValueError("evidence index must contain the parent segment")
    latest = evidence[-1]
    if canonical_sha256(latest["manifest"]) != canonical_sha256(parent_manifest):
        raise ValueError("parent manifest is not the latest indexed manifest")
    if canonical_sha256(latest["plan"]) != canonical_sha256(parent_plan):
        raise ValueError("parent plan is not the latest indexed plan")
    requests = build_continuation_requests(
        campaign,
        evidence,
        quality_exclusions=(
            _read(args.quality_exclusions)
            if args.quality_exclusions is not None
            else None
        ),
    )
    materializer = None
    if any(request.get("request_kind") == "replacement" for request in requests):
        if args.source_formal_config is None:
            raise ValueError(
                "--source-formal-config is required when replacement scenes are needed"
            )
        formal_config, base_config = load_v010_formal_config(
            args.source_formal_config, project_root=PROJECT_ROOT
        )
        materializer = partial(
            materialize_v010_recovery_replacement_trial,
            formal_config,
            base_config,
            formal_config["pilot_authorization"],
            campaign,
            segment_id=args.segment_id,
            source_plan_sha256=(campaign.get("variation_contract") or {})["source_plan_sha256"],
            allowed_splits=(
                ("train", "validation")
                if config.get("schema_version") == "farpoint.recovery-plan-config.v2"
                else ("train",)
            ),
        )
    plan, runtime = build_recovery_continuation_plan(
        parent_plan,
        config,
        campaign,
        requests,
        segment_id=args.segment_id,
        replacement_materializer=materializer,
    )
    segment = create_continuation_segment(
        campaign,
        parent_segment,
        parent_manifest,
        segment_id=args.segment_id,
        git_commit=args.git_commit,
        plan_sha256=plan["plan_sha256"],
        oracle_profile_allowlist=args.oracle_profile,
    )
    destination = args.output_root / "segments" / args.segment_id
    values = {
        "replacement-requests.json": requests,
        "plan.json": plan,
        "segment.json": segment,
        "recovery-runtime.json": runtime,
    }
    for name, value in values.items():
        _write_new(destination / name, value)
    load_recovery_runtime(destination / "recovery-runtime.json")
    if evidence_index.get("campaign_id") != campaign["campaign_id"]:
        raise ValueError("evidence index campaign identity mismatch")
    evidence_entry = {
        "segment": f"segments/{args.segment_id}/segment.json",
        "plan": f"segments/{args.segment_id}/plan.json",
        "manifest": f"segments/{args.segment_id}/manifest.json",
        "episodes_root": f"segments/{args.segment_id}/episodes",
        "recovery_runtime": f"segments/{args.segment_id}/recovery-runtime.json",
    }
    evidence_index.setdefault("segments", []).append(evidence_entry)
    _replace(index_path, evidence_index)
    print(
        json.dumps(
            {
                "campaign_id": campaign["campaign_id"],
                "segment_id": args.segment_id,
                "plan_sha256": plan["plan_sha256"],
                "parent_manifest_sha256": segment["parent_manifest_sha256"],
                "request_count": len(requests),
                "maximum_attempts": plan["collection"]["maximum_attempts"],
                "runtime_id": runtime["runtime_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
