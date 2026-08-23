#!/usr/bin/env python3
"""Create a frozen v0.2.0 formal authorization from passed pilot reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.campaign_recovery import build_campaign_export_selection  # noqa: E402
from farpoint.v020_plan import (  # noqa: E402
    build_v020_pilot_authorization,
    load_v020_config,
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("evidence paths must be relative and cannot contain ..")
    resolved_base = base.resolve()
    resolved = (resolved_base / path).resolve()
    if resolved != resolved_base and resolved_base not in resolved.parents:
        raise ValueError("evidence path escapes its index root")
    return resolved


def _combined_campaign_record(
    campaign_path: Path,
    evidence_index_path: Path,
    report_paths: list[Path],
) -> dict:
    campaign = _json(campaign_path)
    index = _json(evidence_index_path)
    rows = index.get("segments") or []
    if not rows:
        raise ValueError("combined campaign evidence index has no segments")
    if len(report_paths) != len(rows):
        raise ValueError("combined campaign requires one full-frame report per segment")
    reports = {}
    for report_path in report_paths:
        report = _json(report_path)
        plan_sha256 = report.get("plan_sha256")
        if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
            raise ValueError("combined segment report is missing plan SHA256")
        if plan_sha256 in reports:
            raise ValueError("combined segment reports contain duplicate plan SHA256")
        reports[plan_sha256] = (report_path, report)

    base = evidence_index_path.parent
    evidence = []
    plan_sha256s = []
    manifest_sha256s = []
    report_sha256s = []
    segment_ids = []
    total_attempts = 0
    allowed_acceptance_errors = {
        "selected_success_count_below_threshold",
        "selected_success_count_mismatch",
    }
    for row in rows:
        segment_path = _resolve(base, row["segment"])
        plan_path = _resolve(base, row["plan"])
        manifest_path = _resolve(base, row["manifest"])
        episodes_root = _resolve(
            base,
            row.get("episodes_root")
            or str(Path(row["manifest"]).parent / "episodes"),
        )
        segment = _json(segment_path)
        plan = _json(plan_path)
        manifest = _json(manifest_path)
        plan_sha256 = plan.get("plan_sha256")
        if plan_sha256 not in reports:
            raise ValueError("combined segment is missing its full-frame report")
        report_path, report = reports[plan_sha256]
        evidence_errors = report.get("evidence_errors") or []
        acceptance_errors = set(report.get("acceptance_errors") or [])
        if evidence_errors:
            raise ValueError("combined segment report contains evidence errors")
        if not acceptance_errors.issubset(allowed_acceptance_errors):
            raise ValueError("combined segment report has non-aggregate acceptance errors")
        attempts = manifest.get("attempts") or []
        selected = manifest.get("selected_variations") or {}
        episode_evidence = report.get("episode_evidence") or {}
        if report.get("success_count") != len(selected):
            raise ValueError("combined segment report success count mismatch")
        if episode_evidence.get("episode_count") != len(attempts):
            raise ValueError("combined segment report attempt evidence count mismatch")
        if report.get("independent_episode_identity_count") != len(attempts):
            raise ValueError("combined segment report identity count mismatch")
        if set(report.get("required_cameras") or []) != {"front", "wrist"}:
            raise ValueError("combined segment report must require front and wrist")
        evidence.append(
            {
                "segment": segment,
                "plan": plan,
                "manifest": manifest,
                "episodes_root": str(episodes_root),
            }
        )
        plan_sha256s.append(plan_sha256)
        manifest_sha256s.append(_file_sha256(manifest_path))
        report_sha256s.append(_file_sha256(report_path))
        segment_ids.append(segment["segment_id"])
        total_attempts += len(attempts)

    if set(reports) != set(plan_sha256s):
        raise ValueError("combined segment reports do not exactly match evidence index")
    selection = build_campaign_export_selection(
        campaign,
        evidence,
        dataset_id="farpoint-so101-v020-combined-pilot",
    )
    if len(selection.get("episodes") or []) != 30:
        raise ValueError("combined campaign must select exactly 30 quotas")
    return {
        "evidence_kind": "self_healing_campaign",
        "evidence_index_sha256": _file_sha256(evidence_index_path),
        "campaign_sha256": campaign["campaign_sha256"],
        "segment_ids": segment_ids,
        "plan_sha256s": plan_sha256s,
        "manifest_sha256s": manifest_sha256s,
        "report_sha256s": report_sha256s,
        "pilot_status": "PASS",
        "success_count": 30,
        "required_successes": 30,
        "attempted_count": total_attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/variations/so101_v020_nominal300.json")
    parser.add_argument("--pad-plan", type=Path, required=True)
    parser.add_argument("--pad-manifest", type=Path, required=True)
    parser.add_argument("--pad-report", type=Path, required=True)
    parser.add_argument("--combined-plan", type=Path, required=True)
    parser.add_argument("--combined-manifest", type=Path)
    parser.add_argument("--combined-report", type=Path)
    parser.add_argument("--combined-campaign", type=Path)
    parser.add_argument("--combined-evidence-index", type=Path)
    parser.add_argument("--combined-segment-report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite authorization: {args.output}")
    aggregate_values = (
        args.combined_campaign,
        args.combined_evidence_index,
        *args.combined_segment_report,
    )
    use_aggregate = any(value is not None for value in aggregate_values)
    if use_aggregate:
        if args.combined_manifest is not None or args.combined_report is not None:
            raise ValueError("combined pilot evidence modes are mutually exclusive")
        if (
            args.combined_campaign is None
            or args.combined_evidence_index is None
            or not args.combined_segment_report
        ):
            raise ValueError("aggregate combined pilot evidence is incomplete")
        combined_campaign_record = _combined_campaign_record(
            args.combined_campaign,
            args.combined_evidence_index,
            args.combined_segment_report,
        )
    else:
        if args.combined_manifest is None or args.combined_report is None:
            raise ValueError("legacy combined pilot manifest and report are required")
        combined_campaign_record = None
    authorization = build_v020_pilot_authorization(
        load_v020_config(args.config, project_root=PROJECT_ROOT),
        pad_plan=_json(args.pad_plan),
        pad_manifest_path=args.pad_manifest,
        pad_report_path=args.pad_report,
        combined_plan=_json(args.combined_plan),
        combined_manifest_path=args.combined_manifest,
        combined_report_path=args.combined_report,
        combined_campaign_record=combined_campaign_record,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(authorization, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
