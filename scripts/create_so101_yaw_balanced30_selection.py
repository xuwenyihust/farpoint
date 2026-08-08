#!/usr/bin/env python3
"""Create and verify the formal balanced30 candidate from an aborted source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.so101_yaw_balanced_selection import (  # noqa: E402
    SELECTION_SEED,
    build_artifacts,
    build_validation_report,
    file_sha256,
    render_validation_markdown,
    select_balanced30,
    validate_aborted_source,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variation-plan", type=Path, required=True)
    parser.add_argument("--abort-record", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(
            f"refusing to overwrite non-empty output directory: {args.output_dir}"
        )
    manifest = _read_json(args.manifest)
    plan = _read_json(args.variation_plan)
    abort_record = _read_json(args.abort_record)
    source_manifest_sha256 = file_sha256(args.manifest)
    abort_record_sha256 = file_sha256(args.abort_record)
    source_errors = validate_aborted_source(manifest, plan, abort_record)
    if source_errors:
        raise ValueError("invalid aborted source: " + "; ".join(source_errors))
    selected, stats = select_balanced30(manifest, plan, seed=args.seed)
    report = build_validation_report(
        collection_id=args.collection_id,
        source_manifest=manifest,
        plan=plan,
        abort_record=abort_record,
        selected=selected,
        stats=stats,
        episodes_root=args.episodes_root,
        source_manifest_path=args.manifest,
        abort_record_path=args.abort_record,
        source_manifest_file_sha256=source_manifest_sha256,
        abort_record_file_sha256=abort_record_sha256,
    )
    _write_json(args.output_dir / "selection-validation.json", report)
    _write_text(
        args.output_dir / "selection-validation.md",
        render_validation_markdown(report),
    )
    if not report["valid"]:
        raise ValueError(
            "balanced30 evidence validation failed: " + "; ".join(report["errors"])
        )
    candidate, selection = build_artifacts(
        manifest,
        plan,
        abort_record,
        selected,
        stats,
        collection_id=args.collection_id,
        dataset_id=args.dataset_id,
        episodes_root=args.episodes_root,
        git_commit=args.git_commit,
        source_manifest_file_sha256=source_manifest_sha256,
        abort_record_file_sha256=abort_record_sha256,
    )
    _write_json(args.output_dir / "manifest.json", candidate)
    _write_json(args.output_dir / "export-selection.json", selection)
    print(json.dumps({"valid": True, "balance": stats}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
