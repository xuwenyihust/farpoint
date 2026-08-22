#!/usr/bin/env python3
"""Create a frozen v0.2.0 formal authorization from passed pilot reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.v020_plan import (  # noqa: E402
    build_v020_pilot_authorization,
    load_v020_config,
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/variations/so101_v020_nominal300.json")
    parser.add_argument("--pad-plan", type=Path, required=True)
    parser.add_argument("--pad-manifest", type=Path, required=True)
    parser.add_argument("--pad-report", type=Path, required=True)
    parser.add_argument("--combined-plan", type=Path, required=True)
    parser.add_argument("--combined-manifest", type=Path, required=True)
    parser.add_argument("--combined-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite authorization: {args.output}")
    authorization = build_v020_pilot_authorization(
        load_v020_config(args.config, project_root=PROJECT_ROOT),
        pad_plan=_json(args.pad_plan), pad_manifest_path=args.pad_manifest,
        pad_report_path=args.pad_report, combined_plan=_json(args.combined_plan),
        combined_manifest_path=args.combined_manifest,
        combined_report_path=args.combined_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(authorization, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
