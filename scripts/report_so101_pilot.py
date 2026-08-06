#!/usr/bin/env python3
"""Cross-check an SO-101 pilot manifest and emit JSON/Markdown evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.so101_pilot_report import (  # noqa: E402
    build_so101_pilot_report,
    render_so101_pilot_report_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--episodes-root", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = build_so101_pilot_report(plan, manifest, args.episodes_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            render_so101_pilot_report_markdown(report), encoding="utf-8"
        )
    print(
        f"SO101_PILOT_REPORT status={report['pilot_status']} "
        f"success={report['success_count']}/{report['required_successes']}"
    )
    return 0 if report["pilot_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
