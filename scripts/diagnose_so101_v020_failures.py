#!/usr/bin/env python3
"""Write a stratified immutable failure diagnosis for an SO-101 v0.2.0 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.v020_diagnosis import build_v020_failure_diagnosis  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable diagnosis: {args.output}")
    report = build_v020_failure_diagnosis(
        json.loads(args.plan.read_text(encoding="utf-8")),
        json.loads(args.manifest.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"SO101_V020_DIAGNOSIS failures={report['failure_count']} "
        f"sha256={report['diagnosis_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
