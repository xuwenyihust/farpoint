#!/usr/bin/env python3
"""Mark an interrupted SO-101 collection and its live episode as ABORTED."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.so101_collection import (  # noqa: E402
    abort_collection_artifacts,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    audit_output = args.audit_output or args.manifest.with_name("abort_record.json")
    report = abort_collection_artifacts(
        args.manifest,
        args.episodes_root,
        args.reason,
    )
    write_manifest(audit_output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote abort audit to {audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
