#!/usr/bin/env python3
"""Write a machine-readable recovery expert-replay integrity gate report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farpoint.recovery_replay_audit import build_recovery_replay_integrity_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_recovery_replay_integrity_report(
        selection_path=args.selection,
        spec_path=args.spec,
        replay_manifest_path=args.replay_manifest,
        run_root=args.run_root,
        expected_git_commit=args.expected_git_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
