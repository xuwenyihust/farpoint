#!/usr/bin/env python3
"""Resolve explicit cleanup candidates into a hashed, non-destructive manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.cleanup_manifest import build_cleanup_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="JSON array; every path must explicitly declare retain/disposable and a reason")
    parser.add_argument("--protected-root", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite cleanup manifest: {args.output}")
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    manifest = build_cleanup_manifest(candidates, protected_roots=args.protected_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_sha256": manifest["manifest_sha256"], "entries": len(manifest["entries"]), "disposable": sum(row["disposition"] == "disposable" for row in manifest["entries"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
