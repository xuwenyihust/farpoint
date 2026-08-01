#!/usr/bin/env python3
"""Create a successful-only release selection from an accepted benchmark v2 manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.formal_benchmark import build_release_selection  # noqa: E402
from farpoint.release_spec import load_release_spec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-root", default="outputs/episodes")
    parser.add_argument("--dataset-id")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    benchmark = json.loads(args.benchmark_manifest.read_text(encoding="utf-8"))
    dataset_id = args.dataset_id or load_release_spec()["dataset_id"]
    selection = build_release_selection(
        benchmark,
        dataset_id=dataset_id,
        episode_root=args.episode_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"SELECTION_OK: {len(selection['episodes'])} episodes -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
