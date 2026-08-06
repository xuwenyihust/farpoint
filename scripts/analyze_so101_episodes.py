#!/usr/bin/env python3
"""Build JSON and Markdown evidence reports for raw SO-101 episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.so101_episode_analysis import (  # noqa: E402
    analyze_so101_episodes,
    render_so101_analysis_markdown,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dirs", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    analysis = analyze_so101_episodes(args.episode_dirs)
    encoded = json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            render_so101_analysis_markdown(analysis), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
