#!/usr/bin/env python3
"""Stage a static Hugging Face Space from a quality report artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.dataset_quality_space import audit_quality_space, stage_quality_space  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = stage_quality_space(args.template, args.report, args.output)
    print(json.dumps({"index": index, "audit": audit_quality_space(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
