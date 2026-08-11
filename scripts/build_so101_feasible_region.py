#!/usr/bin/env python3
"""Build a frozen feasible-region record from headless Isaac probe evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.workspace_mapping import (  # noqa: E402
    WorkspaceProbe,
    derive_feasible_region,
    feasible_region_record,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    probes = [
        WorkspaceProbe(
            x_m=float(row["position_xy_m"][0]),
            y_m=float(row["position_xy_m"][1]),
            checks=tuple(sorted(row["checks"].items())),
            probe_id=row["probe_id"],
        )
        for row in evidence["probes"]
    ]
    region = derive_feasible_region(
        probes,
        region_id=evidence["region_id"],
        version=evidence["version"],
        frame_id=evidence["frame_id"],
        object_anchor=evidence["object_anchor"],
        footprint_xy_m=tuple(evidence["footprint_xy_m"]),
        generator_identity=evidence["generator_identity"],
    )
    output = {
        "region": feasible_region_record(region),
        "probe_count": len(probes),
        "passing_probe_count": sum(probe.passed for probe in probes),
        "source_evidence_sha256": region.constraints_sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
