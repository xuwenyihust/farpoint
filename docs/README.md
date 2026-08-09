# Farpoint Documentation

## Start Here

- [Architecture](architecture.md): system boundaries and data flow
- [Isaac Sim Setup](isaac-sim.md): public container and runtime requirements
- [Task Schema](task-schema.md): simulation task configuration
- [Cube Position Baseline](cube-position-baseline.md): deterministic 5 by 5
  position planning, immutable trial manifests, and pilot acceptance

## Dataset V1

- [Data Contract](dataset-v1/data-contract.md): LeRobot-compatible schema
- Dataset Cards are generated from each release specification and the exported
  dataset metadata by `scripts/release_dataset.py`; they are not maintained as
  hand-written documentation.

## Dataset Contract V2

- [Data Contract](dataset-v2/data-contract.md): multi-task, multi-split,
  reproducible episode and benchmark metadata for future releases

Machine-specific setup notes, benchmark run records, and data-platform
operations are kept locally under `.codex/local/`. Internal roadmaps,
sub-goals, and release checklists are kept there as well and are not part of
the public repository.

Generated episodes, reports, benchmark artifacts, and caches are stored under
`outputs/` and are intentionally excluded from the public source repository.

Source code is licensed under Apache-2.0. The separately published
`farpoint-ur10e-robotiq-2f85` dataset is intended to use CC BY 4.0, subject to third-party
asset provenance and redistribution terms.
