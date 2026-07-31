# Farpoint Documentation

## Start Here

- [Architecture](architecture.md): system boundaries and data flow
- [Isaac Sim Setup](isaac-sim.md): public container and runtime requirements
- [Task Schema](task-schema.md): simulation task configuration

## Dataset V1

- [Data Contract](dataset-v1/data-contract.md): LeRobot-compatible schema
- [Hugging Face Dataset Card](dataset-v1/huggingface-dataset-card.md): release
  metadata and dataset licensing template

Machine-specific setup notes, benchmark run records, and data-platform
operations are kept locally under `.codex/local/`. Internal roadmaps,
sub-goals, and release checklists are kept there as well and are not part of
the public repository.

Generated episodes, reports, benchmark artifacts, and caches are stored under
`outputs/` and are intentionally excluded from the public source repository.

Source code is licensed under Apache-2.0. The separately published
`farpoint-v1` dataset is intended to use CC BY 4.0, subject to third-party
asset provenance and redistribution terms.
