# Farpoint Pipelines

This directory contains the public orchestration layer for Farpoint. Pipeline
entry points should be small, composable commands that call reusable code from
`src/farpoint/`.

The stable release command reads its version and dataset identity from the
repository-level `release.toml`:

```text
python scripts/release_dataset.py build \
  --source-dataset outputs/datasets/<dataset-candidate> \
  --output-dir outputs/releases/<candidate-id>
```

For a fresh release from raw episodes, replace `--source-dataset` with
`--episode-root` and repeated `--episode-id` values. The pipeline creates a
benchmark manifest, exports canonical LeRobot data, validates it, creates a
Dataset Viewer-compatible public package, and writes `release.json`. Validate
and stage the immutable candidate before publishing:

```text
python scripts/release_dataset.py validate outputs/releases/<candidate-id>
python scripts/release_dataset.py stage outputs/releases/<candidate-id>
python scripts/release_dataset.py publish outputs/releases/<candidate-id> \
  --confirm-version <tag-from-release.toml>
```

The publish command refuses candidates that do not have a successful staging
record or whose version differs from `release.toml`.

The source dataset path is intentionally explicit. Simulation execution remains
an upstream step so a release command never starts Isaac Sim unexpectedly.

For the intra-task diversity pilot, audit a deterministic two-seed matrix and
create its benchmark manifest with:

```text
python scripts/create_variation_pilot.py \
  --episode-root outputs/episodes \
  --config configs/variations/ur10e_robotiq_2f85_pickup.json \
  --output outputs/benchmarks/farpoint_v1_1_pilot/manifest.json \
  --benchmark-id farpoint_v1_1_pilot
```

Isaac Sim containers, DGX paths, credentials, and generated artifacts belong in
configuration or environment variables, never in reusable library modules.
