# Farpoint Pipelines

This directory contains the public orchestration layer for Farpoint. Pipeline
entry points should be small, composable commands that call reusable code from
`src/farpoint/`.

The first stable release command is available now:

```text
python scripts/release_farpoint_v1_2.py \\
  --source-dataset outputs/datasets/farpoint-ur10e-robotiq-2f85-v1.1-merged \\
  --output-dir outputs/releases/farpoint_v1_2_0 \\
  --dataset-id farpoint_ur10e_robotiq_2f85 \\
  --release-version v1.2.0
```

For a fresh release from raw episodes, replace `--source-dataset` with
`--episode-root` and repeated `--episode-id` values. The pipeline creates a
benchmark manifest, exports canonical LeRobot data, validates it, creates a
Dataset Viewer-compatible public package, and writes `release.json`. Add
`--publish --hf-repo-id USER/DATASET` only in an authenticated release
environment. Publishing creates the requested Hub tag after the upload.

The source dataset path is intentionally explicit. Simulation execution remains
an upstream step so a release command never starts Isaac Sim unexpectedly.

Isaac Sim containers, DGX paths, credentials, and generated artifacts belong in
configuration or environment variables, never in reusable library modules.
