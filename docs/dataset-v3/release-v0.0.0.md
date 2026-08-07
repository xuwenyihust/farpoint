# Farpoint SO-101 v0.0.0 release candidate

## Scope

This release introduces the extensible `wenyixu101/farpoint-so101` dataset
repository. The initial version contains the accepted 50-episode SO-101 cube
pick-and-place Balanced50 candidate. Future versions may add objects, targets,
tasks, and sensors to the same dataset repository.

## Source evidence

- Candidate: `so101_cube_pick_place_formal_v0_0_0_balanced50_candidate_20260807_2413c47`
- Source collection: `so101_cube_pick_place_formal_v0_0_0_20260806_b5c924e`
- Episode-generating revision: `b5c924e2fdf20106b2a480de533e5b3f8e9abc4a`
- Selection/export revision: `2413c47fd023ced4cc74281d2a0c6d3a490dc6bc`
- Candidate final audit: PASS
- LeRobot validator: PASS
- LeRobot readback: PASS
- Visual readback: PASS

## Release contents

- 50 episodes and 34,757 frames
- 40 train, 5 validation, and 5 test episodes
- LeRobot Dataset v3
- One 640 x 480, 30 Hz front-camera AV1 video stream
- Six-dimensional `observation.state` and `action`
- Farpoint v3 episode and variation metadata

Generated episodes, videos, Parquet shards, local paths, and credentials are
not committed to the Farpoint code repository.

## Promotion plan

1. Build, validate, and stage the immutable public package.
2. Upload it to the `v0.0.0-rc1` branch of
   `wenyixu101/farpoint-so101`.
3. Validate the Dataset Card, branch downloads, checksums, Parquet tables,
   video decoding, and `LeRobotDataset` loading.
4. Upload the same package to `main` without creating a release tag.
5. Validate the Hugging Face Dataset Viewer against `main`.
6. After explicit owner approval, tag the validated main commit as `v0.0.0`.

If an RC gate fails, do not tag the dataset. Fix the package and publish a new
`v0.0.0-rcN` branch.
