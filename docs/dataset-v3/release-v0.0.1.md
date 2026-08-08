# Farpoint SO-101 v0.0.1 release candidate

## Scope

This release extends `wenyixu101/farpoint-so101` from 50 to 100 successful
SO-101 cube pick-and-place demonstrations. It adds a 0.03 kg cube stratum that
mirrors the existing 0.04 kg position, size, color, and split distribution.
The policy feature schema remains unchanged.

## Source evidence

- Candidate: `farpoint_so101_v0_0_1_candidate_20260808_ec9f8c9`
- Candidate assembly revision: `ec9f8c9cb5f7296f17b059af6d00bc1264526089`
- Original 0.04 kg source: `so101_cube_pick_place_formal_v0_0_0_balanced50_candidate_20260807_2413c47`
- New 0.03 kg completion: `so101_cube_mass_003_completion50_v0_0_1_20260808_ec9f8c9`
- Recovery subset: 6 successes from 6 attempts
- Candidate manifest: PASS
- LeRobot validator: PASS
- `LeRobotDataset` 0.4.4 readback: PASS
- Parquet numeric and metadata audit: PASS
- Dashboard report and preview readback: PASS

## Release contents

- 100 episodes and 72,433 frames
- 80 train, 10 validation, and 10 test episodes
- 50 episodes at 0.03 kg and 50 episodes at 0.04 kg
- Complete 5 x 5 position coverage at each mass
- Balanced 0.03 m / 0.04 m cube sizes and red / blue appearances at each mass
- LeRobot Dataset v3
- One 640 x 480, 30 Hz front-camera video stream
- Six-dimensional `observation.state` and `action`
- Farpoint v3 episode and variation metadata

Generated episodes, videos, Parquet shards, local paths, and credentials are
not committed to the Farpoint code repository.

## Promotion plan

1. Build, validate, and stage the immutable public package.
2. Upload it to the `v0.0.1-rc1` branch of
   `wenyixu101/farpoint-so101`.
3. Validate the Dataset Card, Dataset Viewer, checksums, Parquet tables, video
   decoding, and `LeRobotDataset` loading from the RC branch.
4. Promote the same staged package to `main`.
5. Re-run the Viewer and loader checks against `main`.
6. Tag the validated Hub commit as `v0.0.1`.

If an RC gate fails, do not update `main` or create the version tag. Fix the
package and publish a new `v0.0.1-rcN` branch.
