# Farpoint SO-101 v0.0.2 release candidate

## Scope

This release extends `wenyixu101/farpoint-so101` from 100 to 130 successful
SO-101 cube pick-and-place demonstrations. It adds a 30-episode increment with
0.03 m cube edges and cube yaw fixed at 0°, balanced across mass and color.
The existing 100 episodes use 45° cube yaw. The policy feature schema remains
unchanged.

## Source evidence

- Candidate: `farpoint_so101_v0_0_2_candidate_20260808_985b72f`
- Candidate assembly revision: `985b72f4f2e01ff43c03dbe0b28ede72e395a754`
- Existing v0.0.1 source: `farpoint_so101_v0_0_1_candidate_20260808_ec9f8c9`
- New yaw source: `so101_cube_yaw0_30mm_balanced30_candidate_20260808_985b72f`
- Candidate manifest SHA256:
  `a895062be80fdb33c1e0166d51bfe77a832dc6796f5aefbb9bc644e5c6612ce7`
- Export selection SHA256:
  `3eca6cd7c466e90a0bd5a22c7ca988a2b08a78bce17ca878710bc25c6c4e3a18`
- Candidate manifest: PASS
- LeRobot validator: PASS with no errors or warnings
- `LeRobotDataset` 0.4.4 readback: PASS
- Full AV1 video decode: PASS, 93,812 of 93,812 frames
- Parquet numeric, timestamp, boundary, and identity audit: PASS

## Release contents

- 130 episodes and 93,812 frames
- 104 train, 11 validation, and 15 test episodes
- 30 episodes at 0° cube yaw and 100 episodes at 45° cube yaw
- 65 episodes at 0.03 kg and 65 episodes at 0.04 kg
- 80 episodes with 0.03 m cube edges and 50 with 0.04 m cube edges
- 65 red-cube and 65 blue-cube episodes
- Complete 5 x 5 position coverage overall; the 0° increment covers 23 of
  25 cells
- LeRobot Dataset v3
- One 640 x 480, 30 Hz front-camera video stream
- Six-dimensional `observation.state` and `action`
- Farpoint v3 episode and variation metadata

Generated episodes, videos, Parquet shards, local paths, and credentials are
not committed to the Farpoint code repository.

## Promotion plan

1. Build, validate, and stage the immutable public package.
2. Upload it to the `v0.0.2-rc1` branch of
   `wenyixu101/farpoint-so101`.
3. Validate the Dataset Card, Dataset Viewer, checksums, Parquet tables, video
   decoding, and `LeRobotDataset` loading from the RC branch.
4. After owner review and merge of this release PR, promote the same validated
   contents to `main`.
5. Re-run the Viewer and loader checks against `main`.
6. Tag the validated Hub commit as `v0.0.2`.

If an RC gate fails, do not update `main` or create the version tag. Fix the
package and publish a new `v0.0.2-rcN` branch.
