# Farpoint SO-101 v0.0.3 release candidate

## Scope

This release extends `wenyixu101/farpoint-so101` from 130 to 160 successful
SO-101 cube pick-and-place demonstrations. It adds 30 demonstrations with
0.03 m cube edges and cube yaw fixed at 30°, including every cell in the
5 x 5 position grid. The policy feature schema remains unchanged.

## Source evidence

- Candidate: `farpoint_so101_v0_0_3_candidate_20260810_afa6244`
- Candidate assembly revision: `afa62445c299adba73a3ac06701103ca32c8e7fe`
- Existing v0.0.2 source: `farpoint_so101_v0_0_2_candidate_20260808_985b72f`
- New yaw source: `so101_cube_yaw30_30mm_completion30_v0_0_3_20260809_ebe231c`
- Existing v0.0.2 manifest SHA256:
  `a895062be80fdb33c1e0166d51bfe77a832dc6796f5aefbb9bc644e5c6612ce7`
- Existing v0.0.2 export selection SHA256:
  `3eca6cd7c466e90a0bd5a22c7ca988a2b08a78bce17ca878710bc25c6c4e3a18`
- New yaw manifest SHA256:
  `19daac1ef98d369722ace48253f4ade4520df037973ffe4bb95c46c6de0094cf`
- New yaw selection SHA256:
  `93854819f5841a0f1f3e8e625e92f4dc04e3cdf38de408be49fca640dadd8d10`
- Combined candidate manifest SHA256:
  `d120c687cd04c00c26b97baef5e312108896019abce75927817683040754adca`
- Combined export selection SHA256:
  `311c48b09b1abef0e53a61c2ad3c3ea6443f9ec31dc69d37d9ed1f15e5de16ad`
- Composition validation SHA256:
  `eccb66ccdef7fb7273da96b668d132a9cccd3c518042f21c2513939f518b8c24`
- Candidate composition validation: PASS
- LeRobot validator: PASS with no errors or warnings; report SHA256:
  `e45ca1f7d57af8076b842af64502cdae1d500e5797c3d2896318d5b430f7d53a`
- `LeRobotDataset` 0.4.4 readback: PASS; report SHA256:
  `d97a7504de613aa6bcb41f088be2b06875e0dc8c67b6ed177c3536a0225f09f0`
- Full AV1 video decode: PASS, 116,240 of 116,240 frames
- Parquet numeric, timestamp, boundary, identity, and split audits: PASS
- Export checksum manifest verified; SHA256:
  `1f6361cf3d1e8c105f02d81d07c1c1c9db083d2a419c72cd50ed6c4843975360`

## Release contents

- 160 episodes and 116,240 frames
- 128 train, 14 validation, and 18 test episodes
- 100 episodes at 45° cube yaw, 30 at 0°, and 30 at 30°
- 80 episodes at 0.03 kg and 80 at 0.04 kg
- 110 episodes with 0.03 m cube edges and 50 with 0.04 m cube edges
- 80 red-cube and 80 blue-cube episodes
- Complete 5 x 5 position coverage overall and within the 30° increment
- LeRobot Dataset v3
- One 640 x 480, 30 Hz front-camera video stream
- Six-dimensional `observation.state` and `action`
- Farpoint v3 episode and variation metadata

Generated episodes, videos, Parquet shards, local paths, and credentials are
not committed to the Farpoint code repository. The Dataset Card remains
managed directly on Hugging Face and is not a Git repository artifact.

## Promotion plan

1. Build, validate, and stage the immutable public package.
2. After explicit owner approval, upload it to the `v0.0.3-rc1` branch of
   `wenyixu101/farpoint-so101`.
3. Validate the Dataset Card, Dataset Viewer, checksums, Parquet tables, video
   decoding, and `LeRobotDataset` loading from the RC branch.
4. After owner review and merge of this release PR, promote the same validated
   contents to `main` only with explicit owner approval.
5. Re-run the Viewer and loader checks against `main`.
6. Tag the validated Hub commit as `v0.0.3` only with explicit owner approval.

If an RC gate fails, do not update `main` or create the version tag. Fix the
package and publish a new `v0.0.3-rcN` branch.
