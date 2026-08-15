# Farpoint SO-101 v0.1.2 release candidate

## Scope

This release preserves the immutable `v0.1.1` parent and adds 20 successful
grasp-stage recovery demonstrations. The policy feature schema remains
unchanged. No `v0.1.1` episode was rerun or modified.

## Source evidence

- Parent dataset tag: `v0.1.1`
- Parent dataset commit: `ff1a812584b677b02998a722ac2a446ce1003e55`
- Recovery campaign: `so101_v012_grasp_recovery20_formal_20260815_df7f0fb`
- Recovery implementation revision:
  `df7f0fbeaca7efc4f65b0efbb1fc870cbf9f0aa9`
- Recovery campaign report SHA256:
  `b45d855758bbba365d2a9b04137f00d595b9ab2ba790db87637253076d2e689c`
- Recovery formal validation SHA256:
  `32a75807361e455b78e9fc0ed6b5f1144997c31b1d4941083498898209cc112d`
- Recovery selection SHA256:
  `c2d0fa845effec8c1af2d88b7517d6865b4e299b2f9c5324ddddae36c818beef`
- Combined 240-episode export selection SHA256:
  `ce28c3ab8d01fdfef2ea1b57451b5062618e62e9ee82137e1f2336491f4ddd46`
- LeRobot v3 validation report SHA256:
  `3293ffd85c27791f3c28449bf44caf09af1aea2c76fb265bd971dc61d411a37c`
- LeRobot 0.4.4 loader QA report SHA256:
  `bb3a5cff91161b6f2e8213f0c1eab6e3cfe351a11d5cb085f45536cda8e94091`
- Candidate checksum manifest SHA256:
  `b53c484a4dae40b5d0060c83243b784933655f3e51ac8fd6bbbfa89f316bc775`
- Recovery campaign result: PASS, 20 selected episodes from 72 attempts across
  five immutable segments
- Recovery evidence: 20/20 measured `handoff_stage=grasp`, 20/20
  `failure_class=contact_without_lift`, 20/20 real cube contact at handoff,
  and 0/20 previously lifted
- Recovery integrity: PASS for 17,309 frames and 40 source front/wrist videos

## Release contents

- 240 successful and dataset-valid demonstrations
- 183,914 policy frames at 30 Hz
- 220 train and 20 validation episodes
- 200 nominal demonstrations, 20 approach-stage recovery demonstrations, and
  20 grasp-stage recovery demonstrations
- all 40 recovery demonstrations are in the train split
- grasp recovery quotas: red/blue 10 each, middle/outer 10 each, and four
  demonstrations in each of five yaw strata
- LeRobot Dataset v3
- synchronized front and wrist RGB streams at 640 x 480 and 30 Hz
- six-dimensional `observation.state` and joint-position `action`
- normalized Farpoint v3 episode metadata with handoff-stage v1 evidence

Generated episodes, videos, Parquet shards, local paths, and credentials are
not committed to the Farpoint code repository. The Dataset Card remains
managed directly on Hugging Face and is not a Git repository artifact.

## Promotion plan

1. Export and validate the local 240-episode candidate, including full dual-
   camera decoding and `LeRobotDataset` 0.4.4 readback. Complete.
2. Register and verify the recovery campaign as a Dashboard Benchmark.
   Complete.
3. Build and validate the public package from the reviewed release source.
4. Upload the package to the isolated `v0.1.2-rc1` Hub branch.
5. Validate the RC Dataset Viewer, Parquet/video artifacts, metadata, and
   direct loader access.
6. Promote the exact validated RC contents without rewriting `v0.1.1`, then
   create the immutable `v0.1.2` dataset tag.
7. Update the Dataset Card and Quality Space only after tag verification.

If an RC gate fails, do not update the release tag. Preserve the failed RC
evidence, fix the package through a reviewed change, and publish a new RC
revision.
