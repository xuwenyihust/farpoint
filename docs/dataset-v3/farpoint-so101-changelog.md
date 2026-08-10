# Farpoint SO-101 dataset changelog

This changelog tracks published versions of
`wenyixu101/farpoint-so101`. Dataset versions are independent of Farpoint
Python package and schema versions.

## v0.0.3

Adds a 30° cube-yaw increment while preserving the policy feature schema:

- 160 successful SO-101 cube pick-and-place demonstrations
- 116,240 policy frames at 30 Hz
- 128 train, 14 validation, and 18 test episodes
- 100 demonstrations at 45° cube yaw, 30 at 0°, and 30 at 30°
- 80 demonstrations at 0.03 kg and 80 at 0.04 kg
- 110 demonstrations with 0.03 m cube edges and 50 with 0.04 m cube edges
- 80 red-cube and 80 blue-cube demonstrations
- complete 5 x 5 cube-position grid coverage overall and within the new 30°
  increment
- one 640 x 480 front-camera RGB stream; no wrist-camera feature
- unchanged six-dimensional joint state and joint-position action schema

## v0.0.2

Adds a 0° cube-yaw increment while preserving the policy feature schema:

- 130 successful SO-101 cube pick-and-place demonstrations
- 93,812 policy frames at 30 Hz
- 104 train, 11 validation, and 15 test episodes
- 30 demonstrations at 0° cube yaw and 100 at 45°
- 65 demonstrations at 0.03 kg and 65 at 0.04 kg
- 80 demonstrations with 0.03 m cube edges and 50 with 0.04 m cube edges
- 65 red-cube and 65 blue-cube demonstrations
- complete 5 x 5 cube-position grid coverage overall; the 0° increment covers
  23 of 25 cells
- one 640 x 480 front-camera RGB stream; no wrist-camera feature
- unchanged six-dimensional joint state and joint-position action schema

## v0.0.1

Adds a mirrored cube-mass stratum while preserving the v0.0.0 policy feature
schema:

- 100 successful SO-101 cube pick-and-place demonstrations
- 72,433 policy frames at 30 Hz
- 80 train, 10 validation, and 10 test episodes
- 50 demonstrations at 0.03 kg and 50 at 0.04 kg
- matched position, cube-size, color, and split coverage across both masses
- complete 5 x 5 cube-position grid coverage at each mass
- one 640 x 480 front-camera RGB stream; no wrist-camera feature
- unchanged six-dimensional joint state and joint-position action schema

## v0.0.0

Initial experimental simulation baseline:

- 50 successful SO-101 cube pick-and-place demonstrations
- 34,757 policy frames at 30 Hz
- 40 train, 5 validation, and 5 test episodes
- one 640 x 480 front-camera RGB stream
- six-dimensional joint state and joint-position action
- complete 5 x 5 cube-position grid coverage
- balanced 0.03 m / 0.04 m cube sizes and red / blue appearances
- structured Farpoint v3 episode metadata
