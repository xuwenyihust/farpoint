# Farpoint SO-101 dataset changelog

This changelog tracks published versions of
`wenyixu101/so101-sim-oracle-pick-and-place`. Dataset versions are independent of Farpoint
Python package and schema versions.

## v0.2.0

Starts a fresh nominal-only baseline with broader environment variation; no
v0.1.x nominal or recovery episode is inherited:

- 300 successful SO-101 cube pick-and-place demonstrations
- 255,043 policy frames at 30 Hz
- 270 train and 30 validation episodes
- 30 balanced cells crossing two cube variants, three target anchors, and five
  front-camera extrinsic profiles, with ten demonstrations per cell
- deterministic continuous Latin-hypercube sampling of cube X, Y, and yaw
  within every cell
- a compact 90 x 90 x 10 mm target pad selected by the frozen pad pilot
- two synchronized 640 x 480 RGB streams: front and wrist
- unchanged six-dimensional joint state and joint-position action schema
- candidate tree SHA256
  `1201462db640a8cdff9c938c95cf67044e5550d41ca6dc43a800dd680493749d`

## v0.1.4

Adds transport-stage recovery demonstrations while preserving the v0.1.x
policy feature schema:

- 280 successful SO-101 cube pick-and-place demonstrations
- 212,606 policy frames at 30 Hz
- 260 train and 20 validation episodes
- 200 nominal demonstrations, 20 strict grasp-stage recovery demonstrations,
  40 strict approach-stage recovery demonstrations, and 20 strict
  transport-stage recovery demonstrations
- transport recovery handoff requires a prior validated lift, no prior target
  entry, and continuous live-state execution without a scene reset
- the transport recovery set covers the frozen `transport_stall` failure
  subclass; it does not claim balanced transport-failure subclass coverage
- all legacy pre-lift recovery demonstrations remain excluded
- two synchronized 640 x 480 RGB streams: front and wrist
- unchanged six-dimensional joint state and joint-position action schema

## v0.1.3

Replaces legacy pre-lift recovery demonstrations with strict approach-stage
coverage while preserving the v0.1.x policy feature schema:

- 260 successful SO-101 cube pick-and-place demonstrations
- 198,571 policy frames at 30 Hz
- 240 train and 20 validation episodes
- 200 nominal demonstrations, 20 strict grasp-stage recovery demonstrations,
  and 40 strict approach-stage recovery demonstrations
- all legacy pre-lift recovery demonstrations are excluded
- two synchronized 640 x 480 RGB streams: front and wrist
- unchanged six-dimensional joint state and joint-position action schema

## v0.1.2

Adds grasp-stage recovery demonstrations while preserving the v0.1.x policy
feature schema:

- 240 successful SO-101 cube pick-and-place demonstrations
- 183,914 policy frames at 30 Hz
- 220 train and 20 validation episodes
- 200 nominal demonstrations, 20 approach-stage recovery demonstrations, and
  20 grasp-stage recovery demonstrations
- grasp recovery handoff requires measured cube contact without a prior lift
- two synchronized 640 x 480 RGB streams: front and wrist
- unchanged six-dimensional joint state and joint-position action schema

## v0.1.1

Adds approach-stage recovery demonstrations to the v0.1.0 lineage:

- 220 successful SO-101 cube pick-and-place demonstrations
- 166,605 policy frames at 30 Hz
- 200 train and 20 validation episodes
- 200 nominal demonstrations and 20 approach-stage recovery demonstrations
- two synchronized 640 x 480 RGB streams: front and wrist
- unchanged six-dimensional joint state and joint-position action schema

## v0.1.0

Starts the dual-camera continuous-position lineage:

- 200 successful SO-101 cube pick-and-place demonstrations
- 149,948 policy frames at 30 Hz
- 180 train and 20 validation episodes
- two object variants across continuous positions, five yaw strata, and three
  feasible-region bands
- two synchronized 640 x 480 RGB streams: front and wrist
- six-dimensional joint state and joint-position action schema

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
