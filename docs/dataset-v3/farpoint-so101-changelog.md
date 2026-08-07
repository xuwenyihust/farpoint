# Farpoint SO-101 dataset changelog

This changelog tracks published versions of
`wenyixu101/farpoint-so101`. Dataset versions are independent of Farpoint
Python package and schema versions.

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
