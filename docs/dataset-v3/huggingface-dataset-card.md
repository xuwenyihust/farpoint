---
pretty_name: Farpoint SO-101
license: cc-by-4.0
library_name: lerobot
task_categories:
- robotics
configs:
- config_name: default
  default: true
  data_files:
  - split: train
    path: "data/**/*.parquet"
- config_name: episode_metadata
  data_files:
  - split: train
    path: "meta/episode_metadata.parquet"
tags:
- LeRobotDataset-v3
- format:parquet
- modality:video
- robotics
- robot-learning
- isaac-sim
- isaac-lab
- manipulation
- so101
- synthetic-data
---

# Farpoint SO-101

Farpoint SO-101 is an extensible LeRobot v3 dataset of physics-based SO-101
robot manipulation episodes generated with Isaac Lab, Isaac Sim, and the
Farpoint simulation pipeline.

- Dataset: [wenyixu101/farpoint-so101](https://huggingface.co/datasets/wenyixu101/farpoint-so101)
- Source: [xuwenyihust/farpoint](https://github.com/xuwenyihust/farpoint)
- Dataset version: `v0.0.0`
- Status: experimental simulation baseline

The repository name intentionally describes the robot rather than one object or
task. Future versions may add new manipulated objects, placement targets,
scene variations, cameras, and SO-101 task families while preserving immutable
version tags.

## v0.0.0 dataset summary

The first release contains 50 successful, dataset-valid cube pick-and-place
demonstrations selected for workspace and appearance diversity:

- Robot: SO-101 follower arm with five arm joints and one jaw joint
- Simulator: Isaac Sim 6.0.0 with PhysX through Isaac Lab 3.0.0-beta2
- Controller: simulation-truth contact-aware DLS IK oracle
- Task: contact-only cube pickup, transport, release, and stable placement
- Episodes: 50
- Frames: 34,757 at 30 Hz
- Logical episode splits: 40 train, 5 validation, and 5 test
- Observation: one 640 x 480 front-camera RGB stream and six joint positions
- Action: the six joint-position targets actually sent at the current control step
- Joint order: shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper
- Export scale: the five arm joints use `[-100, 100]`; the gripper uses `[0, 100]`

There is no wrist-camera feature in `v0.0.0`.

## Variation coverage

The release covers all cells of a 5 x 5 stratified object-position grid. Its
selected demonstrations contain:

- Cube edge lengths: 0.03 m and 0.04 m, 25 episodes each
- Cube colors: red and blue, 25 episodes each
- Cube center X range: approximately 0.147 m to 0.255 m
- Cube center Y range: approximately -0.116 m to -0.024 m
- Cube mass: fixed at 0.04 kg
- Placement target: a fixed green 0.16 m x 0.14 m raised pad

The normalized `episode_metadata` configuration records requested and resolved
scene entities, geometry, appearance, pose, physics, task relationships,
variation axes, seeds, splits, and provenance. This entity-based contract is
designed to represent future cylinders, meshes, toys, boxes, movable targets,
and other task objects without changing the policy feature schema.

## Dataset Viewer and LeRobot splits

The default Dataset Viewer configuration displays frame-level data. Select the
`episode_metadata` configuration to inspect one structured row per episode.

LeRobot v3 stores all frame rows in one physical Parquet table. For that reason,
the Hub Viewer exposes the frame table as one physical `train` split. The
logical train, validation, and test episode ranges are recorded in
`meta/info.json`, and each metadata row includes its logical split.

## Provenance and success policy

Only successful, complete demonstrations selected by the accepted Balanced50
collection are included. Failed and unselected source attempts remain in the
Farpoint collection evidence and are not silently relabeled as demonstrations.

Grasping uses simulated contact and friction. No temporary fixed joint,
attachment constraint, or suction-style shortcut is used.

## Intended use

The dataset is intended for robot-learning pipeline development, behavior
cloning experiments, simulation evaluation, data-loader testing, and research
on variation-aware manipulation. Its small scale and simulation-only origin
make it unsuitable as a standalone basis for real-world robot deployment or
safety claims.

## Limitations

- Simulation only; no sim-to-real performance is claimed.
- Successful demonstrations only; this is not a failure-learning dataset.
- `v0.0.0` contains one cube pick-and-place task family.
- Object position, size, and color vary; mass, friction, lighting, and target
  geometry are fixed in this release.
- The policy observations contain one external RGB camera and robot state.

## License and third-party components

Farpoint-authored dataset content is released under the Creative Commons
Attribution 4.0 International license (CC BY 4.0):

https://creativecommons.org/licenses/by/4.0/

This license applies only to content that Farpoint has the right to license.
Isaac Sim, Isaac Lab, NVIDIA Omniverse components, and other third-party assets
are not relicensed by Farpoint. The SO-101 USD originates from NVIDIA's
Sim-to-Real SO-101 Workshop at commit
`ce807d99724cb65671abec01f908a2fcb4a6eab7`; its upstream Apache-2.0 notice and
Farpoint's third-party attribution apply.

## Versioning and attribution

Dataset versions are independent of the Farpoint Python package version.
Please cite the Farpoint source repository and identify both the dataset
repository and immutable version tag used in your work.

`v0.0.0` is the initial experimental release. Pre-1.0 versions may extend the
task and scene distribution and may evolve metadata contracts with explicit
release notes.
