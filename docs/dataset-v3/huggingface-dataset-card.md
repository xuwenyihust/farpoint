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
- Dataset version: `v0.0.2`
- Status: experimental multi-mass, multi-yaw simulation baseline

## Dataset summary

| Metric | Value |
|---|---|
| **Task** | Cube pickup, transport, release, and stable placement |
| **Robot** | SO-101 follower arm: 5 arm joints + 1 gripper joint |
| **Simulator** | Isaac Sim 6.0.0 + PhysX, through Isaac Lab 3.0.0-beta2 |
| **Controller** | Simulation-truth, contact-aware DLS IK oracle |
| **Episodes** | 130 successful demonstrations |
| **Frames** | 93,812 at 30 Hz (about 52 min 7 s) |
| **Camera** | One 640 x 480 front RGB stream |

### Policy features

| Feature | Shape / format | Description |
|---|---|---|
| `observation.state` | `float32[6]` | Current SO-101 joint positions |
| `observation.images.front` | video, `480 x 640 x 3` | External front-camera RGB |
| `action` | `float32[6]` | Joint-position target sent during the current control step |

### Logical episode splits

| Split | Episodes | Share |
|---|---:|---:|
| Train | 104 | 80.0% |
| Validation | 11 | 8.5% |
| Test | 15 | 11.5% |

There is no wrist-camera feature in `v0.0.2`.

## Variation coverage

| Variation axis | Values in this release | Distribution |
|---|---|---:|
| Position | 5 x 5 stratified XY grid; cube centers span approximately `x = 0.147–0.255 m` and `y = -0.116–-0.024 m` | 25 / 25 cells covered overall; the new 0° stratum covers 23 / 25 cells |
| Cube yaw | 0°, 45° | 30 at 0°; 100 at 45° |
| Cube edge length | 0.03 m, 0.04 m | 80 at 0.03 m; 50 at 0.04 m |
| Cube color | Red, blue | 65 episodes each |
| Cube mass | 0.03 kg, 0.04 kg | 65 episodes each |
| Placement target | Green raised pad, 0.16 m x 0.14 m | Fixed |

The normalized `episode_metadata` configuration records requested and resolved
scene entities, geometry, appearance, pose, physics, task relationships,
variation axes, seeds, splits, and provenance. This entity-based contract is
designed to represent future cylinders, meshes, toys, boxes, movable targets,
and other task objects without changing the policy feature schema.

## Intended use

The dataset is intended for robot-learning pipeline development, behavior
cloning experiments, simulation evaluation, data-loader testing, and research
on variation-aware manipulation. Its small scale and simulation-only origin
make it unsuitable as a standalone basis for real-world robot deployment or
safety claims.

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

See the
[dataset changelog](https://github.com/xuwenyihust/farpoint/blob/main/docs/dataset-v3/farpoint-so101-changelog.md)
for version-by-version changes. Pre-1.0 versions may extend the task and scene
distribution and may evolve metadata contracts with explicit release notes.
