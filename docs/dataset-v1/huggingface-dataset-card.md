---
pretty_name: Farpoint UR10e Robotiq 2F-85
license: cc-by-4.0
library_name: lerobot
task_categories:
- robotics
tags:
- LeRobotDataset-v3
- format:parquet
- modality:video
- robotics
- robot-learning
- isaac-sim
- manipulation
---

# Farpoint UR10e Robotiq 2F-85

Farpoint UR10e Robotiq 2F-85 is an experimental LeRobot v3 dataset of
physics-based cube pick-and-place episodes generated with Isaac Sim and the
Farpoint simulation pipeline.

- Dataset: [farpoint-ur10e-robotiq-2f85](https://huggingface.co/datasets/wenyixu101/farpoint-ur10e-robotiq-2f85)
- Source: [xuwenyihust/farpoint](https://github.com/xuwenyihust/farpoint)
- Dataset version: `v0.0.0`
- Status: experimental baseline

## Dataset summary

The release contains all 55 successful, dataset-valid episodes from an
accepted cube-position collection:

- Robot: Universal Robots UR10e
- Gripper: Robotiq 2F-85
- Task: contact-only cube pick, transport, and placement
- Workspace: 0.26 m by 0.20 m, divided into a 5 by 5 position grid
- Coverage: all 25 cells, with at least two successful episodes per cell
- Collection outcome: 55 successful episodes from 66 task attempts (83.3%)
- Public splits: 44 train, 6 validation, and 5 test episodes
- Observation: front-camera RGB video and robot joint state
- Action: commanded robot joint position

The release includes every successful, dataset-valid collection episode. It
does not hide failed attempts in the collection evidence, but failed attempts
are not demonstrations and are not included in the LeRobot dataset.

## Versioning

Dataset versions are independent of the Farpoint GitHub repository and Python
package version. The dataset records the exact Farpoint Git revision and
simulation provenance used to generate each episode.

`v0.0.0` is the first experimental clean baseline. No compatibility guarantee
is made for future pre-1.0 dataset releases.

## Intended use

This dataset is intended for robot-learning data pipeline development,
behavior-cloning experiments, simulation evaluation, and analysis of
position-conditioned manipulation. Its small scale and simulation-only origin
make it unsuitable as a standalone foundation for real-world robot deployment.

## License

Farpoint-authored dataset content is released under the Creative Commons
Attribution 4.0 International license (CC BY 4.0):

https://creativecommons.org/licenses/by/4.0/

This license applies only to content that Farpoint has the right to license.
Isaac Sim, NVIDIA Omniverse, NVIDIA-provided robot assets, textures, and other
third-party components are not relicensed or redistributed by Farpoint. Their
use remains subject to the original provider terms.

## Limitations

- Simulation-only data; no sim-to-real performance is claimed.
- One robot and gripper configuration.
- One cube shape and one task family.
- Position variation is broader than other scene variation dimensions.
- Successful demonstrations only; failures remain in collection evidence.

## Attribution

Please cite the Farpoint source repository and identify the dataset repository
and revision used in your work.
