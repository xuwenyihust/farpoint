---
pretty_name: Farpoint UR10e Robotiq 2F-85
license: cc-by-4.0
library_name: lerobot
task_categories:
- robotics
tags:
- LeRobotDataset-v3
- format:parquet
- modality:tabular
- robotics
- robot-learning
- isaac-sim
- lerobot
- manipulation
---

# Farpoint UR10e Robotiq 2F-85

Farpoint UR10e Robotiq 2F-85 is a LeRobot-compatible dataset of physics-based
robot manipulation episodes generated with the Farpoint simulation pipeline.

- Dataset: [farpoint-ur10e-robotiq-2f85](https://huggingface.co/datasets/wenyixu101/farpoint-ur10e-robotiq-2f85)
- Source repository: [xuwenyihust/farpoint](https://github.com/xuwenyihust/farpoint)

## License

The original Farpoint dataset content is released under the **Creative
Commons Attribution 4.0 International (CC BY 4.0)** license:

https://creativecommons.org/licenses/by/4.0/

You must provide appropriate credit, link to the license, and indicate if
changes were made.

This license applies only to content that Farpoint has the right to license.
Isaac Sim, NVIDIA Omniverse, NVIDIA-provided robot assets, textures, and
other third-party components are not relicensed by Farpoint. Their use and
redistribution remain subject to the original terms from their providers.

## Data and format

See the Farpoint source repository's
[`docs/dataset-v1/data-contract.md`](https://github.com/xuwenyihust/farpoint/blob/main/docs/dataset-v1/data-contract.md)
for the data contract and LeRobot compatibility policy.

The dataset is intended to be consumed independently of the source repository;
the source repository contains the generation and validation pipeline, while
this Hugging Face repository contains the released dataset artifacts.

## Releases

- `v1.0.0`: the original 12 successful legacy randomized episodes.
- `v1.1.0`: 29 successful episodes, combining the original 12 episodes with 17
  V1.1 profiled episodes. Legacy records are labeled
  `farpoint_legacy_randomized_v0`; profiled records are labeled
  `farpoint_v1_1_profiled` in `meta/episode_metadata.jsonl`.
- `v1.1.1`: the same 29 episodes with Viewer-compatible metadata in
  `meta/episode_metadata.parquet`.
- `v1.2.0`: the same validated 29-episode corpus, published through the
  reproducible Farpoint release pipeline with a Viewer-safe public package.

The default revision tracks the latest release. Earlier releases remain
available through their Hugging Face revision tags.

## Intended use

The dataset is intended for robot learning research, simulation evaluation,
benchmarking, and development of data pipelines. It is not a guarantee of
real-world robot performance or safety.

## Attribution

Please cite the Farpoint repository and identify the dataset release version
used in your work.
