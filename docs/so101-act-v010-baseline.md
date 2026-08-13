# SO-101 ACT v0.1.0 baseline

This experiment fine-tunes LeRobot 0.4.4 ACT on the immutable
`wenyixu101/farpoint-so101@v0.1.0` dataset tag. It is a reproducible baseline,
not a claim of closed-loop task performance.

## Frozen experiment

- Dataset commit: `4f0dd8d8b39d8065e86ca3ea5eea8b4ce1ffd77b`
- Inputs: `observation.state`, front RGB, and wrist RGB
- Output: 6-dimensional joint-position action chunks
- Logical splits: episodes `0:180` train and `180:200` validation
- ACT: ImageNet-pretrained ResNet-18, chunk size 100, 100 action steps
- Pilot: 1,000 steps, batch size 8, checkpoints every 250 steps
- Formal baseline: 20,000 steps, batch size 8, checkpoints every 5,000 steps

The empty `test: 200:200` range in LeRobot metadata is retained only for
metadata compatibility. It is not a logical demonstration split. Final task
evaluation belongs to the independent Isaac Lab rollout suite, not this
offline checkpoint-selection procedure.

## Gates

Run a single-step smoke and then the 1,000-step pilot from an exact PR commit:

```bash
export FARPOINT_GIT_COMMIT="$(git rev-parse HEAD)"
export FARPOINT_TRAINING_DATA_ROOT=/home/wenyixu/datasets/farpoint-so101-training-v010
export FARPOINT_TRAINING_MODEL_ROOT=/home/wenyixu/models/farpoint-so101-training-v010
export FARPOINT_TRAINING_LOG_ROOT=/home/wenyixu/logs/farpoint-so101-training-v010
export FARPOINT_TRAINING_CACHE_ROOT=/home/wenyixu/.cache/farpoint-so101-training-v010

scripts/run_so101_act_experiment.sh \
  act-v010-smoke-001 so101_act_v0_1_0_pilot.json smoke v0.1.0

scripts/run_so101_act_experiment.sh \
  act-v010-pilot-001 so101_act_v0_1_0_pilot.json pilot v0.1.0
```

The pilot passes only when the pinned dataset, CUDA runtime, AV1 decoder,
train-only normalization view, both camera samples, training process, four
checkpoints, and deterministic validation report all pass. Validation chooses
the lowest fixed-sample ACT objective and requires at least 5% relative
improvement over the first checkpoint.

After the implementation PR is owner-reviewed and merged, run the formal
baseline from the exact merged commit:

```bash
scripts/run_so101_act_experiment.sh \
  act-v010-baseline-20k-001 \
  so101_act_v0_1_0_baseline_20k.json training v0.1.0
```

Generated dataset views, models, logs, checkpoints, and validation reports stay
outside Git. Model publication and Hugging Face uploads require a separate
owner approval.
