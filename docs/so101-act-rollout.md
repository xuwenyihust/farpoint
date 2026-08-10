# SO-101 ACT closed-loop rollout

The first rollout gate evaluates the 1,000-step ACT pilot checkpoint in the
same Isaac Lab SO-101 environment used to generate the demonstrations. Its
frozen contract is
`configs/evaluations/so101_act_v0_0_3_rollout_smoke.json`.

## Scope

This is an interface smoke, not a policy benchmark. It freezes five new scene
seeds spanning position, size, color, mass, and yaw. All five episodes must
finish with finite actions and no calibrated hard-range violation. Task success
and stage progress are measured, but the minimum required success count is zero
because the checkpoint saw only 1,000 optimizer steps (about 0.086 train
epochs). Raising a task-success threshold belongs to a later benchmark spec and
must not mutate this smoke after observing its results.

The suite consumes no dataset rows. The training and validation episode ranges
are provenance only; test episodes `142:160` remain explicitly excluded.

## Control path

At 30 Hz the runner:

1. reads front RGB and the six simulator joint positions;
2. converts joint radians to the calibrated SO-101 LeRobot convention;
3. runs the saved ACT preprocessor, policy, and postprocessor;
4. enforces calibrated hard limits and a six-unit per-step safety bound;
5. converts the applied target back to radians and holds it for four 120 Hz
   physics ticks.

Each episode records an MP4, a per-step action/truth trace, reset and mass
audits, action-safety counts, contact/lift/target/release evidence, and a task
result. The report distinguishes interface acceptance from task success.

## DGX image and run

Build the combined Isaac Lab + LeRobot inference image on the DGX Spark:

```bash
scripts/build_so101_rollout_image.sh
```

The build refuses an unexpected Isaac base image ID and preserves its CUDA
PyTorch and torchvision packages while adding exactly LeRobot 0.4.4. It also
pins a standard ARM64 Pillow 12.1.1 build so LeRobot does not import the binary copy
inside Isaac's extension cache, whose private shared-library path is valid only
for selected Kit extensions.

Run from one exact reviewed commit and a complete checkpoint directory:

```bash
export FARPOINT_GIT_COMMIT=<40-character-rollout-commit>
export FARPOINT_ACT_CHECKPOINT=/home/wenyixu/models/farpoint-so101-training/\
act-v0.0.3-pilot-1c036f9/checkpoints/001000/pretrained_model
export FARPOINT_DATA_ROOT=/home/wenyixu/logs/farpoint-so101-rollouts

scripts/run_so101_act_rollout.sh headless \
  --spec /workspace/project/configs/evaluations/so101_act_v0_0_3_rollout_smoke.json \
  --output-root /workspace/farpoint-data/<new-run-id>
```

Generated videos, traces, checkpoints, and reports remain outside Git. A PASS
does not authorize longer training, model publication, or a formal rollout
benchmark.
