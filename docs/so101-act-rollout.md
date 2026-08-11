# SO-101 ACT closed-loop rollout

The rollout gate evaluates saved ACT checkpoints in the same Isaac Lab SO-101
environment used to generate the demonstrations. The immutable contracts are:

- `configs/evaluations/so101_act_v0_0_3_rollout_smoke.json` for the 1,000-step
  pilot checkpoint;
- `configs/evaluations/so101_act_v0_0_3_baseline_20k_rollout_smoke.json` for
  the validation-selected 20,000-step baseline checkpoint.

Both contracts use the same five scenes, seeds, control limits, and acceptance
criteria so their closed-loop results can be compared directly.

## Scope

This is an interface smoke, not a policy benchmark. It freezes five scene
seeds spanning position, size, color, mass, and yaw. All five episodes must
finish with finite actions and no calibrated hard-range violation. Task success
and stage progress are measured, but the minimum required success count is zero
to preserve comparability with the original 1,000-step smoke. Raising a
task-success threshold belongs to a later benchmark spec and must not mutate
either smoke after observing its results.

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
Checkpoint loading disables redundant ImageNet backbone initialization because
the complete backbone is already stored in `model.safetensors`; inference does
not require network access.

## DGX containers and run

The launcher deliberately uses two existing, independently verified images:

- `farpoint-so101-isaaclab:3.0-beta2` runs Isaac Lab and the environment;
- `farpoint-so101-lerobot-training:0.4.4` loads ACT and serves actions on a
  loopback-only HTTP endpoint.

This boundary prevents LeRobot dependencies from replacing Python packages in
Isaac Kit. The launcher verifies both image IDs, the exact model SHA256, the
LeRobot version, and the source commit before starting the five scenes. The
policy server is stopped when the Isaac runner exits.

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
does not authorize model publication or a formal rollout benchmark.
