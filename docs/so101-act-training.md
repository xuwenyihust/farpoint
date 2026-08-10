# SO-101 ACT training foundation

Farpoint pins the first ACT baseline to
`wenyixu101/farpoint-so101@v0.0.3`. The machine-independent experiment contract
is `configs/training/so101_act_v0_0_3_baseline.json`.

## Data policy

- Train uses episodes `0:128` only.
- Validation uses episodes `128:142` to compare checkpoints in a later gate.
- Test uses episodes `142:160` only for final offline checks and later simulator rollouts.
- Preflight verifies the immutable tag's frozen Hub commit, features, frame
  counts, FPS, and split metadata.
- Published global normalization statistics are not used for training.
  Preflight creates a separate local dataset view whose `meta/stats.json` is
  aggregated from train episodes only. Data and video files remain linked to a
  tag-specific immutable source cache.

This prevents validation and test observations from influencing training even
indirectly through normalization. A later checkpoint-selection workflow must
load validation explicitly; `lerobot-train` does not perform that evaluation
for this experiment.

## DGX Spark environment and smoke gate

Build the independent ARM64 GPU image on the DGX Spark:

```bash
scripts/build_so101_training_image.sh
```

The image uses an immutable digest of NVIDIA's 26.01 CUDA base, whose PyTorch
2.10 pre-release is inside LeRobot 0.4.4's supported range. During the build it
records the exact preinstalled torch and torchvision versions as pip
constraints, preventing their replacement with CPU-only ARM64 wheels.

Set writable storage roots if the repository cache is not appropriate:

```bash
export FARPOINT_TRAINING_DATA_ROOT=/path/to/datasets
export FARPOINT_TRAINING_MODEL_ROOT=/path/to/models
export FARPOINT_TRAINING_LOG_ROOT=/path/to/logs
export FARPOINT_TRAINING_CACHE_ROOT=/path/to/cache
```

Run preflight plus exactly one ACT optimizer step with a new immutable run ID:

```bash
scripts/smoke_so101_act_training.sh act-v0.0.3-smoke-<git-sha>
```

The wrapper resolves the Git commit from a normal checkout. When source is
synced without `.git`, set `FARPOINT_GIT_COMMIT` to the exact reviewed commit
before running it. The evidence also records the locally built Docker image ID.

The gate fails before training if the tag moved, metadata changed, CUDA or AV1
decoding is unavailable, train-only frame totals differ, sample shapes are
wrong, values are non-finite, or output/view/report paths already exist. Its
JSON evidence is written under the configured log root. Generated datasets,
views, checkpoints, reports, credentials, and machine-specific paths stay out
of Git.

## What this foundation does not do

It does not start the 20,000-step run, select a checkpoint, evaluate validation
or test, run Isaac Lab rollouts, publish a model, or upload training artifacts.
Those are separate, evidence-gated steps after this smoke gate passes.

## Short training pilot

The deterministic pilot contract is
`configs/training/so101_act_v0_0_3_pilot.json`. It trains on the same train-only
view for 1,000 optimizer steps and saves checkpoints every 250 steps. It never
pushes to the Hub or enables external experiment tracking.

Run it in a detached DGX session from one exact reviewed commit:

```bash
FARPOINT_GIT_COMMIT=<40-character-commit> \
  scripts/pilot_so101_act_training.sh act-v0.0.3-pilot-<git-sha>
```

After training, the wrapper scores all four checkpoints on 128 deterministic,
evenly spaced frames from validation episodes `128:142`. It selects the lowest
mean ACT training objective and requires at least 5% improvement relative to
the first checkpoint. Test episodes `142:160` are excluded and recorded as such
in the validation report.

The resulting validation loss is a teacher-forced offline diagnostic. It can
show that optimization is learning a held-out action mapping, but it is not a
robot-task success rate. Isaac Lab rollouts remain a later, independent gate.
