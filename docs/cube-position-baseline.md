# Cube Position Baseline

Farpoint's clean cube position baseline varies only the initial horizontal
position of one cube. It keeps the UR10e, Robotiq 2F-85, target, object shape,
yaw, appearance, camera, lighting, controller, physics, and recording policy
fixed.

This is an experiment plan, not a published dataset release. `release.toml`
remains at the current public release until the release workflow is complete.

## Trial Design

The workspace is divided into a 5 by 5 grid:

- X range: `[0.94, 1.00]` meters
- Y range: `[0.22, 0.28]` meters
- Three primary slots per cell
- Two predeclared reserve candidates per primary slot
- Samples remain within the 10% to 90% interior of each cell

The 75 primary trials use this fixed split assignment:

- Slot 0 and slot 1 in every cell: train (50 trials)
- Slot 2 in cells where `(row + column) % 2 == 0`: validation (13 trials)
- Remaining slot 2 trials: test (12 trials)

Reserve candidates inherit the primary slot's cell and split. They are not
primary benchmark attempts and do not change the primary success rate.

## Immutable Manifest

The input configuration is
`configs/variations/farpoint_v1_3_cube_position.json`. Generate or verify the
committed manifest with:

```bash
python3 scripts/create_position_plan.py
```

The command records:

- Every primary and reserve coordinate
- Trial, variation, cell, slot, and split identity
- Deterministic seed material and derived seeds
- Frozen factors and varied axes
- Configuration, planner implementation, and complete plan SHA256 values

Running the command again verifies the existing file. It refuses to overwrite
a different manifest at the same path. Change the plan identity and output path
when intentionally creating a new experiment design.

## Scene Binding

The Isaac runner accepts three position-plan environment variables:

```bash
export FARPOINT_POSITION_PLAN=configs/plans/farpoint_v1_3_cube_position_baseline.json
export FARPOINT_TRIAL_ID=primary_r00_c00_s00
export FARPOINT_RESERVE_INDEX=0
./scripts/run_isaac_example.sh examples/isaac_perception_contact_scene
```

The scene resolves its seed and object position from the manifest, fixes the
target position, and disables the legacy random sampler. Episode metadata
records the plan SHA, trial, split, cell, slot, and reserve lineage.

## Nine-Episode Pilot

The pilot uses slot 0 in the nine cells formed by rows and columns `0`, `2`,
and `4`. Run the pilot on the configured GPU host after committing the code:

```bash
python3 scripts/run_position_pilot.py \
  --pilot-id cube_position_pilot_YYYYMMDD_<git-sha> \
  --git-commit <full-git-sha>
```

The pilot passes only at 9/9 accepted episodes. Each episode must prove:

- Contact-only grasp with no temporary grasp joint
- RGB-D perception XY error at most 0.02 m
- Lift height at least 0.15 m
- At least 20 bilateral-contact frames
- At least 120 continuous transport-contact frames
- Final target XY error at most 0.05 m
- At least 120 settling frames
- Valid dataset observations, RGB replay source, depth, previews, and telemetry
- Exact plan, trial, and requested-position identity

Generated pilot manifests and episode artifacts remain under ignored
`outputs/`. Formal 75-attempt benchmarking and dataset publication happen only
after the position-planner PR is reviewed and merged.
