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

The pilot runner resolves and records the immutable Isaac image digest and
uses an explicit 900-second per-episode timeout so full diagnostics can be
written even on terminal task failures.

Generated pilot manifests and episode artifacts remain under ignored
`outputs/`. A passing pilot gates the formal 75-attempt benchmark; it is not a
dataset release by itself.

## Accepted Goal 2 Pilot

`cube_position_pilot_20260801_a645c7c` passed all nine planned trials with no
failed acceptance checks. Its immutable runtime identity is:

- Code revision: `a645c7c94b1f3a0acc62a78f5ff33d1db4243816`
- Position plan SHA256:
  `736cb89f26a8d2a943d54f381cec5fd2f7e5c86d258b995dbdaf0fde28185993`
- Isaac Sim image: `nvcr.io/nvidia/isaac-sim:6.0.0`
- Image digest:
  `sha256:68735a60b6c15c85e0dd0098570c6d2cc79e928f2d068ce2790aa43284ac165d`

Across the nine accepted episodes, perception XY error was
`0.015893-0.016689 m`, lift height was `0.2262-0.2283 m`, continuous transport
contact was `1201-1879` frames, final target XY error was
`0.00116-0.00809 m`, and every release was observed for 120 settling frames.
Every episode included valid RGB-D observations, preview frames, telemetry,
and contact-only grasp evidence without a temporary grasp joint.

## Expanded Workspace Feasibility

The accepted Goal 2 pilot proves the pipeline over a deliberately narrow
`0.06 x 0.06 m` sampling region. It does not claim that the whole tabletop is
reachable. The next candidate expands the cube-center sampling workspace to:

- X range: `[0.84, 1.10]` meters
- Y range: `[0.18, 0.38]` meters
- Candidate area: `0.26 x 0.20 m`

The original configuration and manifest remain immutable. The expanded
candidate uses
`configs/variations/farpoint_v1_3_cube_position_expanded.json` and
`configs/plans/farpoint_v1_3_cube_position_expanded_candidate.json`.

Before any formal 75-trial benchmark, run the nine edge/center trials with:

```bash
python3 scripts/run_position_pilot.py \
  --plan configs/plans/farpoint_v1_3_cube_position_expanded_candidate.json \
  --pilot-id cube_position_workspace_feasibility_YYYYMMDD_<git-sha> \
  --git-commit <full-git-sha>
```

The feasibility gate requires all nine episodes to pass the same contact,
perception, lift, transport, placement, settling, dataset, preview, and
telemetry checks as the narrow pilot. The selected positions must additionally
span at least `0.20 m` in X and `0.16 m` in Y. Task failures reject the
candidate; they are not replaced with reserve trials. Only infrastructure
failures may rerun the same trial and seed.

If the feasibility gate passes, this exact candidate manifest and SHA become
the formal 75-primary plan. If it fails, do not run the formal benchmark;
change the bounds or controller under a new plan identity and repeat the gate.

### Accepted Expanded Workspace Pilot

`cube_position_workspace_feasibility_20260801_d095f5d` passed all nine
edge/center trials. Its immutable runtime identity is:

- Code revision: `d095f5de33a5df958b22782eb73f14c004a679f4`
- Position plan SHA256:
  `f13bb891d6044145a0e2c5b65982f91810298f0a8387328cc933fb51bd0da8db`
- Isaac Sim image: `nvcr.io/nvidia/isaac-sim:6.0.0`
- Image digest:
  `sha256:68735a60b6c15c85e0dd0098570c6d2cc79e928f2d068ce2790aa43284ac165d`

The selected positions spanned `0.224391 m` in X and `0.182146 m` in Y,
exceeding both workspace coverage gates. Across the nine accepted episodes,
perception XY error was `0.015421-0.017307 m`, lift height was
`0.2251-0.2548 m`, continuous transport contact was `1204-1982` frames,
final target XY error was `0.00096-0.01746 m`, and every release was observed
for 120 settling frames. Each episode contained `352-455` synchronized dataset
observations and passed all contact-only, RGB-D, preview, and telemetry checks.

The run recorded 11 infrastructure attempts. Two attempts reached the
five-minute Isaac Kit startup timeout before producing an episode; both were
retained as infrastructure failures and retried with the identical trial and
seed. No task failure was retried or replaced, and no reserve candidate was
used. This accepted manifest is therefore eligible for the formal 75-primary
v1.3 cube-position benchmark.
