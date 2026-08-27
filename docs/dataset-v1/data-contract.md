# Farpoint Dataset V1 Data Contract

## Status

This document defines the contract for **Farpoint Dataset V1**. It is a
design and validation contract, not a generated dataset.

The contract is intentionally separate from the operational registry. The
registry tracks lineage and run status; this contract describes the dataset
that a LeRobot loader consumes.

## Canonical Dataset Layout

Farpoint V1 targets the LeRobot Dataset v3 layout:

```text
farpoint_v1/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   ├── farpoint_v1.json
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet
├── data/
│   └── chunk-000/
│       └── file-000.parquet
└── videos/
    └── observation.images.front/
        └── chunk-000/
            └── file-000.mp4
```

`meta/farpoint_v1.json` is the Farpoint-specific contract sidecar. The
standard LeRobot metadata remains authoritative for loading the dataset.

The V1 exporter may use `tasks.jsonl` during development if the installed
LeRobot version requires it, but the validator must record which task metadata
representation was used. The exporter should ultimately use the official
LeRobot writer rather than constructing Parquet metadata manually.

## Dataset Identity

The Farpoint sidecar must contain:

```json
{
  "schema_version": "farpoint.dataset.v1",
  "dataset_id": "farpoint_ur10e_robotiq_2f85",
  "format": "lerobot",
  "format_version": "v3",
  "split": "train",
  "task": {
    "name": "ur10e_robotiq_single_cube_pick_place",
    "instruction": "Pick up the cube and place it in the target zone."
  },
  "robot": {
    "name": "ur10e",
    "gripper": "robotiq_2f85",
    "arm_dof": 6,
    "gripper_dof": 1
  },
  "simulation": {
    "simulator": "Isaac Sim",
    "image": "nvcr.io/nvidia/isaac-sim:6.0.0",
    "physics": "PhysX",
    "control_mode": "articulation_drive"
  },
  "recording": {
    "fps": 20,
    "cameras": ["observation.images.front"],
    "image_width": 640,
    "image_height": 360
  }
}
```

The concrete contract is machine-readable in:

```text
schemas/farpoint_v1.schema.json
```

## Feature Contract

### Required features

| Feature | Meaning | Expected shape | Unit |
| --- | --- | --- | --- |
| `observation.state` | Robot proprioception | `[7]` minimum | radians / meters |
| `action` | Command applied at the current control step | `[7]` minimum | radians / meters |
| `observation.images.front` | Front RGB camera observation | `[H, W, 3]` | uint8 |
| `timestamp` | Monotonic sample time | scalar | seconds |
| `frame_index` | Frame index within the episode | scalar | integer |
| `episode_index` | Dataset episode index | scalar | integer |
| `task_index` | Index into task metadata | scalar | integer |
| `next.done` | Terminal flag for the transition | scalar | boolean |

The minimum seven-dimensional state/action vector is:

```text
[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
```

V1 may append additional state features, but it must not change the meaning or
ordering of these first seven values. Any additional feature must be declared
in `meta/info.json` and `meta/farpoint_v1.json`.

### State semantics

- Arm joint positions are ordered by the UR10e articulation joint order.
- Arm positions use radians.
- The gripper value is the normalized or physical gripper command documented
  by the generated metadata.
- If velocities are added, they must use a separate feature such as
  `observation.velocity` and must use radians per second or meters per second.
- The coordinate frame for object and end-effector labels is the Isaac Sim
  world frame unless metadata explicitly says otherwise.

### Action semantics

`action` records the command actually sent to the articulation controller, not
the desired Cartesian target before planning. This distinction matters when
evaluating controller tracking and reproducing a trajectory.

The metadata must record:

- action type: joint position, joint velocity, or effort
- control frequency
- action delay, if any
- joint ordering
- gripper command semantics

## Episode Semantics

An episode is one complete attempt at the fixed V1 task.

Each episode must have:

- a unique `episode_index` within the dataset
- a stable source `episode_id` from the Farpoint registry
- one `task_index`
- one seed
- a contiguous frame range
- exactly one terminal frame with `next.done = true`
- a success/failure result and failure category

The episode metadata must include at least:

```text
episode_index
episode_id
source_episode_id
task_index
seed
length
success
failure_category
started_at
finished_at
```

The first frame is indexed `0`. `timestamp` must be non-negative and strictly
increasing within an episode. `frame_index` must increase by one unless the
dataset writer documents a deliberate subsampling rule.

## Task Semantics

V1 contains one task definition:

```text
name: ur10e_robotiq_single_cube_pick_place
instruction: Pick up the cube and place it in the target zone.
```

Task metadata must also identify:

- object type and nominal dimensions
- target-zone definition
- required lift height
- maximum final XY error
- minimum contact and settling requirements
- whether the grasp is physics contact-only

`task_index` refers to a stable task table row. It must not be inferred from a
free-form string at load time.

## Video Contract

The V1 front camera is stored as:

```text
videos/observation.images.front/<chunk>/file-<index>.mp4
```

The dataset metadata must record:

- width and height
- RGB channel order
- frame rate
- codec/container information
- episode frame offsets
- whether frames are resized or cropped

Video and tabular data must describe the same control timeline. A validator
must reject an episode when a video stream cannot be decoded or when the
episode frame range cannot be mapped to the corresponding video frames.

## Registry Lineage

The registry should store or expose the following dataset-level fields:

```text
dataset_id
dataset_version
format
format_version
source_benchmark_id
episode_count
total_frames
export_status
validation_status
dataset_path
created_at
```

Each exported episode should retain:

```text
source_episode_id
source_artifact_path
source_seed
source_task_name
source_simulator
```

The dataset must remain usable if the original registry database is rebuilt.
The dataset should not depend on the registry SQLite file to load samples.

## Validation Rules

The V1 validator must perform the following checks without starting Isaac Sim:

### Contract checks

- `meta/farpoint_v1.json` exists and validates against the JSON Schema.
- `meta/info.json` exists and declares the required features.
- The format is LeRobot v3-compatible.
- Dataset identity and task identity are present.

### Layout checks

- `data/` contains at least one Parquet shard.
- `meta/episodes/` contains at least one episode metadata shard.
- `meta/stats.json` exists.
- Task metadata exists as `tasks.parquet` or `tasks.jsonl`.
- Each declared camera has a corresponding video directory.
- Each video directory contains at least one MP4 shard.

### Consistency checks

- Required features are declared exactly once.
- Shapes and dtypes are internally consistent.
- Episode indices are unique and contiguous.
- Task indices resolve to task metadata.
- State/action dimensions match the feature declaration.
- Timestamps are monotonic within each episode.
- Frame indices are contiguous within each episode.
- Terminal flags appear at episode boundaries.
- Statistics exist for numeric state and action features.

### Validation result

The validator returns exit code `0` only when all required checks pass. It must
write a machine-readable result containing:

```json
{
  "valid": true,
  "schema_version": "farpoint.dataset.v1",
  "dataset_id": "farpoint_ur10e_robotiq_2f85",
  "errors": [],
  "warnings": [],
  "checks": {}
}
```

Warnings must not hide required failures. A missing video decoder may be a
warning only in structural mode; full validation must fail when video decoding
cannot be verified.

## Compatibility Policy

V1 is compatible with LeRobot v3 at the dataset API boundary. It is not enough
for the directory names to look similar. A valid V1 release must be loaded by
the installed LeRobot `LeRobotDataset` implementation and must pass the
Farpoint validator.

If LeRobot changes its on-disk metadata representation, the exporter may
adapt while keeping the Farpoint sidecar schema versioned. A compatibility
adapter must record the exact LeRobot version used during export and
validation.

Legacy datasets containing `meta/robotsim_v1.json` and
`robotsim.dataset.v1` remain readable by the validator. New exports must
always use the Farpoint names above; the legacy names are not written by the
current exporter.

## Licensing and Provenance

The separately published `ur10e-robotiq-2f85` dataset is intended to be released
under CC BY 4.0 for original Farpoint content. Each release must document
the provenance and redistribution terms of simulator assets, robot models,
textures, and other third-party content. NVIDIA Isaac Sim, Omniverse, and
NVIDIA-provided assets are not relicensed by Farpoint and must remain under
their original terms.

## Explicit Non-goals for V1

- No multi-object task catalog.
- No domain randomization matrix.
- No wrist camera requirement.
- No real-robot calibration data.
- No policy-quality claim based only on dataset validity.
- No manual editing of exported episode rows.

## Definition of Done

Sub-goal 1 is complete when:

1. This contract is reviewed and versioned.
2. `schemas/farpoint_v1.schema.json` validates the sidecar example.
3. A validator command runs without Isaac Sim.
4. A deliberately invalid fixture produces a non-zero exit code.
5. A structurally valid fixture produces exit code `0`.
6. The contract clearly maps Farpoint source episodes to LeRobot features.
