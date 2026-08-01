# Farpoint Dataset Contract v2

## Status

Dataset Contract v2 is the metadata and validation foundation for future
Farpoint releases. It does not change the currently published release in
`release.toml`, and it does not imply that a v1.3 dataset has been generated.

Release versions and contract versions are independent:

- Release identity: `release.toml`, currently `1.2.0`
- Dataset contract: `farpoint.dataset.v2`
- Episode contract: `farpoint.episode.v2`
- Variation contract: `farpoint.variation.v2`
- Benchmark contract: `farpoint.benchmark.v2`

The machine-readable schemas are packaged in
[`src/farpoint/schemas/`](../../src/farpoint/schemas/).

## Design Goals

Contract v2 supports multiple tasks and explicit `train`, `validation`, and
`test` splits. It records enough provenance to reproduce an episode and enough
structured scene metadata to query position, shape, yaw, appearance, camera,
and lighting variations without parsing opaque JSON strings.

Historical `farpoint.dataset.v1` and `robotsim.dataset.v1` exports remain
readable by the validator. They are not silently rewritten as v2 records.

## Canonical Layout

The canonical working export uses LeRobot Dataset v3 plus Farpoint sidecars:

```text
dataset/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   ├── farpoint_v2.json
│   ├── episode_metadata.jsonl
│   └── episodes/
├── data/
└── videos/
```

`farpoint_v2.json` and `episode_metadata.jsonl` are canonical validation
inputs. The public Hugging Face package converts normalized episode metadata
to `meta/episode_metadata.parquet` and removes non-standard JSON/JSONL inputs
so the Dataset Viewer does not mistake them for dataset splits.

Seed fields are stored as decimal strings in the public Parquet metadata. This
keeps one stable Arrow type while preserving simulator seeds beyond int64.

## Episode Metadata

Each v2 episode contains these typed sections:

| Section | Purpose |
| --- | --- |
| `identity` | Episode, trial, task, split, and dataset index |
| `provenance` | Git commit, config hash, image digest, assets, and seeds |
| `task` | Stable task ID, instruction, object shape, and success criteria |
| `embodiment` | Robot, gripper, controller, control mode, and grasp mode |
| `scene` | Object and target poses, camera, lighting, and appearance |
| `variation` | Varied/frozen axes, cell/slot, requested and resolved values |
| `recording` | Rate, cameras, resolution, and frame count |
| `outcome` | Success, failure taxonomy, validity, and quality metrics |

Requested variation values describe the plan. Resolved values describe what
the simulator actually instantiated. The validator requires resolved object
shape, dimensions, and position to match the recorded scene.

## Dynamic Tasks

The exporter derives the LeRobot task instruction from each episode. It does
not contain a global cube instruction. A dataset task has:

```json
{
  "task_id": "pick_place_cube_v1",
  "instruction": "Pick up the cube and place it in the target zone.",
  "object_shape": "cube",
  "success_criteria_id": "contact_pick_place_v1"
}
```

The task ID must resolve in the dataset sidecar, the instruction must resolve
in LeRobot's task table, and the object shape must agree with episode scene and
variation metadata.

Task IDs and instructions form a one-to-one mapping. Two task IDs may not
collapse onto the same LeRobot instruction and task index.

## Splits

Selections explicitly assign every episode to `train`, `validation`, or
`test`. The exporter orders episodes by split and writes contiguous LeRobot
split ranges. The validator compares:

- Dataset sidecar split counts
- Episode metadata split identities
- `meta/info.json` split ranges
- Optional benchmark trial split assignments

No seed-derived or filename-derived split is accepted implicitly.

Full v2 validation reads every data, episode, and task Parquet shard; checks
episode boundaries, timestamps, terminal flags, finite state/action vectors,
and task indexes; decodes the front-camera MP4 files; and compares decoded
frame counts with the tabular trajectory.

## Provenance and Benchmark Links

V2 requires the exact Git commit, task config SHA256, Isaac Sim image digest,
robot asset identity/path, and simulator seeds. An optional benchmark manifest
can be supplied to the validator; every selected episode must then match a
benchmark trial by trial ID, episode ID, variation ID, and split.

## Export and Validation

The v2 exporter consumes an explicit selection manifest:

```json
{
  "schema_version": "farpoint.export-selection.v1",
  "dataset_id": "farpoint-ur10e-robotiq-2f85",
  "episodes": [
    {
      "episode_dir": "/path/to/episode",
      "trial_id": "cube-r00-c00-s00",
      "split": "train"
    }
  ]
}
```

Run the exporter and validator with:

```bash
python scripts/export_lerobot_dataset.py selection.json /path/to/candidate
python scripts/validate_lerobot_dataset.py /path/to/candidate \
  --benchmark-manifest /path/to/benchmark.json
```

The selection manifest is operational input and must not be published when it
contains machine-local paths.
