# SO-101 Cube Pick-and-place

This pilot adds the manager-based Isaac Lab environment
`Farpoint-SO101-PickPlace-Cube-v0` and a versioned Farpoint v3 contract for
exporting successful demonstrations to LeRobot v3. It is simulation-only; no
real arm, policy training, or Hugging Face upload is required.

## Reproducible workflow

From the feature worktree:

```bash
python scripts/create_so101_variation_plan.py \
  --config configs/variations/so101_cube_pick_place_v1.json \
  artifacts/so101/variation_plan.json

scripts/run_so101_isaaclab.sh viewer \
  --plan artifacts/so101/variation_plan.json
scripts/run_so101_isaaclab.sh headless \
  --plan artifacts/so101/variation_plan.json \
  --manifest artifacts/so101/collection_manifest.json \
  --output-root artifacts/so101/episodes

# Code-review pilot: ten distinct successes from at most fifteen frozen trials.
python scripts/create_so101_pilot_plan.py \
  artifacts/so101/pilot_plan.json \
  --pilot-id so101_cube_pick_place_pilot_v1
scripts/run_so101_isaaclab.sh headless \
  --pilot-plan \
  --plan artifacts/so101/pilot_plan.json \
  --manifest artifacts/so101/pilot_manifest.json \
  --output-root artifacts/so101/pilot_episodes \
  --max-attempts-this-run 15
python scripts/report_so101_pilot.py \
  --plan artifacts/so101/pilot_plan.json \
  --manifest artifacts/so101/pilot_manifest.json \
  --episodes-root artifacts/so101/pilot_episodes \
  --json-output artifacts/so101/pilot_report.json \
  --markdown-output artifacts/so101/pilot_report.md

python scripts/export_lerobot_dataset.py \
  artifacts/so101/export_selection.json artifacts/so101/lerobot_v3
python scripts/validate_lerobot_dataset.py artifacts/so101/lerobot_v3
```

To copy source/configuration to the DGX Spark without copying generated data:

```bash
FARPOINT_DGX_HOST=dgx-spark scripts/sync_so101_to_dgx.sh
ssh dgx-spark 'cd /home/wenyixu/projects/farpoint && scripts/run_so101_isaaclab.sh headless'
```

Set `FARPOINT_DGX_HOST` to an IP address if the `.local` hostname is not
available. Set `FARPOINT_DGX_ROOT` to use a different remote project path.

The launcher downloads the pinned NVIDIA workshop USD into the ignored
`.cache/farpoint/assets` directory and verifies its SHA256 before starting the
Isaac Sim 6.0 container. `viewer` and `headless` use the same collector; the
latter keeps one simulator process alive while it resets episodes.

The collection manifest records every attempt, including failures. Only
successful and `dataset_valid` episodes listed by the selection manifest are
exported. Variation splits are fixed at 80 train, 10 validation, and 10 test;
the collector does not rebalance them based on outcomes.

The code-review pilot is deliberately separate from the workspace gate and
formal 100-episode collection. Its frozen ordering starts with ten trials that
cover the workspace, both cube sizes, both colors, and an 8/1/1 train,
validation, and test mix. Five frozen fallback variations provide a strict
15-attempt ceiling; collection stops immediately after ten distinct successful
variations. Failed attempts remain in the pilot manifest and raw episode root.

## Extensible scene metadata

New v3 episodes keep the legacy `scene.object` and `scene.target` fields for
existing readers, and add canonical `scene.entities`. An entity has a stable
identity and role plus open-ended `entity_type` and `asset_id` fields, a pose,
physical geometry, appearance, and physics. Object types are not restricted to
cube primitives; for example a cylinder or doll asset can use an arbitrary XYZ
`dimension_profiles_m` entry in a future variation config.

Placement targets separate their physical geometry from their acceptance
regions. A flat pad can use an `on` region, while an open box can record its
outer collision dimensions and a smaller, independently positioned `inside`
region. This permits target position, size, shape, and success semantics to vary
without changing policy features or the exporter.

Every episode records requested and simulator-resolved entity values. The
LeRobot dataset keeps the complete records in `meta/episode_metadata.jsonl` and
adds an `episode_scene_metadata` index to `meta/farpoint_v3.json`. These values
remain sidecar metadata rather than observations. The v0 Isaac scene adapter
still spawns the four cube variants and one procedural pad; adding a cylinder,
mesh/doll, or box requires a corresponding spawn adapter, but not another
metadata contract version.

## Runtime gate

The repository tests validate the contracts, deterministic variation plan,
oracle state machine, resume logic, and six-dimensional/two-camera exporter.
Isaac runtime checks require an ARM64 DGX Spark with Isaac Lab 3.0 beta2 and
Isaac Sim 6.0. If the `dgx-spark` SSH alias is unavailable, treat the Isaac
viewer/headless checks as pending rather than claiming a successful collection.

The robot asset is not committed to Git. Source and pinned commit are kept in
`examples/isaaclab_so101_pick_place/farpoint_so101_env/assets.py`; the Docker
image and launcher provide the repeatable acquisition path.
