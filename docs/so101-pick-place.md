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
  --output artifacts/so101/variation_plan.json

scripts/run_so101_isaaclab.sh viewer \
  --plan artifacts/so101/variation_plan.json
scripts/run_so101_isaaclab.sh headless \
  --plan artifacts/so101/variation_plan.json \
  --manifest artifacts/so101/collection_manifest.json \
  --output-root artifacts/so101/episodes

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

## Runtime gate

The repository tests validate the contracts, deterministic variation plan,
oracle state machine, resume logic, and six-dimensional/two-camera exporter.
Isaac runtime checks require an ARM64 DGX Spark with Isaac Lab 3.0 beta2 and
Isaac Sim 6.0. If the `dgx-spark` SSH alias is unavailable, treat the Isaac
viewer/headless checks as pending rather than claiming a successful collection.

The robot asset is not committed to Git. Source and pinned commit are kept in
`examples/isaaclab_so101_pick_place/farpoint_so101_env/assets.py`; the Docker
image and launcher provide the repeatable acquisition path.
