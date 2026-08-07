# SO-101 Cube Pick-and-place

This pilot adds the manager-based Isaac Lab environment
`Farpoint-SO101-PickPlace-Cube-v0` and a versioned Farpoint v3 contract for
exporting successful demonstrations to LeRobot v3. It is simulation-only; no
real arm, policy training, or Hugging Face upload is required.

The v0.0.0 policy interface is intentionally front-camera-only. The wrist
camera is not spawned, rendered, recorded, or exported. Historical dual-camera
episodes remain readable, but new SO-101 v0.0.0 collections advertise only
`observation.images.front`.

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

`viewer` defaults to WebRTC livestream mode 2 at 1280×720 and selects Isaac
Lab's Kit visualizer. The matching renderer/stream resolution avoids rejected
frames, while Kit pumps application updates during sensor synchronization so a
remote inspection run does not wait 30 seconds per control step. Explicit
`--livestream`, `--visualizer`, or `--kit_args` values override these defaults;
`--livestream=0` retains the local-window path. Launcher arguments containing
spaces are preserved as one collector argument across the Docker boundary.

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
The selected-success split can differ from 8/1/1 if a validation or test trial
fails and a fallback reaches the success target first. Report that as a pilot
coverage limitation; do not relabel or rebalance episodes after collection.

## P0 collection watchdog and gate workflow

Long SO-101 runs should use the frozen P0 watchdog policy rather than relying
on an operator to notice an unrecoverable collection. The watchdog evaluates
only persisted state: the plan, collection manifest, episode run-state
sidecars, the remaining attempt budget, and stable failure classes. It never
changes scene parameters, controller thresholds, trial ordering, splits, or
the attempt budget.

Its decisions are `CONTINUE`, `STOP`, `COMPLETE`, and `INVALID`. A stop is
required when the success target is mathematically unreachable, one structural
failure class repeats beyond the frozen window, a live episode becomes stale,
or the collection itself stops making progress. The collector evaluates this
policy after an attempt and its manifest are fully written. On `STOP` or
`INVALID`, it preserves every artifact, marks a still-running manifest
`ABORTED` with the watchdog reason, and does not start the next attempt.

Initialize the complete stage sequence before using Isaac:

```bash
python scripts/run_so101_gate_workflow.py init \
  artifacts/so101/oracle_gate_v1 \
  --workflow-id so101_oracle_gate_v1 \
  --git-commit "$(git rev-parse HEAD)"

python scripts/run_so101_gate_workflow.py status \
  artifacts/so101/oracle_gate_v1/workflow.json
```

Keep the artifact root repository-relative so the same action paths resolve in
the `/workspace/project` container mount and after syncing to DGX Spark.

The status JSON exposes exactly one admissible `next_action`, including the
required `FARPOINT_GIT_COMMIT`, command arguments, plan, manifest, episode
root, and copied watchdog policy. Run it from the repository root. Re-run
`status` after collection or evidence reporting; do not skip a locked stage.

The frozen P0 sequence is:

1. 20-of-20 fixed 30 mm repeatability gate.
2. 20-of-20 fixed 40 mm repeatability gate.
3. Two-size by five-position workspace matrix at a 90% threshold.
4. Ten-success stratified pilot with a 15-attempt ceiling.

Every completed stage needs a `PASS` evidence report before the next stage is
unlocked. A watchdog abort, failed report, invalid evidence, altered plan or
policy, or mismatched Git commit blocks the workflow. Passing all four stages
produces `READY_FOR_FORMAL_REVIEW`; it does not authorize or launch a formal
collection. Formal collection still requires merged `main`, a new frozen
collection identity, and the normal owner-reviewed workflow.

For an existing collection, the read-only one-shot check is:

```bash
python scripts/watch_so101_collection.py \
  --plan artifacts/so101/gate/plan.json \
  --manifest artifacts/so101/gate/manifest.json \
  --episodes-root artifacts/so101/gate/episodes \
  --output artifacts/so101/gate/watchdog.json
```

The command exits 0 for `CONTINUE` or `COMPLETE`, 2 for `STOP`, and 3 for
`INVALID`. The one-shot CLI does not mutate the collection; safe automatic
stopping is enabled by passing `--watchdog-policy` to the long-lived collector,
as emitted by the gate workflow.

## Dashboard lifecycle

Episode IDs include the collection ID as well as the attempt ID, so two
collections can reuse one frozen variation without colliding in the registry.
The collector writes a small `farpoint.episode-run.v1` `run-state.json` before
recording the first frame and updates it when the attempt finishes or the runner
fails. This lets the Dashboard expose RUNNING, FAIL, and incomplete attempts
with their existing front-frame preview before final `metadata.json` exists.

The Dashboard indexes the original episode directory in place. It does not copy
the RGB sequence or mutate the episode. Once final v3 metadata exists, that
metadata remains authoritative. The detail view includes outcome, failure
reason, `dataset_valid`, variation, split, collection provenance, and requested
and resolved scene entities. Old episodes without run-state sidecars continue
to use the existing v1/v2/v3 discovery path.

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
oracle state machine, resume logic, six-dimensional front-only exporter, and
historical dual-camera read compatibility. A runtime gate must additionally
prove bilateral contact, lift, transport, release, stable placement, and
retreat from recorded evidence; a success flag alone is insufficient.
Isaac runtime checks require an ARM64 DGX Spark with Isaac Lab 3.0 beta2 and
Isaac Sim 6.0. Run Viewer and headless evidence separately. If Viewer streaming
connects but RTX frame completion stalls, retain the diagnostic run as
incomplete and do not claim the close-up collision/joint-direction inspection
from the connection state alone.

The robot asset is not committed to Git. Source and pinned commit are kept in
`examples/isaaclab_so101_pick_place/farpoint_so101_env/assets.py`; the Docker
image and launcher provide the repeatable acquisition path.
