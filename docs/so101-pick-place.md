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

The P0 structural-failure tolerance permits up to eleven consecutive failures
of one class and stops on the twelfth. The recent ten-attempt window remains in
the watchdog report as diagnostic evidence, but the default policy does not
stop on a recent-window fraction. Strict gates can still stop earlier when
their frozen success target becomes mathematically unreachable. Budget and
liveness checks are unchanged.

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

Controller fixes for a known subset of variations use a separate targeted
diagnostic pilot instead of resuming or modifying formal collection evidence.
The weak-contact capture pilot freezes two structural-failure sentinels first,
then two weak-contact variations and one known-success control at 0.03 kg:

```bash
python scripts/run_so101_gate_workflow.py init \
  artifacts/so101/weak_contact_capture_pilot_<sha> \
  --workflow-id weak_contact_capture_pilot_<sha> \
  --git-commit "$(git rev-parse HEAD)" \
  --workflow-config \
    configs/workflows/so101_weak_contact_capture_pilot.json
```

The pilot always executes all five frozen trials, even if its three-success
target becomes mathematically unreachable. Passing requires the collision and
non-cube-contact sentinels to retain their exact expected failure reasons, both
weak-contact variations to succeed, and the known-pass control to remain
successful. This role-aware contract prevents an unrelated sentinel success
from masking a weak-contact regression. A pass supports the weak-contact
hypothesis only; it does not show that the two structural failure modes were
fixed and does not authorize formal collection or reuse of the pilot episodes
in a release candidate.

The structural contact-handoff pilot tests those two former structural
failures plus one known-success control at 0.03 kg. During `DESCEND`, the
controller now stops insertion at the first cube-filtered fingertip force of
at least 0.1 N. The separate generic 2 N contact signal remains the collision
input, so table or target-pad contact cannot advance the grasp state. Aperture
alignment constrains the two local axes across the opening, while bilateral
capture force, rigidity, and proof lift validate contact along the long-finger
insertion axis. All three frozen trials must succeed:

```bash
python scripts/run_so101_gate_workflow.py init \
  artifacts/so101/structural_contact_handoff_pilot_<sha> \
  --workflow-id structural_contact_handoff_pilot_<sha> \
  --git-commit "$(git rev-parse HEAD)" \
  --workflow-config \
    configs/workflows/so101_structural_contact_handoff_pilot.json
```

The pilot is diagnostic evidence only. Its episodes remain `PILOT` artifacts
and are not eligible for the v0.0.1 dataset candidate.

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

## Cube-mass feasibility profile

Before adding mass as a dataset variation axis, run the bounded paired profile:

```bash
python scripts/run_so101_gate_workflow.py init \
  artifacts/so101/cube_mass_003_feasibility \
  --workflow-id cube_mass_003_feasibility_<sha> \
  --git-commit "$(git rev-parse HEAD)" \
  --workflow-config configs/workflows/so101_cube_mass_003_feasibility.json

python scripts/run_so101_gate_workflow.py status \
  artifacts/so101/cube_mass_003_feasibility/workflow.json
```

This profile runs five matched pairs (ten attempts total) at the proven 30 mm
red-cube pose. Each pair shares its environment seed and compares the existing
0.04 kg baseline with the proposed 0.03 kg mass. Both masses must achieve at
least four successes; the workflow never expands the attempt budget.

The collector applies the requested mass to the active rigid body after every
reset and reads the value back from the PhysX rigid-body view. Requested,
resolved, and actual values are stored in the episode sidecars. A mismatch
beyond `1e-6 kg` is a runner failure, so changing metadata alone cannot pass the
profile. The report also compares successful pairs using action-path length,
frame count, and bilateral lift force. It recommends a larger
physics-robustness pilot only when feasibility passes and a frozen behavior
threshold is met; otherwise it reports either insufficient evidence or no
measurable signal. `FEASIBILITY_COMPLETE` does not authorize a formal
collection or dataset release.

After the fixed-pose profile passes, use the candidate-only workspace pilot to
avoid recollecting baseline demonstrations already present in v0.0.0:

```bash
python scripts/run_so101_gate_workflow.py init \
  artifacts/so101/cube_mass_003_workspace_pilot \
  --workflow-id cube_mass_003_workspace_pilot_<sha> \
  --git-commit "$(git rev-parse HEAD)" \
  --workflow-config configs/workflows/so101_cube_mass_003_workspace_pilot.json
```

This second profile runs only the 0.03 kg, 30 mm red cube at five positions
whose corresponding 0.04 kg episodes succeeded in the v0.0.0 collection. The
historical episode IDs, positions, mass, collection ID, and generating commit
are frozen into the plan. Four of five candidate successes are required and no
retry budget is added. Every new episode must still pass the actual PhysX mass
audit.

The historical episodes establish only that the selected positions were
previously solvable. Because their generating commit differs from the candidate
pilot, they are not a contemporaneous control and must not be used to claim a
causal trajectory difference between 0.04 and 0.03 kg. A passing report supports
adding 0.03 kg as a dataset variation axis; it does not authorize formal
collection.

The 40 mm admission gate uses the same candidate-only contract with five
historically successful large-cube positions:

```bash
python scripts/run_so101_gate_workflow.py init \
  artifacts/so101/cube_mass_003_workspace_gate_40mm \
  --workflow-id cube_mass_003_workspace_gate_40mm_<sha> \
  --git-commit "$(git rev-parse HEAD)" \
  --workflow-config configs/workflows/so101_cube_mass_003_workspace_gate_40mm.json
```

At least four of five attempts must pass, including the actual PhysX mass
audit. Passing the earlier 30 mm pilot does not substitute for this large-cube
gate.

## v0.0.1 mirrored mass collection

The formal 0.03 kg collection mirrors the exact 50 trial identities selected
for the v0.0.0 balanced50 candidate. It therefore freezes the same XY samples,
25/25 size balance, 25/25 color balance, 40/5/5 split, and coverage of every
workspace cell. Only `entities.pick_object.physics.mass_kg` changes. The
collection requires 50 eligible successes and has a hard ceiling of 150
attempts; failed attempts remain in the raw manifest and episode root.

Formal collection must use an exact commit already merged to `main` and an
owner-approved collection ID:

```bash
python scripts/create_so101_mass_collection_plan.py \
  artifacts/so101/mass_v0_0_1/plan.json

FARPOINT_GIT_COMMIT="$(git rev-parse HEAD)" \
scripts/run_so101_isaaclab.sh headless \
  --plan artifacts/so101/mass_v0_0_1/plan.json \
  --manifest artifacts/so101/mass_v0_0_1/manifest.json \
  --output-root artifacts/so101/mass_v0_0_1/episodes \
  --collection-id so101_cube_mass_003_formal_v0_0_1_<date>_<sha> \
  --max-attempts-this-run 150 \
  --watchdog-policy configs/workflows/so101_watchdog_p0.json

python scripts/report_so101_mass_collection.py \
  --plan artifacts/so101/mass_v0_0_1/plan.json \
  --manifest artifacts/so101/mass_v0_0_1/manifest.json \
  --episodes-root artifacts/so101/mass_v0_0_1/episodes \
  --json-output artifacts/so101/mass_v0_0_1/report.json \
  --markdown-output artifacts/so101/mass_v0_0_1/report.md
```

The report rejects missing raw artifacts, balance drift, sidecar disagreement,
or any selected episode whose requested, resolved, and actual PhysX masses do
not agree within `1e-6 kg`. After it passes, combine the existing 0.04 kg
balanced50 selection and the new collection without relabeling splits:

```bash
python scripts/create_so101_mass_dataset_candidate.py \
  --baseline-manifest <balanced50-manifest.json> \
  --candidate-manifest artifacts/so101/mass_v0_0_1/manifest.json \
  --candidate-plan artifacts/so101/mass_v0_0_1/plan.json \
  --baseline-episodes-root <v0.0.0-episodes> \
  --candidate-episodes-root artifacts/so101/mass_v0_0_1/episodes \
  --collection-id farpoint_so101_v0_0_1_candidate_<date>_<sha> \
  --manifest-output artifacts/so101/mass_v0_0_1/candidate/manifest.json \
  --selection-output artifacts/so101/mass_v0_0_1/candidate/export-selection.json
```

The resulting candidate contains 100 episodes: 50 at 0.04 kg and 50 at
0.03 kg, with an 80/10/10 split and 50 exact mirrored trial pairs. Export,
Dashboard registration, release-candidate staging, and Hugging Face publishing
remain separate validation and owner-approval steps.

### Continuing an aborted mass collection

An `ABORTED` manifest is immutable evidence and must not be changed back to
`RUNNING`. After the watchdog policy change is reviewed and merged, freeze a
new continuation plan from the terminal parent evidence:

```bash
python scripts/create_so101_mass_continuation_plan.py \
  --parent-plan <aborted-root>/plan.json \
  --parent-manifest <aborted-root>/manifest.json \
  --continuation-id so101_cube_mass_003_continuation_<date>_<sha> \
  --output <continuation-root>/plan.json
```

The continuation contains only uncovered variations and inherits only the
unused portion of the parent's 150-attempt budget. It has a new collection ID,
manifest, episode root, and generating Git commit. The original successes and
failures remain untouched.

After the continuation reaches `PASS`, compose the two immutable sources:

```bash
python scripts/create_so101_mass_completion.py \
  --parent-plan <aborted-root>/plan.json \
  --parent-manifest <aborted-root>/manifest.json \
  --parent-episodes-root <aborted-root>/episodes \
  --continuation-plan <continuation-root>/plan.json \
  --continuation-manifest <continuation-root>/manifest.json \
  --continuation-episodes-root <continuation-root>/episodes \
  --collection-id so101_cube_mass_003_completion50_<date>_<sha> \
  --manifest-output <completion-root>/manifest.json \
  --selection-output <completion-root>/export-selection.json \
  --report-output <completion-root>/report.json
```

The completion command verifies all 50 selected episode artifacts, requested,
resolved, and actual PhysX mass, exact variation identity, split assignment,
25/25 workspace cells, and the frozen size/color balance. Only its passing
selection manifest may be registered in the Dashboard Benchmarks tree.

If both the original collection and its first continuation are terminal
`ABORTED` evidence, do not resume either manifest and do not nest another
continuation inside the exhausted old budget. Freeze a new recovery from the
union of their selected variations:

```bash
python scripts/create_so101_mass_recovery_plan.py \
  --reference-plan <original-root>/plan.json \
  --source-plan <original-root>/plan.json \
  --source-manifest <original-root>/manifest.json \
  --source-plan <continuation-root>/plan.json \
  --source-manifest <continuation-root>/manifest.json \
  --recovery-id so101_cube_mass_003_recovery13_<date>_<sha> \
  --maximum-attempts 150 \
  --output <recovery-root>/plan.json
```

The recovery plan binds both source plan and manifest hashes, rejects overlap,
and contains only the exact union-missing variations. Its 150-attempt ceiling
is a new frozen recovery budget; it does not rewrite or borrow from the old
continuation budget.

After the recovery itself reaches `FINISHED / PASS`, validate and compose all
three immutable sources:

```bash
python scripts/create_so101_mass_multi_source_completion.py \
  --reference-plan <original-root>/plan.json \
  --source-plan <original-root>/plan.json \
  --source-manifest <original-root>/manifest.json \
  --source-episodes-root <original-root>/episodes \
  --source-plan <continuation-root>/plan.json \
  --source-manifest <continuation-root>/manifest.json \
  --source-episodes-root <continuation-root>/episodes \
  --recovery-plan <recovery-root>/plan.json \
  --recovery-manifest <recovery-root>/manifest.json \
  --recovery-episodes-root <recovery-root>/episodes \
  --collection-id so101_cube_mass_003_completion50_<date>_<sha> \
  --manifest-output <completion-root>/manifest.json \
  --selection-output <completion-root>/export-selection.json \
  --report-output <completion-root>/report.json
```

This path audits 35 + 2 + 13 episode artifacts independently, preserves each
source collection ID and attempt identity, and still requires exact 50/50
variation coverage before producing a candidate selection.

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
