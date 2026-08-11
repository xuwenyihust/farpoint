# SO-101 v0.1.0 Camera and Workspace Gate

This gate is code-review evidence for the new dataset lineage. It does not
authorize a formal collection or create a benchmark.

## Camera contract

`configs/cameras/so101_front_wrist_v1.json` is the versioned source for the
front and wrist RGB sensors. A v0.1.0 run uses `--require-dual-camera`; the
collector then fails before environment construction if either Isaac camera
drifts from the profile. Legacy v0.0.x runs keep their front-only default.

Every completed dual-camera attempt writes `camera-evidence.json` containing:

- the complete profile and its canonical SHA-256;
- resolved runtime 3x3 intrinsic matrices;
- local parent-frame mount transforms;
- a declaration that both images come from the same 120 Hz control tick at a
  recording stride of four.

Viewer QA must inspect reset, pregrasp, contact, lift, placement, settle, and
retreat. It must reject gripper occlusion, an empty task view, invalid focus,
or a wrist-camera collision. Headless QA additionally decodes every front and
wrist frame and checks one-to-one timestamps.

## Continuous feasible region

The Isaac probe runner emits one result per object anchor and XY location. Each
probe must independently report full-path IK, joint limits, table collision,
self-collision, front visibility, and wrist visibility. A point passes only
when all six checks pass.

`scripts/build_so101_feasible_region.py` turns those immutable probes into a
continuous polygon. It deliberately refuses a convex hull that contains any
known failed probe; this prevents a grid projection from hiding holes. The
result freezes the object footprint, generator identity, probe evidence hash,
and maximum boundary clearance used for outer/middle/core classification.

The 5x5 or 7x7 Dashboard views are projections only. They are never collection
or metadata identities.
