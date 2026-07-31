# Farpoint V1.1 Goal: Intra-Task Diversity

## Objective

Expand `farpoint-ur10e-robotiq-2f85` from a small proof-of-pipeline release into
a useful first research dataset while keeping one task and one robot setup
fixed:

> UR10e + Robotiq 2F-85 performs a physics-based tabletop pickup.

The goal is to test whether policies and data tooling can handle controlled
variation within the same task. This is a data-generation and evaluation goal,
not yet a general-purpose VLA training dataset.

## Scope

Keep fixed:

- UR10e and Robotiq 2F-85 assets
- tabletop pickup success definition
- observation and action feature names
- camera placement and image resolution
- LeRobot Dataset v3 layout

Vary only parameters that are safe and easy to audit:

- object type: cube and cylinder
- object position: six reachable tabletop zones
- object yaw: at least three bins
- object appearance: two color/material profiles
- deterministic episode seed

Do not add new robots, new task families, wrist cameras, real-robot data, or
unbounded domain randomization in this goal.

## Dataset Target

Generate **60 new successful episodes**, arranged as a balanced matrix of
variation profiles and at least 10 independent seeds per profile. Keep the
existing 12-episode release unchanged and publish the expanded result as a
new dataset revision or versioned release.

Every episode must retain:

- a unique `episode_index`
- the simulation and exporter versions
- the deterministic seed
- the full variation configuration
- success and failure metadata
- the original Farpoint and LeRobot-compatible metadata

Recommended sidecar fields are `variation_id`, `object_type`,
`object_position_bin`, `object_yaw_bin`, `appearance_profile`, and `seed`.

## Acceptance Criteria

The goal is complete when:

1. At least 60 new episodes pass the existing structural validator.
2. At least 90% of generated episodes satisfy the physics pickup acceptance
   checks; failed episodes are retained separately as diagnostics or excluded
   from the release with an explicit reason.
3. Every planned variation profile contains at least 10 successful seeds.
4. No episode contains missing frames, invalid action dimensions, broken video
   references, or non-finite numeric values.
5. The dataset loads successfully through the installed LeRobot API.
6. A coverage report lists counts by variation profile, seed, success status,
   and failure reason.
7. Two runs with the same seed and configuration produce equivalent metadata
   and numerically close initial conditions.
8. The Dataset Card documents the variation matrix, generation image/version,
   known limitations, and NVIDIA/third-party asset boundaries.
9. The release remains usable by existing Farpoint validators and dashboard
   tooling without changing the core LeRobot feature contract.

## Implementation Phases

### Phase 1: Configuration and metadata

- Add a versioned variation configuration.
- Add deterministic seed plumbing to the runner.
- Record variation metadata in each episode.
- Add a coverage validator that does not require Isaac Sim.

### Phase 2: Pilot

- Generate 12 episodes across the planned profiles.
- Inspect replay frames and contact metrics.
- Fix spawn, visibility, or grasp failures before scaling.

### Phase 3: Dataset generation

- Generate the remaining episodes in resumable batches.
- Keep logs and raw reports on DGX Spark.
- Export only passing episodes into the release candidate.

### Phase 4: Validation and release

- Run structural, LeRobot, coverage, and replay checks.
- Build a variation summary for the dashboard.
- Publish a versioned Hugging Face release and update the Dataset Card.

## Success Signal

The result should let a community user answer a meaningful question:

> Does a pickup policy or data pipeline work only for one fixed object pose, or
> does it remain reliable across controlled object, pose, appearance, and seed
> variation?

## Explicit Non-Goals

- Training or claiming a production-quality VLA policy.
- Proving sim-to-real transfer.
- Maximizing episode count before the variation matrix is validated.
- Introducing arbitrary randomization that cannot be reproduced or analyzed.
