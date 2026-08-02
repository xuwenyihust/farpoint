# Farpoint Development and Release Workflow

This document is the canonical workflow for Farpoint code, simulation,
benchmark, dataset, and Hugging Face releases.

## Version policy

`release.toml` is the only manually edited source for the current public
release version and dataset identity. Package metadata, Dataset Cards, release
manifests, Git tags, and Hugging Face tags must agree with it.

Release versions use `MAJOR.MINOR.PATCH` in `release.toml` and `vMAJOR.MINOR.PATCH`
for Git and Hugging Face tags. Schema versions are independent compatibility
contracts and do not follow the release number.

For example, a feature branch may introduce `farpoint.dataset.v2` while the
published release remains `1.2.0`. Only the final release PR changes
`release.toml`; contract, pilot, and benchmark PRs must not pre-bump it.

The release currently recorded in `release.toml` is the published baseline.
Ordinary feature PRs do not bump it; the next version is selected in a release PR.

## Code workflow

1. Write the objective, non-goals, acceptance criteria, and data impact.
2. Create a feature branch from `main` and open a Draft PR.
3. Implement focused changes with unit, contract, and UI tests.
4. Pass lint, unit, Dashboard QA, data-contract, coverage, and version checks.
5. Run an owner-approved deterministic pilot from the PR commit when GPU evidence is required.
6. Attach the pilot manifest and report URL to the PR.
7. The repository owner reviews and squash-merges the PR. Agents and automation never merge.

## Simulation and benchmark workflow

Pilot IDs describe the experiment rather than a release, for example
`pickup_diversity_pilot_20260801_<sha>`. Pilots may be debugged and repeated on
the feature branch.

Formal benchmarks run from an exact commit already merged to `main`, use a
frozen config hash and holdout seeds, and may not be tuned in place. A failure
requires a new fix PR, a new pilot, and a new release candidate.

Adaptive dataset collections are reported separately from fixed-sample
benchmarks. Their manifests retain every task outcome used to calculate yield,
distinguish selected episodes from unselected successes, and freeze coverage,
attempt-budget, and stopping rules before execution.

Every run records the Git revision, config hash, Isaac image digest, seeds,
episode IDs, artifact completeness, and distinct execution and quality states.

## Dataset workflow

1. Export only eligible benchmark or collection episodes into a new candidate directory.
   The selection manifest must assign every episode to an explicit split and
   preserve its source trial identity.
2. Validate episode boundaries, finite values, feature dimensions, timestamps,
   Parquet shards, MP4 decoding, frame alignment, metadata, and checksums.
3. Build the Viewer-safe package with `scripts/release_dataset.py build`.
4. Run `validate`, then `stage`; neither command uploads data.
5. QA the Dataset Card, LeRobot loader, Viewer `/is-valid`, and first rows on a
   staging revision or staging dataset.
6. Open a release PR containing the version change, card, release notes, manifest,
   benchmark links, and validation evidence.
7. After the owner merges and separately approves publishing, run `publish`
   with the exact confirmed version.
8. Verify the Hugging Face tag, Dataset Viewer, downloadable artifacts, and
   LeRobot loading from the published revision.

## State model

Do not overload one status field. Data-platform records should distinguish:

```text
execution_status = RUNNING | FINISHED | ABORTED
quality_status   = NOT_EVALUATED | PASS | FAIL
release_status   = EXPERIMENTAL | PILOT | CANDIDATE | PUBLISHED
```

## DGX security

The public repository's pull-request workflows must use GitHub-hosted runners.
Do not execute arbitrary PR code on the DGX Spark as a general self-hosted
runner. DGX jobs require an owner-approved commit SHA and run through the
controlled Farpoint remote workflow.
