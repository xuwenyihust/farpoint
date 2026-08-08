# Farpoint Development and Release Workflow

This document is the canonical workflow for Farpoint code, simulation,
benchmark, dataset, and Hugging Face releases.

## Version policy

Farpoint code and Farpoint datasets are independent products with independent
release histories:

- The GitHub repository and Python package use the Farpoint code version from
  `pyproject.toml` and `farpoint.__version__`.
- Each Hugging Face dataset has its own specification under `configs/datasets/`,
  including its `dataset_version`, Hub repository, schemas, and Dataset Card.
- Schema versions are compatibility contracts and are independent of both code
  and dataset versions.

A dataset release such as
`wenyixu101/farpoint-ur10e-robotiq-2f85@v1.3.0` records the exact generating Git
commit, but it does not change the Farpoint package to `1.3.0` and does not
require a Git tag named `v1.3.0`. Future datasets use separate release specs and
may advance at different rates while sharing this repository.

Only a dataset release PR changes that dataset's specification. Contract,
pilot, benchmark, and collection PRs do not pre-bump a dataset version.

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

An aborted adaptive collection manifest is also immutable. When accepted
episodes should be retained, a merged fix may create a new continuation plan
bound to the aborted manifest hash and containing only uncovered variations.
The parent and continuation may be exposed as one completed selection only
after combined evidence passes the original coverage and quality contract.

Adaptive dataset collections are reported separately from fixed-sample
benchmarks. Their manifests retain every task outcome used to calculate yield,
distinguish selected episodes from unselected successes, and freeze coverage,
attempt-budget, and stopping rules before execution.

Completed formal collection manifests are registered under the Dashboard
`outputs/benchmarks/<collection-id>/manifest.json` tree only after their
collection report passes. Registration exposes the collection in the
Benchmarks tab without copying or mutating its episode artifacts.

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
6. Open a dataset release PR containing that dataset's version change, card,
   release notes, manifest,
   benchmark links, and validation evidence.
7. After the owner merges and separately approves publishing, run `publish`
   with the exact confirmed version.
8. Verify the dataset's Hugging Face tag, Dataset Viewer, downloadable artifacts, and
   LeRobot loading from the published revision.

Dataset publication does not create a Git tag. Farpoint code releases follow a
separate package release process and may use their own Git tags.

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
