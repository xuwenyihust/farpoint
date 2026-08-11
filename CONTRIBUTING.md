# Contributing to Farpoint

Farpoint uses pull requests for every change to `main`. Start with the
[development workflow](docs/development-workflow.md), which defines the code,
simulation, dataset, and release gates.

## Branches and pull requests

- Create `feat/*`, `fix/*`, `data/*`, `docs/*`, or `codex/*` branches from an
  up-to-date `main`.
- Open a Draft PR early. Never push changes directly to `main`.
- Keep generated episodes, reports, datasets, credentials, and machine paths
  out of Git.
- Wait for required checks and resolve review conversations.
- The repository owner normally merges PRs. An agent may merge an exact PR or
  enable auto-merge only when the owner explicitly requests that action in the
  current task. Direct merges require a non-draft, mergeable PR with all
  required checks and reviews passing. Auto-merge may wait for those gates but
  must use the repository's normal branch protection. Agents never use admin or
  branch-protection bypasses, and authorization does not carry across PRs.
- Prefer squash merge so one reviewed PR maps to one `main` commit.

## Local checks

```bash
python3 -m pip install -e '.[dev]'
ruff check src scripts tests
pytest -q --cov=farpoint --cov-report=term-missing --cov-fail-under=75
python scripts/check_release_version.py
python scripts/plan_variations.py --json
```

Run the browser-level Dashboard acceptance test when changing the UI or data
platform server:

```bash
python3 -m pip install -e '.[dev,qa]'
python3 -m playwright install chromium
FARPOINT_RUN_BROWSER_QA=1 pytest -q tests/test_dashboard_browser.py
```

## Dataset changes

A code PR may run a small, deterministic pilot from its exact commit. Formal
benchmarks run only from merged code. Dataset publishing is a separate,
owner-approved operation and must use a staged release candidate that passed
the gates in `docs/development-workflow.md`.
