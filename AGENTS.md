# Farpoint Agent Rules

Follow `CONTRIBUTING.md` and `docs/development-workflow.md` for every change.

- Never commit or push directly to `main`.
- Create a feature branch and a Draft PR.
- An agent may merge a specific PR or enable auto-merge only after the owner
  explicitly requests that action in the current task. The agent must identify
  the exact PR, verify that it is ready for review and mergeable, and never use
  an admin or branch-protection bypass. A direct merge additionally requires
  all required checks and reviews to pass; auto-merge may wait for those gates.
- The sole exception to per-PR auto-merge authorization is a SO-101 Oracle
  repair PR covered by an owner-reviewed campaign grant. Such a grant is valid
  only after this governance rule is merged, must identify one campaign and
  policy version, and expires when that campaign finishes or is abandoned.
  The agent may enable normal GitHub auto-merge (never direct/admin merge) only
  when the PR has the `so101-oracle-repair` label and every path/evidence gate
  in `configs/governance/so101_oracle_repair_v1.json` has passed. A campaign
  grant never authorizes dataset publishing, release changes, or any other PR.
- Never infer merge authorization from approval of code, a pilot, a release,
  or a previous PR. Never create a production release tag or publish to Hugging
  Face without a separate explicit owner request after review.
- Treat `configs/datasets/*.toml` as dataset release-version sources of truth.
- Keep Hugging Face Dataset Cards outside Git; review and edit them directly on
  the Hub only during an owner-approved dataset publication.
- Keep pilot/debug artifacts separate from formal benchmarks and releases.
- Report test, pilot, validation, and publishing evidence separately.

## Architecture

- Keep experiment-specific values in versioned configuration or immutable
  evidence, not reusable runtime modules.
- New variation axes must not require a new collection engine, manifest type,
  selector, or exporter.
- Prefer generic entity paths, validators, samplers, split policies, and
  selection constraints over mass-, yaw-, shape-, or release-specific code.
- Preserve backward reading compatibility for published schemas and manifests.
- Do not change frozen plan hashes, release evidence, or immutable dataset tags.

## SO-101 Oracle repair governance

- Auto-merge-eligible repair diffs are limited to the reusable Oracle/control
  modules, versioned `configs/oracle-profiles/`, and their named tests.
- They may not change success criteria, scene geometry, physics assets,
  metadata schemas, exporters, Dashboard code, workflow/release/dataset specs,
  or GitHub governance. Mixed-purpose PRs are ineligible.
- The controlled DGX gate must bind evidence to the exact PR head commit. For
  every clustered failure class, all three frozen diagnostic seeds must each
  succeed within three attempts. Ten frozen canary seeds must pass 10/10 with
  `success=true` and `dataset_valid=true`.
- GitHub branch protection and all required CI/review checks remain mandatory.
  Agents and automation must never use admin bypass.
