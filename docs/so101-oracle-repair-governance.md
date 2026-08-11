# SO-101 Oracle repair governance

This policy is deliberately narrower than ordinary Farpoint development. It
exists so an owner-approved adaptive campaign can authorize normal GitHub
auto-merge for a repair without authorizing unrelated repository changes.

## Two independent gates

The GitHub-hosted `SO101 Oracle repair path policy` check validates the complete
PR diff. It permits additions or modifications only in the reusable
Oracle/control files, versioned `configs/oracle-profiles/`, and their named
tests. Deletions, renames, copies, mixed-purpose files, schemas, exporters,
Dashboard code, scene/physics/release configs, workflows, and governance are
ineligible.

The controlled DGX workflow runs from the exact PR head commit and validates a
`farpoint.oracle-repair-evidence.v1` document with
`scripts/validate_oracle_repair.py`. It must publish two successful commit
checks named:

- `SO101 Oracle diagnostic`
- `SO101 Oracle canary`

Each clustered failure class contributes exactly three frozen diagnostic
seeds; every seed must succeed and be dataset-valid within at most three
attempts. The independent frozen canary set must pass 10/10. Evidence also
asserts that the scene contract and success criteria did not change.

## Campaign grant

Merging this governance PR does not itself authorize auto-merge. A later
owner-reviewed campaign spec must opt in by campaign ID and policy ID. The grant
expires at campaign completion or abandonment. For a matching repair PR, an
agent may enable repository auto-merge only after the label, path check,
external diagnostic/canary checks, normal CI, reviews, and branch protection
are all satisfied.

Direct merge, admin bypass, dataset release changes, Hugging Face publication,
and reuse of the grant for another campaign or PR remain forbidden.
