# SO-101 v0.1.0 formal campaign

The formal campaign is a new lineage with 200 selected successes and no global
attempt limit across its segments. Segment 000 freezes 200 deterministic Sobol
scenes and a maximum of three attempts per scene. A scene that fails three times
is deferred; a continuation preserves the exact object, yaw stratum, region,
split, and quota ordinal while deriving a new variation seed.

The formal config is `configs/variations/so101_v010_formal200.json`. It binds the
passed 12-scene dual-camera pilot report and manifest by SHA256, the base object
and scene config, and both campaign/runtime watchdog policies. Building fails if
any bound file has changed.

After the implementation PR is merged, initialize from that exact commit:

```bash
python3 scripts/build_v010_formal_campaign.py \
  --config configs/variations/so101_v010_formal200.json \
  --pilot-report <immutable-pilot-report.json> \
  --pilot-manifest <immutable-pilot-manifest.json> \
  --campaign-id <formal-campaign-id> \
  --git-commit <merged-commit> \
  --output-root <new-campaign-root>
```

Run segment 000 with `scripts/run_so101_isaaclab.sh headless`, the frozen plan,
manifest and episode paths under `segments/segment-000`, `--collection-id` set
to the plan id, the bound runtime watchdog, `--require-dual-camera`, and the
campaign/segment identity arguments. The collector may pause but must not
discard selected successes.

Evaluate all immutable segments with `scripts/run_campaign_recovery.py
evaluate`. If it emits `FREEZE_REPLACEMENT_SEGMENT`, build the new scenes with
`scripts/build_v010_replacement_segment.py`; the script rejects changed quotas,
unrequested scenes, a mutable parent, or a mismatched parent-manifest hash. An
Oracle repair continuation additionally uses the new merged repair commit and
its governance-approved profile allowlist.

Continuation planning aggregates every indexed segment rather than inspecting
only the latest manifest. A missing quota whose current variation seed has used
fewer than three attempts is carried forward with the same resolved scene and
only its remaining attempt budget. Only a seed that has consumed all three
attempts is marked deferred and replaced by a new deterministic same-quota
seed. This preserves stopped-segment successes and prevents a watchdog pause
from either orphaning a quota or resetting its per-seed attempt limit.

The frozen plan also contains 20 rollout-only scenes. They are disjoint from
collection, replacements, Oracle diagnostics, and validation checkpoint
selection; they never enter the demonstration export.
