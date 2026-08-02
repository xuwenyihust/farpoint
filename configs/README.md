# Farpoint Configuration

Keep versioned, hardware-independent task and release configuration here.
Machine-specific values belong in ignored local files or environment variables.

Recommended configuration layers:

- `tasks/`: simulation task definitions
- `datasets/`: dataset release settings
- `benchmarks/`: benchmark acceptance thresholds
- `collections/`: resource budgets, source imports, coverage, and stopping rules
- `machines/`: local examples only; do not commit private hostnames or tokens

Generated experiment plans live under `plans/`. A committed plan is immutable:
rerunning its generator must reproduce the exact bytes, and changed inputs or
planner code require a new plan identity rather than an in-place overwrite.
