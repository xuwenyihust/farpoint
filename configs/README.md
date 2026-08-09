# Farpoint Configuration

Keep versioned, hardware-independent task and release configuration here.
Machine-specific values belong in ignored local files or environment variables.

Recommended configuration layers:

- `tasks/`: simulation task definitions
- `datasets/`: one independent release specification per Hugging Face dataset;
  dataset versions do not define the Farpoint code version
- `benchmarks/`: benchmark acceptance thresholds
- `collections/`: resource budgets, source imports, coverage, and stopping rules
- `selections/`: reusable balanced-subset policies and coverage constraints
- `workflows/`: frozen multi-stage admission gates and watchdog policies
- `machines/`: local examples only; do not commit private hostnames or tokens

Generated experiment plans live under `plans/`. A committed plan is immutable:
rerunning its generator must reproduce the exact bytes, and changed inputs or
planner code require a new plan identity rather than an in-place overwrite.
