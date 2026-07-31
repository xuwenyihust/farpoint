# Farpoint Pipelines

This directory contains the public orchestration layer for Farpoint. Pipeline
entry points should be small, composable commands that call reusable code from
`src/farpoint/`.

Planned stable commands:

```text
farpoint simulate
farpoint export
farpoint validate
farpoint benchmark
farpoint publish
```

Isaac Sim containers, DGX paths, credentials, and generated artifacts belong in
configuration or environment variables, never in reusable library modules.
