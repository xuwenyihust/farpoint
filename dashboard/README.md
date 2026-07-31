# Farpoint Dashboard

The current dashboard is a static-friendly frontend served by the Farpoint
data platform. It reads generated episode and benchmark reports and is kept
separate from simulation and dataset-generation code so it can later be
published as a Hugging Face Space.

The generated report data remains under `outputs/` and is intentionally ignored
by Git. A public release should publish a curated manifest or dataset-backed
snapshot instead of committing raw simulation artifacts here.
