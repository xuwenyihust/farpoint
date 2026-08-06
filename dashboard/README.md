# Farpoint Dashboard

The current dashboard is a static-friendly frontend served by the Farpoint
data platform. It reads generated episode and benchmark reports and is kept
separate from simulation and dataset-generation code so it can later be
published as a Hugging Face Space.

The generated report data remains under `outputs/` and is intentionally ignored
by Git. A public release should publish a curated manifest or dataset-backed
snapshot instead of committing raw simulation artifacts here.

## External episode roots

The data platform can index completed or in-progress Farpoint episodes outside
its managed `outputs/episodes` directory. Repeat `--episode-root` for each
read-only tree to scan recursively:

```bash
python3 scripts/data_platform_server.py \
  --outputs-root outputs \
  --episode-root /home/wenyixu/datasets/farpoint-so101
```

`FARPOINT_EPISODE_ROOTS` provides the same configuration as an OS-path-separated
list. External episodes remain in place and are never copied or quarantined.
The registry supports legacy top-level episode metadata and Farpoint episode v3
metadata. For v3, the Dashboard reads `identity`, `outcome`, `recording`, and
`variation`, and plays the front camera directly from `rgb/front_*.png`.
The episode details panel displays canonical scene entities plus the requested
and simulator-resolved entity states; these remain read-only sidecar data and
are fetched on demand rather than copied into the registry database.
