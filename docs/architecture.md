# Architecture

Farpoint is organized around a simple loop:

```text
task config -> simulation runner -> episode recorder -> metrics -> dataset/export layer
```

## Core Concepts

### Task Config

A task config describes the scene, objects, capture settings, and success criteria. The first example uses `task.yaml` with `schema_version: "task.v1"` to define:

- Task name and language instruction.
- Number of simulation frames.
- Scene ground and lighting.
- Dynamic cube color, scale, and initial pose.
- Camera preview capture settings.
- Success criteria for PASS/FAIL.
- Output directory inside the mounted workspace.

See [Task Schema v1](task-schema.md) for the current contract.

### Simulation Runner

The current runner is a shell script that launches Isaac Sim in a Docker container. It uses:

- `nvcr.io/nvidia/isaac-sim:6.0.0`
- `SimulationApp({"headless": True})`
- A clean writable cache mount for `/isaac-sim/.cache`
- A mounted project workspace at `/workspace/project`

### Episode Recorder

Each run writes a machine-readable episode folder:

- `metadata.json`: task, simulator, timing, and environment metadata.
- `trajectory.jsonl`: one JSON object per recorded frame.
- `metrics.json`: success flag, frame count, final cube pose, and runtime summary.
- `phase_events.jsonl`: explicit Isaac Sim phase markers from the scene script.
- `preview/`: RGB frames rendered from an Isaac Sim camera for visual inspection.
- `_logs/`: captured process output from the runner.
- `_resources/`: host and container resource telemetry sampled during the run.
- `_phases/`: local and remote runner phase markers.

### Resource Trace

The runner samples host and container telemetry while the Isaac Sim container is active:

- GPU name, utilization, power, and temperature from `nvidia-smi`.
- Host load average and memory usage from Linux system files.
- Container CPU, memory, I/O, and process count from `docker stats`.

The telemetry layer writes two files per run:

- `*_resources.csv`: raw time-series samples.
- `*_resources_summary.json`: peak and aggregate values for dashboards, CI reports, and quick run inspection.

Some unified-memory systems can report active GPU utilization while traditional
`nvidia-smi` GPU memory fields are unavailable. The CSV schema makes that
explicit with:

- `gpu_memory_available`
- `gpu_memory_note`
- blank GPU memory fields when the platform reports `[N/A]`

On unified-memory systems, memory pressure should be interpreted as:

- Peak Workload Memory: maximum Docker/cgroup memory attributed to the simulation container during the run.
- Peak Host Memory Pressure: maximum system memory used relative to total unified memory during the run.
- GPU Load: utilization, power, and temperature from `nvidia-smi`.

Traditional GPU memory counters are still recorded when available on other
platforms.

### Episode Inspector

`scripts/build_episode_report.py` turns one episode folder into a static HTML report. The report links the episode to its matched resource summary and run log, then renders:

- Episode metadata and success metrics.
- In-page PNG frame playback and preview RGB frames.
- Cube height and vertical velocity curves.
- GPU utilization, power, and temperature curves.
- Workload memory and host memory pressure time-series curves.
- Scene, local runner, and remote runner phase timelines.
- Phase swimlanes that align local runner, remote runner, scene, and GPU spike timing.
- Warning and error excerpts from the run log.

`scripts/build_episode_index.py` builds a static dashboard across all complete episodes. It refreshes each episode report and lists run-level comparisons:

- Status, runtime, and frame counts.
- Preview thumbnails.
- Peak GPU utilization.
- Peak workload memory.
- Peak host memory pressure.
- Warning counts.

### Benchmark Runner

`scripts/run_pick_place_benchmark.py` orchestrates deterministic randomized batches on a remote GPU host. It:

- Uses a fixed seed list.
- Creates an isolated Isaac runtime directory for every trial.
- Applies a cooldown between Isaac processes on GB10.
- Uses a process watchdog and retries infrastructure failures that produce no episode.
- Checkpoints a manifest after every completed trial.
- Preserves task failures as benchmark evidence instead of retrying them.

`scripts/build_benchmark_report.py` aggregates the completed episodes into JSON and HTML reports with success rate, target-error statistics, resource peaks, failure taxonomy, acceptance checks, reproducibility checks for repeated seeds, and links to each episode replay.

The first version is deliberately static so it can be opened directly from disk, served by a simple local web server, attached to CI artifacts, or published as generated documentation later.

### Future Layers

The same pattern will support:

- Robot arms and manipulation tasks.
- RGB/depth camera observations.
- Policy and planner interfaces.
- LeRobot-compatible dataset export.

## Execution Model

The local repository is the source of truth. A runner may execute locally or
sync a minimal working copy to a remote GPU host. Runtime caches and generated
artifacts stay outside the source tree and are controlled by configuration.
