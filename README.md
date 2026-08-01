# Farpoint

Farpoint is an open, simulation-first robot learning platform for building,
recording, validating, and evaluating physics-based manipulation tasks.

The project combines Isaac Sim environments, reproducible episode recording,
LeRobot-compatible dataset export, benchmark evaluation, and a static data
dashboard. Generated episodes and future model checkpoints are published
separately from the source repository.

## Repository Layout

```text
src/farpoint/       Reusable Python package
scripts/            CLI and report builders
pipelines/          Pipeline entry-point documentation
examples/           Isaac Sim task examples
configs/            Versioned configuration
schemas/            Dataset and metadata contracts
dashboard/          Dashboard frontend
tests/              Unit and contract tests
docs/               Public architecture and data documentation
outputs/            Local generated artifacts, ignored by Git
```

## Quickstart

Install the package and development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
```

Run the local test suite:

```bash
pytest -q
```

Contributions use feature branches and owner-reviewed pull requests. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the required checks and
[docs/development-workflow.md](docs/development-workflow.md) for the pilot,
benchmark, dataset, and release process.

Isaac Sim examples use the NVIDIA container image
`nvcr.io/nvidia/isaac-sim:6.0.0`. Configure the target GPU host explicitly:

```bash
export FARPOINT_REMOTE_HOST=<gpu-host>
export FARPOINT_REMOTE_ROOT=<remote-farpoint-root>
./scripts/run_isaac_example.sh examples/isaac_cube_scene
```

The runner records episode metadata, trajectories, metrics, preview frames,
phase events, and resource telemetry under `outputs/`. These generated files
are intentionally ignored by Git.

## Dataset V1

Farpoint Dataset V1 targets the LeRobot Dataset v3 layout and contains
physics-based UR10e + Robotiq manipulation episodes. The public contract is
documented in [docs/dataset-v1/data-contract.md](docs/dataset-v1/data-contract.md).

The current release is available on
[Hugging Face](https://huggingface.co/datasets/wenyixu101/farpoint-ur10e-robotiq-2f85).
The Dataset Card is the authoritative release-level license and provenance
notice.

The exporter and validators are available as scripts:

```bash
python3 scripts/export_lerobot_v1_mini.py <output-dir> <episode-dir> ...
python3 scripts/validate_lerobot_dataset.py <dataset-dir>
```

## Dashboard

The dashboard frontend lives under `dashboard/` and can be served by the data
platform or packaged as a static application. It consumes generated episode
and benchmark reports without coupling the UI to a specific machine or GPU
host.

## Documentation

Start with [docs/README.md](docs/README.md). Machine-specific operations,
internal roadmaps, and experiment logs are kept outside the public repository
under the ignored `.codex/` directory.

## License

The Farpoint source code is licensed under the [Apache License 2.0](LICENSE).
The repository's `NOTICE` file describes important third-party boundaries.

The separately published `farpoint-ur10e-robotiq-2f85` dataset is intended to use the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/),
subject to the provenance and redistribution terms of all included assets.
The dataset's Hugging Face Dataset Card is the authoritative release-level
license notice.
