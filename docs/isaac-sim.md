# Isaac Sim Setup

Farpoint runs Isaac Sim through the official NVIDIA container image:

```text
nvcr.io/nvidia/isaac-sim:6.0.0
```

The runner is designed for headless execution and expects:

- Docker with the NVIDIA container runtime.
- A writable Isaac Sim cache mounted at `/isaac-sim/.cache`.
- A writable project workspace mounted at `/workspace/project`.
- `ACCEPT_EULA=Y` and `PRIVACY_CONSENT=Y` in the container environment.

Run a public example with:

```bash
./scripts/run_isaac_example.sh examples/isaac_cube_scene
```

Hardware-specific setup, remote host details, runtime cache paths, and
benchmark logs are intentionally kept in `.codex/local/` rather than this
public document.
