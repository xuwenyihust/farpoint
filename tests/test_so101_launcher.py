from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_so101_isaaclab.sh"


def _launch(tmp_path: Path, *args: str) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "for final_arg in \"$@\"; do :; done\n"
        "printf '%s\\n' \"${final_arg}\"\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    asset = tmp_path / "SO-ARM101-USD.usd"
    asset.touch()
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FARPOINT_DATA_ROOT": str(tmp_path / "data"),
        "FARPOINT_SO101_ASSET": str(asset),
    }
    result = subprocess.run(
        [str(LAUNCHER), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return shlex.split(result.stdout.strip())


def test_viewer_defaults_to_webrtc_kit_at_matching_resolution(tmp_path: Path):
    command = _launch(tmp_path, "viewer", "--plan", "/data/plan.json")

    assert command[:5] == [
        "/workspace/IsaacLab/isaaclab.sh",
        "-p",
        "/workspace/project/examples/isaaclab_so101_pick_place/collect.py",
        "--mode",
        "viewer",
    ]
    assert command[command.index("--livestream") + 1] == "2"
    assert command[command.index("--visualizer") + 1] == "kit"
    assert command[command.index("--kit_args") + 1] == (
        "--/app/window/width=1280 --/app/window/height=720 --no-window"
    )
    assert command.count("--enable_cameras") == 1
    assert command[-2:] == ["--plan", "/data/plan.json"]


def test_viewer_preserves_explicit_multiword_kit_args(tmp_path: Path):
    kit_args = "--/app/window/width=960 --/app/window/height=540 --no-window"
    command = _launch(
        tmp_path,
        "viewer",
        "--livestream",
        "2",
        "--visualizer",
        "kit",
        "--kit_args",
        kit_args,
    )

    assert command[command.index("--kit_args") + 1] == kit_args
    assert command.count("--livestream") == 1
    assert command.count("--visualizer") == 1


def test_local_window_viewer_does_not_force_webrtc_visualizer(tmp_path: Path):
    command = _launch(tmp_path, "viewer", "--livestream=0")

    assert "--visualizer" not in command
    assert "--kit_args" not in command
    assert "--enable_cameras" in command


def test_headless_preserves_collector_argument_boundaries(tmp_path: Path):
    command = _launch(
        tmp_path,
        "headless",
        "--plan",
        "/path with spaces/plan.json",
    )

    assert "--livestream" not in command
    assert command[-2:] == ["--plan", "/path with spaces/plan.json"]
