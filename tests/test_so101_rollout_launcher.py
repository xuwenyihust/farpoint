from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_so101_act_rollout.sh"


def test_rollout_launcher_mounts_checkpoint_read_only_and_preserves_arguments(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == image && \"$2\" == inspect ]]; then\n"
        "  if [[ \"$3\" == farpoint-so101-isaaclab:3.0-beta2 ]]; then\n"
        "    echo sha256:ddcd4daa68cef3ece67f4fbad4eb8f5257d8236a55aba04d0697b55e7679fd04\n"
        "  else\n"
        "    echo sha256:d99274c14bc7e1064f3ad534deb1feecdcbeb271c9d04b3e46377d464c720293\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$1\" == run && \"$2\" == -d ]]; then printf '%s\\n' \"$@\" > \"$TEST_POLICY_ARGS\"; touch \"$TEST_POLICY_STARTED\"; echo policy-container; exit 0; fi\n"
        "if [[ \"$1\" == inspect || \"$1\" == logs || \"$1\" == stop ]]; then exit 0; fi\n"
        "mkdir -p \"$TEST_REPORT_ROOT\"\n"
        "printf '%s\\n' '{\"status\":\"PASS\"}' > \"$TEST_REPORT_ROOT/report.json\"\n"
        "for final_arg in \"$@\"; do :; done\n"
        "printf '%s\\n' \"${final_arg}\"\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    curl = bin_dir / "curl"
    curl.write_text(
        "#!/usr/bin/env bash\n[[ -f \"$TEST_POLICY_STARTED\" ]]\n", encoding="utf-8"
    )
    curl.chmod(0o755)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")
    asset = tmp_path / "SO-ARM101-USD.usd"
    asset.touch()
    data_root = tmp_path / "data"
    data_root.mkdir()
    spec = data_root / "spec with spaces.json"
    spec.write_text('{"control":{"replan_interval_steps":10}}')
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FARPOINT_DATA_ROOT": str(data_root),
        "FARPOINT_SO101_ASSET": str(asset),
        "FARPOINT_ACT_CHECKPOINT": str(checkpoint),
        "FARPOINT_GIT_COMMIT": "a" * 40,
        "TEST_POLICY_STARTED": str(tmp_path / "policy-started"),
        "TEST_POLICY_ARGS": str(tmp_path / "policy-args"),
        "TEST_REPORT_ROOT": str(data_root / "run"),
    }
    completed = subprocess.run(
        [
            str(LAUNCHER),
            "headless",
            "--spec",
            "/workspace/farpoint-data/spec with spaces.json",
            "--output-root",
            "/workspace/farpoint-data/run",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    command = shlex.split(completed.stdout.strip())
    assert command[:5] == [
        "/workspace/IsaacLab/isaaclab.sh",
        "-p",
        "/workspace/project/examples/isaaclab_so101_pick_place/rollout.py",
        "--mode",
        "headless",
    ]
    assert command[command.index("--checkpoint") + 1] == "/workspace/policy"
    assert command[command.index("--spec") + 1].endswith("spec with spaces.json")
    assert command.count("--enable_cameras") == 1
    policy_arguments = (tmp_path / "policy-args").read_text().splitlines()
    assert policy_arguments[policy_arguments.index("--replan-interval-steps") + 1] == "10"
