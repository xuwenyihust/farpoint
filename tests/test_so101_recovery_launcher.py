from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_so101_recovery_collection.sh"


def test_recovery_launcher_binds_checkpoint_replan_and_collector_runtime(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == image && "$2" == inspect ]]; then echo sha256:$(printf a%.0s {1..64}); exit 0; fi\n'
        'if [[ "$1" == run && "$2" == -d ]]; then printf \'%s\\n\' "$@" > "$TEST_POLICY_ARGS"; touch "$TEST_POLICY_STARTED"; echo policy; exit 0; fi\n'
        'if [[ "$1" == inspect || "$1" == logs || "$1" == stop ]]; then exit 0; fi\n'
        'for final_arg in "$@"; do :; done\n'
        "printf '%s\\n' \"${final_arg}\"\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    curl = bin_dir / "curl"
    curl.write_text('#!/usr/bin/env bash\n[[ -f "$TEST_POLICY_STARTED" ]]\n')
    curl.chmod(0o755)

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = checkpoint / "model.safetensors"
    model.write_bytes(b"recovery-model")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    data_root = tmp_path / "data"
    runtime = data_root / "campaign/segments/segment-000/recovery-runtime.json"
    runtime.parent.mkdir(parents=True)
    runtime.write_text(
        json.dumps(
            {
                "source_policy": {"model_sha256": model_sha},
                "control": {"replan_interval_steps": 10},
            }
        )
    )
    asset = tmp_path / "asset.usd"
    asset.touch()
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FARPOINT_DATA_ROOT": str(data_root),
        "FARPOINT_SO101_ASSET": str(asset),
        "FARPOINT_ACT_CHECKPOINT": str(checkpoint),
        "FARPOINT_GIT_COMMIT": "b" * 40,
        "FARPOINT_SIMULATOR_IMAGE_DIGEST": f"sha256:{'c' * 64}",
        "TEST_POLICY_STARTED": str(tmp_path / "policy-started"),
        "TEST_POLICY_ARGS": str(tmp_path / "policy-args"),
    }
    completed = subprocess.run(
        [
            str(LAUNCHER),
            "headless",
            "--recovery-runtime",
            str(runtime),
            "--plan",
            str(data_root / "campaign/segments/segment-000/plan.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    policy_args = (tmp_path / "policy-args").read_text().splitlines()
    assert policy_args[policy_args.index("--replan-interval-steps") + 1] == "10"
    command = shlex.split(completed.stdout.strip().splitlines()[-1])
    assert command[command.index("--recovery-runtime") + 1] == (
        "/workspace/farpoint-data/campaign/segments/segment-000/recovery-runtime.json"
    )
    assert command[:3] == [
        "/workspace/IsaacLab/isaaclab.sh",
        "-p",
        "/workspace/project/examples/isaaclab_so101_pick_place/collect.py",
    ]
