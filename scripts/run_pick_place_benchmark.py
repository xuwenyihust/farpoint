#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPISODES_ROOT = PROJECT_ROOT / "outputs" / "episodes"
BENCHMARKS_ROOT = PROJECT_ROOT / "outputs" / "benchmarks"
DEFAULT_EXAMPLE_PATH = "examples/isaac_ur10e_robotiq_scene"


def utc_now():
    return datetime.now(timezone.utc)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def remote_ssh():
    host = os.environ.get("FARPOINT_REMOTE_HOST", os.environ.get("DGX_SPARK_HOST", ""))
    hostname = os.environ.get(
        "FARPOINT_REMOTE_HOSTNAME",
        os.environ.get("DGX_SPARK_HOSTNAME", ""),
    )
    host_key_alias = os.environ.get(
        "FARPOINT_REMOTE_KEY_ALIAS",
        os.environ.get("DGX_SPARK_HOST_KEY_ALIAS", ""),
    )
    options = []
    if hostname:
        options = [
            "-o",
            f"HostName={hostname}",
            "-o",
            f"HostKeyAlias={host_key_alias}",
        ]
    return host, options


def publish_manifest(manifest_path):
    if os.environ.get("FARPOINT_REMOTE_DATA_PLATFORM", "1") != "1":
        return
    host, options = remote_ssh()
    remote_root = os.environ.get(
        "FARPOINT_REMOTE_ROOT",
        "~/farpoint",
    )
    benchmark_id = manifest_path.parent.name
    remote_dir = f"{remote_root}/outputs/benchmarks/{benchmark_id}"
    subprocess.run(
        ["ssh", *options, host, "mkdir", "-p", remote_dir],
        check=True,
    )
    rsync_ssh = "ssh"
    if options:
        rsync_ssh += " " + " ".join(options)
    subprocess.run(
        [
            "rsync",
            "-az",
            "-e",
            rsync_ssh,
            str(manifest_path),
            f"{host}:{remote_dir}/manifest.json",
        ],
        check=True,
    )


def episode_for_trial(benchmark_id, seed, repeat):
    matches = []
    for metadata_path in EPISODES_ROOT.glob("episode_*/metadata.json"):
        try:
            metadata = read_json(metadata_path)
        except (OSError, ValueError):
            continue
        if (
            metadata.get("benchmark_id") == benchmark_id
            and int(metadata.get("episode_seed", -1)) == seed
            and int(metadata.get("benchmark_repeat", 0)) == repeat
        ):
            matches.append((metadata.get("finished_at", ""), metadata_path.parent))
    return sorted(matches)[-1][1] if matches else None


def load_episode_result(episode_dir, seed, repeat, return_code):
    metrics_path = episode_dir / "metrics.json" if episode_dir else None
    metrics = read_json(metrics_path) if metrics_path and metrics_path.exists() else {}
    return {
        "seed": seed,
        "repeat": repeat,
        "episode_id": episode_dir.name if episode_dir else None,
        "return_code": return_code,
        "success": bool(metrics.get("success")),
        "failure_category": metrics.get("failure_category")
        or ("runner" if return_code else None),
        "failure_reason": metrics.get("failure_reason")
        or ("episode_output_not_found" if episode_dir is None else None),
        "final_target_xy_distance": metrics.get("final_target_xy_distance"),
        "object_lift_height": metrics.get("object_lift_height"),
        "release_settle_frames": metrics.get("release_settle_frames"),
        "post_release_motion": metrics.get("post_release_motion"),
        "elapsed_seconds": metrics.get("elapsed_seconds"),
        "initial_object_perception_xy_error": metrics.get(
            "initial_object_perception_xy_error"
        ),
        "bilateral_contact_frames": metrics.get("bilateral_contact_frames"),
        "transport_contact_frames": metrics.get("transport_contact_frames"),
        "temporary_grasp_joint_created": metrics.get(
            "temporary_grasp_joint_created"
        ),
        "dataset_valid": metrics.get("dataset_valid"),
        "dataset_observation_count": metrics.get("dataset_observation_count"),
    }


def parse_seeds(raw, count):
    if raw:
        seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
        if len(seeds) != len(set(seeds)):
            raise ValueError("seeds must be unique")
        return seeds
    return list(range(count))


def checkpoint(manifest_path, manifest):
    completed = len(manifest["trials"])
    passed = sum(1 for trial in manifest["trials"] if trial["success"])
    manifest["completed_trials"] = completed
    manifest["passed_trials"] = passed
    manifest["success_rate"] = passed / completed if completed else 0.0
    manifest["updated_at"] = utc_now().isoformat()
    write_json(manifest_path, manifest)
    publish_manifest(manifest_path)


def run_remote_trial(seed, repeat, benchmark_id, args, ordinal, total):
    env = os.environ.copy()
    env["FARPOINT_EPISODE_SEED"] = str(seed)
    env["FARPOINT_BENCHMARK_ID"] = benchmark_id
    env["FARPOINT_BENCHMARK_REPEAT"] = str(repeat)
    episode_dir = None
    return_code = 1
    for attempt in range(1, args.max_run_attempts + 1):
        cooldown_seconds = (
            args.cooldown_seconds
            if attempt == 1
            else args.retry_cooldown_seconds
        )
        if cooldown_seconds:
            print(
                f"COOLDOWN {cooldown_seconds}s before seed={seed} "
                f"repeat={repeat} attempt={attempt}",
                flush=True,
            )
            time.sleep(cooldown_seconds)
        print(
            f"RUN {ordinal}/{total} seed={seed} repeat={repeat} attempt={attempt}",
            flush=True,
        )
        completed = subprocess.run(
            ["bash", "scripts/run_isaac_example.sh", args.example_path],
            cwd=PROJECT_ROOT,
            env=env,
            check=False,
        )
        return_code = completed.returncode
        episode_dir = episode_for_trial(benchmark_id, seed, repeat)
        if episode_dir:
            break
        print(
            f"RETRY seed={seed} repeat={repeat}: no complete episode was produced",
            flush=True,
        )
    return load_episode_result(episode_dir, seed, repeat, return_code)


def main():
    parser = argparse.ArgumentParser(
        description="Run a resumable randomized UR10e + Robotiq pick-and-place benchmark."
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seeds", help="Comma-separated fixed seed list; overrides --count.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--benchmark-id")
    parser.add_argument("--min-success-rate", type=float, default=0.80)
    parser.add_argument("--example-path", default=DEFAULT_EXAMPLE_PATH)
    parser.add_argument("--task-name", default="isaac_ur10e_robotiq_scene")
    parser.add_argument("--task-type", default="randomized_pick_and_place_v2")
    parser.add_argument("--max-final-target-xy-distance", type=float, default=0.04)
    parser.add_argument("--max-perception-xy-error", type=float)
    parser.add_argument("--min-bilateral-contact-frames", type=int)
    parser.add_argument("--min-transport-contact-frames", type=int)
    parser.add_argument("--require-contact-only", action="store_true")
    parser.add_argument("--require-dataset", action="store_true")
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=60,
        help="Delay before each new Isaac process to let the DGX GPU/Kit runtime settle.",
    )
    parser.add_argument(
        "--max-run-attempts",
        type=int,
        default=2,
        help="Retry infrastructure failures that do not produce a complete episode.",
    )
    parser.add_argument(
        "--retry-cooldown-seconds",
        type=int,
        default=180,
        help="Recovery delay before retrying an infrastructure failure.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--verify-seeds",
        help="Comma-separated seeds to rerun once for reproducibility evidence.",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds, args.count)
    if not seeds:
        parser.error("at least one seed is required")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds cannot be negative")
    if args.max_run_attempts < 1:
        parser.error("--max-run-attempts must be at least 1")
    if args.retry_cooldown_seconds < 0:
        parser.error("--retry-cooldown-seconds cannot be negative")
    benchmark_id = args.benchmark_id or f"pick_place_v2_{utc_now().strftime('%Y%m%d_%H%M%S')}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", benchmark_id):
        parser.error("--benchmark-id may contain only letters, numbers, dots, underscores, and hyphens")
    manifest_path = BENCHMARKS_ROOT / benchmark_id / "manifest.json"

    if manifest_path.exists():
        if not args.resume:
            parser.error(f"{benchmark_id} already exists; pass --resume to continue it")
        manifest = read_json(manifest_path)
    else:
        manifest = {
            "schema_version": "benchmark.v1",
            "benchmark_id": benchmark_id,
            "task_name": args.task_name,
            "task_type": args.task_type,
            "example_path": args.example_path,
            "created_at": utc_now().isoformat(),
            "seeds": seeds,
            "repeats": args.repeats,
            "planned_trials": len(seeds) * args.repeats,
            "acceptance": {
                "min_success_rate": args.min_success_rate,
                "max_final_target_xy_distance": args.max_final_target_xy_distance,
                "min_object_lift_height": 0.15,
                "min_release_settle_frames": 120,
                "max_perception_xy_error": args.max_perception_xy_error,
                "min_bilateral_contact_frames": args.min_bilateral_contact_frames,
                "min_transport_contact_frames": args.min_transport_contact_frames,
                "require_contact_only": args.require_contact_only,
                "require_dataset": args.require_dataset,
            },
            "trials": [],
        }
        write_json(manifest_path, manifest)

    completed_keys = {
        (int(trial["seed"]), int(trial.get("repeat", 0)))
        for trial in manifest.get("trials", [])
    }
    for repeat in range(args.repeats):
        for seed in seeds:
            key = (seed, repeat)
            if key in completed_keys:
                print(f"SKIP seed={seed} repeat={repeat}: already recorded", flush=True)
                continue
            existing_episode = episode_for_trial(benchmark_id, seed, repeat)
            if existing_episode:
                print(
                    f"ADOPT seed={seed} repeat={repeat}: {existing_episode.name}",
                    flush=True,
                )
                existing_metrics = read_json(existing_episode / "metrics.json")
                manifest["trials"].append(
                    load_episode_result(
                        existing_episode,
                        seed,
                        repeat,
                        0 if existing_metrics.get("success") else 1,
                    )
                )
                completed_keys.add(key)
                checkpoint(manifest_path, manifest)
                continue

            manifest["trials"].append(
                run_remote_trial(
                    seed,
                    repeat,
                    benchmark_id,
                    args,
                    len(manifest["trials"]) + 1,
                    len(seeds) * args.repeats,
                )
            )
            checkpoint(manifest_path, manifest)

    verification_seeds = parse_seeds(args.verify_seeds, 0)
    reproducibility_trials = manifest.setdefault("reproducibility_trials", [])
    reproducibility_keys = {
        (int(trial["seed"]), int(trial.get("repeat", 1)))
        for trial in reproducibility_trials
    }
    for index, seed in enumerate(verification_seeds, start=1):
        repeat = 1
        key = (seed, repeat)
        if key in reproducibility_keys:
            print(f"SKIP reproducibility seed={seed}: already recorded", flush=True)
            continue
        existing_episode = episode_for_trial(benchmark_id, seed, repeat)
        if existing_episode:
            existing_metrics = read_json(existing_episode / "metrics.json")
            result = load_episode_result(
                existing_episode,
                seed,
                repeat,
                0 if existing_metrics.get("success") else 1,
            )
        else:
            result = run_remote_trial(
                seed,
                repeat,
                benchmark_id,
                args,
                index,
                len(verification_seeds),
            )
        reproducibility_trials.append(result)
        checkpoint(manifest_path, manifest)

    manifest["finished_at"] = utc_now().isoformat()
    checkpoint(manifest_path, manifest)
    accepted = (
        manifest["completed_trials"] == manifest["planned_trials"]
        and manifest["success_rate"] >= args.min_success_rate
    )
    manifest["accepted"] = accepted
    write_json(manifest_path, manifest)
    checkpoint(manifest_path, manifest)
    print(
        f"BENCHMARK_RESULT: {'PASS' if accepted else 'FAIL'} {benchmark_id} "
        f"{manifest['passed_trials']}/{manifest['completed_trials']} "
        f"({manifest['success_rate']:.1%})",
        flush=True,
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
