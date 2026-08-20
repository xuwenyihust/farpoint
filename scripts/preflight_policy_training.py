#!/usr/bin/env python3
"""Prepare a split-safe local LeRobot view and optionally run one ACT train step."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from farpoint.policy_training import (
    canonical_sha256,
    create_training_view,
    directory_tree_sha256,
    file_sha256,
    load_training_spec,
    training_arguments,
    unflatten_episode_stats,
    validate_dataset_info,
)
from farpoint.training_sampler import (
    build_sampler_plan,
    expected_group_sample_counts,
    selected_training_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "pilot", "training"), default="smoke")
    parser.add_argument("--run-act-step", action="store_true")
    parser.add_argument("--run-profile", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args()


def validate_resume_checkpoint(spec: dict, checkpoint: Path | None) -> dict | None:
    continuation = spec.get("continuation")
    if continuation is None:
        if checkpoint is not None:
            raise ValueError("resume checkpoint is only valid for a continuation contract")
        return None
    if checkpoint is None:
        raise ValueError("continuation contract requires --resume-checkpoint")
    if not checkpoint.is_absolute() or not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    pretrained = checkpoint / "pretrained_model"
    training_state = checkpoint / "training_state"
    required = {
        "model": pretrained / "model.safetensors",
        "train_config": pretrained / "train_config.json",
        "training_step": training_state / "training_step.json",
        "optimizer_state": training_state / "optimizer_state.safetensors",
        "optimizer_groups": training_state / "optimizer_param_groups.json",
        "rng_state": training_state / "rng_state.safetensors",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    step_payload = json.loads(required["training_step"].read_text(encoding="utf-8"))
    step = int(step_payload.get("step", -1))
    if step != int(continuation["source_step"]):
        raise ValueError(f"resume checkpoint step mismatch: {step}")
    model_sha256 = file_sha256(required["model"])
    if model_sha256 != continuation["source_model_sha256"]:
        raise ValueError("resume checkpoint model hash mismatch")
    source_config = json.loads(required["train_config"].read_text(encoding="utf-8"))
    if source_config.get("job_name") != f"{continuation['source_experiment_id']}_training":
        raise ValueError("resume checkpoint experiment id mismatch")
    if int(source_config.get("steps", -1)) != step:
        raise ValueError("resume checkpoint source steps mismatch")
    if int(source_config.get("seed", -1)) != int(spec["training"]["seed"]):
        raise ValueError("resume checkpoint seed mismatch")
    if int(source_config.get("batch_size", -1)) != int(spec["training"]["batch_size"]):
        raise ValueError("resume checkpoint batch size mismatch")
    return {
        "source_experiment_id": continuation["source_experiment_id"],
        "source_step": step,
        "checkpoint": str(checkpoint),
        "model_sha256": model_sha256,
        "train_config_sha256": file_sha256(required["train_config"]),
        "optimizer_state_sha256": file_sha256(required["optimizer_state"]),
        "optimizer_groups_sha256": file_sha256(required["optimizer_groups"]),
        "rng_state_sha256": file_sha256(required["rng_state"]),
    }


def main() -> int:
    args = parse_args()
    spec = load_training_spec(args.config)
    dataset_spec = spec["dataset"]
    git_commit = os.environ.get("FARPOINT_GIT_COMMIT", "")
    image_id = os.environ.get("FARPOINT_TRAINING_IMAGE_ID", "")
    if len(git_commit) != 40 or any(character not in "0123456789abcdef" for character in git_commit):
        raise RuntimeError("FARPOINT_GIT_COMMIT must bind evidence to an exact source commit")
    if not image_id.startswith("sha256:"):
        raise RuntimeError("FARPOINT_TRAINING_IMAGE_ID must bind evidence to a Docker image ID")
    if args.report.exists() or args.output_dir.exists() or args.view_root.exists():
        raise FileExistsError("report, output directory, and training view must be new paths")
    resume_binding = validate_resume_checkpoint(spec, args.resume_checkpoint)

    import av
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import load_nested_dataset, write_stats

    versions = {
        package: importlib.metadata.version(package)
        for package in ("lerobot", "torch", "torchvision", "av", "huggingface-hub")
    }
    if versions["lerobot"] != spec["environment"]["lerobot_version"]:
        raise RuntimeError(
            f"LeRobot version mismatch: {versions['lerobot']} != "
            f"{spec['environment']['lerobot_version']}"
        )
    if platform.machine() != spec["environment"]["architecture"]:
        raise RuntimeError(
            f"architecture mismatch: {platform.machine()} != "
            f"{spec['environment']['architecture']}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen experiment contract")
    gpu_probe = torch.ones((16, 16), device="cuda") @ torch.ones((16, 16), device="cuda")
    if not torch.isfinite(gpu_probe).all():
        raise RuntimeError("CUDA matrix operation returned non-finite values")
    av.codec.Codec("av1", "r")

    source_spec = dataset_spec.get("source", {"kind": "hub"})
    if source_spec["kind"] == "hub":
        hub_info = HfApi().dataset_info(
            dataset_spec["repo_id"], revision=dataset_spec["revision"]
        )
        if hub_info.sha != dataset_spec["resolved_commit"]:
            raise RuntimeError(
                f"dataset tag moved: {hub_info.sha} != {dataset_spec['resolved_commit']}"
            )
        args.source_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            dataset_spec["repo_id"],
            repo_type="dataset",
            revision=dataset_spec["resolved_commit"],
            local_dir=args.source_root,
        )
        source_binding = {
            "kind": "hub",
            "resolved_commit": hub_info.sha,
        }
    else:
        actual_tree_sha256, file_count = directory_tree_sha256(args.source_root)
        if actual_tree_sha256 != source_spec["tree_sha256"]:
            raise RuntimeError(
                "local dataset tree hash mismatch: "
                f"{actual_tree_sha256} != {source_spec['tree_sha256']}"
            )
        source_binding = {
            "kind": "local_snapshot",
            "tree_sha256": actual_tree_sha256,
            "file_count": file_count,
        }
    info = json.loads((args.source_root / "meta" / "info.json").read_text(encoding="utf-8"))
    validate_dataset_info(spec, info)

    train_episodes = selected_training_episodes(spec)
    all_episode_rows = load_nested_dataset(args.source_root / "meta" / "episodes")
    episode_lengths = {
        int(row["episode_index"]): int(row["length"]) for row in all_episode_rows
    }
    sampler_plan = build_sampler_plan(spec, episode_lengths)
    selected_rows = [all_episode_rows[index] for index in train_episodes]
    selected_frames = int(sampler_plan["selected_frame_count"])
    train_stats = aggregate_stats([unflatten_episode_stats(row) for row in selected_rows])
    source_stats_sha, view_stats_sha = create_training_view(
        args.source_root, args.view_root, train_stats, write_stats
    )
    sampler_plan_path = args.view_root / "meta" / "farpoint_sampler.json"
    sampler_plan_path.write_text(json.dumps(sampler_plan, indent=2) + "\n", encoding="utf-8")

    dataset = LeRobotDataset(
        dataset_spec["repo_id"],
        root=args.view_root,
        revision=dataset_spec["revision"],
        episodes=train_episodes,
        video_backend=dataset_spec["video_backend"],
    )
    if len(dataset) != selected_frames:
        raise RuntimeError(f"loaded train length mismatch: {len(dataset)} != {selected_frames}")
    sample = dataset[0]
    required_shapes = {}
    for feature, contract in dataset_spec["required_features"].items():
        shape = tuple(contract["shape"])
        if contract["dtype"] == "video":
            height, width, channels = shape
            shape = (channels, height, width)
        required_shapes[feature] = shape
    sample_shapes = {}
    for feature, expected_shape in required_shapes.items():
        value = sample[feature]
        shape = tuple(value.shape)
        sample_shapes[feature] = list(shape)
        if shape != expected_shape:
            raise RuntimeError(f"sample {feature} shape mismatch: {shape} != {expected_shape}")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"sample {feature} contains non-finite values")

    if args.run_act_step and args.profile != "smoke":
        raise ValueError("--run-act-step is retained only for the smoke profile")
    should_run = args.run_act_step or args.run_profile
    command = training_arguments(
        spec,
        args.view_root,
        args.output_dir,
        args.profile,
        resume_checkpoint=args.resume_checkpoint,
    )
    return_code = None
    if should_run:
        completed = subprocess.run(command, check=False)
        return_code = completed.returncode
        if return_code != 0:
            raise RuntimeError(f"ACT single-step smoke failed with exit code {return_code}")

    report = {
        "schema_version": "farpoint.policy-training-preflight.v1",
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": spec["experiment_id"],
        "farpoint_git_commit": git_commit,
        "training_image_id": image_id,
        "config_sha256": canonical_sha256(spec),
        "dataset": {
            "repo_id": dataset_spec["repo_id"],
            "revision": dataset_spec["revision"],
            "source": source_binding,
            "selected_split": "train",
            "selected_episode_expression": dataset_spec["splits"]["train"],
            "selected_episode_count": len(train_episodes),
            "selected_frame_count": selected_frames,
            "sampler_kind": sampler_plan["kind"],
            "sampler_plan_sha256": canonical_sha256(sampler_plan),
            "expected_group_sample_counts": expected_group_sample_counts(sampler_plan),
            "excluded_episode_expressions": {
                name: expression
                for name, expression in dataset_spec["splits"].items()
                if name != "train"
            },
            "source_stats_sha256": source_stats_sha,
            "train_only_stats_sha256": view_stats_sha,
        },
        "environment": {
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "versions": versions,
            "cuda_device": torch.cuda.get_device_name(0),
            "av1_decoder": True,
        },
        "sample_shapes": sample_shapes,
        "execution": {
            "profile": args.profile,
            "requested": should_run,
            "return_code": return_code,
            "command": command,
        },
    }
    if resume_binding is not None:
        report["continuation"] = resume_binding
    if source_binding["kind"] == "hub":
        report["dataset"]["resolved_commit"] = source_binding["resolved_commit"]
    if args.profile == "smoke":
        report["act_smoke"] = {
            "requested": should_run,
            "return_code": return_code,
            "command": command,
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
