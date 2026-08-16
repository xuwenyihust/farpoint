#!/usr/bin/env python3
"""Score ACT checkpoints on a deterministic validation sample without touching test."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from farpoint.policy_training import (
    canonical_sha256,
    evenly_spaced_indices,
    file_sha256,
    load_training_spec,
    parse_episode_slice,
    select_validation_checkpoint,
    validation_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", choices=("pilot", "training"))
    return parser.parse_args()


def checkpoint_step(checkpoint_dir: Path) -> int:
    state_path = checkpoint_dir / "training_state" / "training_step.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if isinstance(state, int):
        return state
    for key in ("step", "training_step"):
        if key in state:
            return int(state[key])
    raise ValueError(f"training step missing from {state_path}")


def main() -> int:
    args = parse_args()
    if args.report.exists():
        raise FileExistsError(f"validation report already exists: {args.report}")
    spec = load_training_spec(args.config)
    validation = spec.get("validation")
    if validation is None:
        raise ValueError("training spec does not define validation")
    profile = args.profile or validation_profile(spec)
    if profile != validation_profile(spec):
        raise ValueError("requested profile differs from the frozen validation profile")
    preflight = json.loads(args.preflight_report.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight["execution"]["profile"] != profile:
        raise ValueError(f"{profile} preflight/training evidence is not PASS")
    if preflight.get("config_sha256") != canonical_sha256(spec):
        raise ValueError("preflight report does not bind the requested training config")

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    git_commit = os.environ.get("FARPOINT_GIT_COMMIT", "")
    image_id = os.environ.get("FARPOINT_TRAINING_IMAGE_ID", "")
    if git_commit != preflight["farpoint_git_commit"]:
        raise RuntimeError("validation commit differs from preflight")
    if image_id != preflight["training_image_id"]:
        raise RuntimeError("validation image differs from preflight")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for validation")

    checkpoint_dirs = sorted(
        path
        for path in (args.output_dir / "checkpoints").iterdir()
        if path.name != "last" and (path / "training_state" / "training_step.json").is_file()
    )
    expected_count = spec[profile]["steps"] // spec[profile]["save_freq"]
    if len(checkpoint_dirs) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} checkpoints, found {len(checkpoint_dirs)}"
        )

    first_pretrained = checkpoint_dirs[0] / "pretrained_model"
    policy_config = PreTrainedConfig.from_pretrained(first_pretrained, local_files_only=True)
    metadata = LeRobotDatasetMetadata(
        spec["dataset"]["repo_id"],
        root=args.dataset_root,
        revision=spec["dataset"]["revision"],
    )
    delta_timestamps = resolve_delta_timestamps(policy_config, metadata)
    validation_episodes = parse_episode_slice(spec["dataset"]["splits"][validation["split"]])
    dataset = LeRobotDataset(
        spec["dataset"]["repo_id"],
        root=args.dataset_root,
        revision=spec["dataset"]["revision"],
        episodes=validation_episodes,
        delta_timestamps=delta_timestamps,
        video_backend=spec["dataset"]["video_backend"],
    )
    expected_frames = spec["dataset"]["expected"]["selected_frames"][validation["split"]]
    if len(dataset) != expected_frames:
        raise RuntimeError(f"validation frame count mismatch: {len(dataset)} != {expected_frames}")
    sample_indices = evenly_spaced_indices(len(dataset), validation["sample_count"])
    subset = torch.utils.data.Subset(dataset, sample_indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=validation["batch_size"],
        shuffle=False,
        num_workers=validation["num_workers"],
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if validation["num_workers"] > 0 else None,
    )

    results = []
    for checkpoint_dir in checkpoint_dirs:
        pretrained_dir = checkpoint_dir / "pretrained_model"
        model_file = pretrained_dir / "model.safetensors"
        if not model_file.is_file():
            raise FileNotFoundError(model_file)
        step = checkpoint_step(checkpoint_dir)
        random.seed(validation["seed"])
        np.random.seed(validation["seed"])
        torch.manual_seed(validation["seed"])
        torch.cuda.manual_seed_all(validation["seed"])
        checkpoint_config = PreTrainedConfig.from_pretrained(
            pretrained_dir, local_files_only=True
        )
        checkpoint_config.pretrained_path = str(pretrained_dir)
        checkpoint_config.device = "cuda"
        policy = make_policy(checkpoint_config, ds_meta=metadata)
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=str(pretrained_dir)
        )
        # ACT's teacher-forced VAE objective is only populated in training mode.
        # inference_mode below still disables gradients and optimizer/state writes;
        # a fresh policy is loaded for every checkpoint and never saved again.
        policy.train()
        totals: dict[str, float] = {}
        sample_total = 0
        with torch.inference_mode():
            for batch in loader:
                processed = preprocessor(batch)
                loss, loss_parts = policy(processed)
                batch_size = int(processed["action"].shape[0])
                values = {"loss": float(loss.item()), **loss_parts}
                if not all(np.isfinite(value) for value in values.values()):
                    raise RuntimeError(f"non-finite validation metric at step {step}")
                for name, value in values.items():
                    totals[name] = totals.get(name, 0.0) + value * batch_size
                sample_total += batch_size
        result = {
            "step": step,
            "mean_loss": totals.pop("loss") / sample_total,
            "mean_components": {name: value / sample_total for name, value in totals.items()},
            "sample_count": sample_total,
            "checkpoint": str(checkpoint_dir),
            "model_sha256": file_sha256(model_file),
        }
        results.append(result)
        del policy, preprocessor
        torch.cuda.empty_cache()

    best, improvement = select_validation_checkpoint(
        results, validation["minimum_relative_improvement"]
    )
    report = {
        "schema_version": "farpoint.policy-validation.v1",
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": spec["experiment_id"],
        "training_profile": profile,
        "farpoint_git_commit": git_commit,
        "training_image_id": image_id,
        "config_sha256": canonical_sha256(spec),
        "environment": {
            "architecture": platform.machine(),
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "dataset": {
            "repo_id": spec["dataset"]["repo_id"],
            "revision": spec["dataset"]["revision"],
            "source": preflight["dataset"]["source"],
            "selected_split": validation["split"],
            "selected_episode_expression": spec["dataset"]["splits"][validation["split"]],
            "selected_episode_count": len(validation_episodes),
            "available_frame_count": len(dataset),
            "sample_indices_sha256": canonical_sha256(sample_indices),
            "sample_count": len(sample_indices),
            "excluded_split_expressions": {
                name: expression
                for name, expression in spec["dataset"]["splits"].items()
                if name != validation["split"]
            },
        },
        "checkpoints": sorted(results, key=lambda result: result["step"]),
        "selection": {
            "metric": "mean_act_training_objective",
            "best_step": best["step"],
            "best_checkpoint": best["checkpoint"],
            "best_mean_loss": best["mean_loss"],
            "relative_improvement_from_first": improvement,
            "minimum_relative_improvement": validation["minimum_relative_improvement"],
        },
        "interpretation": (
            "Offline teacher-forced ACT loss on a fixed validation sample; "
            "this is not simulator rollout success."
        ),
    }
    if preflight["dataset"]["source"]["kind"] == "hub":
        report["dataset"]["resolved_commit"] = preflight["dataset"]["source"][
            "resolved_commit"
        ]
    # Preserve the published v0.0.3 report field while allowing later datasets
    # to use train/validation without inventing demonstration test episodes.
    if "test" in spec["dataset"]["splits"]:
        report["dataset"]["excluded_test_expression"] = spec["dataset"]["splits"]["test"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
