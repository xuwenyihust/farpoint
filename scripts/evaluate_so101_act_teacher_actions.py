#!/usr/bin/env python3
"""Measure one-step ACT action error on frozen dataset observations."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from farpoint.policy_rollout import constrain_policy_action, summarize_action_errors
from farpoint.policy_training import canonical_sha256, evenly_spaced_indices, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--resolved-commit", required=True)
    parser.add_argument("--episode-index", type=int, action="append", required=True)
    parser.add_argument("--samples-per-episode", type=int, default=32)
    parser.add_argument("--max-delta-calibrated", type=float, default=6.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _numpy_image(value) -> np.ndarray:
    image = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if image.shape == (3, 480, 640):
        image = np.moveaxis(image, 0, -1)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.shape != (480, 640, 3):
        raise ValueError(f"unexpected dataset image shape: {image.shape}")
    return image


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import importlib.metadata
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.utils.control_utils import predict_action

    if not torch.cuda.is_available():
        raise RuntimeError("teacher-observation evaluation requires CUDA")
    config = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    config.pretrained_path = str(args.checkpoint)
    config.device = "cuda"
    config.n_action_steps = 1
    if hasattr(config, "pretrained_backbone_weights"):
        config.pretrained_backbone_weights = None
    policy = get_policy_class(config.type).from_pretrained(
        args.checkpoint, config=config, local_files_only=True, strict=True
    ).to("cuda")
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(args.checkpoint)
    )
    rows = []
    sample_identity = []
    for episode_index in args.episode_index:
        dataset = LeRobotDataset(
            args.repo_id,
            root=args.dataset_root,
            revision=args.revision,
            episodes=[episode_index],
            video_backend="pyav",
        )
        for local_index in evenly_spaced_indices(len(dataset), min(args.samples_per_episode, len(dataset))):
            sample = dataset[local_index]
            state = np.asarray(sample["observation.state"], dtype=np.float32).reshape(6)
            expert = np.asarray(sample["action"], dtype=np.float32).reshape(6)
            observation = {
                "observation.state": state,
                "observation.images.front": _numpy_image(sample["observation.images.front"]),
                "observation.images.wrist": _numpy_image(sample["observation.images.wrist"]),
            }
            for component in (policy, preprocessor, postprocessor):
                if hasattr(component, "reset"):
                    component.reset()
            predicted = predict_action(
                observation,
                policy,
                torch.device("cuda"),
                preprocessor,
                postprocessor,
                use_amp=False,
                task="Pick up the cube and place it on the green target pad.",
                robot_type="so101",
            ).detach().cpu().numpy().reshape(6)
            applied, safety = constrain_policy_action(
                predicted, state, max_delta=args.max_delta_calibrated
            )
            _, expert_safety = constrain_policy_action(
                expert, state, max_delta=args.max_delta_calibrated
            )
            rows.append(
                {
                    "episode_index": episode_index,
                    "local_index": local_index,
                    "predicted": predicted.tolist(),
                    "applied": applied.tolist(),
                    "expert": expert.tolist(),
                    "prediction_safety": safety,
                    "expert_safety": expert_safety,
                }
            )
            sample_identity.append([episode_index, local_index])
    report = {
        "schema_version": "farpoint.act-teacher-action-evaluation.v1",
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "farpoint_git_commit": os.environ.get("FARPOINT_GIT_COMMIT", ""),
        "policy_image_id": os.environ.get("FARPOINT_POLICY_IMAGE_ID", ""),
        "lerobot_version": importlib.metadata.version("lerobot"),
        "checkpoint": {
            "path": str(args.checkpoint),
            "model_sha256": file_sha256(args.checkpoint / "model.safetensors"),
            "n_action_steps": 1,
        },
        "dataset": {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "resolved_commit": args.resolved_commit,
            "episode_indices": args.episode_index,
            "sample_identity_sha256": canonical_sha256(sample_identity),
            "sample_count": len(rows),
        },
        "metrics": summarize_action_errors(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
