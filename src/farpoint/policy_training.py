"""Split-safe, revision-pinned policy training helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from farpoint.contracts import validate_contract
from farpoint.training_sampler import selected_training_episodes, validate_sampling_contract


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_episode_slice(expression: str, *, allow_empty: bool = False) -> list[int]:
    parts = expression.split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"episode split must be a non-negative start:stop slice: {expression}")
    start, stop = map(int, parts)
    if stop < start or (stop == start and not allow_empty):
        raise ValueError(f"episode split must have stop greater than start: {expression}")
    return list(range(start, stop))


def load_training_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_contract(payload)
    if errors:
        raise ValueError("invalid policy training contract:\n" + "\n".join(errors))
    validate_split_partition(payload)
    validate_sampling_contract(payload)
    validate_training_profiles(payload)
    return payload


def validate_split_partition(spec: dict[str, Any]) -> None:
    split_indices = {
        name: parse_episode_slice(expression)
        for name, expression in spec["dataset"]["splits"].items()
    }
    flat = [episode for values in split_indices.values() for episode in values]
    expected_total = spec["dataset"]["expected"]["total_episodes"]
    if len(flat) != expected_total or sorted(flat) != list(range(expected_total)):
        raise ValueError("logical dataset splits must partition all episode indices exactly")

    metadata_splits = spec["dataset"].get("metadata_splits")
    if metadata_splits is not None:
        metadata_indices = {
            name: parse_episode_slice(expression, allow_empty=True)
            for name, expression in metadata_splits.items()
        }
        metadata_flat = [episode for values in metadata_indices.values() for episode in values]
        if len(metadata_flat) != expected_total or sorted(metadata_flat) != list(
            range(expected_total)
        ):
            raise ValueError("metadata splits must partition all episode indices exactly")
        for name, expression in spec["dataset"]["splits"].items():
            if metadata_splits.get(name) != expression:
                raise ValueError(f"logical split {name} must match its metadata split")


def validate_training_profiles(spec: dict[str, Any]) -> None:
    if "validation" not in spec:
        return
    validation = spec["validation"]
    profile = validation_profile(spec)
    run = spec.get(profile)
    if run is None:
        raise ValueError(f"validation profile is not configured: {profile}")
    save_freq = run.get("save_freq")
    if not run["save_checkpoint"] or save_freq is None:
        raise ValueError(f"{profile} must save checkpoints at a fixed frequency")
    if run["steps"] % save_freq != 0:
        raise ValueError(f"{profile} steps must be divisible by save_freq")
    if run["steps"] // save_freq < 2:
        raise ValueError(f"{profile} must produce at least two checkpoints")
    available_frames = spec["dataset"]["expected"]["selected_frames"][validation["split"]]
    if validation["sample_count"] > available_frames:
        raise ValueError("validation sample_count exceeds available split frames")


def validation_profile(spec: dict[str, Any]) -> str:
    """Resolve the checkpoint-producing profile while preserving pilot v1 specs."""
    validation = spec.get("validation")
    if validation is None:
        raise ValueError("training spec does not define validation")
    explicit = validation.get("profile")
    if explicit is not None:
        return str(explicit)
    if "pilot" in spec:
        return "pilot"
    return "training"


def validate_dataset_info(spec: dict[str, Any], info: dict[str, Any]) -> None:
    dataset = spec["dataset"]
    expected = dataset["expected"]
    checks = {
        "codebase_version": dataset["codebase_version"],
        "total_episodes": expected["total_episodes"],
        "total_frames": expected["total_frames"],
        "fps": expected["fps"],
    }
    errors = [
        f"{key}: expected {value!r}, got {info.get(key)!r}"
        for key, value in checks.items()
        if info.get(key) != value
    ]
    expected_info_splits = dataset.get("metadata_splits", dataset["splits"])
    if info.get("splits") != expected_info_splits:
        errors.append(f"splits: expected {expected_info_splits!r}, got {info.get('splits')!r}")
    for name, required in dataset["required_features"].items():
        actual = (info.get("features") or {}).get(name)
        if actual is None:
            errors.append(f"required feature missing: {name}")
            continue
        for field in ("dtype", "shape"):
            if actual.get(field) != required[field]:
                errors.append(
                    f"feature {name}.{field}: expected {required[field]!r}, "
                    f"got {actual.get(field)!r}"
                )
    if errors:
        raise ValueError("dataset metadata does not match frozen contract:\n" + "\n".join(errors))


def unflatten_episode_stats(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for key, value in row.items():
        if not key.startswith("stats/"):
            continue
        _, feature, statistic = key.split("/", 2)
        stats.setdefault(feature, {})[statistic] = np.asarray(value)
    if not stats:
        raise ValueError("episode record contains no flattened stats")
    return stats


def create_training_view(
    source_root: Path,
    view_root: Path,
    train_stats: dict[str, Any],
    write_stats: Any,
) -> tuple[str, str]:
    """Create a new local dataset view; never mutate the immutable source cache."""
    source_stats = source_root / "meta" / "stats.json"
    if not source_stats.is_file():
        raise FileNotFoundError(source_stats)
    if view_root.exists():
        raise FileExistsError(f"training view already exists: {view_root}")
    source_hash_before = file_sha256(source_stats)
    view_root.mkdir(parents=True)
    try:
        shutil.copytree(source_root / "meta", view_root / "meta")
        for name in ("data", "videos"):
            source = source_root / name
            if source.exists():
                os.symlink(source.resolve(), view_root / name, target_is_directory=True)
        write_stats(train_stats, view_root)
    except Exception:
        shutil.rmtree(view_root)
        raise
    source_hash_after = file_sha256(source_stats)
    if source_hash_before != source_hash_after:
        raise RuntimeError("immutable source dataset stats changed while creating training view")
    return source_hash_before, file_sha256(view_root / "meta" / "stats.json")


def training_arguments(
    spec: dict[str, Any], dataset_root: Path, output_dir: Path, profile: str
) -> list[str]:
    if profile not in spec or profile not in {"training", "pilot", "smoke"}:
        raise ValueError(f"unsupported training profile: {profile}")
    run = spec[profile]
    episodes = selected_training_episodes(spec)
    entrypoint = ["lerobot-train"]
    sampling = spec.get("sampling")
    if (
        profile == "training"
        and sampling is not None
        and sampling["kind"] == "deterministic_grouped_batches"
    ):
        entrypoint = [
            "python",
            "/workspace/project/scripts/train_so101_act_grouped.py",
            f"--farpoint-sampler-plan={dataset_root / 'meta' / 'farpoint_sampler.json'}",
        ]
    arguments = [
        *entrypoint,
        f"--dataset.repo_id={spec['dataset']['repo_id']}",
        f"--dataset.revision={spec['dataset']['revision']}",
        f"--dataset.root={dataset_root}",
        f"--dataset.episodes={json.dumps(episodes, separators=(',', ':'))}",
        f"--dataset.video_backend={spec['dataset']['video_backend']}",
        f"--policy.type={spec['policy']['type']}",
        f"--policy.device={spec['policy']['device']}",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--job_name={spec['experiment_id']}_{profile}",
        f"--batch_size={run['batch_size']}",
        f"--steps={run['steps']}",
        f"--num_workers={run['num_workers']}",
        f"--save_checkpoint={str(run['save_checkpoint']).lower()}",
        f"--log_freq={run['log_freq']}",
        f"--seed={run['seed']}",
        f"--wandb.enable={str(run['wandb_enable']).lower()}",
    ]
    if run.get("save_freq") is not None:
        arguments.append(f"--save_freq={run['save_freq']}")
    for name in (
        "vision_backbone",
        "pretrained_backbone_weights",
        "chunk_size",
        "n_action_steps",
    ):
        if name in spec["policy"]:
            arguments.append(f"--policy.{name}={spec['policy'][name]}")
    return arguments


def evenly_spaced_indices(length: int, sample_count: int) -> list[int]:
    """Return deterministic, unique indices spanning an entire sequence."""
    if length < 1:
        raise ValueError("length must be positive")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if sample_count > length:
        raise ValueError("sample_count cannot exceed length")
    if sample_count == 1:
        return [length // 2]
    return [round(index * (length - 1) / (sample_count - 1)) for index in range(sample_count)]


def select_validation_checkpoint(
    checkpoints: list[dict[str, Any]], minimum_relative_improvement: float
) -> tuple[dict[str, Any], float]:
    """Select the lowest-loss checkpoint and enforce a meaningful pilot trend."""
    if len(checkpoints) < 2:
        raise ValueError("at least two validation checkpoints are required")
    if not 0 <= minimum_relative_improvement < 1:
        raise ValueError("minimum_relative_improvement must be in [0, 1)")
    ordered = sorted(checkpoints, key=lambda result: int(result["step"]))
    losses = [float(result["mean_loss"]) for result in ordered]
    if not np.isfinite(losses).all():
        raise ValueError("validation losses must all be finite")
    first = losses[0]
    if first <= 0:
        raise ValueError("first validation loss must be positive")
    best = min(ordered, key=lambda result: float(result["mean_loss"]))
    improvement = (first - float(best["mean_loss"])) / first
    if int(best["step"]) <= int(ordered[0]["step"]):
        raise ValueError("no later checkpoint improves on the first checkpoint")
    if improvement < minimum_relative_improvement:
        raise ValueError(
            f"relative validation improvement {improvement:.6f} is below "
            f"{minimum_relative_improvement:.6f}"
        )
    return best, improvement
