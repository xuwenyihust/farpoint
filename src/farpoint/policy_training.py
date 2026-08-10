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


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_episode_slice(expression: str) -> list[int]:
    parts = expression.split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"episode split must be a non-negative start:stop slice: {expression}")
    start, stop = map(int, parts)
    if stop <= start:
        raise ValueError(f"episode split must have stop greater than start: {expression}")
    return list(range(start, stop))


def load_training_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_contract(payload)
    if errors:
        raise ValueError("invalid policy training contract:\n" + "\n".join(errors))
    validate_split_partition(payload)
    return payload


def validate_split_partition(spec: dict[str, Any]) -> None:
    split_indices = {
        name: parse_episode_slice(expression)
        for name, expression in spec["dataset"]["splits"].items()
    }
    flat = [episode for values in split_indices.values() for episode in values]
    expected_total = spec["dataset"]["expected"]["total_episodes"]
    if len(flat) != expected_total or sorted(flat) != list(range(expected_total)):
        raise ValueError("train, validation, and test must partition all episode indices exactly")


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
    if info.get("splits") != dataset["splits"]:
        errors.append(f"splits: expected {dataset['splits']!r}, got {info.get('splits')!r}")
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
    if profile not in {"training", "smoke"}:
        raise ValueError(f"unsupported training profile: {profile}")
    run = spec[profile]
    episodes = parse_episode_slice(spec["dataset"]["splits"]["train"])
    arguments = [
        "lerobot-train",
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
    return arguments
