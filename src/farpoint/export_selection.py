"""Compose immutable Farpoint export selections without rewriting episodes."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SELECTION_SCHEMA_VERSION = "farpoint.export-selection.v1"
SPLIT_ORDER = ("train", "validation", "test")
VALID_SPLITS = set(SPLIT_ORDER)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_export_selection(path: Path) -> dict[str, Any]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError(f"selection must use {SELECTION_SCHEMA_VERSION}")
    if not selection.get("dataset_id"):
        raise ValueError("selection must define dataset_id")
    episodes = selection.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("selection must contain episodes")
    return selection


def compose_export_selections(
    sources: Iterable[tuple[str, Path]],
    *,
    dataset_id: str,
    selection_policy: str,
) -> dict[str, Any]:
    """Combine selections while binding every input by role and SHA-256."""
    if not dataset_id or not selection_policy:
        raise ValueError("dataset_id and selection_policy must be non-empty")
    rows = list(sources)
    if len(rows) < 2:
        raise ValueError("composition requires at least two selections")
    roles = [role for role, _ in rows]
    if any(not role for role in roles) or len(set(roles)) != len(roles):
        raise ValueError("selection source roles must be unique and non-empty")

    episodes = []
    source_records = []
    seen_episode_dirs: set[str] = set()
    for role, path in rows:
        selection = load_export_selection(path)
        source_records.append(
            {
                "role": role,
                "dataset_id": selection["dataset_id"],
                "selection_sha256": file_sha256(path),
                "episode_count": len(selection["episodes"]),
            }
        )
        for index, source_episode in enumerate(selection["episodes"]):
            episode = deepcopy(source_episode)
            episode_dir = str(episode.get("episode_dir") or "")
            if not episode_dir or not episode.get("trial_id"):
                raise ValueError(f"{role} episode {index} is missing identity fields")
            if episode.get("split") not in VALID_SPLITS:
                raise ValueError(f"{role} episode {index} has an invalid split")
            resolved_dir = str(Path(episode_dir).expanduser().resolve())
            if resolved_dir in seen_episode_dirs:
                raise ValueError(f"duplicate episode directory: {resolved_dir}")
            seen_episode_dirs.add(resolved_dir)
            episode["selection_source_role"] = role
            episodes.append(episode)

    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "selection_policy": selection_policy,
        "selection_sources": source_records,
        "episodes": episodes,
    }


def repartition_export_selection(
    source_path: Path,
    *,
    dataset_id: str,
    selection_policy: str,
    split_counts: Mapping[str, int],
    seed: int,
) -> dict[str, Any]:
    """Assign dataset splits deterministically without mutating source episodes.

    Raw episode metadata keeps the collection-time split.  Each output entry
    binds that value as ``source_split`` and carries the independently assigned
    dataset ``split``.  The SO-101 exporter validates the binding before
    applying a split override to its exported metadata copy.
    """
    if not dataset_id or not selection_policy:
        raise ValueError("dataset_id and selection_policy must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("split assignment seed must be a non-negative integer")
    if any(not isinstance(split, str) for split in split_counts):
        raise ValueError("split target names must be strings")
    unknown = set(split_counts).difference(VALID_SPLITS)
    if unknown:
        raise ValueError(f"invalid split targets: {sorted(unknown)}")
    raw_counts = {split: split_counts.get(split, 0) for split in SPLIT_ORDER}
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in raw_counts.values()
    ):
        raise ValueError("split targets must be non-negative integers")
    normalized_counts = dict(raw_counts)

    source = load_export_selection(source_path)
    if sum(normalized_counts.values()) != len(source["episodes"]):
        raise ValueError("split targets must sum to the source episode count")

    ranked = []
    for index, episode in enumerate(source["episodes"]):
        trial_id = str(episode.get("trial_id") or "")
        episode_dir = str(episode.get("episode_dir") or "")
        source_split = episode.get("split")
        if not trial_id or not episode_dir or source_split not in VALID_SPLITS:
            raise ValueError(f"source episode {index} has invalid identity or split")
        material = f"{seed}:{trial_id}:{Path(episode_dir).expanduser().resolve()}"
        ranked.append((hashlib.sha256(material.encode()).digest(), index))
    ranked.sort()

    assigned: dict[int, tuple[str, int]] = {}
    rank = 0
    for split in SPLIT_ORDER:
        for _ in range(normalized_counts[split]):
            _, source_index = ranked[rank]
            assigned[source_index] = (split, rank)
            rank += 1

    episodes = []
    for index, source_episode in enumerate(source["episodes"]):
        episode = deepcopy(source_episode)
        episode["source_split"] = source_episode["split"]
        episode["split"], episode["split_assignment_rank"] = assigned[index]
        episodes.append(episode)

    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "selection_policy": selection_policy,
        "selection_sources": [
            {
                "role": "repartition_source",
                "dataset_id": source["dataset_id"],
                "selection_sha256": file_sha256(source_path),
                "episode_count": len(source["episodes"]),
            }
        ],
        "split_assignment": {
            "algorithm": "sha256_rank_v1",
            "seed": seed,
            "targets": normalized_counts,
        },
        "episodes": episodes,
    }
