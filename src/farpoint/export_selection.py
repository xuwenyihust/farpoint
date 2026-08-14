"""Compose immutable Farpoint export selections without rewriting episodes."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SELECTION_SCHEMA_VERSION = "farpoint.export-selection.v1"
VALID_SPLITS = {"train", "validation", "test"}


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
