"""Helpers for building public, Dataset Viewer-compatible releases."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


VIEWER_ALLOWED_JSON = {Path("meta/info.json"), Path("meta/stats.json")}
PARQUET_EMPTY_STRUCT_FIELD = "__farpoint_empty__"
REQUIRED_VIEWER_PATHS = (
    Path("meta/info.json"),
    Path("meta/stats.json"),
    Path("meta/tasks.parquet"),
)


def _parquet_safe(value: Any, field_name: str | None = None) -> Any:
    """Convert metadata to one stable, lossless Arrow representation."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        if field_name == "seed" or (field_name and field_name.endswith("_seed")):
            return str(value)
        if not -(2**63) <= value <= 2**63 - 1:
            raise ValueError(
                f"integer field {field_name or '<unknown>'} exceeds the Parquet int64 range"
            )
        return value
    if isinstance(value, dict):
        if not value:
            return {PARQUET_EMPTY_STRUCT_FIELD: True}
        return {str(key): _parquet_safe(item, str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [_parquet_safe(item, field_name) for item in value]
    return value


def write_episode_metadata_parquet(source_dataset: Path, destination: Path) -> int:
    """Convert the private normalized JSONL sidecar to a public Parquet table."""
    source = source_dataset / "meta" / "episode_metadata.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"missing normalized metadata: {source}")
    records = [
        _parquet_safe(json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("episode metadata JSONL is empty")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - depends on release environment
        raise RuntimeError("pyarrow is required to build public metadata Parquet") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), destination)
    return len(records)


def prepare_viewer_package(source_dataset: Path, destination: Path) -> dict:
    """Copy a canonical export and remove non-standard JSON inputs for HF Viewer."""
    source_dataset = Path(source_dataset).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"viewer package already exists: {destination}")
    shutil.copytree(source_dataset, destination)
    metadata_count = write_episode_metadata_parquet(
        source_dataset, destination / "meta" / "episode_metadata.parquet"
    )
    removed = []
    for path in sorted(destination.rglob("*.json")) + sorted(destination.rglob("*.jsonl")):
        relative = path.relative_to(destination)
        if relative not in VIEWER_ALLOWED_JSON:
            path.unlink()
            removed.append(str(relative))
    audit = audit_viewer_package(destination)
    if not audit["valid"]:
        raise ValueError("viewer package failed audit: " + "; ".join(audit["errors"]))
    return {"metadata_rows": metadata_count, "removed_files": removed, "audit": audit}


def audit_viewer_package(root: Path) -> dict:
    """Check the conservative file contract used by the HF Dataset Viewer."""
    root = Path(root).resolve()
    errors = []
    if not root.is_dir():
        return {"valid": False, "errors": [f"package does not exist: {root}"]}
    for relative in REQUIRED_VIEWER_PATHS:
        if not (root / relative).is_file():
            errors.append(f"missing required file: {relative}")
    if not list((root / "data").rglob("*.parquet")):
        errors.append("data/ must contain Parquet shards")
    if not list((root / "meta" / "episodes").rglob("*.parquet")):
        errors.append("meta/episodes/ must contain Parquet shards")
    if not list((root / "videos").rglob("*.mp4")):
        errors.append("videos/ must contain MP4 shards")
    for path in root.rglob("*.json"):
        if path.relative_to(root) not in VIEWER_ALLOWED_JSON:
            errors.append(f"non-standard JSON input: {path.relative_to(root)}")
    for path in root.rglob("*.jsonl"):
        errors.append(f"non-standard JSONL input: {path.relative_to(root)}")
    metadata = root / "meta" / "episode_metadata.parquet"
    if not metadata.is_file():
        errors.append("missing public episode metadata: meta/episode_metadata.parquet")
    return {"valid": not errors, "errors": errors}
