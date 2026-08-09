"""Generate Hugging Face Dataset Cards from exported dataset metadata."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CELL_PATTERN = re.compile(r"(?:^|_)r(?P<row>[0-9]{2})_c(?P<column>[0-9]{2})(?:_|$)")
SPLIT_ORDER = ("train", "validation", "test")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _episode_records(dataset_root: Path) -> list[dict[str, Any]]:
    path = dataset_root / "meta" / "episode_metadata.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing normalized episode metadata: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("episode metadata must contain JSON objects")
    return rows


def _dataset_sidecar(dataset_root: Path) -> dict[str, Any]:
    candidates = sorted((dataset_root / "meta").glob("farpoint_v*.json"))
    if len(candidates) != 1:
        raise ValueError("dataset must contain exactly one Farpoint sidecar")
    return _read_json(candidates[0])


def _yaml_list(values: Iterable[str]) -> str:
    return "\n".join(f"- {json.dumps(value, ensure_ascii=False)}" for value in values)


def _format_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_shape(definition: dict[str, Any]) -> str:
    dtype = str(definition.get("dtype", "unknown"))
    shape = definition.get("shape")
    if isinstance(shape, list):
        return f"`{dtype}[{', '.join(str(value) for value in shape)}]`"
    return f"`{dtype}`"


def _object_state(record: dict[str, Any]) -> dict[str, Any]:
    variation = record.get("variation") or {}
    resolved = variation.get("resolved") or {}
    entities = resolved.get("entities") or {}
    entity = entities.get("pick_object")
    if isinstance(entity, dict):
        return {
            "shape": entity.get("entity_type"),
            "dimensions_m": (entity.get("geometry") or {}).get("dimensions_m"),
            "position_m": (entity.get("pose") or {}).get("position_m"),
            "orientation_xyzw": (entity.get("pose") or {}).get("orientation_xyzw"),
            "rgba": (entity.get("appearance") or {}).get("rgba"),
            "mass_kg": (entity.get("physics") or {}).get("mass_kg"),
        }
    return {
        "shape": resolved.get("shape"),
        "dimensions_m": resolved.get("dimensions_m"),
        "position_m": resolved.get("position_m"),
        "orientation_xyzw": resolved.get("orientation_xyzw"),
        "rgba": resolved.get("rgba"),
        "mass_kg": resolved.get("mass_kg"),
    }


def _target_state(record: dict[str, Any]) -> dict[str, Any]:
    resolved = (record.get("variation") or {}).get("resolved") or {}
    entity = (resolved.get("entities") or {}).get("placement_target")
    if isinstance(entity, dict):
        return {
            "type": entity.get("entity_type"),
            "dimensions_m": (entity.get("geometry") or {}).get("dimensions_m"),
        }
    target = (record.get("scene") or {}).get("target") or {}
    return {
        "type": target.get("entity_type") or target.get("target_id"),
        "dimensions_m": target.get("dimensions_m"),
    }


def _yaw_degrees(quaternion: Any) -> float | None:
    if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
        return None
    x, y, z, w = (float(value) for value in quaternion)
    yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return round(yaw, 3)


def variation_coverage(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Summarize open-ended entity variation without a release-specific table."""
    objects = [_object_state(record) for record in records]
    rows: list[tuple[str, str]] = []
    cells = set()
    positions = []
    for record, obj in zip(records, objects):
        variation_id = str((record.get("variation") or {}).get("variation_id", ""))
        match = CELL_PATTERN.search(variation_id)
        if match:
            cells.add((int(match.group("row")), int(match.group("column"))))
        position = obj.get("position_m")
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            positions.append((float(position[0]), float(position[1])))
    if positions:
        bounds = (
            f"x = {_format_number(min(value[0] for value in positions))}–"
            f"{_format_number(max(value[0] for value in positions))} m; "
            f"y = {_format_number(min(value[1] for value in positions))}–"
            f"{_format_number(max(value[1] for value in positions))} m"
        )
        coverage = f"{len(cells)} stratified cells; {bounds}" if cells else bounds
        rows.append(("Object position", coverage))

    def unique(field: str) -> list[Any]:
        values = {
            json.dumps(obj.get(field), sort_keys=True)
            for obj in objects
            if obj.get(field) is not None
        }
        return [json.loads(value) for value in sorted(values)]

    shapes = unique("shape")
    if shapes:
        rows.append(("Object type", ", ".join(str(value) for value in shapes)))
    dimensions = unique("dimensions_m")
    if dimensions:
        rows.append(
            (
                "Object dimensions",
                "; ".join(
                    " × ".join(_format_number(float(v)) for v in value) + " m"
                    for value in dimensions
                ),
            )
        )
    masses = unique("mass_kg")
    if masses:
        rows.append(
            ("Object mass", ", ".join(f"{_format_number(float(value))} kg" for value in masses))
        )
    yaws = sorted(
        {
            value
            for obj in objects
            if (value := _yaw_degrees(obj.get("orientation_xyzw"))) is not None
        }
    )
    if yaws:
        rows.append(("Object yaw", ", ".join(f"{_format_number(value)}°" for value in yaws)))
    colors = unique("rgba")
    if colors:
        rows.append(("Object appearance", f"{len(colors)} RGBA profiles"))
    targets = [_target_state(record) for record in records]
    target_values = {
        (str(target.get("type")), json.dumps(target.get("dimensions_m")))
        for target in targets
        if target.get("type")
    }
    if target_values:
        descriptions = []
        for target_type, dimensions_json in sorted(target_values):
            dimensions = json.loads(dimensions_json)
            suffix = ""
            if isinstance(dimensions, list):
                suffix = ", " + " × ".join(_format_number(float(v)) for v in dimensions) + " m"
            descriptions.append(target_type + suffix)
        rows.append(("Placement target", "; ".join(descriptions)))
    return rows


def generate_dataset_card(dataset_root: str | Path, spec: dict[str, Any]) -> str:
    """Render one deterministic Hub README from the exported release itself."""
    root = Path(dataset_root)
    info = _read_json(root / "meta" / "info.json")
    sidecar = _dataset_sidecar(root)
    records = _episode_records(root)
    card = spec.get("card") or {}
    if spec.get("dataset_card_mode") != "generated":
        raise ValueError("generate_dataset_card requires dataset_card_mode='generated'")
    name = str(card.get("pretty_name") or spec["dataset_id"])
    license_id = str(card.get("license") or "other")
    tags = [str(value) for value in card.get("tags") or []]
    features = info.get("features") or {}
    total_episodes = int(info.get("total_episodes", len(records)))
    total_frames = int(info.get("total_frames", 0))
    recording = sidecar.get("recording") or {}
    fps = float(info.get("fps", recording.get("fps", 0)))
    duration = total_frames / fps if fps > 0 else 0.0
    splits = sidecar.get("splits") or Counter(
        (record.get("identity") or {}).get("split") for record in records
    )
    lines = [
        "---",
        f"pretty_name: {json.dumps(name, ensure_ascii=False)}",
        f"license: {json.dumps(license_id, ensure_ascii=False)}",
        "library_name: lerobot",
        "task_categories:",
        "- robotics",
        "configs:",
        "- config_name: default",
        "  default: true",
        "  data_files:",
        "  - split: train",
        '    path: "data/**/*.parquet"',
        "- config_name: episode_metadata",
        "  data_files:",
        "  - split: train",
        '    path: "meta/episode_metadata.parquet"',
        "tags:",
        _yaml_list(tags),
        "---",
        "",
        f"# {name}",
        "",
        str(card.get("description") or "A Farpoint robot-learning dataset."),
        "",
        f"- Dataset: [{spec['hf_repo_id']}](https://huggingface.co/datasets/{spec['hf_repo_id']})",
        f"- Dataset version: `{spec['dataset_tag']}`",
        "",
        "## Dataset summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Episodes** | {total_episodes:,} successful demonstrations |",
        f"| **Frames** | {total_frames:,} at {_format_number(fps)} Hz ({duration / 60:.1f} min) |",
        f"| **Robot** | {(sidecar.get('robot') or {}).get('name', 'unknown')} |",
        f"| **Simulator** | {(sidecar.get('simulation') or {}).get('simulator', 'unknown')} + {(sidecar.get('simulation') or {}).get('physics', 'unknown')} |",
        f"| **Cameras** | {', '.join(recording.get('cameras') or [])} |",
        "",
        "### Logical episode splits",
        "",
        "| Split | Episodes | Share |",
        "|---|---:|---:|",
    ]
    for split in SPLIT_ORDER:
        count = int(splits.get(split, 0))
        share = 100.0 * count / total_episodes if total_episodes else 0.0
        lines.append(f"| {split.title()} | {count} | {share:.1f}% |")
    lines.extend(["", "### Variation coverage", "", "| Variation axis | Coverage |", "|---|---|"])
    lines.extend(f"| {axis} | {coverage} |" for axis, coverage in variation_coverage(records))
    lines.extend(["", "### Policy features", "", "| Feature | Shape / format |", "|---|---|"])
    policy_features = [
        name
        for name in features
        if name in {"observation.state", "action"} or name.startswith("observation.images.")
    ]
    for feature_name in policy_features:
        lines.append(f"| `{feature_name}` | {_format_shape(features[feature_name])} |")
    if card.get("intended_use"):
        lines.extend(["", "## Intended use", "", str(card["intended_use"])])
    if card.get("source_url"):
        lines.extend(
            ["", "## Source", "", f"Generated by [{card['source_url']}]({card['source_url']})."]
        )
    return "\n".join(lines).rstrip() + "\n"
