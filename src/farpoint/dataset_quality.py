"""Generate version-bound, precomputed quality reports for LeRobot datasets."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPORT_SCHEMA_VERSION = "farpoint.dataset-quality-report.v1"
DEFAULT_VISUAL_FRACTIONS = (0.15, 0.4, 0.65, 0.88)
DEFAULT_IDLE_DELTA_THRESHOLD = 0.05
JOINT_LIMIT_MARGIN = 1.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_data_dependencies():
    try:
        import av
        import numpy as np
        import pyarrow.parquet as pq
        from PIL import Image
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError(
            "dataset quality reports require the Farpoint data dependencies"
        ) from error
    return av, np, pq, Image


def quaternion_yaw_degrees(orientation_xyzw: Iterable[float]) -> float:
    x, y, z, w = (float(value) for value in orientation_xyzw)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    value = math.degrees(math.atan2(siny_cosp, cosy_cosp)) % 360.0
    if math.isclose(value, 360.0, abs_tol=1e-6):
        value = 0.0
    return round(value, 3)


def color_label(rgba: Iterable[float]) -> str:
    values = [float(value) for value in rgba]
    if len(values) < 3:
        return "unknown"
    dominant = max(range(3), key=lambda index: values[index])
    return ("red", "green", "blue")[dominant]


def grid_cell(variation_id: str) -> tuple[int | None, int | None]:
    match = re.search(r"(?:^|_)r(\d{2})_c(\d{2})(?:_|$)", variation_id)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def variation_record(metadata: dict[str, Any]) -> dict[str, Any]:
    identity = metadata.get("identity") or {}
    variation = metadata.get("variation") or {}
    resolved = variation.get("resolved") or {}
    position = resolved.get("position_m") or [None, None, None]
    dimensions = resolved.get("dimensions_m") or [None, None, None]
    orientation = resolved.get("orientation_xyzw") or [0.0, 0.0, 0.0, 1.0]
    rgba = resolved.get("rgba") or [0.0, 0.0, 0.0, 1.0]
    row, column = grid_cell(str(variation.get("variation_id") or ""))
    return {
        "episode_index": int(identity["dataset_episode_index"]),
        "episode_id": str(identity["episode_id"]),
        "variation_id": str(variation.get("variation_id") or ""),
        "split": str(identity.get("split") or variation.get("split") or "unknown"),
        "row": row,
        "column": column,
        "x_m": round(float(position[0]), 6),
        "y_m": round(float(position[1]), 6),
        "yaw_deg": quaternion_yaw_degrees(orientation),
        "mass_kg": round(float(resolved["mass_kg"]), 5),
        "size_m": round(float(dimensions[0]), 5),
        "color": color_label(rgba),
        "shape": str(resolved.get("shape") or "unknown"),
    }


def _string_key(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def count_axis(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(_string_key(record[field]) for record in records)
    return dict(sorted(counts.items()))


def select_representative_episodes(
    records: list[dict[str, Any]], count: int
) -> list[int]:
    """Select deterministic samples spanning categorical axes and XY space."""
    if count <= 0 or not records:
        return []
    count = min(count, len(records))
    numeric_fields = ("x_m", "y_m")
    categorical_fields = ("yaw_deg", "mass_kg", "size_m", "color", "split")
    mins = {field: min(float(row[field]) for row in records) for field in numeric_fields}
    maxs = {field: max(float(row[field]) for row in records) for field in numeric_fields}

    def distance(left: dict[str, Any], right: dict[str, Any]) -> float:
        value = 0.0
        for field in numeric_fields:
            span = maxs[field] - mins[field]
            if span:
                value += ((float(left[field]) - float(right[field])) / span) ** 2
        value += sum(left[field] != right[field] for field in categorical_fields)
        return math.sqrt(value)

    remaining = sorted(records, key=lambda row: row["episode_index"])
    selected = [remaining.pop(0)]
    while remaining and len(selected) < count:
        covered = {
            field: {row[field] for row in selected} for field in categorical_fields
        }

        def score(row: dict[str, Any]) -> tuple[float, int]:
            novelty = sum(row[field] not in covered[field] for field in categorical_fields)
            separation = min(distance(row, chosen) for chosen in selected)
            return novelty * 10.0 + separation, -int(row["episode_index"])

        choice = max(remaining, key=score)
        selected.append(choice)
        remaining.remove(choice)
    return [int(row["episode_index"]) for row in selected]


def _quantiles(np, values, probabilities=(0.0, 0.01, 0.5, 0.95, 0.99, 1.0)):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {f"q{int(probability * 100):02d}": None for probability in probabilities}
    result = np.quantile(array, probabilities)
    return {
        f"q{int(probability * 100):02d}": round(float(item), 6)
        for probability, item in zip(probabilities, result, strict=True)
    }


def _joint_statistics(np, values, joint_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for index, name in enumerate(joint_names):
        column = values[:, index]
        rows.append(
            {
                "joint": name,
                **_quantiles(np, column),
                "mean": round(float(column.mean()), 6),
            }
        )
    return rows


def _action_fingerprint(np, actions) -> str:
    sample_indices = np.linspace(0, len(actions) - 1, 64).round().astype(int)
    payload = np.round(actions[sample_indices], 3).astype("<f4").tobytes()
    return hashlib.sha256(payload).hexdigest()


def _build_action_quality(
    np,
    data_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    joint_names: list[str],
    fps: int,
    idle_threshold: float,
) -> dict[str, Any]:
    states = np.asarray([row["observation.state"] for row in data_rows], dtype=np.float64)
    actions = np.asarray([row["action"] for row in data_rows], dtype=np.float64)
    all_deltas = []
    all_velocity = []
    all_acceleration = []
    all_jerk = []
    episode_metrics = []
    fingerprints: defaultdict[str, list[int]] = defaultdict(list)

    for episode in episode_rows:
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        episode_actions = actions[start:stop]
        delta = np.diff(episode_actions, axis=0)
        velocity = delta * fps
        acceleration = np.diff(velocity, axis=0) * fps
        jerk = np.diff(acceleration, axis=0) * fps
        all_deltas.append(delta)
        all_velocity.append(velocity)
        all_acceleration.append(acceleration)
        all_jerk.append(jerk)
        max_delta = np.max(np.abs(delta), axis=1) if len(delta) else np.asarray([])
        idle_ratio = float(np.mean(max_delta <= idle_threshold)) if len(max_delta) else 1.0
        index = int(episode["episode_index"])
        fingerprint = _action_fingerprint(np, episode_actions)
        fingerprints[fingerprint].append(index)
        episode_metrics.append(
            {
                "episode_index": index,
                "frames": int(episode["length"]),
                "duration_s": round(int(episode["length"]) / fps, 3),
                "idle_transition_ratio": round(idle_ratio, 6),
                "max_action_step": round(float(max_delta.max()), 6) if len(max_delta) else 0.0,
            }
        )

    deltas = np.concatenate(all_deltas, axis=0)
    velocity = np.concatenate(all_velocity, axis=0)
    acceleration = np.concatenate(all_acceleration, axis=0)
    jerk = np.concatenate(all_jerk, axis=0)
    absolute_delta = np.max(np.abs(deltas), axis=1)
    tracking_error = np.max(np.abs(actions - states), axis=1)
    arm_saturated = np.any(np.abs(actions[:, :5]) >= 100.0 - JOINT_LIMIT_MARGIN, axis=1)
    gripper_saturated = (actions[:, 5] <= JOINT_LIMIT_MARGIN) | (
        actions[:, 5] >= 100.0 - JOINT_LIMIT_MARGIN
    )
    duplicate_groups = [indices for indices in fingerprints.values() if len(indices) > 1]
    return {
        "definitions": {
            "action_unit": "SO-101 calibrated export units",
            "idle_transition": f"max absolute 6D action delta <= {idle_threshold:g}",
            "joint_limit_margin": JOINT_LIMIT_MARGIN,
            "episode_fingerprint": "64-point action resample rounded to 0.001 then SHA256",
        },
        "episode_length_frames": _quantiles(np, [row["length"] for row in episode_rows]),
        "episode_duration_s": _quantiles(
            np, [int(row["length"]) / fps for row in episode_rows]
        ),
        "state_by_joint": _joint_statistics(np, states, joint_names),
        "action_by_joint": _joint_statistics(np, actions, joint_names),
        "max_action_step": _quantiles(np, absolute_delta),
        "max_velocity_per_s": _quantiles(np, np.max(np.abs(velocity), axis=1)),
        "max_acceleration_per_s2": _quantiles(
            np, np.max(np.abs(acceleration), axis=1)
        ),
        "max_jerk_per_s3": _quantiles(np, np.max(np.abs(jerk), axis=1)),
        "state_action_tracking_error": _quantiles(np, tracking_error),
        "idle_transition_ratio": round(float(np.mean(absolute_delta <= idle_threshold)), 6),
        "arm_limit_ratio": round(float(np.mean(arm_saturated)), 6),
        "gripper_limit_ratio": round(float(np.mean(gripper_saturated)), 6),
        "exact_resampled_action_duplicate_groups": duplicate_groups,
        "episode_metrics": episode_metrics,
    }


def _visual_target_frames(
    episode_rows: list[dict[str, Any]], selected: list[int], fps: int
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    targets: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    by_index = {int(row["episode_index"]): row for row in episode_rows}
    for episode_index in selected:
        row = by_index[episode_index]
        chunk_index = int(row["videos/observation.images.front/chunk_index"])
        file_index = int(row["videos/observation.images.front/file_index"])
        start = round(float(row["videos/observation.images.front/from_timestamp"]) * fps)
        length = int(row["length"])
        for fraction in DEFAULT_VISUAL_FRACTIONS:
            frame_index = start + round((length - 1) * fraction)
            targets[(chunk_index, file_index)].append(
                {
                    "video_frame_index": frame_index,
                    "episode_index": episode_index,
                    "fraction": fraction,
                }
            )
    return targets


def _decode_visual_quality(
    av,
    np,
    Image,
    root: Path,
    output: Path,
    episode_rows: list[dict[str, Any]],
    selected: list[int],
    fps: int,
    sample_stride: int,
) -> dict[str, Any]:
    assets = output / "assets" / "episodes"
    assets.mkdir(parents=True, exist_ok=True)
    targets = _visual_target_frames(episode_rows, selected, fps)
    captured: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    brightness = []
    contrast = []
    dark_rates = []
    bright_rates = []
    decoded_counts = {}
    resolutions = set()

    video_dir = root / "videos" / "observation.images.front"
    for path in sorted(video_dir.rglob("file-*.mp4")):
        chunk_index = int(path.parent.name.split("-")[-1])
        file_index = int(path.stem.split("-")[-1])
        wanted = {
            item["video_frame_index"]: item
            for item in targets.get((chunk_index, file_index), [])
        }
        decoded = 0
        with av.open(str(path)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                decoded += 1
                resolutions.add((int(frame.width), int(frame.height)))
                if frame_index % sample_stride == 0:
                    image = frame.to_image().resize((160, 120))
                    array = np.asarray(image, dtype=np.float32) / 255.0
                    luma = (
                        array[:, :, 0] * 0.2126
                        + array[:, :, 1] * 0.7152
                        + array[:, :, 2] * 0.0722
                    )
                    brightness.append(float(luma.mean()))
                    contrast.append(float(luma.std()))
                    dark_rates.append(float(np.mean(luma <= 0.03)))
                    bright_rates.append(float(np.mean(luma >= 0.97)))
                target = wanted.get(frame_index)
                if target:
                    episode_index = int(target["episode_index"])
                    fraction = float(target["fraction"])
                    relative = Path("assets") / "episodes" / (
                        f"episode-{episode_index:03d}-{round(fraction * 100):02d}.jpg"
                    )
                    frame.to_image().resize((480, 360), Image.Resampling.LANCZOS).save(
                        output / relative, quality=84, optimize=True
                    )
                    captured[episode_index].append(
                        {
                            "fraction": fraction,
                            "label": f"{round(fraction * 100)}%",
                            "path": str(relative),
                        }
                    )
        decoded_counts[str(path.relative_to(video_dir))] = decoded

    samples = []
    for episode_index in selected:
        samples.append(
            {
                "episode_index": episode_index,
                "frames": sorted(captured[episode_index], key=lambda row: row["fraction"]),
            }
        )
    return {
        "definitions": {
            "sample_stride_frames": sample_stride,
            "timeline_fractions": list(DEFAULT_VISUAL_FRACTIONS),
            "dark_pixel_threshold": 0.03,
            "bright_pixel_threshold": 0.97,
        },
        "decoded_frames_by_file": decoded_counts,
        "decoded_frames": sum(decoded_counts.values()),
        "resolutions": [list(value) for value in sorted(resolutions)],
        "sampled_frames": len(brightness),
        "brightness": _quantiles(np, brightness),
        "contrast": _quantiles(np, contrast),
        "dark_pixel_ratio": _quantiles(np, dark_rates),
        "bright_pixel_ratio": _quantiles(np, bright_rates),
        "samples": samples,
    }


def _check(checks: list[dict[str, Any]], check_id: str, passed: bool, detail: str) -> None:
    checks.append(
        {"id": check_id, "status": "PASS" if passed else "FAIL", "detail": detail}
    )


def _read_external_evidence(
    path: Path | None,
    *,
    expected_repo: str,
    expected_commit: str | None = None,
    expected_branch: str | None = None,
) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    actual_repo = value.get("repo_id") or value.get("dataset")
    actual_commit = value.get("hub_head_commit")
    actual_branch = value.get("branch")
    identity_valid = actual_repo == expected_repo
    if expected_commit is not None:
        identity_valid &= actual_commit == expected_commit
    if expected_branch is not None:
        identity_valid &= actual_branch == expected_branch
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "valid": bool(value.get("valid")) and identity_valid,
        "repo": actual_repo,
        "commit": actual_commit,
        "branch": actual_branch,
    }


def generate_quality_report(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    dataset_repo: str,
    dataset_tag: str,
    resolved_dataset_commit: str,
    generator_commit: str,
    tag_validation_path: str | Path | None = None,
    viewer_validation_path: str | Path | None = None,
    visual_episode_count: int = 12,
    visual_sample_stride: int = 30,
    idle_delta_threshold: float = DEFAULT_IDLE_DELTA_THRESHOLD,
) -> dict[str, Any]:
    """Generate one immutable, static-site-ready dataset quality report."""
    av, np, pq, Image = _require_data_dependencies()
    root = Path(dataset_root).resolve()
    output = Path(output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    metadata_rows = pq.read_table(root / "meta" / "episode_metadata.parquet").to_pylist()
    episode_rows = []
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        episode_rows.extend(pq.read_table(path).to_pylist())
    data_rows = []
    for path in sorted((root / "data").rglob("*.parquet")):
        data_rows.extend(pq.read_table(path).to_pylist())
    records = sorted(
        (variation_record(row) for row in metadata_rows), key=lambda row: row["episode_index"]
    )
    record_by_index = {row["episode_index"]: row for row in records}
    episode_by_index = {int(row["episode_index"]): row for row in episode_rows}
    fps = int(info["fps"])
    joint_names = list(info["features"]["action"]["names"])

    action_quality = _build_action_quality(
        np, data_rows, episode_rows, joint_names, fps, idle_delta_threshold
    )
    for row in action_quality["episode_metrics"]:
        row.update(
            {
                key: record_by_index[row["episode_index"]][key]
                for key in ("split", "row", "column", "yaw_deg", "mass_kg", "size_m", "color")
            }
        )

    selected = select_representative_episodes(records, visual_episode_count)
    visual_quality = _decode_visual_quality(
        av,
        np,
        Image,
        root,
        output,
        episode_rows,
        selected,
        fps,
        visual_sample_stride,
    )
    for sample in visual_quality["samples"]:
        sample["variation"] = record_by_index[sample["episode_index"]]

    split_episode_counts = count_axis(records, "split")
    split_frame_counts = Counter()
    for record in records:
        split_frame_counts[record["split"]] += int(
            episode_by_index[record["episode_index"]]["length"]
        )
    position_counts = Counter(
        f"r{record['row']:02d}_c{record['column']:02d}"
        for record in records
        if record["row"] is not None and record["column"] is not None
    )
    combinations = Counter(
        (
            record["row"],
            record["column"],
            record["yaw_deg"],
            record["mass_kg"],
            record["size_m"],
            record["color"],
        )
        for record in records
    )

    checks = []
    total_episodes = int(info["total_episodes"])
    total_frames = int(info["total_frames"])
    _check(
        checks,
        "episode_count",
        len(records) == total_episodes,
        f"{len(records):,} / {total_episodes:,}",
    )
    _check(
        checks,
        "frame_count",
        len(data_rows) == total_frames,
        f"{len(data_rows):,} / {total_frames:,}",
    )
    _check(
        checks,
        "episode_frame_sum",
        sum(int(row["length"]) for row in episode_rows) == total_frames,
        f"{sum(int(row['length']) for row in episode_rows):,} frames",
    )
    outcomes_valid = all(
        bool((row.get("outcome") or {}).get("success"))
        and bool((row.get("outcome") or {}).get("dataset_valid"))
        for row in metadata_rows
    )
    _check(checks, "successful_valid_episodes", outcomes_valid, f"{len(metadata_rows):,} checked")
    finite = all(
        all(math.isfinite(float(value)) for value in row["action"])
        and all(math.isfinite(float(value)) for value in row["observation.state"])
        for row in data_rows
    )
    _check(checks, "finite_state_action", finite, f"{len(data_rows):,} rows checked")
    dimensions_valid = all(
        len(row["action"]) == len(joint_names)
        and len(row["observation.state"]) == len(joint_names)
        for row in data_rows
    )
    _check(checks, "feature_dimensions", dimensions_valid, f"state/action [{len(joint_names)}]")
    timestamp_valid = True
    terminal_valid = True
    for episode in episode_rows:
        start, stop = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        timestamps = [float(row["timestamp"]) for row in data_rows[start:stop]]
        timestamp_valid &= all(right > left for left, right in zip(timestamps, timestamps[1:]))
        terminal_valid &= sum(bool(row["next.done"]) for row in data_rows[start:stop]) == 1
        terminal_valid &= bool(data_rows[stop - 1]["next.done"])
    _check(checks, "monotonic_timestamps", timestamp_valid, f"{total_episodes} episodes checked")
    _check(checks, "episode_terminals", terminal_valid, "exactly one final done per episode")
    decoded_frames = int(visual_quality["decoded_frames"])
    _check(
        checks,
        "video_decode",
        decoded_frames == total_frames,
        f"{decoded_frames:,} / {total_frames:,}",
    )
    _check(
        checks,
        "video_resolution",
        visual_quality["resolutions"] == [[640, 480]],
        str(visual_quality["resolutions"]),
    )
    _check(
        checks,
        "logical_splits",
        split_episode_counts == {"test": 18, "train": 128, "validation": 14},
        ", ".join(f"{key}={value}" for key, value in split_episode_counts.items()),
    )

    tag_evidence = _read_external_evidence(
        Path(tag_validation_path) if tag_validation_path else None,
        expected_repo=dataset_repo,
        expected_commit=resolved_dataset_commit,
        expected_branch=dataset_tag,
    )
    viewer_evidence = _read_external_evidence(
        Path(viewer_validation_path) if viewer_validation_path else None,
        expected_repo=dataset_repo,
    )
    if tag_evidence:
        _check(checks, "published_tag_validation", tag_evidence["valid"], tag_evidence["sha256"])
    if viewer_evidence:
        _check(checks, "dataset_viewer", viewer_evidence["valid"], viewer_evidence["sha256"])

    source_paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"README.md", ".gitattributes"}
    ]
    source_hashes = {str(path.relative_to(root)): file_sha256(path) for path in source_paths}
    integrity_status = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "identity": {
            "dataset_repo": dataset_repo,
            "dataset_tag": dataset_tag,
            "resolved_dataset_commit": resolved_dataset_commit,
            "generator_commit": generator_commit,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_hashes": source_hashes,
            "source_hashes_sha256": canonical_sha256(source_hashes),
        },
        "overview": {
            "title": "Farpoint SO-101 Dataset",
            "episodes": total_episodes,
            "frames": total_frames,
            "fps": fps,
            "duration_hours": round(total_frames / fps / 3600.0, 3),
            "robot": info["robot_type"],
            "task": metadata_rows[0]["task"]["instruction"],
            "camera_features": [
                key for key in info["features"] if key.startswith("observation.images.")
            ],
            "state_dimension": len(joint_names),
            "action_dimension": len(joint_names),
            "split_episodes": split_episode_counts,
            "split_frames": dict(sorted(split_frame_counts.items())),
        },
        "integrity": {
            "status": integrity_status,
            "checks": checks,
            "external_evidence": {"tag": tag_evidence, "viewer": viewer_evidence},
        },
        "variation_coverage": {
            "episodes": records,
            "axis_counts": {
                "split": split_episode_counts,
                "yaw_deg": count_axis(records, "yaw_deg"),
                "mass_kg": count_axis(records, "mass_kg"),
                "size_m": count_axis(records, "size_m"),
                "color": count_axis(records, "color"),
                "shape": count_axis(records, "shape"),
            },
            "position_counts": dict(sorted(position_counts.items())),
            "position_cells_covered": len(position_counts),
            "exact_combinations": len(combinations),
            "combination_repetition": _quantiles(np, list(combinations.values())),
            "position_bounds_m": {
                "x": [min(row["x_m"] for row in records), max(row["x_m"] for row in records)],
                "y": [min(row["y_m"] for row in records), max(row["y_m"] for row in records)],
            },
        },
        "episode_action_quality": action_quality,
        "visual_quality": visual_quality,
    }
    report["report_sha256"] = canonical_sha256(report)
    from farpoint.contracts import validate_contract

    contract_errors = validate_contract(report)
    if contract_errors:
        raise ValueError("quality report contract failed: " + "; ".join(contract_errors))
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
