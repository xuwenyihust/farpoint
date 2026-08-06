"""Evidence-oriented summaries for SO-101 simulation episodes."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from collections import Counter, defaultdict
from pathlib import Path


CAMERA_PATH_KEYS = {
    "rgb_path": "front",
    "wrist_rgb_path": "wrist",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            rows.append(json.loads(raw_line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"episode has no observations: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rgb_png_info(path: Path) -> tuple[int, int, str]:
    """Validate an 8-bit RGB PNG using only the Python standard library."""
    payload = path.read_bytes()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not a PNG file")
    offset = 8
    width = height = bit_depth = color_type = None
    compressed = bytearray()
    saw_end = False
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(payload):
            raise ValueError("truncated PNG chunk")
        chunk_data = payload[data_start:data_end]
        expected_crc = struct.unpack(">I", payload[data_end:crc_end])[0]
        if zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF != expected_crc:
            raise ValueError("invalid PNG chunk CRC")
        if chunk_type == b"IHDR":
            if length != 13:
                raise ValueError("invalid PNG IHDR")
            width, height, bit_depth, color_type = struct.unpack(
                ">IIBB", chunk_data[:10]
            )
        elif chunk_type == b"IDAT":
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            saw_end = True
            break
        offset = crc_end
    if not saw_end or not compressed or None in {width, height, bit_depth, color_type}:
        raise ValueError("incomplete PNG structure")
    if bit_depth != 8 or color_type != 2:
        raise ValueError("PNG must be 8-bit RGB")
    decoded = zlib.decompress(bytes(compressed))
    expected_bytes = int(height) * (1 + 3 * int(width))
    if len(decoded) != expected_bytes:
        raise ValueError("PNG scanline size mismatch")
    return int(width), int(height), "RGB"


def _phase_ranges(rows: list[dict]) -> list[dict]:
    ranges = []
    start = 0
    phase = str(rows[0].get("phase", "unknown"))
    for index, row in enumerate(rows[1:], start=1):
        next_phase = str(row.get("phase", "unknown"))
        if next_phase == phase:
            continue
        ranges.append(
            {
                "phase": phase,
                "start_frame": int(rows[start].get("frame", start)),
                "end_frame": int(rows[index - 1].get("frame", index - 1)),
                "frame_count": index - start,
            }
        )
        start = index
        phase = next_phase
    ranges.append(
        {
            "phase": phase,
            "start_frame": int(rows[start].get("frame", start)),
            "end_frame": int(rows[-1].get("frame", len(rows) - 1)),
            "frame_count": len(rows) - start,
        }
    )
    return ranges


def _named_phase_ranges(rows: list[dict], key: str) -> list[dict]:
    normalized = [
        {**row, "phase": str(row.get(key, "unknown"))} for row in rows
    ]
    return _phase_ranges(normalized)


def classify_so101_failure(reason: str | None, category: str | None = None) -> str:
    """Map detailed runner/oracle reasons to stable experiment-report buckets."""
    value = (reason or "").lower()
    category_value = (category or "").lower()
    if category_value == "runner":
        return "runner_error"
    if "contact_force_limit" in value:
        return "contact_force_limit"
    if "bilateral_contact_lost" in value:
        return "bilateral_contact_lost"
    if "collision" in value:
        return "collision"
    if "drop" in value:
        return "object_dropped"
    if "timeout" in value:
        return "phase_timeout"
    if not reason:
        return "unspecified_failure"
    return "other_oracle_failure"


def _object_pose(row: dict) -> list[float]:
    try:
        pose = row["truth"]["object_root_pose_xyzw"]
    except KeyError as exc:
        raise ValueError("observation is missing truth.object_root_pose_xyzw") from exc
    if len(pose) != 7:
        raise ValueError("object_root_pose_xyzw must contain seven values")
    return [float(value) for value in pose]


def _proof_lift_tracking(rows: list[dict]) -> dict | None:
    verify_rows = [row for row in rows if row.get("phase") == "verify_contact"]
    targets = [
        float(row.get("truth", {}).get("proof_lift_target_m"))
        for row in verify_rows
        if row.get("truth", {}).get("proof_lift_target_m") is not None
    ]
    actual = [
        float(row.get("grasp_evidence", {}).get("proof_lift_m"))
        for row in verify_rows
        if row.get("grasp_evidence", {}).get("proof_lift_m") is not None
    ]
    gripper_z = [
        float(row["truth"]["gripper_link_pose_xyzw"][2])
        for row in verify_rows
        if len(row.get("truth", {}).get("gripper_link_pose_xyzw", [])) == 7
    ]
    if not targets and not actual and not gripper_z:
        return None
    target_max = max(targets, default=0.0)
    actual_max = max(actual, default=0.0)
    return {
        "target_max_m": target_max,
        "actual_max_m": actual_max,
        "target_minus_actual_m": target_max - actual_max,
        "gripper_vertical_displacement_m": (
            max(gripper_z) - gripper_z[0] if gripper_z else None
        ),
        "verify_frame_count": len(verify_rows),
    }


def _contact_alignment_tracking(rows: list[dict]) -> dict:
    unilateral_rows = []
    recenter_norms = []
    for row in rows:
        correction = row.get("truth", {}).get("grasp_xy_correction_m")
        if isinstance(correction, list) and len(correction) == 2:
            recenter_norms.append(math.hypot(float(correction[0]), float(correction[1])))
        if row.get("phase") != "close":
            continue
        forces = row.get("contact_forces_newtons", {})
        left = float(forces.get("left_finger", 0.0))
        right = float(forces.get("right_finger", 0.0))
        if min(left, right) < 0.5 <= max(left, right):
            unilateral_rows.append((left, right))
    return {
        "close_unilateral_contact_frames": len(unilateral_rows),
        "close_peak_unilateral_force_n": max(
            (max(forces) for forces in unilateral_rows), default=0.0
        ),
        "max_grasp_xy_recenter_m": max(recenter_norms, default=0.0),
    }


def _transport_lift_tracking(rows: list[dict]) -> dict | None:
    lift_rows = [row for row in rows if row.get("phase") == "lift"]
    targets = [
        float(row.get("truth", {}).get("transport_lift_target_m"))
        for row in lift_rows
        if row.get("truth", {}).get("transport_lift_target_m") is not None
    ]
    actual = [
        float(row.get("truth", {}).get("transport_lift_actual_m"))
        for row in lift_rows
        if row.get("truth", {}).get("transport_lift_actual_m") is not None
    ]
    if not targets and not actual:
        return None
    target_max = max(targets, default=0.0)
    actual_max = max(actual, default=0.0)
    return {
        "target_max_m": target_max,
        "actual_max_m": actual_max,
        "target_minus_actual_m": target_max - actual_max,
        "lift_frame_count": len(lift_rows),
    }


def _camera_frame_integrity(root: Path, rows: list[dict]) -> dict[str, dict]:
    result = {}
    root_resolved = root.resolve()
    for path_key, camera in CAMERA_PATH_KEYS.items():
        references = [row.get(path_key) for row in rows if row.get(path_key)]
        if not references:
            continue
        existing = 0
        decodable = 0
        resolutions = set()
        modes = set()
        unsafe_paths = []
        for reference in references:
            path = root / str(reference)
            try:
                resolved = path.resolve()
                if not resolved.is_relative_to(root_resolved):
                    unsafe_paths.append(str(reference))
                    continue
                if not resolved.is_file():
                    continue
                existing += 1
                width, height, mode = _read_rgb_png_info(resolved)
                resolutions.add((width, height))
                modes.add(mode)
                decodable += 1
            except (OSError, ValueError):
                continue
        result[camera] = {
            "referenced_frames": len(references),
            "existing_frames": existing,
            "decodable_frames": decodable,
            "resolutions": [list(value) for value in sorted(resolutions)],
            "modes": sorted(modes),
            "unsafe_paths": sorted(unsafe_paths),
        }
    return result


def summarize_so101_episode(episode_dir, *, verify_images: bool = False) -> dict:
    """Summarize one raw episode without treating sensor contact as success."""
    root = Path(episode_dir)
    observations_path = root / "observations.jsonl"
    metadata_path = root / "metadata.json"
    metrics_path = root / "metrics.json"
    for required in (observations_path, metadata_path, metrics_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    rows = _read_jsonl(observations_path)
    metadata = _read_json(metadata_path)
    metrics = _read_json(metrics_path)
    initial_pose = _object_pose(rows[0])
    final_pose = _object_pose(rows[-1])
    object_z = [_object_pose(row)[2] for row in rows]
    timestamps = [float(row.get("timestamp_seconds", 0.0)) for row in rows]
    camera_frame_counts = {
        camera: sum(bool(row.get(key)) for row in rows)
        for key, camera in CAMERA_PATH_KEYS.items()
    }
    camera_frame_counts = {
        camera: count for camera, count in camera_frame_counts.items() if count
    }
    camera_frame_integrity = (
        _camera_frame_integrity(root, rows) if verify_images else None
    )
    phase_ranges = _phase_ranges(rows)
    grasp_phase_ranges = _named_phase_ranges(rows, "grasp_phase")
    action_dimensions = sorted(
        {
            len(row.get("action_joint_positions", []))
            for row in rows
            if row.get("action_joint_positions") is not None
        }
    )
    state_dimensions = sorted(
        {
            len(row.get("joint_positions", []))
            for row in rows
            if row.get("joint_positions") is not None
        }
    )
    contact_alignment = _contact_alignment_tracking(rows)
    return {
        "episode_dir": str(root),
        "variation_id": metadata.get("variation", {}).get("variation_id"),
        "success": bool(metrics.get("success", False)),
        "dataset_valid": bool(metrics.get("dataset_valid", False)),
        "failure_category": metrics.get("failure_category"),
        "failure_reason": metrics.get("failure_reason"),
        "observation_count": len(rows),
        "terminal_phase": str(rows[-1].get("phase", "unknown")),
        "terminal_grasp_phase": str(rows[-1].get("grasp_phase", "unknown")),
        "phase_ranges": phase_ranges,
        "grasp_phase_ranges": grasp_phase_ranges,
        "initial_object_pose_xyzw": initial_pose,
        "final_object_pose_xyzw": final_pose,
        "max_object_z_m": max(object_z),
        "object_lift_above_initial_m": max(object_z) - initial_pose[2],
        "final_object_xy_displacement_m": [
            final_pose[0] - initial_pose[0],
            final_pose[1] - initial_pose[1],
        ],
        "cube_contact_frames": sum(
            bool(row.get("contact", {}).get("cube_contact")) for row in rows
        ),
        "bilateral_contact_frames": sum(
            bool(row.get("contact", {}).get("bilateral_cube_contact"))
            for row in rows
        ),
        "max_contact_force_n": {
            side: max(
                float(row.get("contact_forces_newtons", {}).get(side, 0.0))
                for row in rows
            )
            for side in ("left_finger", "right_finger")
        },
        "proof_lift_tracking": _proof_lift_tracking(rows),
        "transport_lift_tracking": _transport_lift_tracking(rows),
        **contact_alignment,
        "camera_frame_counts": camera_frame_counts,
        "camera_frame_integrity": camera_frame_integrity,
        "state_dimensions": state_dimensions,
        "action_dimensions": action_dimensions,
        "timestamps_strictly_increasing": all(
            later > earlier for earlier, later in zip(timestamps, timestamps[1:])
        ),
        "observations_sha256": _sha256(observations_path),
        "metadata_sha256": _sha256(metadata_path),
        "metrics_sha256": _sha256(metrics_path),
    }


def analyze_so101_episodes(episode_dirs, *, verify_images: bool = False) -> dict:
    """Build aggregate evidence and explicitly identify duplicate artifacts."""
    episodes = [
        summarize_so101_episode(path, verify_images=verify_images)
        for path in episode_dirs
    ]
    duplicate_index = defaultdict(list)
    for episode in episodes:
        duplicate_index[episode["observations_sha256"]].append(
            episode["episode_dir"]
        )
    duplicates = [
        {"observations_sha256": digest, "episode_dirs": paths}
        for digest, paths in sorted(duplicate_index.items())
        if len(paths) > 1
    ]
    failures = Counter(
        episode["failure_reason"] or f"terminal:{episode['terminal_phase']}"
        for episode in episodes
        if not episode["success"]
    )
    failure_classes = Counter(
        classify_so101_failure(
            episode["failure_reason"], episode["failure_category"]
        )
        for episode in episodes
        if not episode["success"]
    )
    return {
        "episode_count": len(episodes),
        "success_count": sum(episode["success"] for episode in episodes),
        "failure_count": sum(not episode["success"] for episode in episodes),
        "independent_observation_artifact_count": len(duplicate_index),
        "duplicate_observation_groups": duplicates,
        "failure_reason_counts": dict(sorted(failures.items())),
        "failure_class_counts": dict(sorted(failure_classes.items())),
        "episodes": episodes,
    }


def render_so101_analysis_markdown(analysis: dict) -> str:
    """Render a compact review report from :func:`analyze_so101_episodes`."""
    lines = [
        "# SO-101 episode evidence report",
        "",
        f"- Episodes: {analysis['episode_count']}",
        f"- Successful: {analysis['success_count']}",
        f"- Failed: {analysis['failure_count']}",
        "- Independent observation artifacts: "
        f"{analysis['independent_observation_artifact_count']}",
        "",
        "| Episode | Result | Frames | Terminal phase | Lift (m) | Proof target/actual (mm) | Transport target/actual (mm) | Max recenter (mm) | Bilateral frames | Cameras |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for episode in analysis["episodes"]:
        cameras = ", ".join(sorted(episode["camera_frame_counts"])) or "none"
        proof = episode.get("proof_lift_tracking")
        proof_text = (
            "n/a"
            if proof is None
            else f"{proof['target_max_m'] * 1000:.2f}/{proof['actual_max_m'] * 1000:.2f}"
        )
        transport = episode.get("transport_lift_tracking")
        transport_text = (
            "n/a"
            if transport is None
            else f"{transport['target_max_m'] * 1000:.2f}/{transport['actual_max_m'] * 1000:.2f}"
        )
        lines.append(
            "| "
            f"{Path(episode['episode_dir']).name} | "
            f"{'PASS' if episode['success'] else 'FAIL'} | "
            f"{episode['observation_count']} | {episode['terminal_phase']} | "
            f"{episode['object_lift_above_initial_m']:.6f} | "
            f"{proof_text} | "
            f"{transport_text} | "
            f"{episode['max_grasp_xy_recenter_m'] * 1000:.2f} | "
            f"{episode['bilateral_contact_frames']} | {cameras} |"
        )
    lines.extend(["", "## Duplicate artifact audit", ""])
    if analysis["duplicate_observation_groups"]:
        for group in analysis["duplicate_observation_groups"]:
            lines.append(
                f"- `{group['observations_sha256']}`: "
                + ", ".join(Path(path).name for path in group["episode_dirs"])
            )
        lines.extend(
            [
                "",
                "Duplicate observation artifacts must not be counted as independent "
                "stability trials unless separate execution provenance proves it.",
            ]
        )
    else:
        lines.append("No duplicate observation artifacts detected.")
    return "\n".join(lines) + "\n"
