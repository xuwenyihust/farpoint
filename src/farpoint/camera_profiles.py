"""Versioned camera profiles and resolved episode camera records."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


CAMERA_PROFILE_SCHEMA = "farpoint.camera-profile.v1"
V010_CAMERA_IDS = ("front", "wrist")
V010_CAMERA_FEATURES = {
    "front": "observation.images.front",
    "wrist": "observation.images.wrist",
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite_numbers(values: Any, length: int) -> bool:
    return (
        isinstance(values, list)
        and len(values) == length
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
    )


def validate_camera_profile(profile: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if profile.get("schema_version") != CAMERA_PROFILE_SCHEMA:
        errors.append(f"camera profile schema_version must be {CAMERA_PROFILE_SCHEMA}")
    if not isinstance(profile.get("profile_id"), str) or not profile["profile_id"]:
        errors.append("camera profile_id must be non-empty")
    if profile.get("recording_hz") != 30:
        errors.append("v0.1.0 camera profile recording_hz must be 30")
    if profile.get("timestamp_source") != "simulation_control_tick":
        errors.append("camera profile timestamp_source must be simulation_control_tick")

    cameras = profile.get("cameras")
    if not isinstance(cameras, list):
        return errors + ["camera profile cameras must be an array"]
    ids = [camera.get("camera_id") for camera in cameras if isinstance(camera, dict)]
    if ids != list(V010_CAMERA_IDS):
        errors.append("v0.1.0 camera profile must contain ordered front and wrist cameras")
    for camera in cameras:
        if not isinstance(camera, dict):
            errors.append("camera entries must be objects")
            continue
        camera_id = camera.get("camera_id")
        if camera.get("feature_key") != V010_CAMERA_FEATURES.get(camera_id):
            errors.append(f"camera {camera_id!r} has an invalid feature_key")
        if camera.get("width") != 640 or camera.get("height") != 480:
            errors.append(f"camera {camera_id!r} must use 640x480 resolution")
        if camera.get("optical_model") != "pinhole":
            errors.append(f"camera {camera_id!r} must use the pinhole optical model")
        for field in ("focal_length_mm", "focus_distance_m"):
            value = camera.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                errors.append(f"camera {camera_id!r} {field} must be positive and finite")
        mount = camera.get("mount")
        if not isinstance(mount, dict):
            errors.append(f"camera {camera_id!r} mount must be an object")
            continue
        if not isinstance(mount.get("parent_frame"), str) or not mount["parent_frame"]:
            errors.append(f"camera {camera_id!r} parent_frame must be non-empty")
        if not _finite_numbers(mount.get("position_m"), 3):
            errors.append(f"camera {camera_id!r} mount position_m must be a finite XYZ vector")
        if not _finite_numbers(mount.get("orientation_xyzw"), 4):
            errors.append(
                f"camera {camera_id!r} mount orientation_xyzw must be a finite quaternion"
            )
        if mount.get("convention") != "opengl":
            errors.append(f"camera {camera_id!r} mount convention must be opengl")
    return errors


def load_camera_profile(path: str | Path) -> dict[str, Any]:
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_camera_profile(profile)
    if errors:
        raise ValueError("invalid camera profile: " + "; ".join(errors))
    return profile


def camera_profile_sha256(profile: dict[str, Any]) -> str:
    errors = validate_camera_profile(profile)
    if errors:
        raise ValueError("invalid camera profile: " + "; ".join(errors))
    return canonical_sha256(profile)


def build_camera_records(
    profile: dict[str, Any],
    *,
    resolved_intrinsics: dict[str, Iterable[Iterable[float]]],
    resolved_mounts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bind runtime calibration and mount evidence to the immutable profile."""
    profile_hash = camera_profile_sha256(profile)
    if set(resolved_intrinsics) != set(V010_CAMERA_IDS):
        raise ValueError("resolved intrinsics must contain exactly front and wrist")
    if set(resolved_mounts) != set(V010_CAMERA_IDS):
        raise ValueError("resolved mounts must contain exactly front and wrist")

    records = []
    for camera in profile["cameras"]:
        camera_id = camera["camera_id"]
        matrix = [list(row) for row in resolved_intrinsics[camera_id]]
        if len(matrix) != 3 or any(not _finite_numbers(row, 3) for row in matrix):
            raise ValueError(f"camera {camera_id!r} intrinsic matrix must be finite 3x3")
        mount = deepcopy(resolved_mounts[camera_id])
        if not _finite_numbers(mount.get("position_m"), 3):
            raise ValueError(f"camera {camera_id!r} resolved mount position must be finite XYZ")
        if not _finite_numbers(mount.get("orientation_xyzw"), 4):
            raise ValueError(f"camera {camera_id!r} resolved mount orientation must be finite XYZW")
        records.append(
            {
                "camera_id": camera_id,
                "feature_key": camera["feature_key"],
                "width": camera["width"],
                "height": camera["height"],
                "config_version": profile["profile_id"],
                "config_sha256": profile_hash,
                "calibration": {
                    "requested": {
                        "model": camera["optical_model"],
                        "focal_length_mm": camera["focal_length_mm"],
                        "focus_distance_m": camera["focus_distance_m"],
                    },
                    "resolved": {"model": "pinhole", "intrinsic_matrix": matrix},
                    "units": {
                        "focal_length_mm": "mm",
                        "focus_distance_m": "m",
                        "intrinsic_matrix": "pixel",
                    },
                },
                "mount_transform": {
                    "requested": deepcopy(camera["mount"]),
                    "resolved": mount,
                    "units": {"position_m": "m", "orientation_xyzw": "unitless"},
                },
                "frame_timestamp_source": profile["timestamp_source"],
            }
        )
    return records


def camera_cfg_drift_errors(profile: dict[str, Any], scene_cfg: Any) -> list[str]:
    """Compare an Isaac Lab scene config to the versioned camera profile."""
    errors = validate_camera_profile(profile)
    if errors:
        return errors
    for requested in profile["cameras"]:
        camera_id = requested["camera_id"]
        cfg = getattr(scene_cfg, f"{camera_id}_camera", None)
        if cfg is None:
            errors.append(f"Isaac scene is missing {camera_id}_camera")
            continue
        checks = (
            ("width", getattr(cfg, "width", None), requested["width"]),
            ("height", getattr(cfg, "height", None), requested["height"]),
            (
                "focal_length_mm",
                getattr(getattr(cfg, "spawn", None), "focal_length", None),
                requested["focal_length_mm"],
            ),
            (
                "focus_distance_m",
                getattr(getattr(cfg, "spawn", None), "focus_distance", None),
                requested["focus_distance_m"],
            ),
            (
                "mount.position_m",
                list(getattr(getattr(cfg, "offset", None), "pos", ())),
                requested["mount"]["position_m"],
            ),
            (
                "mount.orientation_xyzw",
                list(getattr(getattr(cfg, "offset", None), "rot", ())),
                requested["mount"]["orientation_xyzw"],
            ),
            (
                "mount.convention",
                getattr(getattr(cfg, "offset", None), "convention", None),
                requested["mount"]["convention"],
            ),
        )
        for field, actual, expected in checks:
            if actual != expected:
                errors.append(
                    f"Isaac {camera_id}_camera {field} drifted: "
                    f"expected {expected!r}, got {actual!r}"
                )
    return errors


def resolved_mounts_from_profile(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return local mount transforms after the runtime config drift gate passed."""
    errors = validate_camera_profile(profile)
    if errors:
        raise ValueError("invalid camera profile: " + "; ".join(errors))
    return {
        camera["camera_id"]: {
            "parent_frame": camera["mount"]["parent_frame"],
            "position_m": deepcopy(camera["mount"]["position_m"]),
            "orientation_xyzw": deepcopy(camera["mount"]["orientation_xyzw"]),
            "convention": camera["mount"]["convention"],
        }
        for camera in profile["cameras"]
    }
