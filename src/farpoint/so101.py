"""SO-101 joint conventions shared by simulation, recording, and export."""

from __future__ import annotations

from typing import Iterable

import numpy as np


SIM_JOINT_NAMES = (
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
)

LEROBOT_JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

# The NVIDIA SO-101 workshop maps the calibrated LeRobot command range onto
# the actual joint ranges authored in the USD articulation.
USD_MIN_DEGREES = np.asarray((-110.0, -100.0, -100.0, -95.0, -160.0, -10.0))
USD_MAX_DEGREES = np.asarray((110.0, 100.0, 90.0, 95.0, 160.0, 100.0))
LEROBOT_MIN = np.asarray((-100.0, -100.0, -100.0, -100.0, -100.0, 0.0))
LEROBOT_MAX = np.asarray((100.0, 100.0, 100.0, 100.0, 100.0, 100.0))


def _joint_vector(values: Iterable[float]) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=np.float64)
    if vector.shape != (6,):
        raise ValueError(f"SO-101 joint vector must have shape (6,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("SO-101 joint vector must contain only finite values")
    return vector


def lerobot_to_radians(values: Iterable[float], *, clip: bool = False) -> np.ndarray:
    """Convert real-robot compatible LeRobot positions to simulator radians."""
    raw = _joint_vector(values)
    if clip:
        raw = np.clip(raw, LEROBOT_MIN, LEROBOT_MAX)
    elif np.any(raw < LEROBOT_MIN) or np.any(raw > LEROBOT_MAX):
        raise ValueError("SO-101 LeRobot positions are outside the calibrated range")
    fraction = (raw - LEROBOT_MIN) / (LEROBOT_MAX - LEROBOT_MIN)
    degrees = USD_MIN_DEGREES + fraction * (USD_MAX_DEGREES - USD_MIN_DEGREES)
    return np.deg2rad(degrees).astype(np.float32)


def radians_to_lerobot(values: Iterable[float], *, clip: bool = False) -> np.ndarray:
    """Convert simulator radians to the SO-101 LeRobot position convention."""
    radians = _joint_vector(values)
    degrees = np.rad2deg(radians)
    fraction = (degrees - USD_MIN_DEGREES) / (USD_MAX_DEGREES - USD_MIN_DEGREES)
    raw = LEROBOT_MIN + fraction * (LEROBOT_MAX - LEROBOT_MIN)
    if clip:
        raw = np.clip(raw, LEROBOT_MIN, LEROBOT_MAX)
    elif np.any(raw < LEROBOT_MIN - 1e-6) or np.any(raw > LEROBOT_MAX + 1e-6):
        raise ValueError("SO-101 simulator positions are outside the authored USD range")
    return raw.astype(np.float32)


def mapping_metadata() -> dict:
    """Return serializable unit and range metadata for dataset sidecars."""
    return {
        "source_unit": "radian",
        "export_unit": "so101_calibrated_position",
        "sim_joint_names": list(SIM_JOINT_NAMES),
        "feature_names": list(LEROBOT_JOINT_NAMES),
        "lerobot_min": LEROBOT_MIN.tolist(),
        "lerobot_max": LEROBOT_MAX.tolist(),
        "usd_min_degrees": USD_MIN_DEGREES.tolist(),
        "usd_max_degrees": USD_MAX_DEGREES.tolist(),
    }
