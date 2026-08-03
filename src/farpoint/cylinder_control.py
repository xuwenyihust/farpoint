"""Cylinder-specific pre-grasp control helpers."""

from __future__ import annotations


def hold_pregrasp_hover(
    target,
    *,
    motion_frame: int,
    release_frame: int,
    hover_height: float,
):
    """Keep the Cartesian target above the object until aperture calibration settles."""
    if len(target) != 3:
        raise ValueError("target must contain three coordinates")
    result = [float(value) for value in target]
    if int(motion_frame) < int(release_frame):
        result[2] = max(result[2], float(hover_height))
    return result
