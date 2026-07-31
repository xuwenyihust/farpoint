"""Normalize Farpoint episode metadata across legacy and profiled runs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def normalize_episode_metadata(metadata: dict, metrics: dict | None = None) -> dict:
    """Return a stable metadata record without rewriting the raw source metadata.

    Legacy episodes predate the V1.1 variation registry. They are explicitly
    labeled as legacy instead of being assigned a modern profile they did not
    actually use.
    """
    metrics = metrics or {}
    variation = metadata.get("variation") or {}
    randomization = deepcopy(metadata.get("randomization") or {})
    profiled = bool(variation.get("variation_id") or metadata.get("variation_id"))

    if profiled:
        variation_id = variation.get("variation_id") or metadata.get("variation_id")
        object_type = variation.get("object_type", "unknown")
        position_bin = variation.get("object_position_bin", "unknown")
        grasp_profile = variation.get("grasp_profile", "default")
        source_generation = "farpoint_v1_1_profiled"
        normalized_variation = deepcopy(variation)
    else:
        # The legacy task only generated cubes. Keep the original randomization
        # as provenance and avoid pretending it used a V1.1 position profile.
        variation_id = "legacy_cube_randomized"
        object_type = "cube"
        position_bin = "legacy_randomized"
        grasp_profile = "default"
        source_generation = "farpoint_legacy_randomized_v0"
        normalized_variation = {
            "schema_version": "farpoint.variation.v1",
            "variation_id": variation_id,
            "object_type": object_type,
            "object_position_bin": position_bin,
            "grasp_profile": grasp_profile,
            "seed": metadata.get("episode_seed"),
        }

    return {
        "metadata_version": "farpoint.episode.v1",
        "source_generation": source_generation,
        "episode_id": metadata.get("episode_id"),
        "episode_seed": metadata.get("episode_seed"),
        "task_name": metadata.get("task_name"),
        "task_schema_version": metadata.get("task_schema_version"),
        "variation_id": variation_id,
        "object_type": object_type,
        "object_position_bin": position_bin,
        "grasp_profile": grasp_profile,
        "variation": normalized_variation,
        "randomization": randomization,
        "object_position_xy": randomization.get("pick_object_xy"),
        "success": bool(metrics.get("success")),
        "dataset_valid": bool(metrics.get("dataset_valid")),
    }
