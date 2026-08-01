"""Normalize Farpoint episode metadata across legacy and profiled runs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from farpoint.contracts import task_definition, validate_contract, validate_episode_semantics


QUALITY_FIELDS = (
    "final_xy_error_m",
    "perception_error_m",
    "bilateral_contact_frames",
    "lift_height_m",
    "settling_error",
    "joint_smoothness_score",
)


def resolve_measured_object_pose(
    variation: dict[str, Any] | None, position_m: list[float]
) -> dict[str, Any] | None:
    """Copy episode variation metadata and bind its resolved object pose."""
    if variation is None:
        return None
    resolved = deepcopy(variation)
    values = resolved.get("resolved")
    if isinstance(values, dict) and "object_position_m" in values:
        values["object_position_m"] = [float(value) for value in position_m]
    return resolved

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


def normalize_episode_metadata_v2(
    metadata: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    *,
    split: str,
    dataset_episode_index: int,
    trial_id: str | None = None,
) -> dict[str, Any]:
    """Build a strict v2 release record from simulator-authored structured metadata.

    The simulator is responsible for measured scene and provenance values. This
    function adds export identity and outcome fields, but never invents missing
    calibration, asset, pose, or randomization values.
    """
    metrics = metrics or {}
    task = task_definition(metadata)
    source_identity = metadata.get("identity") or {}
    episode_id = metadata.get("episode_id") or source_identity.get("episode_id")
    if not episode_id:
        raise ValueError("episode metadata must define episode_id")

    required_sections = ("provenance", "embodiment", "scene", "variation", "recording")
    missing = [section for section in required_sections if not isinstance(metadata.get(section), dict)]
    if missing:
        raise ValueError("episode metadata is missing structured sections: " + ", ".join(missing))

    outcome_source = metadata.get("outcome") or {}
    quality_source = outcome_source.get("quality") or metrics.get("quality") or metrics
    record = {
        "schema_version": "farpoint.episode.v2",
        "identity": {
            "episode_id": str(episode_id),
            "trial_id": str(
                trial_id or metadata.get("trial_id") or source_identity.get("trial_id") or episode_id
            ),
            "task_id": task["task_id"],
            "split": split,
            "dataset_episode_index": dataset_episode_index,
        },
        "provenance": deepcopy(metadata["provenance"]),
        "task": task,
        "embodiment": deepcopy(metadata["embodiment"]),
        "scene": deepcopy(metadata["scene"]),
        "variation": deepcopy(metadata["variation"]),
        "recording": deepcopy(metadata["recording"]),
        "outcome": {
            "success": bool(outcome_source.get("success", metrics.get("success"))),
            "dataset_valid": bool(
                outcome_source.get("dataset_valid", metrics.get("dataset_valid"))
            ),
            "failure_category": outcome_source.get(
                "failure_category", metrics.get("failure_category")
            ),
            "failure_reason": outcome_source.get("failure_reason", metrics.get("failure_reason")),
            "quality": {field: quality_source.get(field) for field in QUALITY_FIELDS},
        },
    }
    errors = validate_contract(record) + validate_episode_semantics(record)
    if errors:
        raise ValueError("invalid farpoint.episode.v2 metadata: " + "; ".join(errors))
    return record


def validate_simulator_metadata_v2(
    metadata: dict[str, Any], metrics: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate simulator-authored v2 sections before an episode is persisted."""
    split = metadata.get("split")
    if split not in {"train", "validation", "test"}:
        raise ValueError("simulator v2 metadata must define a public dataset split")
    return normalize_episode_metadata_v2(
        metadata,
        metrics,
        split=split,
        dataset_episode_index=0,
        trial_id=metadata.get("trial_id"),
    )
