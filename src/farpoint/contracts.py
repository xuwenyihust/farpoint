"""Versioned metadata contracts and cross-file semantic checks."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from importlib.resources import files
from typing import Any

from farpoint.scene_entities import validate_scene_entities


SCHEMA_FILES = {
    "farpoint.collection-campaign.v1": "collection_campaign_v1.schema.json",
    "farpoint.collection-segment.v1": "collection_segment_v1.schema.json",
    "farpoint.collection-event.v1": "collection_event_v1.schema.json",
    "farpoint.episode.v4": "farpoint_episode_v4.schema.json",
    "farpoint.dataset.v3": "farpoint_dataset_v3.schema.json",
    "farpoint.episode.v3": "farpoint_episode_v3.schema.json",
    "farpoint.variation.v3": "farpoint_variation_v3.schema.json",
    "farpoint.dataset.v2": "farpoint_dataset_v2.schema.json",
    "farpoint.episode.v2": "farpoint_episode_v2.schema.json",
    "farpoint.variation.v2": "farpoint_variation_v2.schema.json",
    "farpoint.benchmark.v2": "farpoint_benchmark_v2.schema.json",
    "farpoint.collection.v1": "farpoint_collection_v1.schema.json",
    "farpoint.dataset-quality-report.v1": "dataset_quality_report_v1.schema.json",
    "farpoint.policy-training.v1": "policy_training_v1.schema.json",
    "farpoint.policy-rollout.v1": "policy_rollout_v1.schema.json",
}
SPLITS = ("train", "validation", "test")
SHAPE_TERMS = {
    "cube": {"cube"},
    "cuboid": {"cuboid", "box"},
    "cylinder": {"cylinder"},
}


def load_schema(schema_version: str) -> dict[str, Any]:
    """Load a public JSON Schema by its stable contract identifier."""
    try:
        filename = SCHEMA_FILES[schema_version]
    except KeyError as error:
        raise ValueError(f"unsupported schema version: {schema_version}") from error
    return json.loads(files("farpoint.schemas").joinpath(filename).read_text(encoding="utf-8"))


def validate_contract(payload: dict[str, Any], schema_version: str | None = None) -> list[str]:
    """Return deterministic JSON Schema errors without hiding all failures behind the first."""
    version = schema_version or payload.get("schema_version")
    if not version:
        return ["schema_version is required"]
    try:
        from jsonschema import Draft202012Validator
    except ImportError as error:  # pragma: no cover - development dependency guard
        raise RuntimeError("jsonschema is required to validate Farpoint contracts") from error
    validator = Draft202012Validator(load_schema(version))
    return [
        f"{'/'.join(str(part) for part in issue.absolute_path) or '<root>'}: {issue.message}"
        for issue in sorted(
            validator.iter_errors(payload),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    ]


def task_definition(metadata: dict[str, Any]) -> dict[str, str]:
    """Resolve an episode task without assuming a cube-only dataset."""
    task = metadata.get("task") or {}
    variation = metadata.get("variation") or {}
    scene = metadata.get("scene") or {}
    archetype = (scene.get("object_archetype") or {}).get("resolved") or {}
    shape = (
        task.get("object_shape")
        or archetype.get("semantic_type")
        or archetype.get("archetype_id")
        or scene.get("object", {}).get("shape")
        or next(
            (
                entity.get("entity_type")
                for entity in scene.get("entities", ())
                if entity.get("role") == "manipulated_object"
            ),
            None,
        )
        or variation.get("object_type")
    )
    task_id = task.get("task_id") or metadata.get("task_name")
    instruction = task.get("instruction") or metadata.get("language_instruction")
    success_criteria_id = task.get("success_criteria_id") or metadata.get("success_criteria_id")
    if not task_id or not instruction or not shape or not success_criteria_id:
        raise ValueError(
            "episode metadata must define task id, instruction, object shape, and success criteria"
        )
    return {
        "task_id": str(task_id),
        "instruction": str(instruction),
        "object_shape": str(shape),
        "success_criteria_id": str(success_criteria_id),
    }


def validate_episode_semantics(record: dict[str, Any]) -> list[str]:
    """Check relationships that JSON Schema cannot express across nested fields."""
    errors = []
    if record.get("schema_version") == "farpoint.episode.v4":
        identity = record.get("identity") or {}
        task = record.get("task") or {}
        scene = record.get("scene") or {}
        variation = record.get("variation") or {}
        recording = record.get("recording") or {}
        if identity.get("task_id") != task.get("task_id"):
            errors.append("identity.task_id does not match task.task_id")
        if identity.get("split") != variation.get("split"):
            errors.append("variation.split does not match identity.split")
        if set(variation.get("varied_axes") or ()) & set(variation.get("frozen_axes") or ()):
            errors.append("variation axes cannot be both varied and frozen")
        entities = scene.get("entities") or []
        try:
            validate_scene_entities(entities)
        except ValueError as error:
            errors.append(str(error))
        entity_index = {
            entity.get("entity_id"): entity
            for entity in entities
            if isinstance(entity, dict) and entity.get("entity_id")
        }
        for task_field, role in (
            ("manipulated_entity_id", "manipulated_object"),
            ("target_entity_id", "placement_target"),
        ):
            entity_id = task.get(task_field)
            if entity_id not in entity_index:
                errors.append(f"task.{task_field} is missing from scene.entities")
            elif entity_index[entity_id].get("role") != role:
                errors.append(f"task.{task_field} does not name a {role}")
        target = entity_index.get(task.get("target_entity_id")) or {}
        if task.get("acceptance_region_id") not in {
            region.get("region_id")
            for region in target.get("regions", ())
            if isinstance(region, dict)
        }:
            errors.append("task.acceptance_region_id is missing from the target entity")
        cameras = recording.get("cameras") or []
        camera_ids = {camera.get("camera_id") for camera in cameras if isinstance(camera, dict)}
        feature_keys = {
            camera.get("feature_key") for camera in cameras if isinstance(camera, dict)
        }
        if camera_ids != {"front", "wrist"}:
            errors.append("episode v4 requires exactly front and wrist cameras")
        if feature_keys != {
            "observation.images.front",
            "observation.images.wrist",
        }:
            errors.append("episode v4 camera feature keys are incomplete")
        if not (recording.get("synchronization") or {}).get("same_control_tick"):
            errors.append("episode v4 cameras must share the same control tick")
        return errors
    if record.get("schema_version") == "farpoint.episode.v3":
        identity = record.get("identity") or {}
        task = record.get("task") or {}
        scene = record.get("scene") or {}
        obj = scene.get("object") or {}
        variation = record.get("variation") or {}
        resolved = variation.get("resolved") or {}
        pose = obj.get("initial_pose") or {}
        if identity.get("task_id") != task.get("task_id"):
            errors.append("identity.task_id does not match task.task_id")
        if task.get("object_shape") != obj.get("shape"):
            errors.append("task.object_shape does not match scene.object.shape")
        for field, scene_field in (("shape", "shape"), ("dimensions_m", "dimensions_m"), ("position_m", None), ("rgba", "rgba"), ("mass_kg", "mass_kg"), ("static_friction", "static_friction"), ("dynamic_friction", "dynamic_friction")):
            expected = pose.get("position_m") if field == "position_m" else obj.get(scene_field)
            if resolved.get(field) != expected:
                errors.append(f"variation.resolved.{field} does not match the scene object")
        if set(variation.get("varied_axes") or ()) & set(variation.get("frozen_axes") or ()):
            errors.append("variation axes cannot be both varied and frozen")
        if variation.get("split") != identity.get("split"):
            errors.append("variation.split does not match identity.split")
        entities = scene.get("entities")
        if entities is not None:
            try:
                validate_scene_entities(entities)
            except ValueError as error:
                errors.append(str(error))
            entity_index = {
                entity.get("entity_id"): entity
                for entity in entities
                if isinstance(entity, dict) and entity.get("entity_id")
            }
            manipulated_id = task.get("manipulated_entity_id")
            target_id = task.get("target_entity_id")
            if manipulated_id and manipulated_id not in entity_index:
                errors.append("task.manipulated_entity_id is missing from scene.entities")
            if target_id and target_id not in entity_index:
                errors.append("task.target_entity_id is missing from scene.entities")
            if manipulated_id in entity_index:
                entity = entity_index[manipulated_id]
                if entity.get("role") != "manipulated_object":
                    errors.append("task.manipulated_entity_id does not name a manipulated object")
                if task.get("object_shape") != entity.get("entity_type"):
                    errors.append(
                        "task.object_shape does not match the manipulated entity type"
                    )
            if target_id in entity_index:
                target_entity = entity_index[target_id]
                if target_entity.get("role") != "placement_target":
                    errors.append("task.target_entity_id does not name a placement target")
                region_id = task.get("acceptance_region_id")
                if region_id and region_id not in {
                    region.get("region_id")
                    for region in target_entity.get("regions", ())
                    if isinstance(region, dict)
                }:
                    errors.append(
                        "task.acceptance_region_id is missing from the target entity"
                    )
            resolved_entities = resolved.get("entities")
            if resolved_entities is not None:
                if not isinstance(resolved_entities, dict):
                    errors.append("variation.resolved.entities must be an object")
                else:
                    for entity_id, entity in entity_index.items():
                        if resolved_entities.get(entity_id) != entity:
                            errors.append(
                                f"variation.resolved.entities.{entity_id} does not "
                                "match scene.entities"
                            )
        return errors
    identity = record.get("identity") or {}
    task = record.get("task") or {}
    scene = record.get("scene") or {}
    obj = scene.get("object") or {}
    variation = record.get("variation") or {}
    resolved = variation.get("resolved") or {}

    if identity.get("task_id") != task.get("task_id"):
        errors.append("identity.task_id does not match task.task_id")
    if task.get("object_shape") != obj.get("shape"):
        errors.append("task.object_shape does not match scene.object.shape")
    if resolved.get("object_shape") != obj.get("shape"):
        errors.append("variation.resolved.object_shape does not match scene.object.shape")
    if resolved.get("object_position_m") != (obj.get("initial_pose") or {}).get("position_m"):
        errors.append("variation.resolved.object_position_m does not match the scene object pose")
    if resolved.get("object_dimensions_m") != obj.get("dimensions_m"):
        errors.append("variation.resolved.object_dimensions_m does not match the scene object")
    if resolved.get("mass_kg") is not None and resolved.get("mass_kg") != obj.get("mass_kg"):
        errors.append("variation.resolved.mass_kg does not match the scene object")
    if resolved.get("rgba") is not None and resolved.get("rgba") != obj.get("rgba"):
        errors.append("variation.resolved.rgba does not match the scene object")
    if set(variation.get("varied_axes") or ()) & set(variation.get("frozen_axes") or ()):
        errors.append("variation axes cannot be both varied and frozen")
    if variation.get("split") is not None and variation.get("split") != identity.get("split"):
        errors.append("variation.split does not match identity.split")

    shape = str(task.get("object_shape") or "").lower()
    instruction_words = set(re.findall(r"[a-z0-9]+", str(task.get("instruction") or "").lower()))
    accepted_terms = SHAPE_TERMS.get(shape, {shape})
    if shape and not accepted_terms.intersection(instruction_words):
        errors.append("task instruction does not name its object shape")
    return errors


def validate_benchmark_episode_links(
    benchmark: dict[str, Any], episodes: list[dict[str, Any]]
) -> list[str]:
    """Validate provenance links between selected dataset episodes and a benchmark manifest."""
    errors = []
    trials = {trial.get("trial_id"): trial for trial in benchmark.get("trials", [])}
    if len(trials) != len(benchmark.get("trials", [])):
        errors.append("benchmark trial ids must be unique")
    for episode in episodes:
        identity = episode.get("identity") or {}
        trial_id = identity.get("trial_id")
        trial = trials.get(trial_id)
        if trial is None:
            errors.append(f"episode trial_id is missing from benchmark: {trial_id}")
            continue
        if trial.get("episode_id") != identity.get("episode_id"):
            errors.append(f"benchmark episode_id mismatch for trial: {trial_id}")
        if trial.get("variation_id") != (episode.get("variation") or {}).get("variation_id"):
            errors.append(f"benchmark variation_id mismatch for trial: {trial_id}")
        if trial.get("split") != identity.get("split"):
            errors.append(f"benchmark split mismatch for trial: {trial_id}")
        if benchmark.get("task_id") != (episode.get("task") or {}).get("task_id"):
            errors.append(f"benchmark task_id mismatch for trial: {trial_id}")
        provenance = episode.get("provenance") or {}
        for key in ("git_commit", "config_sha256", "simulator_image_digest"):
            if benchmark.get(key) != provenance.get(key):
                errors.append(f"benchmark {key} mismatch for trial: {trial_id}")
        outcome = episode.get("outcome") or {}
        if trial.get("success") != outcome.get("success"):
            errors.append(f"benchmark success mismatch for trial: {trial_id}")
        if trial.get("dataset_valid") != outcome.get("dataset_valid"):
            errors.append(f"benchmark dataset_valid mismatch for trial: {trial_id}")
    return errors


def validate_benchmark_semantics(benchmark: dict[str, Any]) -> list[str]:
    """Check that benchmark acceptance is derived from its recorded trials."""
    errors = []
    trials = benchmark.get("trials") or []
    acceptance = benchmark.get("acceptance") or {}
    observed_successes = sum(trial.get("success") is True for trial in trials)
    observed_rate = observed_successes / len(trials) if trials else 0.0
    if acceptance.get("observed_successes") != observed_successes:
        errors.append("benchmark observed_successes does not match its trials")
    reported_rate = acceptance.get("observed_success_rate")
    if not isinstance(reported_rate, (int, float)) or not math.isclose(
        reported_rate, observed_rate, abs_tol=1e-9
    ):
        errors.append("benchmark observed_success_rate does not match its trials")
    expected_accepted = observed_successes >= acceptance.get(
        "required_successes", 0
    ) and observed_rate >= acceptance.get("required_success_rate", 0.0)
    if acceptance.get("accepted") is not expected_accepted:
        errors.append("benchmark accepted does not match its acceptance thresholds")
    return errors


def validate_collection_semantics(collection: dict[str, Any]) -> list[str]:
    """Check balanced selection, yield, coverage, and collection acceptance."""
    errors = []
    attempts = collection.get("attempts") or []
    acceptance = collection.get("acceptance") or {}
    successes = sum(row.get("outcome_success") is True for row in attempts)
    observed_yield = successes / len(attempts) if attempts else 0.0
    selected = [row for row in attempts if row.get("selected_for_dataset") is True]
    selected_per_cell = Counter(row.get("cell_id") for row in selected)
    splits = Counter(row.get("dataset_split") for row in selected)
    if acceptance.get("observed_task_attempts") != len(attempts):
        errors.append("collection observed_task_attempts does not match attempts")
    if acceptance.get("observed_task_successes") != successes:
        errors.append("collection observed_task_successes does not match attempts")
    reported_yield = acceptance.get("observed_task_yield")
    if not isinstance(reported_yield, (int, float)) or not math.isclose(
        reported_yield, observed_yield, abs_tol=1e-9
    ):
        errors.append("collection observed_task_yield does not match attempts")
    if acceptance.get("observed_selected_episodes") != len(selected):
        errors.append("collection observed_selected_episodes does not match attempts")
    if acceptance.get("observed_covered_cells") != len(selected_per_cell):
        errors.append("collection observed_covered_cells does not match attempts")
    if acceptance.get("selected_per_cell") != dict(sorted(selected_per_cell.items())):
        errors.append("collection selected_per_cell does not match attempts")
    observed_splits = {split: splits[split] for split in SPLITS}
    if acceptance.get("observed_splits") != observed_splits:
        errors.append("collection observed_splits does not match attempts")
    expected_accepted = (
        len(attempts) <= acceptance.get("maximum_task_attempts", 0)
        and observed_yield >= acceptance.get("required_task_yield", 1.0)
        and len(selected) == acceptance.get("required_selected_episodes")
        and len(selected_per_cell) == acceptance.get("required_cells")
        and bool(selected_per_cell)
        and set(selected_per_cell.values()) == {acceptance.get("required_selected_per_cell")}
        and observed_splits == acceptance.get("required_splits")
    )
    if acceptance.get("accepted") is not expected_accepted:
        errors.append("collection accepted does not match collection evidence")
    for row in selected:
        if not row.get("outcome_success") or not row.get("dataset_valid"):
            errors.append(f"selected collection attempt is not eligible: {row.get('trial_id')}")
    if collection.get("release_policy") == "coverage_first_all_successful":
        release = collection.get("release_acceptance") or {}
        eligible = [
            row
            for row in attempts
            if row.get("outcome_success") is True and row.get("dataset_valid") is True
        ]
        successes_per_cell = dict(sorted(Counter(row.get("cell_id") for row in eligible).items()))
        minimum = release.get("minimum_successes_per_cell", 0)
        covered = sum(count >= minimum for count in successes_per_cell.values())
        split_counts = {
            split: sum(row.get("source_split") == split for row in eligible) for split in SPLITS
        }
        if release.get("successes_per_cell") != successes_per_cell:
            errors.append("coverage release successes_per_cell does not match attempts")
        if release.get("eligible_episodes") != len(eligible):
            errors.append("coverage release eligible_episodes does not match attempts")
        if release.get("observed_covered_cells") != covered:
            errors.append("coverage release observed_covered_cells does not match attempts")
        if release.get("split_counts") != split_counts:
            errors.append("coverage release split_counts does not match attempts")
        if release.get("required_cells") != 25:
            errors.append("coverage release must require all 25 cells")
        if minimum != 2:
            errors.append("coverage release must require two successes per cell")
        expected_release_accepted = covered == 25 and minimum == 2
        if release.get("accepted") is not expected_release_accepted:
            errors.append("coverage release accepted does not match coverage evidence")
    return errors


def validate_collection_episode_links(
    collection: dict[str, Any], episodes: list[dict[str, Any]]
) -> list[str]:
    """Validate selected dataset episodes against a collection manifest."""
    errors = []
    coverage_first = collection.get("release_policy") == "coverage_first_all_successful"
    selected = {
        row.get("trial_id"): row
        for row in collection.get("attempts", [])
        if (
            row.get("outcome_success") is True and row.get("dataset_valid") is True
            if coverage_first
            else row.get("selected_for_dataset") is True
        )
    }
    if len(selected) != sum(
        (
            row.get("outcome_success") is True and row.get("dataset_valid") is True
            if coverage_first
            else row.get("selected_for_dataset") is True
        )
        for row in collection.get("attempts", [])
    ):
        errors.append("selected collection trial ids must be unique")
    episode_trial_ids = [(episode.get("identity") or {}).get("trial_id") for episode in episodes]
    if len(set(episode_trial_ids)) != len(episode_trial_ids):
        errors.append("dataset episode trial ids must be unique")
    missing = sorted(set(selected) - set(episode_trial_ids))
    if missing:
        errors.append(
            "selected collection trials are missing from the dataset: " + ", ".join(missing)
        )
    for episode in episodes:
        identity = episode.get("identity") or {}
        trial_id = identity.get("trial_id")
        attempt = selected.get(trial_id)
        if attempt is None:
            errors.append(f"episode trial_id is missing from collection: {trial_id}")
            continue
        if attempt.get("episode_id") != identity.get("episode_id"):
            errors.append(f"collection episode_id mismatch for trial: {trial_id}")
        if attempt.get("variation_id") != (episode.get("variation") or {}).get("variation_id"):
            errors.append(f"collection variation_id mismatch for trial: {trial_id}")
        expected_split = (
            attempt.get("source_split") if coverage_first else attempt.get("dataset_split")
        )
        if expected_split != identity.get("split"):
            errors.append(f"collection split mismatch for trial: {trial_id}")
        if collection.get("task_id") != (episode.get("task") or {}).get("task_id"):
            errors.append(f"collection task_id mismatch for trial: {trial_id}")
        outcome = episode.get("outcome") or {}
        if outcome.get("success") is not True or outcome.get("dataset_valid") is not True:
            errors.append(f"collection selected episode is not successful: {trial_id}")
    return errors
