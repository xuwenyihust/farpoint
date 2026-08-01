"""Versioned metadata contracts and cross-file semantic checks."""

from __future__ import annotations

import json
import math
import re
from importlib.resources import files
from typing import Any


SCHEMA_FILES = {
    "farpoint.dataset.v2": "farpoint_dataset_v2.schema.json",
    "farpoint.episode.v2": "farpoint_episode_v2.schema.json",
    "farpoint.variation.v2": "farpoint_variation_v2.schema.json",
    "farpoint.benchmark.v2": "farpoint_benchmark_v2.schema.json",
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
    shape = (
        task.get("object_shape")
        or (metadata.get("scene") or {}).get("object", {}).get("shape")
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
    if set(variation.get("varied_axes") or ()) & set(variation.get("frozen_axes") or ()):
        errors.append("variation axes cannot be both varied and frozen")

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
    expected_accepted = (
        observed_successes >= acceptance.get("required_successes", 0)
        and observed_rate >= acceptance.get("required_success_rate", 0.0)
    )
    if acceptance.get("accepted") is not expected_accepted:
        errors.append("benchmark accepted does not match its acceptance thresholds")
    return errors
