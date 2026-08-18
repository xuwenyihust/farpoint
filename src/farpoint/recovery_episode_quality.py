"""Campaign-owned eligibility checks for live recovery demonstrations."""

from __future__ import annotations

from typing import Any


ELIGIBILITY_SCHEMA = "farpoint.recovery-episode-eligibility.v1"


def validate_recovery_episode_eligibility(
    metadata: dict[str, Any],
    trial: dict[str, Any],
    eligibility: dict[str, Any],
) -> list[str]:
    """Validate measured handoff evidence against one frozen campaign rule."""
    errors = []
    if eligibility.get("schema_version") != ELIGIBILITY_SCHEMA:
        return ["unsupported recovery episode eligibility schema"]
    trigger_class = trial.get("recovery_trigger_class")
    requirements = (eligibility.get("by_trigger_class") or {}).get(trigger_class)
    if not trigger_class or requirements is None:
        return ["recovery trial has no eligibility rule for its trigger class"]

    demonstration = metadata.get("demonstration") or {}
    intervention = demonstration.get("intervention") or {}
    trigger = intervention.get("trigger") or {}
    evidence = trigger.get("evidence") or {}
    handoff = intervention.get("handoff") or {}
    if demonstration.get("type") != "recovery":
        errors.append("demonstration.type must be recovery")
    if trigger.get("failure_class") != trigger_class:
        errors.append("recovery failure_class does not match the assigned trigger class")
    expected_stage = requirements.get("handoff_stage")
    if expected_stage is not None and trigger.get("handoff_stage") != expected_stage:
        errors.append("recovery handoff_stage does not match campaign eligibility")
    for field, expected in (requirements.get("trigger_fields") or {}).items():
        if trigger.get(field) != expected:
            errors.append(f"recovery trigger {field} does not match campaign eligibility")
    allowed_subclasses = requirements.get("allowed_failure_subclasses")
    if allowed_subclasses is not None and trigger.get("failure_subclass") not in set(
        allowed_subclasses
    ):
        errors.append("recovery failure_subclass is not allowed by campaign eligibility")
    for field, expected in (requirements.get("trigger_evidence") or {}).items():
        if evidence.get(field) != expected:
            errors.append(
                f"recovery trigger evidence {field} does not match campaign eligibility"
            )
    for field, expected in (requirements.get("handoff") or {}).items():
        if handoff.get(field) != expected:
            errors.append(f"recovery handoff {field} does not match campaign eligibility")
    return errors
