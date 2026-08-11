"""Immutable campaign/segment contracts and append-only campaign events."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import time
import threading
from typing import Any, Iterable

from farpoint.contracts import validate_contract


CAMPAIGN_SCHEMA = "farpoint.collection-campaign.v1"
SEGMENT_SCHEMA = "farpoint.collection-segment.v1"
EVENT_SCHEMA = "farpoint.collection-event.v1"
EXECUTION_STATUSES = {"RUNNING", "FINISHED", "ABORTED", "PAUSED"}
QUALITY_STATUSES = {"NOT_EVALUATED", "PASS", "FAIL"}


def canonical_sha256(value: Any, *, omit: Iterable[str] = ()) -> str:
    payload = deepcopy(value)
    for key in omit:
        payload.pop(key, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stable_rank(seed: int, *parts: str) -> bytes:
    return hashlib.sha256((f"{seed}:" + ":".join(parts)).encode()).digest()


def build_crossed_quotas(
    *,
    object_variant_ids: Iterable[str],
    yaw_stratum_ids: Iterable[str],
    population_per_yaw_region: dict[str, int],
    validation_per_object_region: dict[str, int],
    validation_per_object_yaw: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build exact object × yaw × region × split quotas without a test split."""
    objects = tuple(object_variant_ids)
    yaws = tuple(yaw_stratum_ids)
    regions = tuple(population_per_yaw_region)
    if not objects or not yaws or not regions:
        raise ValueError("quota axes must be non-empty")
    if any(value <= 0 for value in population_per_yaw_region.values()):
        raise ValueError("population quotas must be positive")
    if set(validation_per_object_region) != set(regions):
        raise ValueError("validation region quotas must match population regions")
    if sum(validation_per_object_region.values()) != len(yaws) * validation_per_object_yaw:
        raise ValueError("validation region and yaw marginals do not have the same total")
    if any(
        count < 0 or count > population_per_yaw_region[region] * len(yaws)
        for region, count in validation_per_object_region.items()
    ):
        raise ValueError("validation region quota is outside its population")

    rows = []
    for object_id in objects:
        remaining = dict(validation_per_object_region)
        validation_matrix: dict[tuple[str, str], int] = {}
        for yaw_index, yaw_id in enumerate(yaws):
            for slot in range(validation_per_object_yaw):
                eligible = [
                    region
                    for region in regions
                    if remaining[region] > 0
                    and validation_matrix.get((yaw_id, region), 0)
                    < population_per_yaw_region[region]
                ]
                if not eligible:
                    raise ValueError("validation marginals cannot be allocated")
                region = min(
                    eligible,
                    key=lambda candidate: (
                        -remaining[candidate],
                        _stable_rank(seed, object_id, str(yaw_index), str(slot), candidate),
                    ),
                )
                validation_matrix[(yaw_id, region)] = (
                    validation_matrix.get((yaw_id, region), 0) + 1
                )
                remaining[region] -= 1
        if any(remaining.values()):
            raise ValueError("validation region marginals were not exhausted")
        for yaw_id in yaws:
            for region in regions:
                validation_count = validation_matrix.get((yaw_id, region), 0)
                train_count = population_per_yaw_region[region] - validation_count
                for split, count in (("train", train_count), ("validation", validation_count)):
                    if count:
                        rows.append(
                            {
                                "object_variant_id": object_id,
                                "yaw_stratum_id": yaw_id,
                                "region_band": region,
                                "split": split,
                                "count": count,
                            }
                        )
    return rows


def variation_seed(
    campaign_sha256: str,
    *,
    object_variant_id: str,
    yaw_stratum_id: str,
    region_band: str,
    split: str,
    quota_ordinal: int,
    replacement_index: int = 0,
) -> int:
    """Bind scene identity permanently; replacements change only their index."""
    _check_hash(campaign_sha256, "campaign_sha256", errors := [])
    if errors:
        raise ValueError(errors[0])
    if quota_ordinal < 0 or replacement_index < 0:
        raise ValueError("quota ordinal and replacement index must be non-negative")
    material = ":".join(
        (
            "farpoint-variation-seed-v1",
            campaign_sha256,
            object_variant_id,
            yaw_stratum_id,
            region_band,
            split,
            str(quota_ordinal),
            str(replacement_index),
        )
    )
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def attempt_seed(variation_seed_value: int, attempt_index: int) -> int:
    if variation_seed_value < 0:
        raise ValueError("variation seed must be non-negative")
    if attempt_index not in (0, 1, 2):
        raise ValueError("attempt index must be 0, 1, or 2")
    material = f"farpoint-attempt-seed-v1:{variation_seed_value}:{attempt_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def _check_hash(value: Any, name: str, errors: list[str]) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        errors.append(f"{name} must be a lowercase SHA256")


def validate_campaign_semantics(campaign: dict[str, Any]) -> list[str]:
    errors = validate_contract(campaign)
    if errors:
        return errors
    if campaign["campaign_sha256"] != canonical_sha256(campaign, omit=("campaign_sha256",)):
        errors.append("campaign_sha256 does not match canonical campaign content")
    target = campaign["target"]
    quotas = campaign["quotas"]
    if sum(int(row["count"]) for row in quotas) != int(target["successful_episodes"]):
        errors.append("campaign quotas do not sum to the success target")
    split_counts: dict[str, int] = {}
    for row in quotas:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + int(row["count"])
    if split_counts != target["splits"]:
        errors.append("campaign quotas do not match target split counts")
    if campaign["attempt_policy"].get("maximum_attempts_per_variation") != 3:
        errors.append("campaign must allow exactly three attempts per variation")
    if campaign["attempt_policy"].get("global_attempt_limit") is not None:
        errors.append("campaign must not define a global attempt limit")
    return errors


def validate_segment_semantics(segment: dict[str, Any]) -> list[str]:
    errors = validate_contract(segment)
    if errors:
        return errors
    if segment["segment_sha256"] != canonical_sha256(segment, omit=("segment_sha256",)):
        errors.append("segment_sha256 does not match canonical segment content")
    if segment["segment_index"] == 0 and segment.get("parent_manifest_sha256") is not None:
        errors.append("the first segment cannot have a parent manifest")
    if segment["segment_index"] > 0 and segment.get("parent_manifest_sha256") is None:
        errors.append("continuation segments must bind a parent manifest")
    return errors


def validate_event_sequence(events: Iterable[dict[str, Any]]) -> list[str]:
    errors = []
    previous_sequence = -1
    campaign_id = None
    for index, event in enumerate(events):
        schema_errors = validate_contract(event)
        errors.extend(f"event[{index}] {error}" for error in schema_errors)
        if schema_errors:
            continue
        if event["sequence"] != previous_sequence + 1:
            errors.append(f"event[{index}] sequence is not contiguous")
        previous_sequence = event["sequence"]
        if campaign_id is None:
            campaign_id = event["campaign_id"]
        elif event["campaign_id"] != campaign_id:
            errors.append(f"event[{index}] campaign_id changed")
    return errors


def create_campaign(spec: dict[str, Any]) -> dict[str, Any]:
    campaign = deepcopy(spec)
    campaign["schema_version"] = CAMPAIGN_SCHEMA
    campaign["campaign_sha256"] = canonical_sha256(campaign, omit=("campaign_sha256",))
    errors = validate_campaign_semantics(campaign)
    if errors:
        raise ValueError("invalid campaign: " + "; ".join(errors))
    return campaign


def create_segment(spec: dict[str, Any]) -> dict[str, Any]:
    segment = deepcopy(spec)
    segment["schema_version"] = SEGMENT_SCHEMA
    segment["segment_sha256"] = canonical_sha256(segment, omit=("segment_sha256",))
    errors = validate_segment_semantics(segment)
    if errors:
        raise ValueError("invalid segment: " + "; ".join(errors))
    return segment


@dataclass
class CampaignEventLog:
    """Durable JSONL event writer with contiguous monotonic sequence numbers."""

    path: Path
    campaign_id: str
    _next_sequence: int | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        segment_id: str | None = None,
        timestamp_unix: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._next_sequence is None:
                events = self._events()
                errors = validate_event_sequence(events)
                if errors:
                    raise ValueError(
                        "cannot append to invalid event log: " + "; ".join(errors)
                    )
                self._next_sequence = len(events)
            event = {
                "schema_version": EVENT_SCHEMA,
                "campaign_id": self.campaign_id,
                "segment_id": segment_id,
                "sequence": self._next_sequence,
                "timestamp_unix": float(
                    time.time() if timestamp_unix is None else timestamp_unix
                ),
                "event_type": event_type,
                "payload": deepcopy(payload),
            }
            errors = validate_contract(event)
            if errors:
                raise ValueError("invalid campaign event: " + "; ".join(errors))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644
            )
            try:
                os.write(
                    descriptor, (json.dumps(event, sort_keys=True) + "\n").encode()
                )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._next_sequence += 1
            return event


def atomic_status_write(path: Path, status: dict[str, Any]) -> None:
    """Atomically replace campaign status without mutating the event history."""
    if status.get("execution_status") not in EXECUTION_STATUSES:
        raise ValueError("invalid execution_status")
    if status.get("quality_status") not in QUALITY_STATUSES:
        raise ValueError("invalid quality_status")
    heartbeat = status.get("heartbeat_unix")
    if not isinstance(heartbeat, (int, float)) or not math.isfinite(float(heartbeat)):
        raise ValueError("heartbeat_unix must be finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
