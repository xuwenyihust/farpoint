"""Deterministic grouped batch sampling for policy-training ablations."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator, Mapping, Sequence
from typing import Any


def parse_episode_slices(expressions: Sequence[str]) -> list[int]:
    """Expand ordered ``start:stop`` slices while rejecting overlap."""
    episodes: list[int] = []
    seen: set[int] = set()
    for expression in expressions:
        parts = expression.split(":")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(f"episode slice must be a non-negative start:stop slice: {expression}")
        start, stop = map(int, parts)
        if stop <= start:
            raise ValueError(f"episode slice must have stop greater than start: {expression}")
        for episode in range(start, stop):
            if episode in seen:
                raise ValueError(f"episode selection overlaps at index {episode}")
            seen.add(episode)
            episodes.append(episode)
    return episodes


def selected_training_episodes(spec: Mapping[str, Any]) -> list[int]:
    """Resolve the actual training selection, falling back to the logical train split."""
    sampling = spec.get("sampling")
    if sampling is None:
        return parse_episode_slices([spec["dataset"]["splits"]["train"]])
    return parse_episode_slices(sampling["episode_slices"])


def validate_sampling_contract(spec: Mapping[str, Any]) -> None:
    """Validate selection and grouped-batch invariants not expressible in JSON Schema."""
    sampling = spec.get("sampling")
    if sampling is None:
        return
    selected = selected_training_episodes(spec)
    logical_train = set(parse_episode_slices([spec["dataset"]["splits"]["train"]]))
    if not set(selected).issubset(logical_train):
        raise ValueError("sampling episode selection must be a subset of the logical train split")
    if len(selected) != sampling["expected_episode_count"]:
        raise ValueError("sampling expected_episode_count does not match its episode selection")

    if sampling["kind"] == "uniform_frames":
        return

    groups = sampling["groups"]
    group_ids = [group["group_id"] for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("sampling group ids must be unique")
    grouped: list[int] = []
    for group in groups:
        grouped.extend(parse_episode_slices(group["episode_slices"]))
    if len(grouped) != len(set(grouped)) or set(grouped) != set(selected):
        raise ValueError("sampling groups must partition the selected episodes exactly")

    batch_size = int(spec["training"]["batch_size"])
    known = set(group_ids)
    for template in sampling["batch_cycle"]:
        if set(template) - known:
            raise ValueError("batch cycle references an unknown sampling group")
        if sum(int(count) for count in template.values()) != batch_size:
            raise ValueError("each sampling batch template must match the training batch size")


def build_sampler_plan(spec: Mapping[str, Any], episode_lengths: Mapping[int, int]) -> dict[str, Any]:
    """Bind a config-owned sampling contract to observed episode lengths."""
    selected = selected_training_episodes(spec)
    missing = sorted(set(selected) - set(episode_lengths))
    if missing:
        raise ValueError(f"selected episode metadata is missing: {missing}")
    selected_frames = sum(int(episode_lengths[index]) for index in selected)
    sampling = spec.get("sampling")
    expected_frames = (
        int(sampling["expected_frame_count"])
        if sampling is not None
        else int(spec["dataset"]["expected"]["selected_frames"]["train"])
    )
    if selected_frames != expected_frames:
        raise ValueError(
            f"selected train frame count mismatch: {selected_frames} != {expected_frames}"
        )
    start_step = int((spec.get("continuation") or {}).get("source_step", 0))
    steps = int(spec["training"]["steps"])
    if not 0 <= start_step < steps:
        raise ValueError("sampler continuation start_step must be in [0, steps)")
    plan: dict[str, Any] = {
        "schema_version": "farpoint.training-sampler-plan.v1",
        "kind": "uniform_frames" if sampling is None else sampling["kind"],
        "seed": int(spec["training"]["seed"]),
        "start_step": start_step,
        "steps": steps,
        "batch_size": int(spec["training"]["batch_size"]),
        "selected_episodes": selected,
        "selected_episode_count": len(selected),
        "selected_frame_count": selected_frames,
    }
    if sampling is not None and sampling["kind"] == "deterministic_grouped_batches":
        plan["groups"] = sampling["groups"]
        plan["batch_cycle"] = sampling["batch_cycle"]
    return plan


def expected_group_sample_counts(plan: Mapping[str, Any]) -> dict[str, int]:
    """Return exact group draws made by a finite grouped sampler plan."""
    if plan["kind"] != "deterministic_grouped_batches":
        return {}
    cycle = plan["batch_cycle"]
    steps = int(plan["steps"])
    counts = {group["group_id"]: 0 for group in plan["groups"]}
    for step in range(int(plan.get("start_step", 0)), steps):
        for group_id, count in cycle[step % len(cycle)].items():
            counts[group_id] += int(count)
    return counts


class DeterministicGroupedBatchSampler:
    """Yield fixed-composition batches while cycling shuffled group frame pools."""

    def __init__(
        self,
        plan: Mapping[str, Any],
        dataset_episode_order: Sequence[int],
        episode_lengths: Mapping[int, int],
    ) -> None:
        if plan["kind"] != "deterministic_grouped_batches":
            raise ValueError("grouped batch sampler requires a grouped sampler plan")
        if list(dataset_episode_order) != list(plan["selected_episodes"]):
            raise ValueError("dataset episode order does not match the frozen sampler plan")
        self.plan = dict(plan)
        frame_ranges: dict[int, range] = {}
        offset = 0
        for episode in dataset_episode_order:
            length = int(episode_lengths[episode])
            frame_ranges[episode] = range(offset, offset + length)
            offset += length
        if offset != int(plan["selected_frame_count"]):
            raise ValueError("dataset frame count does not match the frozen sampler plan")

        self.group_frames: dict[str, list[int]] = {}
        for group in plan["groups"]:
            episodes = parse_episode_slices(group["episode_slices"])
            frames = [index for episode in episodes for index in frame_ranges[episode]]
            if not frames:
                raise ValueError(f"sampling group contains no frames: {group['group_id']}")
            self.group_frames[group["group_id"]] = frames

    def __len__(self) -> int:
        return int(self.plan["steps"]) - int(self.plan.get("start_step", 0))

    def __iter__(self) -> Iterator[list[int]]:
        pools: dict[str, list[int]] = {}
        cursors: dict[str, int] = {}
        rngs: dict[str, random.Random] = {}
        for group_id, frames in self.group_frames.items():
            salt = int.from_bytes(hashlib.sha256(group_id.encode()).digest()[:8], "big")
            rng = random.Random(int(self.plan["seed"]) ^ salt)
            pool = list(frames)
            rng.shuffle(pool)
            pools[group_id] = pool
            cursors[group_id] = 0
            rngs[group_id] = rng

        start_step = int(self.plan.get("start_step", 0))
        for step in range(int(self.plan["steps"])):
            batch: list[int] = []
            template = self.plan["batch_cycle"][step % len(self.plan["batch_cycle"])]
            for group_id, requested in template.items():
                for _ in range(int(requested)):
                    if cursors[group_id] >= len(pools[group_id]):
                        rngs[group_id].shuffle(pools[group_id])
                        cursors[group_id] = 0
                    batch.append(pools[group_id][cursors[group_id]])
                    cursors[group_id] += 1
            if len(batch) != int(self.plan["batch_size"]):
                raise RuntimeError("generated batch does not match the frozen batch size")
            if step >= start_step:
                yield batch
