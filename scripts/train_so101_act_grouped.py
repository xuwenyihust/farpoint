#!/usr/bin/env python3
"""Run LeRobot ACT training with a frozen deterministic grouped batch sampler."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from farpoint.training_sampler import DeterministicGroupedBatchSampler


def extract_sampler_plan_argument(arguments: list[str]) -> tuple[Path, list[str]]:
    prefix = "--farpoint-sampler-plan="
    matches = [argument for argument in arguments if argument.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("exactly one --farpoint-sampler-plan argument is required")
    path = Path(matches[0].removeprefix(prefix))
    return path, [argument for argument in arguments if not argument.startswith(prefix)]


def make_grouped_dataloader_class(original_dataloader, plan, state, restore=None):
    """Create a real DataLoader subclass so Accelerate type checks remain valid."""

    class GroupedDataLoader(original_dataloader):
        def __init__(self, dataset, *args, **kwargs):
            try:
                if state["used"]:
                    super().__init__(dataset, *args, **kwargs)
                    return
                if kwargs.get("sampler") is not None:
                    raise ValueError("Farpoint grouped sampling cannot be combined with another sampler")
                if kwargs.get("batch_size") != int(plan["batch_size"]):
                    raise ValueError("LeRobot batch size does not match the frozen sampler plan")

                episode_rows = dataset.meta.episodes
                episode_lengths = {
                    int(row["episode_index"]): int(row["length"])
                    for row in episode_rows.select_columns(["episode_index", "length"])
                }
                sampler = DeterministicGroupedBatchSampler(plan, dataset.episodes, episode_lengths)
                state["used"] = True
                options = dict(kwargs)
                for key in ("batch_size", "shuffle", "sampler", "drop_last"):
                    options.pop(key, None)
                super().__init__(dataset, *args, batch_sampler=sampler, **options)
            finally:
                if restore is not None:
                    restore()

    return GroupedDataLoader


def main() -> None:
    plan_path, sys.argv = extract_sampler_plan_argument(sys.argv)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "farpoint.training-sampler-plan.v1":
        raise ValueError("unsupported Farpoint sampler plan")
    if plan.get("kind") != "deterministic_grouped_batches":
        raise ValueError("grouped ACT entrypoint requires deterministic_grouped_batches")

    import torch
    from lerobot.scripts import lerobot_train

    original_dataloader = torch.utils.data.DataLoader
    state = {"used": False}

    def restore():
        torch.utils.data.DataLoader = original_dataloader

    GroupedDataLoader = make_grouped_dataloader_class(
        original_dataloader,
        plan,
        state,
        restore=restore,
    )
    torch.utils.data.DataLoader = GroupedDataLoader
    try:
        lerobot_train.train()
    finally:
        restore()
    if not state["used"]:
        raise RuntimeError("LeRobot training did not construct the grouped DataLoader")


if __name__ == "__main__":
    main()
