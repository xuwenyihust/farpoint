"""Dependency-light serving state for frozen expert action replays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from farpoint.policy_training import file_sha256
from farpoint.so101 import radians_to_lerobot


class ExpertActionReplay:
    """Serve policy-rate actions with optional exact physics-rate targets."""

    def __init__(self, manifest_path: Path):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "farpoint.expert-action-replay.v1":
            raise ValueError("unsupported expert action replay manifest")
        scenes = payload.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("expert action replay manifest has no scenes")
        self.manifest_sha256 = file_sha256(manifest_path)
        self.camera_features = payload["camera_features"]
        self.scenes = {scene["scene_id"]: scene for scene in scenes}
        if len(self.scenes) != len(scenes):
            raise ValueError("expert action replay scene IDs must be unique")
        self.scene_id: str | None = None
        self.step = 0
        replay = payload.get("physics_replay") or {}
        self.physics_replay_mode = replay.get("mode", "policy_rate_endpoints")
        self.physics_steps_per_policy = int(replay.get("maximum_targets_per_policy_step", 0))
        if self.physics_replay_mode == "exact_trace":
            if replay.get("unit") != "radian":
                raise ValueError("exact expert replay physics unit must be radian")
            if self.physics_steps_per_policy < 1:
                raise ValueError("expert replay physics cadence is invalid")
            self._validate_exact_groups(scenes)
        elif self.physics_replay_mode != "policy_rate_endpoints":
            raise ValueError("unsupported expert replay physics mode")

    def _validate_exact_groups(self, scenes: list[dict[str, Any]]) -> None:
        for scene in scenes:
            actions = scene.get("actions_calibrated") or []
            groups = scene.get("physics_action_groups_radians")
            if not isinstance(groups, list):
                raise ValueError("expert replay requires exact physics action groups")
            if len(groups) != len(actions):
                raise ValueError("expert replay physics groups do not match policy actions")
            for group in groups:
                values = np.asarray(group, dtype=np.float64)
                if (
                    values.ndim != 2
                    or values.shape[1:] != (6,)
                    or not 1 <= values.shape[0] <= self.physics_steps_per_policy
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("expert replay contains an invalid physics action group")

    def reset(self, scene_id: str) -> None:
        if scene_id not in self.scenes:
            raise KeyError(f"scene is absent from expert replay: {scene_id}")
        self.scene_id = scene_id
        self.step = 0

    def next_action(self) -> tuple[np.ndarray, dict[str, Any]]:
        if self.scene_id is None:
            raise RuntimeError("expert replay was not reset for a scene")
        scene = self.scenes[self.scene_id]
        actions = scene["actions_calibrated"]
        if not actions:
            raise RuntimeError(f"expert replay scene has no actions: {self.scene_id}")
        source_step = min(self.step, len(actions) - 1)
        exhausted = self.step >= len(actions)
        action = np.asarray(actions[source_step], dtype=np.float32)
        self.step += 1
        execution: dict[str, Any] = {
            "source": "dataset_replay",
            "source_step": source_step,
            "source_steps": len(actions),
            "source_exhausted": exhausted,
        }
        if self.physics_replay_mode == "exact_trace":
            groups = scene["physics_action_groups_radians"]
            physics_actions = (
                [groups[-1][-1]] * self.physics_steps_per_policy
                if exhausted
                else groups[source_step]
            )
            if exhausted:
                # A source policy action denotes the first target in its
                # physics group.  After exhaustion we instead hold the last
                # target, so return the calibrated form of that held target
                # to keep the policy and physics commands identical.
                action = radians_to_lerobot(
                    np.asarray(groups[-1][-1], dtype=np.float64), clip=True
                ).astype(np.float32)
            execution.update(
                {
                    "physics_action_source": "exact_trace",
                    "physics_actions_radians": physics_actions,
                }
            )
        return action, execution
