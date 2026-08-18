"""Read-only Dashboard index for immutable policy rollout evidence."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _layout_paths(root: Path) -> tuple[Path, Path, Path] | None:
    """Resolve supported immutable rollout layouts without modifying evidence."""
    legacy = (
        root / "spec.json",
        root / "run" / "report.json",
        root / "run" / "episodes",
    )
    if legacy[0].is_file() and legacy[1].is_file():
        return legacy
    flat = (
        root.parent / "specs" / f"{root.name}.spec.json",
        root / "report.json",
        root / "episodes",
    )
    if flat[0].is_file() and flat[1].is_file():
        return flat
    return None


class PolicyRolloutDashboardIndex:
    """Discover rollout suites without mutating their frozen evidence."""

    def __init__(self, roots: list[str | Path]):
        self.roots = tuple(dict.fromkeys(Path(root).resolve() for root in roots))

    def _runs(self) -> list[Path]:
        runs: list[Path] = []
        for root in self.roots:
            if not root.is_dir():
                continue
            candidates = [root] if _layout_paths(root) is not None else root.iterdir()
            for candidate in candidates:
                if (
                    candidate.is_dir()
                    and SAFE_ID.fullmatch(candidate.name)
                    and _layout_paths(candidate) is not None
                ):
                    runs.append(candidate.resolve())
        return sorted(set(runs), key=lambda path: path.stat().st_mtime, reverse=True)

    def _record(self, root: Path, *, include_episodes: bool) -> dict[str, Any]:
        layout = _layout_paths(root)
        if layout is None:
            raise FileNotFoundError(root)
        spec_path, report_path, _ = layout
        spec = _read_json(spec_path)
        report = _read_json(report_path)
        if report.get("suite_id") != spec.get("suite_id"):
            raise ValueError(f"rollout suite identity mismatch: {root}")
        source = report.get("holdout_source") or spec.get("holdout_source") or {}
        acceptance = report.get("acceptance") or {}
        acceptance_contract = spec.get("acceptance") or {}
        record: dict[str, Any] = {
            "rollout_id": root.name,
            "suite_id": report["suite_id"],
            "status": report.get("status", "UNKNOWN"),
            "created_at": report.get("created_at"),
            "task_id": (spec.get("task") or {}).get("task_id"),
            "evaluation_class": (spec.get("task") or {}).get("evaluation_class"),
            "checkpoint_step": (report.get("checkpoint") or {}).get("step"),
            "model_sha256": (report.get("checkpoint") or {}).get("model_sha256"),
            "rollout_git_commit": report.get("rollout_git_commit"),
            "campaign_id": source.get("campaign_id"),
            "completed_episodes": acceptance.get("completed_episodes", 0),
            "task_successes": acceptance.get("task_successes", 0),
            "task_success_rate": acceptance.get("task_success_rate", 0.0),
            "stage_progress": acceptance.get("stage_progress") or {},
            "terminal_reason_counts": acceptance.get("terminal_reason_counts") or {},
            "acceptance_errors": acceptance.get("acceptance_errors") or [],
            "nonfinite_action_count": acceptance.get("nonfinite_action_count", 0),
            "hard_range_violation_count": acceptance.get("hard_range_violation_count", 0),
            "maximum_hard_range_excess_calibrated": acceptance.get(
                "maximum_hard_range_excess_calibrated", 0.0
            ),
            "maximum_hard_range_excess_limit_calibrated": acceptance_contract.get(
                "maximum_hard_range_excess_calibrated"
            ),
            "detail_url": f"/api/policy-rollouts/{quote(root.name, safe='')}",
        }
        if not include_episodes:
            return record
        scenes = {scene["scene_id"]: scene for scene in spec.get("scenes", [])}
        episodes = []
        for result in report.get("episodes", []):
            scene_id = result["scene_id"]
            scene = scenes.get(scene_id, {})
            videos = {}
            for camera_id, evidence in (result.get("videos") or {}).items():
                relative = Path(str(evidence.get("path", "")))
                expected = Path("episodes") / scene_id / f"{camera_id}.mp4"
                if relative != expected:
                    continue
                videos[camera_id] = {
                    **evidence,
                    "url": (
                        f"/policy-rollouts/{quote(root.name, safe='')}/episodes/"
                        f"{quote(scene_id, safe='')}/{quote(camera_id, safe='')}.mp4"
                    ),
                }
            episodes.append(
                {
                    "scene_id": scene_id,
                    "task_success": bool(result.get("task_success")),
                    "terminal_reason": result.get("terminal_reason"),
                    "policy_steps": result.get("policy_steps"),
                    "object_variant_id": scene.get("object_variant_id"),
                    "region_band": scene.get("region_band"),
                    "yaw_stratum_id": scene.get("yaw_stratum_id"),
                    "yaw_degrees": scene.get("yaw_degrees"),
                    "stage_evidence": result.get("stage_evidence") or {},
                    "maximum_hard_range_excess_calibrated": result.get(
                        "maximum_hard_range_excess_calibrated", 0.0
                    ),
                    "videos": videos,
                }
            )
        record["episodes"] = episodes
        return record

    def list_rollouts(self) -> list[dict[str, Any]]:
        records = []
        for root in self._runs():
            try:
                records.append(self._record(root, include_episodes=False))
            except (OSError, KeyError, TypeError, ValueError):
                continue
        return records

    def detail(self, rollout_id: str) -> dict[str, Any]:
        root = self.rollout_root(rollout_id)
        return self._record(root, include_episodes=True)

    def rollout_root(self, rollout_id: str) -> Path:
        if not SAFE_ID.fullmatch(rollout_id or ""):
            raise FileNotFoundError(rollout_id)
        for root in self._runs():
            if root.name == rollout_id:
                return root
        raise FileNotFoundError(rollout_id)

    def video_path(self, rollout_id: str, scene_id: str, camera_id: str) -> Path:
        if not SAFE_ID.fullmatch(scene_id or "") or camera_id not in {"front", "wrist"}:
            raise FileNotFoundError(scene_id)
        root = self.rollout_root(rollout_id)
        layout = _layout_paths(root)
        if layout is None:
            raise FileNotFoundError(root)
        _, _, episodes_root = layout
        detail = self._record(root, include_episodes=True)
        episode = next((row for row in detail["episodes"] if row["scene_id"] == scene_id), None)
        if episode is None or camera_id not in episode["videos"]:
            raise FileNotFoundError(scene_id)
        path = (episodes_root / scene_id / f"{camera_id}.mp4").resolve()
        evidence_root = (episodes_root / scene_id).resolve()
        if evidence_root not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        return path
