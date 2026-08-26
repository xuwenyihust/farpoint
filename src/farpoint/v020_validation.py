"""Release-independent selection gates for the SO-101 v0.2.0 candidate."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
from typing import Any


SCHEMA = "farpoint.so101-v020-candidate-validation.v1"


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(row for row in root.rglob("*") if row.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_v020_selection(
    plan: dict[str, Any] | list[dict[str, Any]], selection: dict[str, Any]
) -> list[str]:
    errors = []
    episodes = selection.get("episodes") or []
    if len(episodes) != 300:
        errors.append(f"selected_episode_count:{len(episodes)}!=300")
    episode_dirs = [row.get("episode_dir") for row in episodes]
    if len(set(episode_dirs)) != len(episode_dirs):
        errors.append("selected_episode_paths_not_unique")
    plans = plan if isinstance(plan, list) else [plan]
    trials = {
        row["variation_id"]: row
        for segment_plan in plans
        for row in segment_plan.get("trials") or []
    }
    split_counts = Counter()
    cell_counts = Counter()
    strata = defaultdict(lambda: defaultdict(set))
    for index, episode in enumerate(episodes):
        variation_id = episode.get("variation_id")
        trial = trials.get(variation_id)
        if trial is None:
            errors.append(f"episode[{index}]:variation_outside_v020_plan")
            continue
        if (trial.get("continuation_provenance") or {}).get("source_plan_sha256") not in {
            None,
            *{segment_plan.get("plan_sha256") for segment_plan in plans},
        }:
            errors.append(f"episode[{index}]:continuation_source_mismatch")
        split_counts[trial["split"]] += 1
        cell = (
            trial["object_variant_id"],
            trial["target_profile_id"],
            trial["camera_profile_id"],
        )
        cell_counts[cell] += 1
        sample = (trial.get("sampler") or {}).get("resolved") or {}
        for axis, stratum in (sample.get("strata") or {}).items():
            strata[cell][axis].add(int(stratum))
        values = trial.get("variation_resolved") or {}
        position = values.get("position_xy_m") or [math.nan, math.nan]
        yaw = float(values.get("yaw_degrees", math.nan))
        if not all(math.isfinite(float(value)) for value in (*position, yaw)):
            errors.append(f"episode[{index}]:nonfinite_continuous_variation")
        if not 0.14 <= float(position[0]) <= 0.26 or not -0.12 <= float(position[1]) <= -0.02:
            errors.append(f"episode[{index}]:position_outside_frozen_bounds")
        if not 0.0 <= yaw < 90.0:
            errors.append(f"episode[{index}]:yaw_outside_frozen_bounds")
    if split_counts != Counter({"train": 270, "validation": 30}):
        errors.append(f"split_counts:{dict(split_counts)}")
    if len(cell_counts) != 30 or set(cell_counts.values()) != {10}:
        errors.append(f"cell_balance:{dict(cell_counts)}")
    # Exact LHS stratum coverage is mandatory for the initial plan. A repaired
    # replacement remains valid when it preserves bounds but is reported as a
    # coverage change for explicit release review.
    for cell, axes in strata.items():
        for axis, values in axes.items():
            if values != set(range(10)):
                errors.append(f"lhs_stratum_gap:{cell}:{axis}:{sorted(values)}")
    return errors


def build_v020_candidate_validation(
    plan: dict[str, Any] | list[dict[str, Any]],
    selection: dict[str, Any],
    *,
    candidate_root: str | Path | None = None,
    lerobot_validation: dict[str, Any] | None = None,
    loader_replays: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    plans = plan if isinstance(plan, list) else [plan]
    errors = validate_v020_selection(plans, selection)
    primary = plans[0]
    if lerobot_validation is not None:
        native_valid = lerobot_validation.get("valid") is True
        bound_pass = lerobot_validation.get("status") == "PASS"
        if not native_valid and not bound_pass:
            errors.append("lerobot_validation_not_pass")
    replay_rows = loader_replays or []
    replay_groups = {
        (row.get("object_variant_id"), row.get("target_profile_id"), row.get("camera_profile_id"))
        for row in replay_rows
        if row.get("status") == "PASS"
    }
    if loader_replays is not None and len(replay_groups) < 3:
        errors.append("cross_group_loader_replay_below_three")
    root = Path(candidate_root) if candidate_root is not None else None
    if root is not None and not root.is_dir():
        errors.append("candidate_root_missing")
    return {
        "schema_version": SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "plan_sha256": primary.get("plan_sha256"),
        "segment_plan_sha256s": [row.get("plan_sha256") for row in plans],
        "campaign_sha256": primary.get("campaign_sha256"),
        "candidate_tree_sha256": _tree_sha256(root) if root is not None and root.is_dir() else None,
        "selected_episode_count": len(selection.get("episodes") or []),
        "errors": sorted(set(errors)),
        "lerobot_validation": lerobot_validation,
        "loader_replays": replay_rows,
    }
