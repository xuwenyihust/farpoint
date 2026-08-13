"""Build auditable expert replays for exported recovery demonstrations."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from farpoint.contracts import validate_contract, validate_episode_semantics
from farpoint.demonstration import state_snapshot_sha256
from farpoint.policy_rollout import load_rollout_spec
from farpoint.policy_training import file_sha256
from farpoint.so101 import USD_MAX_DEGREES, USD_MIN_DEGREES, radians_to_lerobot


def _select_evenly(values: list[Any], count: int) -> list[Any]:
    if not 1 <= count <= len(values):
        raise ValueError("scene_count must select at least one available recovery episode")
    indexes = [math.floor(index * len(values) / count) for index in range(count)]
    return [values[index] for index in indexes]


def _pick_object(metadata: dict[str, Any]) -> dict[str, Any]:
    matches = [
        entity
        for entity in (metadata.get("scene") or {}).get("entities") or []
        if entity.get("entity_id") == "pick_object"
    ]
    if len(matches) != 1:
        raise ValueError("recovery episode must contain one pick_object entity")
    return matches[0]


def _relative_artifact(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("recovery command trace path must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / path).resolve()
    if resolved_root not in resolved.parents:
        raise ValueError("recovery command trace escapes its episode root")
    return resolved


def _physics_action_groups(
    root: Path,
    metadata: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    physics_hz: int,
    physics_steps_per_policy: int,
) -> tuple[list[list[list[float]]], dict[str, Any], int]:
    intervention = (metadata.get("demonstration") or {}).get("intervention") or {}
    descriptor = intervention.get("command_trace")
    if not isinstance(descriptor, dict):
        raise ValueError(f"recovery episode lacks an audited command trace: {root}")
    if descriptor.get("control_hz") != physics_hz:
        raise ValueError("recovery command trace control_hz mismatch")
    trace_path = _relative_artifact(root, str(descriptor.get("path") or ""))
    if file_sha256(trace_path) != descriptor.get("sha256"):
        raise ValueError(f"recovery command trace hash mismatch: {root}")
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != descriptor.get("sample_count") or not rows:
        raise ValueError("recovery command trace sample count mismatch")
    if [row.get("control_step") for row in rows] != list(range(len(rows))):
        raise ValueError("recovery command trace control steps are not contiguous")
    radians_actions = []
    clipped = 0
    for row in rows:
        radians = np.asarray(row.get("action_joint_positions"), dtype=np.float64)
        if radians.shape != (6,) or not np.isfinite(radians).all():
            raise ValueError("recovery command trace contains an invalid action")
        degrees = np.rad2deg(radians)
        clipped += int(np.count_nonzero((degrees < USD_MIN_DEGREES) | (degrees > USD_MAX_DEGREES)))
        radians_actions.append(radians.tolist())
    groups = []
    for index, observation in enumerate(observations):
        start = observation.get("control_step")
        if start != index * physics_steps_per_policy:
            raise ValueError("recovery observation control steps do not match policy cadence")
        end = (
            observations[index + 1]["control_step"] if index + 1 < len(observations) else len(rows)
        )
        if not 1 <= end - start <= physics_steps_per_policy:
            raise ValueError("recovery command trace cannot be grouped at policy cadence")
        source_action = np.asarray(observation.get("action_joint_positions"), dtype=np.float64)
        trace_action = np.asarray(rows[start]["action_joint_positions"], dtype=np.float64)
        if source_action.shape != (6,) or not np.allclose(
            source_action, trace_action, rtol=0.0, atol=1e-7
        ):
            raise ValueError("recovery observation action does not match command trace")
        groups.append(radians_actions[start:end])
    return groups, deepcopy(descriptor), clipped


def build_recovery_replay(
    selection: dict[str, Any],
    template: dict[str, Any],
    runtime: dict[str, Any],
    *,
    selection_sha256: str,
    scene_count: int,
    suite_id: str,
    action_safety_calibration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze state-restored replay scenes and their exported action streams."""
    if selection.get("schema_version") != "farpoint.export-selection.v1":
        raise ValueError("unsupported recovery selection schema")
    physics_hz = int(runtime["control"]["physics_hz"])
    policy_hz = int(runtime["control"]["policy_hz"])
    if physics_hz % policy_hz:
        raise ValueError("recovery replay physics_hz must be divisible by policy_hz")
    physics_steps_per_policy = physics_hz // policy_hz
    episodes = selection.get("episodes") or []
    chosen = _select_evenly(episodes, scene_count)
    reference_minimum = int(
        action_safety_calibration["reference_minimum_delta_limited_actions_per_episode"]
    )
    reference_maximum = int(
        action_safety_calibration["reference_maximum_delta_limited_actions_per_episode"]
    )
    allowed_maximum = int(
        action_safety_calibration["allowed_maximum_delta_limited_actions_per_episode"]
    )
    if not 0 <= reference_minimum <= reference_maximum <= allowed_maximum:
        raise ValueError("invalid recovery replay action-safety calibration bounds")
    if len({row["episode_dir"] for row in chosen}) != len(chosen):
        raise ValueError("recovery replay selected duplicate episodes")
    spec_scenes = []
    replay_scenes = []
    maximum_actions = 0
    for index, selected in enumerate(chosen):
        root = Path(selected["episode_dir"])
        metadata_path = root / "metadata.json"
        observations_path = root / "observations.jsonl"
        handoff_path = root / "handoff.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        errors = [*validate_contract(metadata), *validate_episode_semantics(metadata)]
        if errors:
            raise ValueError(f"invalid recovery metadata {root}: {'; '.join(errors)}")
        demonstration = metadata.get("demonstration") or {}
        if demonstration.get("type") != "recovery":
            raise ValueError(f"selection contains a non-recovery episode: {root}")
        if (metadata.get("identity") or {}).get("split") != "train":
            raise ValueError("recovery replay may only consume training episodes")
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        snapshot = handoff["state_snapshot"]
        expected_snapshot_sha = demonstration["intervention"]["handoff"]["state_snapshot_sha256"]
        if state_snapshot_sha256(snapshot) != expected_snapshot_sha:
            raise ValueError(f"recovery handoff snapshot hash mismatch: {root}")
        observation_rows = [
            json.loads(line) for line in observations_path.read_text(encoding="utf-8").splitlines()
        ]
        physics_action_groups, command_trace, physics_clipped = _physics_action_groups(
            root,
            metadata,
            observation_rows,
            physics_hz=physics_hz,
            physics_steps_per_policy=physics_steps_per_policy,
        )
        actions = []
        phases = []
        clipped = 0
        for row in observation_rows:
            radians = np.asarray(row["action_joint_positions"], dtype=np.float64)
            if radians.shape != (6,) or not np.isfinite(radians).all():
                raise ValueError(f"invalid recovery action: {observations_path}")
            degrees = np.rad2deg(radians)
            clipped += int(
                np.count_nonzero((degrees < USD_MIN_DEGREES) | (degrees > USD_MAX_DEGREES))
            )
            actions.append(radians_to_lerobot(radians, clip=True).tolist())
            phases.append(str(row.get("phase", "unknown")))
        if not actions:
            raise ValueError(f"recovery episode contains no actions: {root}")
        maximum_actions = max(maximum_actions, len(actions))
        obj = _pick_object(metadata)
        scene_id = f"recovery_replay_{index:02d}_{expected_snapshot_sha[:8]}"
        spec_scenes.append(
            {
                "scene_id": scene_id,
                # The collector seeds Isaac/PhysX with the attempt seed.  A
                # variation seed identifies the frozen scene, but it is not
                # the simulator reset seed and can initialize different
                # solver state.  State-restored replay must reproduce the
                # original reset before applying the handoff snapshot.
                "seed": int((metadata.get("identity") or {})["attempt_seed"]),
                "object_variant_id": (
                    ((metadata.get("scene") or {}).get("object_variant") or {}).get("resolved", {})
                ).get("variant_id"),
                "region_band": ((metadata.get("variation") or {}).get("resolved") or {}).get(
                    "region_band"
                ),
                "yaw_stratum_id": ((metadata.get("variation") or {}).get("resolved") or {}).get(
                    "yaw_stratum_id"
                ),
                "yaw_degrees": float(
                    ((metadata.get("variation") or {}).get("resolved") or {})["yaw_degrees"]
                ),
                "object": {
                    "shape": obj["geometry"]["shape"],
                    "dimensions_m": obj["geometry"]["dimensions_m"],
                    "position_m": obj["pose"]["position_m"],
                    "orientation_xyzw": obj["pose"]["orientation_xyzw"],
                    "rgba": obj["appearance"]["rgba"],
                    "mass_kg": obj["physics"]["mass_kg"],
                },
                "initial_state": {"snapshot_sha256": expected_snapshot_sha, **snapshot},
            }
        )
        replay_scenes.append(
            {
                "scene_id": scene_id,
                "source_recovery_episode_id": metadata["identity"]["episode_id"],
                "source_metadata_sha256": file_sha256(metadata_path),
                "source_handoff_sha256": file_sha256(handoff_path),
                "source_observations_sha256": file_sha256(observations_path),
                "source_command_trace": command_trace,
                "state_snapshot_sha256": expected_snapshot_sha,
                "actions_calibrated": actions,
                "physics_action_groups_radians": physics_action_groups,
                "phases": phases,
                "source_values_clipped_by_exporter": clipped,
                "source_physics_values_clipped_by_exporter": physics_clipped,
            }
        )
    spec = deepcopy(template)
    spec["schema_version"] = "farpoint.policy-rollout.v1"
    spec["suite_id"] = suite_id
    spec["task"]["evaluation_class"] = "recovery_expert_replay"
    spec.pop("holdout_source", None)
    spec["recovery_replay_source"] = {
        "campaign_id": selection["collection_id"],
        "selection_sha256": selection_sha256,
        "evaluated_episode_count": len(spec_scenes),
        "state_restore": "handoff_snapshot_v1",
        "command_replay": "physics_rate_trace_v1",
        "action_safety_calibration": deepcopy(action_safety_calibration),
    }
    spec["control"] = {
        **deepcopy(runtime["control"]),
        "max_policy_steps": maximum_actions + 60,
        "stable_steps": 15,
    }
    spec["acceptance"] = {
        "required_completed_episodes": len(spec_scenes),
        "minimum_task_successes": len(spec_scenes),
        "maximum_nonfinite_actions": 0,
        "maximum_hard_range_violations": 0,
        "maximum_hard_range_excess_calibrated": 0.0,
        "maximum_delta_limited_actions": allowed_maximum * len(spec_scenes),
    }
    spec["scenes"] = spec_scenes
    replay = {
        "schema_version": "farpoint.expert-action-replay.v1",
        "dataset_revision": "v0.1.1-local-candidate",
        "camera_features": deepcopy(spec["environment"]["camera_features"]),
        "action_conversion": {
            "source_unit": "radian",
            "output_unit": "so101_calibrated_position",
            "clip_to_calibrated_range": True,
        },
        "physics_replay": {
            "mode": "exact_trace",
            "unit": "radian",
            "physics_hz": physics_hz,
            "policy_hz": policy_hz,
            "maximum_targets_per_policy_step": physics_steps_per_policy,
        },
        "source": deepcopy(spec["recovery_replay_source"]),
        "scenes": replay_scenes,
    }
    errors = validate_contract(spec)
    if errors:
        raise ValueError("invalid recovery replay spec:\n" + "\n".join(errors))
    return spec, replay


def write_recovery_replay_bundle(
    selection_path: Path,
    template_path: Path,
    runtime_path: Path,
    output_root: Path,
    *,
    scene_count: int,
    suite_id: str,
    action_safety_calibration: dict[str, Any],
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    template = json.loads(template_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    spec, replay = build_recovery_replay(
        selection,
        template,
        runtime,
        selection_sha256=file_sha256(selection_path),
        scene_count=scene_count,
        suite_id=suite_id,
        action_safety_calibration=action_safety_calibration,
    )
    output_root.mkdir(parents=True)
    spec_path = output_root / "spec.json"
    replay_path = output_root / "replay-manifest.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    replay_path.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")
    load_rollout_spec(spec_path)
    return {
        "spec": str(spec_path),
        "spec_sha256": file_sha256(spec_path),
        "replay_manifest": str(replay_path),
        "replay_manifest_sha256": file_sha256(replay_path),
        "scene_count": len(spec["scenes"]),
    }
