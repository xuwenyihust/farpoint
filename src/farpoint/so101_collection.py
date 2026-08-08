"""Resumable, coverage-preserving collection state for SO-101 demonstrations."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import signal
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "farpoint.collection.v2"
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


class CollectionSignalAbort(BaseException):
    """Signal-safe escape used to unwind Isaac collection on SIGTERM."""

    def __init__(self, signum: int):
        self.signum = int(signum)
        super().__init__(signal.Signals(self.signum).name)


def raise_collection_signal_abort(signum: int, _frame: Any = None) -> None:
    raise CollectionSignalAbort(signum)


def collection_interruption_reason(error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "SIGINT"
    if isinstance(error, CollectionSignalAbort):
        return signal.Signals(error.signum).name
    raise ValueError("unsupported collection interruption")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def episode_id_for_attempt(collection_id: str, attempt_id: str) -> str:
    """Return a collection-scoped, path-safe episode identifier.

    Attempt identifiers repeat when a frozen plan is rerun.  Including the
    collection identifier keeps those independent runs distinct in the raw
    episode store and in registries whose primary key is ``episode_id``.
    """
    for label, value in (("collection_id", collection_id), ("attempt_id", attempt_id)):
        if not value or _IDENTIFIER_PATTERN.fullmatch(value) is None:
            raise ValueError(
                f"{label} must contain only letters, numbers, '.', '_' or '-'"
            )
    return f"episode_{collection_id}__{attempt_id}"


def build_attempt_run_state(
    attempt: dict[str, Any], *, collection_id: str, git_commit: str
) -> dict[str, Any]:
    """Build the live sidecar used before final episode metadata is available."""
    return {
        "schema_version": "farpoint.episode-run.v1",
        "execution_status": "RUNNING",
        "identity": {
            "episode_id": episode_id_for_attempt(collection_id, attempt["attempt_id"]),
            "trial_id": attempt["trial_id"],
            "task_id": "so101_cube_pick_place",
            "split": attempt["split"],
            "episode_seed": int(
                attempt.get("environment_seed", attempt["attempt_seed"])
            )
            % (2**32),
        },
        "provenance": {
            "collection_id": collection_id,
            "git_commit": git_commit,
        },
        "variation": {
            "variation_id": attempt["variation_id"],
            "split": attempt["split"],
            "varied_axes": copy.deepcopy(attempt["varied_axes"]),
            "frozen_axes": copy.deepcopy(attempt["frozen_axes"]),
            "requested": copy.deepcopy(attempt.get("requested") or {}),
            "resolved": copy.deepcopy(attempt.get("resolved") or {}),
        },
        "recording": {
            "cameras": ["observation.images.front"],
            "frame_count": 0,
        },
        "outcome": {
            "success": None,
            "dataset_valid": False,
            "failure_category": None,
            "failure_reason": None,
        },
    }


def _new_manifest(
    plan: dict[str, Any],
    *,
    collection_id: str,
    git_commit: str,
    required_successes: int,
    maximum_attempts: int,
    release_status: str,
    completion_policy: str = "success_target",
    stop_when_success_target_unreachable: bool = True,
) -> dict[str, Any]:
    if completion_policy not in {"success_target", "all_planned_trials"}:
        raise ValueError("unsupported completion_policy")
    return {
        "schema_version": SCHEMA_VERSION,
        "collection_id": collection_id,
        "task_id": plan["task_id"],
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "git_commit": git_commit,
        "required_successes": required_successes,
        "maximum_attempts": maximum_attempts,
        "attempts": [],
        "selected_variations": {},
        "execution_status": "RUNNING",
        "quality_status": "NOT_EVALUATED",
        "release_status": release_status,
        "completion_policy": completion_policy,
        "stop_when_success_target_unreachable": bool(
            stop_when_success_target_unreachable
        ),
        "created_at": _now(),
        "updated_at": _now(),
    }


def create_manifest(
    plan: dict[str, Any],
    *,
    collection_id: str,
    git_commit: str,
    maximum_attempts: int = 150,
) -> dict[str, Any]:
    trials = plan.get("trials") or []
    profile = plan.get("collection") or {}
    if profile:
        if profile.get("kind") not in {
            "mirrored_mass_success_collection",
            "balanced_yaw_success_collection",
        }:
            raise ValueError("unsupported SO-101 collection profile")
        required_successes = int(profile.get("required_successes", 0))
        frozen_maximum = int(profile.get("maximum_attempts", 0))
        if required_successes != len(trials) or required_successes <= 0:
            raise ValueError("mirrored mass collection must require every planned variation")
        if frozen_maximum < required_successes:
            raise ValueError("mirrored mass collection attempt budget is unreachable")
        if maximum_attempts != frozen_maximum:
            raise ValueError("maximum_attempts does not match the frozen collection profile")
        release_status = "CANDIDATE"
    else:
        if len(trials) != 100:
            raise ValueError("SO-101 collection requires exactly 100 planned variations")
        if maximum_attempts < len(trials):
            raise ValueError("maximum_attempts cannot be less than the planned variation count")
        required_successes = 100
        frozen_maximum = maximum_attempts
        release_status = "PILOT"
    manifest = _new_manifest(
        plan,
        collection_id=collection_id,
        git_commit=git_commit,
        required_successes=required_successes,
        maximum_attempts=frozen_maximum,
        release_status=release_status,
    )
    if profile:
        manifest["collection_profile"] = copy.deepcopy(profile)
    return manifest


def create_pilot_manifest(
    plan: dict[str, Any], *, collection_id: str, git_commit: str
) -> dict[str, Any]:
    """Create a bounded manifest from a frozen stratified or diagnostic pilot."""
    trials = plan.get("trials") or []
    pilot = plan.get("pilot") or {}
    required_successes = int(pilot.get("required_successes", 0))
    maximum_attempts = int(pilot.get("maximum_attempts", 0))
    kind = pilot.get("kind")
    if kind in {"targeted_mass_diagnostic_pilot", "targeted_yaw_pilot"}:
        frozen_ids = pilot.get("trial_ids") or []
        if len(trials) != 100:
            raise ValueError("targeted pilot plan must retain all 100 variations")
        if not 0 < required_successes <= maximum_attempts:
            raise ValueError("targeted pilot success threshold is invalid")
        if maximum_attempts != len(frozen_ids):
            raise ValueError("targeted pilot attempt budget must match its trial ids")
        if [trial["trial_id"] for trial in trials[:maximum_attempts]] != frozen_ids:
            raise ValueError("targeted pilot ordering does not match its frozen ids")
        masses = {
            float(trial["resolved"]["mass_kg"])
            for trial in trials[:maximum_attempts]
        }
        if kind == "targeted_mass_diagnostic_pilot":
            target_mass = float(pilot.get("target_mass_kg", 0.0))
            if target_mass <= 0.0 or masses != {target_mass}:
                raise ValueError("targeted pilot mass is inconsistent")
        else:
            if maximum_attempts != 12 or required_successes != 10:
                raise ValueError("yaw pilot must freeze a 10-of-12 acceptance gate")
            if masses != {0.03, 0.04}:
                raise ValueError("yaw pilot masses are inconsistent")
            expected_orientation = pilot.get("orientation_xyzw")
            if any(
                trial["resolved"]["orientation_xyzw"] != expected_orientation
                for trial in trials[:maximum_attempts]
            ):
                raise ValueError("yaw pilot orientation is inconsistent")
        return _new_manifest(
            plan,
            collection_id=collection_id,
            git_commit=git_commit,
            required_successes=required_successes,
            maximum_attempts=maximum_attempts,
            release_status="PILOT",
            completion_policy="all_planned_trials",
            stop_when_success_target_unreachable=False,
        )

    primary_ids = pilot.get("primary_trial_ids") or []
    fallback_ids = pilot.get("fallback_trial_ids") or []
    if len(trials) != 100:
        raise ValueError("SO-101 pilot plan must retain all 100 variations")
    if kind != "stratified_success_pilot":
        raise ValueError("unsupported SO-101 pilot kind")
    if required_successes != 10 or len(primary_ids) != required_successes:
        raise ValueError("SO-101 pilot must freeze exactly 10 primary successes")
    if maximum_attempts != 15 or len(primary_ids) + len(fallback_ids) != maximum_attempts:
        raise ValueError("SO-101 pilot must freeze exactly 15 maximum attempts")
    frozen_ids = list(primary_ids) + list(fallback_ids)
    if [trial["trial_id"] for trial in trials[:maximum_attempts]] != frozen_ids:
        raise ValueError("SO-101 pilot trial ordering does not match its frozen ids")
    return _new_manifest(
        plan,
        collection_id=collection_id,
        git_commit=git_commit,
        required_successes=required_successes,
        maximum_attempts=maximum_attempts,
        release_status="PILOT",
    )


def create_gate_manifest(
    plan: dict[str, Any], *, collection_id: str, git_commit: str
) -> dict[str, Any]:
    gate = plan.get("gate") or {}
    trials = plan.get("trials") or []
    required_successes = int(gate.get("required_successes", 0))
    maximum_attempts = int(gate.get("maximum_attempts", 0))
    kind = gate.get("kind")
    if kind == "fixed_cube_repeatability":
        if required_successes != len(trials) or maximum_attempts != len(trials):
            raise ValueError("fixed gate must require one success for every frozen trial")
    elif kind == "cube_workspace_matrix":
        minimum_success_rate = float(gate.get("minimum_success_rate", 0.0))
        if not 0.0 < minimum_success_rate <= 1.0:
            raise ValueError("matrix gate minimum_success_rate must be in (0, 1]")
        if maximum_attempts != len(trials):
            raise ValueError("matrix gate must run each frozen trial exactly once")
        if required_successes != math.ceil(minimum_success_rate * len(trials)):
            raise ValueError("matrix gate required_successes does not match its threshold")
    elif kind == "cube_mass_feasibility":
        repetitions = int(gate.get("repetitions_per_mass", 0))
        minimum_per_mass = int(gate.get("minimum_successes_per_mass", 0))
        if repetitions <= 0 or len(trials) != repetitions * 2:
            raise ValueError("mass feasibility gate requires two trials per pair")
        if not 0 < minimum_per_mass <= repetitions:
            raise ValueError("invalid mass feasibility success threshold")
        if maximum_attempts != len(trials):
            raise ValueError("mass feasibility gate must run every frozen trial")
        if required_successes != minimum_per_mass * 2:
            raise ValueError("mass feasibility total threshold is inconsistent")
        masses = {
            float(trial["resolved"]["mass_kg"])
            for trial in trials
        }
        if masses != {
            float(gate["baseline_mass_kg"]),
            float(gate["candidate_mass_kg"]),
        }:
            raise ValueError("mass feasibility trials do not match frozen masses")
        pair_counts = Counter(trial.get("mass_pair_id") for trial in trials)
        if len(pair_counts) != repetitions or set(pair_counts.values()) != {2}:
            raise ValueError("mass feasibility pairs must contain two trials")
        for pair_id in pair_counts:
            pair = [trial for trial in trials if trial.get("mass_pair_id") == pair_id]
            if {trial.get("mass_role") for trial in pair} != {
                "baseline",
                "candidate",
            }:
                raise ValueError("mass feasibility pair roles are invalid")
            if len({trial.get("environment_seed") for trial in pair}) != 1:
                raise ValueError("mass feasibility pairs must share an environment seed")
        tolerance = float(gate.get("actual_mass_tolerance_kg", -1.0))
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("mass feasibility actual-mass tolerance is invalid")
        thresholds = gate.get("behavior_change_thresholds") or {}
        if set(thresholds) != {
            "action_path_relative",
            "mean_lift_bilateral_force_relative",
            "frame_count_absolute",
        } or any(float(value) < 0.0 for value in thresholds.values()):
            raise ValueError("mass feasibility behavior thresholds are invalid")
    elif kind == "cube_mass_workspace_pilot":
        if len(trials) != 5 or maximum_attempts != 5:
            raise ValueError("mass workspace pilot requires exactly five trials")
        minimum_successes = int(gate.get("minimum_successes", 0))
        if required_successes != minimum_successes or not 0 < minimum_successes <= 5:
            raise ValueError("mass workspace pilot success threshold is invalid")
        candidate_mass = float(gate.get("candidate_mass_kg", 0.0))
        if candidate_mass <= 0.0 or {
            float(trial["resolved"]["mass_kg"]) for trial in trials
        } != {candidate_mass}:
            raise ValueError("mass workspace pilot candidate mass is inconsistent")
        positions = [
            tuple(float(value) for value in trial["resolved"]["position_m"][:2])
            for trial in trials
        ]
        if len(set(positions)) != 5 or positions != [
            tuple(float(value) for value in position)
            for position in gate.get("positions_xy_m", [])
        ]:
            raise ValueError("mass workspace pilot positions are inconsistent")
        historical = (gate.get("historical_baseline") or {}).get("episodes") or []
        if len(historical) != 5 or any(
            trial.get("historical_baseline") != baseline
            for trial, baseline in zip(trials, historical)
        ):
            raise ValueError("mass workspace historical baselines are inconsistent")
    else:
        raise ValueError("unsupported gate kind")
    return _new_manifest(
        plan,
        collection_id=collection_id,
        git_commit=git_commit,
        required_successes=required_successes,
        maximum_attempts=maximum_attempts,
        release_status="EXPERIMENTAL",
        completion_policy="all_planned_trials",
    )


def load_manifest(path: str | Path, plan: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"collection manifest must use {SCHEMA_VERSION}")
    if manifest.get("plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("collection manifest does not match the variation plan")
    validate_manifest(manifest, plan)
    return manifest


def validate_manifest(manifest: dict[str, Any], plan: dict[str, Any]) -> None:
    known = {trial["variation_id"] for trial in plan["trials"]}
    selected = manifest.get("selected_variations") or {}
    unknown = set(selected) - known
    if unknown:
        raise ValueError(f"collection selected unknown variations: {sorted(unknown)}")
    attempts = manifest.get("attempts") or []
    completion_policy = manifest.get("completion_policy", "success_target")
    if completion_policy not in {"success_target", "all_planned_trials"}:
        raise ValueError("collection has unsupported completion_policy")
    if not isinstance(
        manifest.get("stop_when_success_target_unreachable", True), bool
    ):
        raise ValueError("collection stop-when-unreachable policy must be boolean")
    if len(attempts) > int(manifest["maximum_attempts"]):
        raise ValueError("collection exceeds its frozen maximum attempt budget")
    attempt_ids = [row.get("attempt_id") for row in attempts]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("collection attempt ids must be unique")
    for variation_id, attempt_id in selected.items():
        matches = [row for row in attempts if row.get("attempt_id") == attempt_id]
        if len(matches) != 1 or matches[0].get("variation_id") != variation_id:
            raise ValueError(f"invalid selected attempt for variation {variation_id}")
        if not matches[0].get("success") or not matches[0].get("dataset_valid"):
            raise ValueError(f"selected attempt is not eligible: {attempt_id}")


def next_attempt(manifest: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any] | None:
    """Return the next uncovered variation, preserving the original split."""
    validate_manifest(manifest, plan)
    completion_policy = manifest.get("completion_policy", "success_target")
    if (
        completion_policy == "success_target"
        and len(manifest["selected_variations"])
        == int(manifest["required_successes"])
    ):
        return None
    if len(manifest["attempts"]) >= int(manifest["maximum_attempts"]):
        return None
    counts = Counter(row["variation_id"] for row in manifest["attempts"])
    selected = manifest["selected_variations"]
    candidates = [trial for trial in plan["trials"] if trial["variation_id"] not in selected]
    # Preserve the frozen plan order within each retry round. Sorting by the
    # human-readable trial id silently defeated deliberately stratified or
    # diagnostic plan ordering (for example a large-cube-first smoke test).
    minimum_attempt_count = min(counts[trial["variation_id"]] for trial in candidates)
    trial = copy.deepcopy(
        next(
            trial
            for trial in candidates
            if counts[trial["variation_id"]] == minimum_attempt_count
        )
    )
    attempt_index = counts[trial["variation_id"]]
    attempt_id = f"{trial['trial_id']}__attempt{attempt_index:02d}"
    retry_material = {
        "plan_sha256": plan["plan_sha256"],
        "trial_id": trial["trial_id"],
        "attempt_index": attempt_index,
    }
    # Isaac Lab 3.0 forwards this value to NumPy's legacy RandomState API,
    # which accepts only unsigned 32-bit seeds.
    retry_seed = int.from_bytes(
        hashlib.sha256(
            json.dumps(retry_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()[:4],
        "big",
    )
    trial.update(
        {
            "attempt_id": attempt_id,
            "attempt_index": attempt_index,
            "attempt_seed": retry_seed,
            "varied_axes": copy.deepcopy(plan["varied_axes"]),
            "frozen_axes": copy.deepcopy(plan["frozen_axes"]),
        }
    )
    return trial


def record_attempt(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    attempt: dict[str, Any],
    *,
    episode_id: str | None,
    success: bool,
    dataset_valid: bool,
    failure_category: str | None = None,
    failure_reason: str | None = None,
) -> None:
    expected = next_attempt(manifest, plan)
    if expected is None or expected["attempt_id"] != attempt.get("attempt_id"):
        raise ValueError("attempt is not the next frozen collection attempt")
    row = {
        "attempt_id": attempt["attempt_id"],
        "trial_id": attempt["trial_id"],
        "variation_id": attempt["variation_id"],
        "split": attempt["split"],
        "attempt_index": attempt["attempt_index"],
        "attempt_seed": attempt["attempt_seed"],
        "episode_id": episode_id,
        "success": bool(success),
        "dataset_valid": bool(dataset_valid),
        "failure_category": failure_category,
        "failure_reason": failure_reason,
        "selected_for_dataset": bool(success and dataset_valid),
        "finished_at": _now(),
    }
    manifest["attempts"].append(row)
    if row["selected_for_dataset"]:
        manifest["selected_variations"][attempt["variation_id"]] = attempt["attempt_id"]
    eligible_successes = len(manifest["selected_variations"])
    exhausted = len(manifest["attempts"]) >= int(manifest["maximum_attempts"])
    completion_policy = manifest.get("completion_policy", "success_target")
    complete = (
        exhausted
        if completion_policy == "all_planned_trials"
        else eligible_successes == int(manifest["required_successes"])
    )
    if complete or exhausted:
        manifest["execution_status"] = "FINISHED"
        manifest["quality_status"] = (
            "PASS"
            if complete
            and eligible_successes >= int(manifest["required_successes"])
            else "FAIL"
        )
    manifest["updated_at"] = _now()
    validate_manifest(manifest, plan)


def abort_collection_manifest(manifest: dict[str, Any], reason: str) -> None:
    """Mark an in-progress collection terminal without scoring its quality."""
    message = str(reason).strip()
    if not message:
        raise ValueError("abort reason must be non-empty")
    status = manifest.get("execution_status")
    if status not in {"RUNNING", "ABORTED"}:
        raise ValueError(f"cannot abort collection in {status!r} state")
    manifest["execution_status"] = "ABORTED"
    manifest["quality_status"] = "NOT_EVALUATED"
    manifest["abort_reason"] = message
    manifest["aborted_at"] = _now()
    manifest["updated_at"] = manifest["aborted_at"]


def finish_diagnostic_manifest(
    manifest: dict[str, Any], diagnostic_name: str, *, succeeded: bool
) -> None:
    """Terminate a diagnostic-only manifest without scoring collection quality.

    Isaac diagnostics share the collector startup path so signal handling and
    environment construction remain identical to collection runs.  They do
    not execute plan attempts, however, and therefore must end as ``ABORTED``
    with ``NOT_EVALUATED`` quality instead of remaining spuriously ``RUNNING``
    or claiming a collection result.
    """
    name = str(diagnostic_name).strip()
    if not name:
        raise ValueError("diagnostic_name must be non-empty")
    outcome = "completed" if bool(succeeded) else "failed"
    abort_collection_manifest(manifest, f"diagnostic_{outcome}:{name}")


def abort_attempt_run_state(run_state: dict[str, Any], reason: str) -> None:
    """Mark a live episode sidecar as interrupted and ineligible for data."""
    message = str(reason).strip()
    if not message:
        raise ValueError("abort reason must be non-empty")
    status = run_state.get("execution_status")
    if status not in {"RUNNING", "ABORTED"}:
        raise ValueError(f"cannot abort attempt in {status!r} state")
    run_state["execution_status"] = "ABORTED"
    run_state["outcome"] = {
        "success": False,
        "dataset_valid": False,
        "failure_category": "interrupted",
        "failure_reason": message,
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def abort_collection_artifacts(
    manifest_path: str | Path,
    episodes_root: str | Path,
    reason: str,
) -> dict[str, Any]:
    """Atomically terminate live collection records without deleting artifacts."""
    manifest_destination = Path(manifest_path)
    manifest = json.loads(manifest_destination.read_text(encoding="utf-8"))
    previous_manifest_status = manifest.get("execution_status")
    abort_collection_manifest(manifest, reason)
    aborted_episode_ids = []
    for run_state_path in sorted(Path(episodes_root).glob("*/run-state.json")):
        run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
        if run_state.get("execution_status") != "RUNNING":
            continue
        abort_attempt_run_state(run_state, reason)
        write_manifest(run_state_path, run_state)
        aborted_episode_ids.append(
            (run_state.get("identity") or {}).get("episode_id")
            or run_state_path.parent.name
        )
    write_manifest(manifest_destination, manifest)
    return {
        "schema_version": "farpoint.collection-abort-record.v1",
        "collection_id": manifest.get("collection_id"),
        "reason": str(reason).strip(),
        "previous_manifest_status": previous_manifest_status,
        "execution_status": manifest["execution_status"],
        "completed_attempt_count": len(manifest.get("attempts") or []),
        "selected_variation_count": len(manifest.get("selected_variations") or {}),
        "aborted_episode_ids": aborted_episode_ids,
        "manifest_path": str(manifest_destination),
        "episodes_root": str(Path(episodes_root)),
    }


def build_export_selection(manifest: dict[str, Any], episodes_root: str = "outputs/episodes") -> dict:
    attempts = {row["attempt_id"]: row for row in manifest["attempts"]}
    selected = []
    for variation_id, attempt_id in sorted(manifest["selected_variations"].items()):
        row = attempts[attempt_id]
        selected.append(
            {
                "episode_dir": f"{episodes_root}/{row['episode_id']}",
                "trial_id": row["trial_id"],
                "variation_id": variation_id,
                "split": row["split"],
            }
        )
    return {
        "schema_version": "farpoint.export-selection.v1",
        "dataset_id": (manifest.get("collection_profile") or {}).get(
            "dataset_id", "farpoint-so101-cube-pick-place"
        ),
        "collection_id": manifest["collection_id"],
        "selection_policy": "one_success_per_stratified_variation",
        "episodes": selected,
    }
