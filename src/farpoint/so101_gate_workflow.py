"""Frozen stage ordering and admission checks for SO-101 oracle operations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from farpoint.so101_collection import validate_manifest
from farpoint.so101_gate import (
    build_cube_workspace_matrix_plan,
    build_fixed_cube_gate_plan,
)
from farpoint.so101_pilot import (
    build_so101_pilot_plan,
    build_so101_yaw_pilot_plan,
    build_targeted_mass_diagnostic_pilot_plan,
)
from farpoint.so101_mass_feasibility import (
    build_cube_mass_feasibility_plan,
    build_cube_mass_workspace_pilot_plan,
)
from farpoint.so101_watchdog import validate_watchdog_policy


WORKFLOW_SCHEMA_VERSION = "farpoint.so101-gate-workflow.v1"
WORKFLOW_CONFIG_SCHEMA_VERSION = "farpoint.so101-gate-workflow-config.v1"


def _sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_gate_workflow_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != WORKFLOW_CONFIG_SCHEMA_VERSION:
        raise ValueError(f"gate workflow config must use {WORKFLOW_CONFIG_SCHEMA_VERSION}")
    completion_status = config.get("completion_status", "READY_FOR_FORMAL_REVIEW")
    if not isinstance(completion_status, str) or not completion_status:
        raise ValueError("gate workflow completion_status must be non-empty")
    stages = config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("gate workflow config requires stages")
    stage_ids = [stage.get("stage_id") for stage in stages]
    if any(not isinstance(value, str) or not value for value in stage_ids):
        raise ValueError("every gate workflow stage requires a stage_id")
    if len(stage_ids) != len(set(stage_ids)):
        raise ValueError("gate workflow stage ids must be unique")
    supported = {
        "fixed_cube_repeatability",
        "cube_workspace_matrix",
        "cube_mass_feasibility",
        "cube_mass_workspace_pilot",
        "targeted_mass_diagnostic_pilot",
        "targeted_yaw_pilot",
        "stratified_success_pilot",
    }
    for stage in stages:
        if stage.get("kind") not in supported:
            raise ValueError(f"unsupported gate workflow stage: {stage.get('kind')}")


def build_so101_gate_workflow(
    workflow_config: dict[str, Any],
    variation_config: dict[str, Any],
    watchdog_policy: dict[str, Any],
    *,
    workflow_id: str,
    git_commit: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Freeze all gate plans and their order before any simulator work starts."""
    validate_gate_workflow_config(workflow_config)
    validate_watchdog_policy(watchdog_policy)
    if not workflow_id:
        raise ValueError("workflow_id must be non-empty")
    if not git_commit:
        raise ValueError("git_commit must be non-empty")
    plans: dict[str, dict[str, Any]] = {}
    stages = []
    for index, stage_config in enumerate(workflow_config["stages"], start=1):
        stage_id = stage_config["stage_id"]
        plan_id = f"{workflow_id}_{stage_id}"
        kind = stage_config["kind"]
        if kind == "fixed_cube_repeatability":
            plan = build_fixed_cube_gate_plan(
                variation_config,
                gate_id=plan_id,
                edge_m=float(stage_config["edge_m"]),
                position_xy_m=tuple(stage_config["position_xy_m"]),
                repetitions=int(stage_config["repetitions"]),
            )
            collector_mode = "gate"
            report_kind = "gate"
        elif kind == "cube_workspace_matrix":
            plan = build_cube_workspace_matrix_plan(
                variation_config,
                gate_id=plan_id,
                positions_xy_m=[tuple(position) for position in stage_config["positions_xy_m"]],
                minimum_success_rate=float(stage_config["minimum_success_rate"]),
            )
            collector_mode = "gate"
            report_kind = "gate"
        elif kind == "cube_mass_feasibility":
            plan = build_cube_mass_feasibility_plan(
                variation_config,
                profile_id=plan_id,
                baseline_mass_kg=float(stage_config.get("baseline_mass_kg", 0.04)),
                candidate_mass_kg=float(stage_config.get("candidate_mass_kg", 0.03)),
                edge_m=float(stage_config.get("edge_m", 0.03)),
                position_xy_m=tuple(stage_config.get("position_xy_m", (0.20, -0.095))),
                repetitions_per_mass=int(stage_config.get("repetitions_per_mass", 5)),
                minimum_successes_per_mass=int(stage_config.get("minimum_successes_per_mass", 4)),
            )
            collector_mode = "gate"
            report_kind = "mass_feasibility"
        elif kind == "cube_mass_workspace_pilot":
            plan = build_cube_mass_workspace_pilot_plan(
                variation_config,
                pilot_id=plan_id,
                candidate_mass_kg=float(stage_config["candidate_mass_kg"]),
                edge_m=float(stage_config["edge_m"]),
                historical_baseline_commit=str(stage_config["historical_baseline_commit"]),
                historical_baseline_collection_id=str(
                    stage_config["historical_baseline_collection_id"]
                ),
                historical_baselines=stage_config["historical_baselines"],
                minimum_successes=int(stage_config.get("minimum_successes", 4)),
            )
            collector_mode = "gate"
            report_kind = "mass_workspace_pilot"
        elif kind == "targeted_mass_diagnostic_pilot":
            plan = build_targeted_mass_diagnostic_pilot_plan(
                variation_config,
                pilot_id=plan_id,
                source_trial_ids=tuple(stage_config["source_trial_ids"]),
                target_mass_kg=float(stage_config["target_mass_kg"]),
                required_successes=int(stage_config["required_successes"]),
                expectations=stage_config["expectations"],
            )
            collector_mode = "pilot"
            report_kind = "pilot"
        elif kind == "targeted_yaw_pilot":
            plan = build_so101_yaw_pilot_plan(
                variation_config,
                pilot_id=plan_id,
                yaw_degrees=float(stage_config["yaw_degrees"]),
                trial_profiles=stage_config["trial_profiles"],
                required_successes=int(stage_config.get("required_successes", 10)),
                size_scope=str(stage_config.get("size_scope", "balanced")),
                required_success_cells=stage_config.get("required_success_cells"),
            )
            collector_mode = "pilot"
            report_kind = "pilot"
        else:
            plan = build_so101_pilot_plan(
                variation_config,
                pilot_id=plan_id,
            )
            collector_mode = "pilot"
            report_kind = "pilot"
        directory = f"stages/{index:02d}_{stage_id}"
        plans[stage_id] = plan
        stages.append(
            {
                "order": index,
                "stage_id": stage_id,
                "kind": kind,
                "collector_mode": collector_mode,
                "report_kind": report_kind,
                "directory": directory,
                "plan_path": f"{directory}/plan.json",
                "manifest_path": f"{directory}/manifest.json",
                "episodes_root": f"{directory}/episodes",
                "watchdog_report_path": f"{directory}/watchdog.json",
                "report_json_path": f"{directory}/report.json",
                "report_markdown_path": f"{directory}/report.md",
                "plan_id": plan["plan_id"],
                "plan_sha256": plan["plan_sha256"],
                "maximum_attempts": (
                    int((plan.get("gate") or plan.get("pilot"))["maximum_attempts"])
                ),
            }
        )
    workflow = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "workflow_id": workflow_id,
        "git_commit": git_commit,
        "task_id": variation_config["task_id"],
        "variation_config_revision": variation_config["config_revision"],
        "workflow_config_sha256": _sha256(workflow_config),
        "watchdog_policy_sha256": _sha256(watchdog_policy),
        "watchdog_policy_path": "watchdog-policy.json",
        "completion_status": str(
            workflow_config.get("completion_status", "READY_FOR_FORMAL_REVIEW")
        ),
        "stages": stages,
        "formal_collection_policy": (
            "outside_workflow_requires_merged_main_and_owner_authorization"
        ),
    }
    workflow["workflow_sha256"] = _sha256(workflow)
    return workflow, plans


def write_so101_gate_workflow(
    root: str | Path,
    workflow: dict[str, Any],
    plans: dict[str, dict[str, Any]],
    watchdog_policy: dict[str, Any],
) -> Path:
    destination = Path(root)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"workflow root is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for stage in workflow["stages"]:
        path = destination / stage["plan_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(plans[stage["stage_id"]], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (destination / workflow["watchdog_policy_path"]).write_text(
        json.dumps(watchdog_policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    workflow_path = destination / "workflow.json"
    workflow_path.write_text(
        json.dumps(workflow, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return workflow_path


def _stage_action(stage: dict[str, Any], workflow: dict[str, Any], root: Path) -> dict[str, Any]:
    plan = str(root / stage["plan_path"])
    manifest = str(root / stage["manifest_path"])
    episodes = str(root / stage["episodes_root"])
    if stage["state"] in {"READY", "RUNNING"}:
        mode_flag = "--gate-plan" if stage["collector_mode"] == "gate" else "--pilot-plan"
        return {
            "kind": "COLLECT",
            "working_directory": "farpoint_repository_root",
            "environment": {"FARPOINT_GIT_COMMIT": workflow["git_commit"]},
            "command": [
                "scripts/run_so101_isaaclab.sh",
                "headless",
                mode_flag,
                "--plan",
                plan,
                "--manifest",
                manifest,
                "--output-root",
                episodes,
                "--max-attempts-this-run",
                str(stage["maximum_attempts"]),
                "--watchdog-policy",
                str(root / workflow["watchdog_policy_path"]),
            ],
        }
    if stage["state"] == "NEEDS_REPORT":
        scripts = {
            "gate": "scripts/report_so101_gate.py",
            "pilot": "scripts/report_so101_pilot.py",
            "mass_feasibility": "scripts/report_so101_mass_feasibility.py",
            "mass_workspace_pilot": "scripts/report_so101_mass_workspace_pilot.py",
        }
        script = scripts[stage["report_kind"]]
        return {
            "kind": "REPORT",
            "working_directory": "farpoint_repository_root",
            "command": [
                "python3",
                script,
                "--plan",
                plan,
                "--manifest",
                manifest,
                "--episodes-root",
                episodes,
                "--json-output",
                str(root / stage["report_json_path"]),
                "--markdown-output",
                str(root / stage["report_markdown_path"]),
            ],
        }
    return {"kind": "NONE", "command": []}


def evaluate_so101_gate_workflow(
    workflow_path: str | Path,
) -> dict[str, Any]:
    """Evaluate stage evidence and return only the next admissible operation."""
    path = Path(workflow_path)
    root = path.parent
    workflow = _read_json(path)
    errors: list[str] = []
    if workflow.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
        errors.append("invalid_workflow_schema")
    stored_hash = workflow.get("workflow_sha256")
    hash_material = dict(workflow)
    hash_material.pop("workflow_sha256", None)
    if stored_hash != _sha256(hash_material):
        errors.append("workflow_hash_mismatch")
    try:
        policy = _read_json(root / workflow["watchdog_policy_path"])
        validate_watchdog_policy(policy)
        if _sha256(policy) != workflow.get("watchdog_policy_sha256"):
            errors.append("watchdog_policy_hash_mismatch")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"invalid_watchdog_policy:{type(error).__name__}:{error}")

    unlocked = not errors
    stage_results = []
    active_stage = None
    for stage in workflow.get("stages") or []:
        result = dict(stage)
        result["errors"] = []
        if not unlocked:
            result["state"] = "LOCKED"
            stage_results.append(result)
            continue
        try:
            plan = _read_json(root / stage["plan_path"])
            if plan.get("plan_sha256") != stage.get("plan_sha256"):
                raise ValueError("stage plan hash does not match workflow")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            result["state"] = "INVALID"
            result["errors"].append(f"invalid_plan:{type(error).__name__}:{error}")
            errors.extend(result["errors"])
            unlocked = False
            stage_results.append(result)
            continue

        manifest_path = root / stage["manifest_path"]
        report_path = root / stage["report_json_path"]
        if not manifest_path.exists():
            result["state"] = "READY"
        else:
            try:
                manifest = _read_json(manifest_path)
                validate_manifest(manifest, plan)
                if manifest.get("git_commit") != workflow.get("git_commit"):
                    raise ValueError("manifest git commit does not match workflow")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                result["state"] = "INVALID"
                result["errors"].append(f"invalid_manifest:{type(error).__name__}:{error}")
                errors.extend(result["errors"])
                unlocked = False
                stage_results.append(result)
                continue
            execution_status = manifest.get("execution_status")
            if execution_status == "RUNNING":
                result["state"] = "RUNNING"
            elif execution_status == "ABORTED":
                result["state"] = "BLOCKED"
                result["errors"].append(
                    f"collection_aborted:{manifest.get('abort_reason') or 'unknown'}"
                )
            elif execution_status == "FINISHED" and not report_path.exists():
                result["state"] = "NEEDS_REPORT"
            elif execution_status == "FINISHED":
                try:
                    report = _read_json(report_path)
                    status_key = {
                        "gate": "gate_status",
                        "pilot": "pilot_status",
                        "mass_feasibility": "feasibility_status",
                        "mass_workspace_pilot": "pilot_status",
                    }[stage["report_kind"]]
                    if report.get("plan_sha256") != stage["plan_sha256"]:
                        raise ValueError("report plan hash does not match workflow")
                    if report.get("git_commit") != workflow.get("git_commit"):
                        raise ValueError("report git commit does not match workflow")
                    result["report_status"] = report.get(status_key)
                    result["state"] = "PASS" if report.get(status_key) == "PASS" else "BLOCKED"
                    if result["state"] == "BLOCKED":
                        result["errors"].append(f"stage_report_status:{report.get(status_key)}")
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                    result["state"] = "INVALID"
                    result["errors"].append(f"invalid_report:{type(error).__name__}:{error}")
                    errors.extend(result["errors"])
            else:
                result["state"] = "INVALID"
                result["errors"].append(f"unsupported_execution_status:{execution_status}")
                errors.extend(result["errors"])

        stage_results.append(result)
        if result["state"] != "PASS":
            active_stage = result
            unlocked = False

    if errors:
        status = "INVALID"
    elif all(stage["state"] == "PASS" for stage in stage_results):
        status = workflow.get("completion_status", "READY_FOR_FORMAL_REVIEW")
    elif active_stage is None:
        status = "INVALID"
        errors.append("missing_active_stage")
    elif active_stage["state"] == "BLOCKED":
        status = "BLOCKED"
    else:
        status = active_stage["state"]
    next_action = (
        _stage_action(active_stage, workflow, root)
        if active_stage is not None and status not in {"INVALID", "BLOCKED"}
        else {"kind": "NONE", "command": []}
    )
    return {
        "schema_version": "farpoint.so101-gate-workflow-status.v1",
        "workflow_id": workflow.get("workflow_id"),
        "git_commit": workflow.get("git_commit"),
        "status": status,
        "active_stage_id": active_stage.get("stage_id") if active_stage else None,
        "stages": stage_results,
        "errors": errors,
        "next_action": next_action,
        "formal_collection_authorized": False,
    }
