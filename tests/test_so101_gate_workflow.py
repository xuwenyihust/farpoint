import json
from pathlib import Path

from farpoint.object_variation import load_variation_config
from farpoint.so101_collection import (
    abort_collection_manifest,
    create_gate_manifest,
    next_attempt,
    record_attempt,
    write_manifest,
)
from farpoint.so101_gate_workflow import (
    build_so101_gate_workflow,
    evaluate_so101_gate_workflow,
    write_so101_gate_workflow,
)
from farpoint.so101_watchdog import load_watchdog_policy


ROOT = Path(__file__).resolve().parents[1]
GIT_COMMIT = "a" * 40


def workflow_config():
    return {
        "schema_version": "farpoint.so101-gate-workflow-config.v1",
        "stages": [
            {
                "stage_id": "fixed_30mm",
                "kind": "fixed_cube_repeatability",
                "edge_m": 0.03,
                "position_xy_m": [0.20, -0.095],
                "repetitions": 1,
            },
            {
                "stage_id": "fixed_40mm",
                "kind": "fixed_cube_repeatability",
                "edge_m": 0.04,
                "position_xy_m": [0.20, -0.095],
                "repetitions": 1,
            },
        ],
    }


def initialize(tmp_path):
    config = load_variation_config(
        ROOT / "configs/variations/so101_cube_pick_place_v1.json"
    )
    policy = load_watchdog_policy(
        ROOT / "configs/workflows/so101_watchdog_p0.json"
    )
    workflow, plans = build_so101_gate_workflow(
        workflow_config(),
        config,
        policy,
        workflow_id="workflow_test",
        git_commit=GIT_COMMIT,
    )
    path = write_so101_gate_workflow(tmp_path / "workflow", workflow, plans, policy)
    return path, workflow, plans


def finish_stage(workflow_path, workflow, plan, stage, *, success=True):
    root = workflow_path.parent
    manifest = create_gate_manifest(
        plan,
        collection_id=plan["plan_id"],
        git_commit=workflow["git_commit"],
    )
    attempt = next_attempt(manifest, plan)
    record_attempt(
        manifest,
        plan,
        attempt,
        episode_id=f"episode_{attempt['attempt_id']}",
        success=success,
        dataset_valid=True,
        failure_category=None if success else "oracle",
        failure_reason=None if success else "bilateral_contact_lost:static_hold",
    )
    manifest_path = root / stage["manifest_path"]
    write_manifest(manifest_path, manifest)
    return manifest


def write_pass_report(workflow_path, workflow, stage):
    report = {
        "schema_version": "farpoint.so101-gate-report.v1",
        "plan_sha256": stage["plan_sha256"],
        "git_commit": workflow["git_commit"],
        "gate_status": "PASS",
    }
    path = workflow_path.parent / stage["report_json_path"]
    path.write_text(json.dumps(report), encoding="utf-8")


def test_workflow_exposes_only_first_stage_and_frozen_watchdog(tmp_path):
    path, workflow, _plans = initialize(tmp_path)

    status = evaluate_so101_gate_workflow(path)

    assert status["status"] == "READY"
    assert status["active_stage_id"] == "fixed_30mm"
    assert [row["state"] for row in status["stages"]] == ["READY", "LOCKED"]
    action = status["next_action"]
    assert action["kind"] == "COLLECT"
    assert action["working_directory"] == "farpoint_repository_root"
    assert action["environment"] == {"FARPOINT_GIT_COMMIT": GIT_COMMIT}
    assert "--watchdog-policy" in action["command"]
    assert str(path.parent / "watchdog-policy.json") in action["command"]
    assert workflow["formal_collection_policy"].startswith("outside_workflow")
    assert status["formal_collection_authorized"] is False


def test_workflow_requires_report_pass_before_unlocking_next_stage(tmp_path):
    path, workflow, plans = initialize(tmp_path)
    first = workflow["stages"][0]
    finish_stage(path, workflow, plans[first["stage_id"]], first)

    needs_report = evaluate_so101_gate_workflow(path)
    assert needs_report["status"] == "NEEDS_REPORT"
    assert needs_report["next_action"]["kind"] == "REPORT"

    write_pass_report(path, workflow, first)
    status = evaluate_so101_gate_workflow(path)
    assert status["status"] == "READY"
    assert status["active_stage_id"] == "fixed_40mm"
    assert [row["state"] for row in status["stages"]] == ["PASS", "READY"]


def test_workflow_blocks_after_watchdog_aborts_a_stage(tmp_path):
    path, workflow, plans = initialize(tmp_path)
    first = workflow["stages"][0]
    plan = plans[first["stage_id"]]
    manifest = create_gate_manifest(
        plan, collection_id=plan["plan_id"], git_commit=GIT_COMMIT
    )
    abort_collection_manifest(
        manifest, "watchdog:stop:success_target_unreachable"
    )
    write_manifest(path.parent / first["manifest_path"], manifest)

    status = evaluate_so101_gate_workflow(path)

    assert status["status"] == "BLOCKED"
    assert status["next_action"] == {"kind": "NONE", "command": []}
    assert status["stages"][1]["state"] == "LOCKED"


def test_workflow_rejects_commit_drift(tmp_path):
    path, workflow, plans = initialize(tmp_path)
    first = workflow["stages"][0]
    manifest = create_gate_manifest(
        plans[first["stage_id"]],
        collection_id="wrong_commit",
        git_commit="b" * 40,
    )
    write_manifest(path.parent / first["manifest_path"], manifest)

    status = evaluate_so101_gate_workflow(path)

    assert status["status"] == "INVALID"
    assert "manifest git commit does not match workflow" in status["errors"][0]


def test_mass_feasibility_profile_uses_special_report_and_completion(tmp_path):
    config = load_variation_config(
        ROOT / "configs/variations/so101_cube_pick_place_v1.json"
    )
    policy = load_watchdog_policy(
        ROOT / "configs/workflows/so101_watchdog_p0.json"
    )
    profile = {
        "schema_version": "farpoint.so101-gate-workflow-config.v1",
        "completion_status": "FEASIBILITY_COMPLETE",
        "stages": [
            {
                "stage_id": "cube_mass_30g",
                "kind": "cube_mass_feasibility",
                "repetitions_per_mass": 1,
                "minimum_successes_per_mass": 1,
            }
        ],
    }
    workflow, plans = build_so101_gate_workflow(
        profile,
        config,
        policy,
        workflow_id="mass_workflow",
        git_commit=GIT_COMMIT,
    )
    path = write_so101_gate_workflow(
        tmp_path / "mass_workflow", workflow, plans, policy
    )
    stage = workflow["stages"][0]
    assert stage["report_kind"] == "mass_feasibility"
    manifest = create_gate_manifest(
        plans[stage["stage_id"]],
        collection_id=plans[stage["stage_id"]]["plan_id"],
        git_commit=GIT_COMMIT,
    )
    for _ in range(2):
        attempt = next_attempt(manifest, plans[stage["stage_id"]])
        record_attempt(
            manifest,
            plans[stage["stage_id"]],
            attempt,
            episode_id=f"episode_{attempt['attempt_id']}",
            success=True,
            dataset_valid=True,
        )
    write_manifest(path.parent / stage["manifest_path"], manifest)
    needs_report = evaluate_so101_gate_workflow(path)
    assert needs_report["next_action"]["command"][1] == (
        "scripts/report_so101_mass_feasibility.py"
    )
    report_path = path.parent / stage["report_json_path"]
    report_path.write_text(
        json.dumps(
            {
                "plan_sha256": stage["plan_sha256"],
                "git_commit": GIT_COMMIT,
                "feasibility_status": "PASS",
            }
        ),
        encoding="utf-8",
    )
    assert evaluate_so101_gate_workflow(path)["status"] == "FEASIBILITY_COMPLETE"


def test_candidate_mass_workspace_profile_freezes_five_trials(tmp_path):
    config = load_variation_config(
        ROOT / "configs/variations/so101_cube_pick_place_v1.json"
    )
    policy = load_watchdog_policy(
        ROOT / "configs/workflows/so101_watchdog_p0.json"
    )
    baselines = [
        {
            "episode_id": f"episode_baseline_{index}",
            "position_xy_m": [0.15 + 0.02 * index, -0.11 + 0.02 * index],
            "mass_kg": 0.04,
            "success": True,
        }
        for index in range(5)
    ]
    profile = {
        "schema_version": "farpoint.so101-gate-workflow-config.v1",
        "completion_status": "PILOT_COMPLETE",
        "stages": [
            {
                "stage_id": "candidate_workspace",
                "kind": "cube_mass_workspace_pilot",
                "candidate_mass_kg": 0.03,
                "edge_m": 0.03,
                "minimum_successes": 4,
                "historical_baseline_commit": "b" * 40,
                "historical_baseline_collection_id": "formal_v0_0_0",
                "historical_baselines": baselines,
            }
        ],
    }
    workflow, plans = build_so101_gate_workflow(
        profile,
        config,
        policy,
        workflow_id="candidate_workspace",
        git_commit=GIT_COMMIT,
    )
    path = write_so101_gate_workflow(
        tmp_path / "candidate_workspace", workflow, plans, policy
    )
    stage = workflow["stages"][0]

    assert stage["maximum_attempts"] == 5
    assert stage["report_kind"] == "mass_workspace_pilot"
    status = evaluate_so101_gate_workflow(path)
    assert status["next_action"]["command"][2:4] == ["--gate-plan", "--plan"]


def test_targeted_mass_diagnostic_profile_uses_pilot_contract(tmp_path):
    config = load_variation_config(
        ROOT / "configs/variations/so101_cube_pick_place_v1.json"
    )
    policy = load_watchdog_policy(
        ROOT / "configs/workflows/so101_watchdog_p0.json"
    )
    profile = {
        "schema_version": "farpoint.so101-gate-workflow-config.v1",
        "completion_status": "PILOT_COMPLETE",
        "stages": [
            {
                "stage_id": "capture_fix",
                "kind": "targeted_mass_diagnostic_pilot",
                "target_mass_kg": 0.03,
                "required_successes": 3,
                "source_trial_ids": [
                    "cube_r03_c00_s1_k1",
                    "cube_r04_c01_s0_k0",
                    "cube_r02_c00_s1_k0",
                    "cube_r03_c03_s1_k0",
                    "cube_r01_c01_s0_k1",
                ],
                "expectations": {
                    "cube_r03_c00_s1_k1": {
                        "success": False,
                        "failure_reason": "collision",
                    },
                    "cube_r04_c01_s0_k0": {
                        "success": False,
                        "failure_reason": "grasp_phase_timeout:slow_close",
                    },
                    "cube_r02_c00_s1_k0": {"success": True},
                    "cube_r03_c03_s1_k0": {"success": True},
                    "cube_r01_c01_s0_k1": {"success": True},
                },
            }
        ],
    }

    workflow, plans = build_so101_gate_workflow(
        profile,
        config,
        policy,
        workflow_id="capture_fix",
        git_commit=GIT_COMMIT,
    )
    path = write_so101_gate_workflow(
        tmp_path / "capture_fix", workflow, plans, policy
    )
    stage = workflow["stages"][0]

    assert stage["collector_mode"] == "pilot"
    assert stage["report_kind"] == "pilot"
    assert stage["maximum_attempts"] == 5
    assert plans[stage["stage_id"]]["pilot"]["expectations"]
    status = evaluate_so101_gate_workflow(path)
    assert status["next_action"]["command"][2:4] == ["--pilot-plan", "--plan"]


def test_structural_contact_handoff_profile_freezes_two_fixes_and_control(tmp_path):
    config = load_variation_config(
        ROOT / "configs/variations/so101_cube_pick_place_v1.json"
    )
    policy = load_watchdog_policy(
        ROOT / "configs/workflows/so101_watchdog_p0.json"
    )
    profile = json.loads(
        (ROOT / "configs/workflows/so101_structural_contact_handoff_pilot.json")
        .read_text()
    )

    workflow, plans = build_so101_gate_workflow(
        profile,
        config,
        policy,
        workflow_id="structural_contact_handoff",
        git_commit=GIT_COMMIT,
    )
    path = write_so101_gate_workflow(
        tmp_path / "structural_contact_handoff", workflow, plans, policy
    )
    stage = workflow["stages"][0]
    plan = plans[stage["stage_id"]]

    assert stage["collector_mode"] == "pilot"
    assert stage["maximum_attempts"] == 3
    assert plan["pilot"]["required_successes"] == 3
    assert plan["pilot"]["source_trial_ids"] == [
        "cube_r03_c00_s1_k1",
        "cube_r04_c01_s0_k0",
        "cube_r03_c03_s1_k0",
    ]
    assert all(
        expectation == {"success": True}
        for expectation in plan["pilot"]["expectations"].values()
    )
    status = evaluate_so101_gate_workflow(path)
    assert status["next_action"]["command"][2:4] == ["--pilot-plan", "--plan"]


def test_recovery6_contact_profile_freezes_failures_and_success_controls(tmp_path):
    config = load_variation_config(
        ROOT / "configs/variations/so101_cube_pick_place_v1.json"
    )
    policy = load_watchdog_policy(
        ROOT / "configs/workflows/so101_watchdog_p0.json"
    )
    profile = json.loads(
        (ROOT / "configs/workflows/so101_recovery6_contact_pilot.json")
        .read_text()
    )

    workflow, plans = build_so101_gate_workflow(
        profile,
        config,
        policy,
        workflow_id="recovery6_contact",
        git_commit=GIT_COMMIT,
    )
    path = write_so101_gate_workflow(
        tmp_path / "recovery6_contact", workflow, plans, policy
    )
    stage = workflow["stages"][0]
    plan = plans[stage["stage_id"]]

    assert stage["collector_mode"] == "pilot"
    assert stage["maximum_attempts"] == 8
    assert plan["pilot"]["required_successes"] == 8
    assert plan["pilot"]["source_trial_ids"][:6] == [
        "cube_r02_c01_s0_k1",
        "cube_r03_c01_s0_k0",
        "cube_r03_c01_s0_k1",
        "cube_r04_c02_s0_k0",
        "cube_r04_c02_s1_k0",
        "cube_r04_c03_s0_k1",
    ]
    assert plan["pilot"]["source_trial_ids"][-2:] == [
        "cube_r03_c01_s1_k0",
        "cube_r04_c04_s0_k0",
    ]
    assert all(
        expectation == {"success": True}
        for expectation in plan["pilot"]["expectations"].values()
    )
    status = evaluate_so101_gate_workflow(path)
    assert status["next_action"]["command"][2:4] == ["--pilot-plan", "--plan"]
