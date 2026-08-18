import json
import struct
import zlib

import pytest

from farpoint.campaign import canonical_sha256
from farpoint.object_variation import load_variation_config
from farpoint.so101_collection import (
    create_manifest,
    create_pilot_manifest,
    next_attempt,
    record_attempt,
)
from farpoint.so101_pilot import build_so101_pilot_plan, build_so101_yaw_pilot_plan
from farpoint.so101_pilot_report import (
    _expectation_errors,
    _pilot_status,
    _required_success_cell_errors,
    build_so101_pilot_report,
    render_so101_pilot_report_markdown,
)


def _config():
    return load_variation_config("configs/variations/so101_cube_pick_place_v1.json")


def _write_rgb_png(path):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    scanline = b"\x00" + bytes((255, 0, 0)) * 640
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 640, 480, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanline * 480))
        + chunk(b"IEND", b"")
    )


def _write_success_episode(root, variation_id, marker, *, wrist=False):
    root.mkdir(parents=True)
    (root / "rgb").mkdir()
    _write_rgb_png(root / "rgb/front.png")
    if wrist:
        _write_rgb_png(root / "rgb/wrist.png")
    phases = ["home", "verify_contact", "lift"] + ["settle"] * 15 + ["retreat"]
    rows = []
    for frame, phase in enumerate(phases):
        row = {
                "frame": frame,
                "timestamp_seconds": frame / 30,
                "phase": phase,
                "grasp_phase": "approach" if frame == 0 else "validated",
                "joint_positions": [0.0] * 6,
                "action_joint_positions": [0.0] * 6,
                "rgb_path": "rgb/front.png",
                "contact": {
                    "cube_contact": frame > 0,
                    "bilateral_cube_contact": frame > 0,
                },
                "contact_forces_newtons": {
                    "left_finger": float(frame > 0),
                    "right_finger": float(frame > 0),
                },
                "grasp_evidence": {"proof_lift_m": 0.006 if frame > 0 else 0.0},
                "truth": {
                    "object_root_pose_xyzw": [
                        0.15 + marker,
                        -0.11,
                        0.05 + min(frame, 2) * 0.006,
                        0,
                        0,
                        0,
                        1,
                    ],
                    "gripper_link_pose_xyzw": [0, 0, 0.1 + frame * 0.001, 0, 0, 0, 1],
                    "proof_lift_target_m": 0.006 if frame > 0 else 0.0,
                    "transport_lift_target_m": 0.04 if phase == "lift" else 0.0,
                    "transport_lift_actual_m": 0.04 if phase == "lift" else 0.0,
                    "grasp_xy_correction_m": [0.001, 0.0],
                },
            }
        if wrist:
            row["wrist_rgb_path"] = "rgb/wrist.png"
        rows.append(row)
    (root / "observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (root / "metadata.json").write_text(
        json.dumps({"variation": {"variation_id": variation_id}}), encoding="utf-8"
    )
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "success": True,
                "dataset_valid": True,
                "failure_category": None,
                "failure_reason": None,
            }
        ),
        encoding="utf-8",
    )


def test_pilot_report_passes_ten_independent_physical_front_only_episodes(tmp_path):
    config = load_variation_config("configs/variations/so101_cube_pick_place_v1.json")
    plan = build_so101_pilot_plan(config, pilot_id="pilot_report")
    manifest = create_pilot_manifest(
        plan, collection_id=plan["plan_id"], git_commit="a" * 40
    )
    for index in range(10):
        attempt = next_attempt(manifest, plan)
        episode_id = f"episode_{index}"
        _write_success_episode(
            tmp_path / episode_id, attempt["variation_id"], index / 1000
        )
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=episode_id,
            success=True,
            dataset_valid=True,
        )

    report = build_so101_pilot_report(plan, manifest, tmp_path)

    assert report["pilot_status"] == "PASS"
    assert report["attempted_count"] == 10
    assert report["success_count"] == 10
    assert report["attempt_seed_count"] == 10
    assert report["variation_seed_count"] == 10
    assert report["independent_episode_identity_count"] == 10
    assert report["minimum_selected_proof_lift_m"] == 0.006
    assert report["minimum_selected_settle_frames"] == 15
    assert report["evidence_errors"] == []
    assert report["acceptance_errors"] == []
    assert "Pilot status: **PASS**" in render_so101_pilot_report_markdown(report)


def test_pilot_report_allows_retry_to_reuse_frozen_variation_seed(tmp_path):
    config = load_variation_config("configs/variations/so101_cube_pick_place_v1.json")
    plan = build_so101_pilot_plan(config, pilot_id="retry_seed_report")
    plan["trials"] = plan["trials"][:10]
    plan["pilot"]["maximum_attempts"] = 30
    plan["pilot"]["primary_trial_ids"] = [
        trial["trial_id"] for trial in plan["trials"]
    ]
    plan["pilot"]["fallback_trial_ids"] = []
    plan["collection"] = {
        "kind": "self_healing_campaign_segment",
        "required_successes": 10,
        "maximum_attempts": 30,
        "attempt_policy": {"maximum_attempts_per_variation": 3},
    }
    plan["plan_sha256"] = canonical_sha256(plan, omit=("plan_sha256",))
    manifest = create_manifest(
        plan, collection_id=plan["plan_id"], git_commit="a" * 40
    )
    failed_variation_id = None
    for index in range(10):
        attempt = next_attempt(manifest, plan)
        episode_id = f"episode_{index}"
        _write_success_episode(
            tmp_path / episode_id, attempt["variation_id"], index / 1000
        )
        success = index != 0
        if not success:
            failed_variation_id = attempt["variation_id"]
            (tmp_path / episode_id / "metrics.json").write_text(
                json.dumps(
                    {
                        "success": False,
                        "dataset_valid": True,
                        "failure_category": "oracle",
                        "failure_reason": "grasp_phase_timeout:slow_close",
                    }
                ),
                encoding="utf-8",
            )
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=episode_id,
            success=success,
            dataset_valid=True,
            failure_category=None if success else "oracle",
            failure_reason=None if success else "grasp_phase_timeout:slow_close",
        )
    retry = next_attempt(manifest, plan)
    assert retry["variation_id"] == failed_variation_id
    _write_success_episode(tmp_path / "episode_retry", retry["variation_id"], 0.02)
    (tmp_path / "episode_retry" / "metadata.json").write_text(
        json.dumps(
            {
                "variation": {"variation_id": retry["variation_id"]},
                "attempt_id": retry["attempt_id"],
            }
        ),
        encoding="utf-8",
    )
    record_attempt(
        manifest,
        plan,
        retry,
        episode_id="episode_retry",
        success=True,
        dataset_valid=True,
    )

    report = build_so101_pilot_report(plan, manifest, tmp_path)

    assert report["evidence_errors"] == []
    assert report["pilot_status"] == "PASS"
    assert report["attempted_count"] == 11
    assert report["attempt_seed_count"] == 11
    assert report["variation_seed_count"] == 10


def test_pilot_report_accepts_bound_zero_frame_runner_failure(tmp_path):
    plan = build_so101_pilot_plan(_config(), pilot_id="runner_failure_report")
    plan["trials"] = plan["trials"][:10]
    plan["pilot"]["maximum_attempts"] = 30
    plan["pilot"]["primary_trial_ids"] = [
        trial["trial_id"] for trial in plan["trials"]
    ]
    plan["pilot"]["fallback_trial_ids"] = []
    plan["collection"] = {
        "kind": "self_healing_campaign_segment",
        "required_successes": 10,
        "maximum_attempts": 30,
        "attempt_policy": {"maximum_attempts_per_variation": 3},
    }
    plan["plan_sha256"] = canonical_sha256(plan, omit=("plan_sha256",))
    manifest = create_manifest(
        plan, collection_id=plan["plan_id"], git_commit="a" * 40
    )
    failed = next_attempt(manifest, plan)
    failed_episode_id = "episode_runner_failure"
    failed_root = tmp_path / failed_episode_id
    failed_root.mkdir()
    failure_reason = "RuntimeError: recovery handoff deadline"
    (failed_root / "run-state.json").write_text(
        json.dumps(
            {
                "schema_version": "farpoint.episode-run.v1",
                "execution_status": "FAILED",
                "identity": {
                    "episode_id": failed_episode_id,
                    "trial_id": failed["trial_id"],
                },
                "outcome": {
                    "success": False,
                    "dataset_valid": False,
                    "failure_category": "runner",
                    "failure_reason": failure_reason,
                },
                "provenance": {
                    "collection_id": manifest["collection_id"],
                    "git_commit": manifest["git_commit"],
                },
                "recording": {"frame_count": 0},
            }
        ),
        encoding="utf-8",
    )
    (failed_root / "runner_error.json").write_text(
        json.dumps({"error": "RuntimeError(...)" , "traceback": "trace"}),
        encoding="utf-8",
    )
    record_attempt(
        manifest,
        plan,
        failed,
        episode_id=failed_episode_id,
        success=False,
        dataset_valid=False,
        failure_category="runner",
        failure_reason=failure_reason,
    )
    for index in range(10):
        attempt = next_attempt(manifest, plan)
        episode_id = f"episode_success_{index}"
        _write_success_episode(
            tmp_path / episode_id, attempt["variation_id"], index / 1000
        )
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=episode_id,
            success=True,
            dataset_valid=True,
        )

    report = build_so101_pilot_report(plan, manifest, tmp_path)

    assert report["pilot_status"] == "PASS"
    assert report["evidence_errors"] == []
    assert report["terminal_runner_attempts"] == [failed_episode_id]
    assert report["episode_evidence"]["episode_count"] == 10

    run_state_path = failed_root / "run-state.json"
    run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
    run_state["outcome"]["failure_reason"] = "different runner failure"
    run_state_path.write_text(json.dumps(run_state), encoding="utf-8")

    invalid = build_so101_pilot_report(plan, manifest, tmp_path)

    assert invalid["pilot_status"] == "INVALID_EVIDENCE"
    assert f"{failed_episode_id}:runner_outcome_mismatch" in invalid["evidence_errors"]


def test_pilot_report_accepts_explicit_dual_camera_contract(tmp_path):
    plan = build_so101_pilot_plan(_config(), pilot_id="dual_camera_report")
    manifest = create_pilot_manifest(
        plan, collection_id=plan["plan_id"], git_commit="d" * 40
    )
    for index in range(10):
        attempt = next_attempt(manifest, plan)
        episode_id = f"dual_episode_{index}"
        _write_success_episode(
            tmp_path / episode_id,
            attempt["variation_id"],
            index / 1000,
            wrist=True,
        )
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=episode_id,
            success=True,
            dataset_valid=True,
        )

    report = build_so101_pilot_report(
        plan,
        manifest,
        tmp_path,
        required_cameras=("front", "wrist"),
    )

    assert report["pilot_status"] == "PASS"
    assert report["required_cameras"] == ["front", "wrist"]
    assert report["evidence_errors"] == []
    assert "front, wrist camera artifacts passed" in render_so101_pilot_report_markdown(
        report
    )


def test_pilot_gate_failure_is_not_invalid_evidence():
    assert _pilot_status("FINISHED", "FAIL", 9, 10, [], []) == "FAIL"
    assert _pilot_status("FINISHED", "PASS", 10, 10, [], ["role_mismatch"]) == "FAIL"
    assert (
        _pilot_status("FINISHED", "FAIL", 9, 10, ["missing_episode"], [])
        == "INVALID_EVIDENCE"
    )


def test_targeted_pilot_expectations_check_success_and_failure_roles():
    expectations = {
        "collision_trial": {"success": False, "failure_reason": "collision"},
        "fixed_trial": {"success": True},
    }
    passing = [
        {"trial_id": "collision_trial", "success": False, "failure_reason": "collision"},
        {"trial_id": "fixed_trial", "success": True, "failure_reason": None},
    ]
    assert _expectation_errors(expectations, passing) == []

    failing = [
        {"trial_id": "collision_trial", "success": True, "failure_reason": None},
    ]
    assert _expectation_errors(expectations, failing) == [
        "collision_trial:expected_success_false",
        "fixed_trial:missing_expected_attempt",
    ]


def test_required_yaw_success_cells_cannot_be_masked_by_other_successes():
    plan = {
        "pilot": {"required_success_cells": ["r04_c00", "r04_c01"]},
        "trials": [
            {"variation_id": "critical-a", "cell_id": "r04_c00"},
            {"variation_id": "critical-b", "cell_id": "r04_c01"},
            {"variation_id": "control", "cell_id": "r02_c02"},
        ],
    }

    errors = _required_success_cell_errors(
        plan,
        [{"variation_id": "critical-a"}, {"variation_id": "control"}],
    )

    assert errors == ["required_success_cell_failed:r04_c01"]


def test_yaw_pilot_report_accepts_twelve_successes_and_audits_pose_and_mass(tmp_path):
    workflow = json.loads(
        open("configs/workflows/so101_cube_yaw0_pilot.json", encoding="utf-8").read()
    )
    profiles = workflow["stages"][0]["trial_profiles"]
    plan = build_so101_yaw_pilot_plan(
        _config(),
        pilot_id="yaw0_report",
        yaw_degrees=0.0,
        trial_profiles=profiles,
        size_scope=workflow["stages"][0]["size_scope"],
    )
    manifest = create_pilot_manifest(
        plan, collection_id=plan["plan_id"], git_commit="a" * 40
    )
    for index in range(12):
        attempt = next_attempt(manifest, plan)
        episode_id = f"episode_yaw_{index}"
        root = tmp_path / episode_id
        _write_success_episode(root, attempt["variation_id"], index / 1000)
        metadata = json.loads((root / "metadata.json").read_text())
        audit = {
            "requested_mass_kg": attempt["requested"]["mass_kg"],
            "resolved_mass_kg": attempt["resolved"]["mass_kg"],
            "physx_actual_mass_kg": attempt["resolved"]["mass_kg"],
            "requested_resolved_absolute_error_kg": 0.0,
            "resolved_physx_absolute_error_kg": 0.0,
            "tolerance_kg": 1e-6,
            "verified": True,
        }
        metadata.update(
            {
                "variation": {
                    "variation_id": attempt["variation_id"],
                    "requested": attempt["requested"],
                    "resolved": attempt["resolved"],
                },
                "scene": {
                    "object": {
                        "initial_pose": {"orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
                        "mass_audit": audit,
                    }
                },
            }
        )
        (root / "metadata.json").write_text(json.dumps(metadata))
        metrics = json.loads((root / "metrics.json").read_text())
        metrics["physics_audit"] = {"mass": audit}
        (root / "metrics.json").write_text(json.dumps(metrics))
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=episode_id,
            success=True,
            dataset_valid=True,
        )

    report = build_so101_pilot_report(plan, manifest, tmp_path)

    assert report["pilot_status"] == "PASS"
    assert report["success_count"] == 12
    assert report["required_successes"] == 10
    assert report["yaw_audit_count"] == 12
    assert all(row["orientation_verified"] and row["mass_verified"] for row in report["yaw_audits"])
    assert all(row["orientation_tolerance_degrees"] == 2.0 for row in report["yaw_audits"])
    assert report["evidence_errors"] == []

    first_observations = tmp_path / "episode_yaw_0" / "observations.jsonl"
    rows = [json.loads(line) for line in first_observations.read_text().splitlines()]
    rows[0]["truth"]["object_root_pose_xyzw"][3:] = [
        0.0,
        0.0,
        0.0043633093,
        0.9999904807,
    ]
    first_observations.write_text("".join(json.dumps(row) + "\n" for row in rows))

    settled = build_so101_pilot_report(plan, manifest, tmp_path)
    assert settled["pilot_status"] == "PASS"
    assert settled["yaw_audits"][0]["initial_orientation_error_degrees"] == pytest.approx(
        0.5
    )

    rows[0]["truth"]["object_root_pose_xyzw"][3:] = [0.0, 0.0, 0.3826834324, 0.9238795325]
    first_observations.write_text("".join(json.dumps(row) + "\n" for row in rows))

    invalid = build_so101_pilot_report(plan, manifest, tmp_path)
    assert invalid["pilot_status"] == "INVALID_EVIDENCE"
    assert "episode_yaw_0:yaw_orientation_audit_failed" in invalid["evidence_errors"]
