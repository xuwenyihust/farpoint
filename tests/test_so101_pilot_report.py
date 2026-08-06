import json
import struct
import zlib

from farpoint.object_variation import load_variation_config
from farpoint.so101_collection import create_pilot_manifest, next_attempt, record_attempt
from farpoint.so101_pilot import build_so101_pilot_plan
from farpoint.so101_pilot_report import (
    _pilot_status,
    build_so101_pilot_report,
    render_so101_pilot_report_markdown,
)


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


def _write_success_episode(root, variation_id, marker):
    root.mkdir(parents=True)
    (root / "rgb").mkdir()
    _write_rgb_png(root / "rgb/front.png")
    phases = ["home", "verify_contact", "lift"] + ["settle"] * 15 + ["retreat"]
    rows = []
    for frame, phase in enumerate(phases):
        rows.append(
            {
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
        )
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


def test_pilot_gate_failure_is_not_invalid_evidence():
    assert _pilot_status("FINISHED", "FAIL", 9, 10, []) == "FAIL"
    assert (
        _pilot_status("FINISHED", "FAIL", 9, 10, ["missing_episode"])
        == "INVALID_EVIDENCE"
    )
