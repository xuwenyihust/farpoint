import json
import struct
import zlib
from pathlib import Path

from farpoint.object_variation import load_variation_config
from farpoint.so101_collection import (
    create_gate_manifest,
    next_attempt,
    record_attempt,
)
from farpoint.so101_mass_feasibility import (
    build_cube_mass_feasibility_plan,
    build_cube_mass_workspace_pilot_plan,
)
from farpoint.so101_mass_feasibility_report import (
    build_so101_mass_feasibility_report,
    render_so101_mass_feasibility_report_markdown,
)
from farpoint.so101_mass_workspace_report import (
    build_so101_mass_workspace_report,
    render_so101_mass_workspace_report_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_rgb_png(path, color):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    scanline = b"\x00" + bytes(color) * 640
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 640, 480, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanline * 480))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _plan():
    config = load_variation_config(
        ROOT / "configs/variations/so101_cube_pick_place_v1.json"
    )
    return build_cube_mass_feasibility_plan(
        config,
        profile_id="mass_report_test",
        repetitions_per_mass=3,
        minimum_successes_per_mass=3,
    )


def _write_episode(root, trial, *, actual_mass=None):
    root.mkdir(parents=True)
    (root / "rgb").mkdir()
    mass = float(trial["resolved"]["mass_kg"])
    position = trial["resolved"]["position_m"]
    role_scale = 1.2 if trial["mass_role"] == "candidate" else 1.0
    rows = []
    for frame in range(3):
        rgb_path = f"rgb/front_{frame:06d}.png"
        _write_rgb_png(root / rgb_path, (frame, 0, 0))
        rows.append(
            {
                "frame": frame,
                "timestamp_seconds": frame / 30,
                "phase": "lift" if frame else "home",
                "grasp_phase": "validated" if frame else "approach",
                "joint_positions": [0.0] * 6,
                "action_joint_positions": [role_scale * frame] + [0.0] * 5,
                "rgb_path": rgb_path,
                "contact": {
                    "cube_contact": bool(frame),
                    "bilateral_cube_contact": bool(frame),
                },
                "contact_forces_newtons": {
                    "left_finger": role_scale * 2.0,
                    "right_finger": role_scale * 2.0,
                },
                "truth": {
                    "object_root_pose_xyzw": [
                        position[0],
                        position[1],
                        position[2] + frame / 100,
                        0,
                        0,
                        0,
                        1,
                    ]
                },
            }
        )
    audit = {
        "requested_mass_kg": mass,
        "resolved_mass_kg": mass,
        "physx_actual_mass_kg": mass if actual_mass is None else actual_mass,
        "tolerance_kg": 1e-6,
        "verified": actual_mass is None,
    }
    (root / "observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "variation": {"variation_id": trial["variation_id"]},
                "scene": {"object": {"mass_audit": audit}},
            }
        ),
        encoding="utf-8",
    )
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "success": True,
                "dataset_valid": True,
                "failure_category": None,
                "failure_reason": None,
                "physics_audit": {"mass": audit},
            }
        ),
        encoding="utf-8",
    )


def _complete(tmp_path, *, bad_actual_variation=None):
    plan = _plan()
    manifest = create_gate_manifest(
        plan, collection_id="mass_report", git_commit="b" * 40
    )
    while (attempt := next_attempt(manifest, plan)) is not None:
        episode_id = f"episode_{attempt['variation_id']}"
        _write_episode(
            tmp_path / episode_id,
            attempt,
            actual_mass=(
                0.04 if attempt["variation_id"] == bad_actual_variation else None
            ),
        )
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=episode_id,
            success=True,
            dataset_valid=True,
        )
    return plan, manifest


def test_report_passes_and_recommends_expansion_when_behavior_changes(tmp_path):
    plan, manifest = _complete(tmp_path)

    report = build_so101_mass_feasibility_report(plan, manifest, tmp_path)

    assert report["feasibility_status"] == "PASS"
    assert report["success_by_role"] == {"baseline": 3, "candidate": 3}
    assert report["successful_pair_count"] == 3
    assert report["behavior_signal_detected"] is True
    assert report["recommendation"] == "EXPAND_PHYSICS_ROBUSTNESS_PILOT"
    assert report["evidence_errors"] == []
    assert "PhysX masses agree" in render_so101_mass_feasibility_report_markdown(
        report
    )


def test_report_rejects_metadata_only_mass_change(tmp_path):
    plan = _plan()
    candidate = next(
        trial for trial in plan["trials"] if trial["mass_role"] == "candidate"
    )
    plan, manifest = _complete(
        tmp_path, bad_actual_variation=candidate["variation_id"]
    )

    report = build_so101_mass_feasibility_report(plan, manifest, tmp_path)

    assert report["feasibility_status"] == "INVALID_EVIDENCE"
    assert report["recommendation"] == "DO_NOT_EXPAND_FEASIBILITY_FAILED"
    assert any(error.endswith(":mass_audit_failed") for error in report["evidence_errors"])


def test_candidate_workspace_report_passes_with_historical_references(tmp_path):
    config = load_variation_config(
        ROOT / "configs/variations/so101_cube_pick_place_v1.json"
    )
    baselines = [
        {
            "episode_id": f"historical_{index}",
            "position_xy_m": [0.15 + 0.02 * index, -0.11 + 0.02 * index],
            "mass_kg": 0.04,
            "success": True,
        }
        for index in range(5)
    ]
    plan = build_cube_mass_workspace_pilot_plan(
        config,
        pilot_id="candidate_workspace_report",
        candidate_mass_kg=0.03,
        edge_m=0.03,
        historical_baseline_commit="b" * 40,
        historical_baseline_collection_id="formal_v0_0_0",
        historical_baselines=baselines,
    )
    manifest = create_gate_manifest(
        plan, collection_id="candidate_workspace", git_commit="c" * 40
    )
    while (attempt := next_attempt(manifest, plan)) is not None:
        episode_id = f"episode_{attempt['variation_id']}"
        _write_episode(tmp_path / episode_id, attempt)
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=episode_id,
            success=True,
            dataset_valid=True,
        )

    report = build_so101_mass_workspace_report(plan, manifest, tmp_path)

    assert report["pilot_status"] == "PASS"
    assert report["recommendation"] == "EXPAND_MASS_AXIS"
    assert report["success_count"] == 5
    assert len(report["successful_positions_xy_m"]) == 5
    assert len(report["mass_audits"]) == 5
    assert report["historical_comparison_is_contemporaneous"] is False
    assert report["evidence_errors"] == []
    assert "not a contemporaneous control" in (
        render_so101_mass_workspace_report_markdown(report)
    )
