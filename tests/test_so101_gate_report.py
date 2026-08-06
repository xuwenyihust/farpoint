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
from farpoint.so101_gate import build_fixed_cube_gate_plan
from farpoint.so101_gate_report import (
    build_so101_gate_report,
    render_so101_gate_report_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_rgb_png(path, width=640, height=480, color=(255, 0, 0)):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    scanline = b"\x00" + bytes(color) * width
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanline * height))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _gate(repetitions=2):
    config = load_variation_config(
        ROOT / "configs/variations/so101_cube_pick_place_v1.json"
    )
    return build_fixed_cube_gate_plan(
        config,
        gate_id="report_gate",
        edge_m=0.04,
        position_xy_m=(0.150233, -0.114276),
        repetitions=repetitions,
    )


def _write_episode(root, variation_id, *, success=True, wrist=False):
    root.mkdir(parents=True)
    (root / "rgb").mkdir()
    variation_marker = sum(variation_id.encode("utf-8")) / 1_000_000
    rows = []
    for frame in range(2):
        row = {
            "frame": frame,
            "timestamp_seconds": frame / 30,
            "phase": "retreat" if frame else "home",
            "grasp_phase": "validated" if frame else "approach",
            "joint_positions": [0.0] * 6,
            "action_joint_positions": [0.0] * 6,
            "rgb_path": f"rgb/front_{frame:06d}.png",
            "contact": {
                "cube_contact": bool(frame),
                "bilateral_cube_contact": bool(frame),
            },
            "contact_forces_newtons": {
                "left_finger": float(frame),
                "right_finger": float(frame),
            },
            "truth": {
                "object_root_pose_xyzw": [
                    0.15 + variation_marker,
                    -0.11,
                    0.05 + 0.01 * frame,
                    0,
                    0,
                    0,
                    1,
                ]
            },
        }
        if wrist:
            row["wrist_rgb_path"] = f"rgb/wrist_{frame:06d}.png"
            _write_rgb_png(root / row["wrist_rgb_path"], color=(0, 255, frame))
        _write_rgb_png(root / row["rgb_path"], color=(255, 0, frame))
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
                "success": success,
                "dataset_valid": True,
                "failure_category": None if success else "oracle",
                "failure_reason": None if success else "bilateral_contact_lost:static_hold",
            }
        ),
        encoding="utf-8",
    )


def test_gate_report_passes_only_with_complete_independent_front_evidence(tmp_path):
    plan = _gate()
    manifest = create_gate_manifest(
        plan, collection_id="report_run", git_commit="a" * 40
    )
    for index in range(2):
        attempt = next_attempt(manifest, plan)
        episode_id = f"episode_{index}"
        _write_episode(tmp_path / episode_id, attempt["variation_id"])
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=episode_id,
            success=True,
            dataset_valid=True,
        )

    report = build_so101_gate_report(plan, manifest, tmp_path)

    assert report["gate_status"] == "PASS"
    assert report["success_count"] == 2
    assert report["attempt_seed_count"] == 2
    assert report["variation_seed_count"] == 2
    assert report["evidence_errors"] == []
    assert "Gate status: **PASS**" in render_so101_gate_report_markdown(report)


def test_fixed_repeatability_gate_allows_deterministic_duplicate_observations(
    tmp_path,
):
    plan = _gate()
    manifest = create_gate_manifest(
        plan, collection_id="deterministic_run", git_commit="d" * 40
    )
    episode_dirs = []
    for index in range(2):
        attempt = next_attempt(manifest, plan)
        episode_id = f"episode_{index}"
        episode_dir = tmp_path / episode_id
        _write_episode(episode_dir, attempt["variation_id"])
        episode_dirs.append(episode_dir)
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=episode_id,
            success=True,
            dataset_valid=True,
        )
    # Deterministic PhysX with identical resolved scene factors can produce
    # byte-identical observation streams. Episode identity remains independent
    # through the distinct metadata, attempt seed, and variation seed.
    (episode_dirs[1] / "observations.jsonl").write_bytes(
        (episode_dirs[0] / "observations.jsonl").read_bytes()
    )

    report = build_so101_gate_report(plan, manifest, tmp_path)

    assert report["gate_status"] == "PASS"
    assert report["deterministic_observation_duplicates_allowed"] is True
    assert report["episode_evidence"]["independent_observation_artifact_count"] == 1
    assert report["independent_episode_identity_count"] == 2
    assert report["attempt_seed_count"] == 2
    assert report["variation_seed_count"] == 2


def test_gate_report_classifies_physical_failure(tmp_path):
    plan = _gate(repetitions=1)
    manifest = create_gate_manifest(
        plan, collection_id="failed_run", git_commit="b" * 40
    )
    attempt = next_attempt(manifest, plan)
    _write_episode(tmp_path / "failed_episode", attempt["variation_id"], success=False)
    record_attempt(
        manifest,
        plan,
        attempt,
        episode_id="failed_episode",
        success=False,
        dataset_valid=True,
        failure_category="oracle",
        failure_reason="bilateral_contact_lost:static_hold",
    )

    report = build_so101_gate_report(plan, manifest, tmp_path)

    assert report["gate_status"] == "FAIL"
    assert report["failure_class_counts"] == {"bilateral_contact_lost": 1}


def test_gate_report_rejects_wrist_camera_evidence_in_v0(tmp_path):
    plan = _gate(repetitions=1)
    manifest = create_gate_manifest(
        plan, collection_id="wrist_run", git_commit="c" * 40
    )
    attempt = next_attempt(manifest, plan)
    _write_episode(
        tmp_path / "wrist_episode", attempt["variation_id"], wrist=True
    )
    record_attempt(
        manifest,
        plan,
        attempt,
        episode_id="wrist_episode",
        success=True,
        dataset_valid=True,
    )

    report = build_so101_gate_report(plan, manifest, tmp_path)

    assert report["gate_status"] == "INVALID_EVIDENCE"
    assert report["evidence_errors"] == ["wrist_episode:not_front_only_complete"]
