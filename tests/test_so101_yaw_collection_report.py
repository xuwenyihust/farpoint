from pathlib import Path

import pytest

from farpoint.object_variation import load_variation_config
from farpoint.so101_collection import create_manifest, next_attempt, record_attempt
from farpoint.so101_yaw_collection import build_yaw_collection_plan, load_yaw_collection_config
from farpoint.so101_yaw_collection_report import build_so101_yaw_collection_report


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("config_name", "expected_count", "expected_splits", "expected_yaw"),
    [
        (
            "so101_cube_yaw0_30mm_v0_0_2.json",
            50,
            {"test": 5, "train": 40, "validation": 5},
            0.0,
        ),
        (
            "so101_cube_yaw30_30mm_v0_0_3.json",
            30,
            {"test": 3, "train": 24, "validation": 3},
            30.0,
        ),
    ],
)
def test_formal_yaw_report_requires_complete_balanced_evidence(
    tmp_path,
    monkeypatch,
    config_name,
    expected_count,
    expected_splits,
    expected_yaw,
):
    plan = build_yaw_collection_plan(
        load_variation_config(ROOT / "configs/variations/so101_cube_pick_place_v1.json"),
        load_yaw_collection_config(ROOT / "configs/collections" / config_name),
    )
    manifest = create_manifest(
        plan,
        collection_id="yaw_formal",
        git_commit="a" * 40,
        maximum_attempts=plan["collection"]["maximum_attempts"],
    )
    fake_episodes = []
    for index in range(expected_count):
        attempt = next_attempt(manifest, plan)
        episode_id = f"episode_{index}"
        (tmp_path / episode_id).mkdir()
        record_attempt(manifest, plan, attempt, episode_id=episode_id, success=True, dataset_valid=True)
        fake_episodes.append({
            "episode_dir": str(tmp_path / episode_id),
            "success": True,
            "dataset_valid": True,
            "terminal_phase": "retreat",
            "terminal_grasp_phase": "validated",
            "proof_lift_tracking": {"actual_max_m": 0.006},
            "phase_ranges": [{"phase": "settle", "frame_count": 15}],
        })
    monkeypatch.setattr(
        "farpoint.so101_yaw_collection_report.analyze_so101_episodes",
        lambda *_args, **_kwargs: {"episodes": fake_episodes},
    )
    monkeypatch.setattr(
        "farpoint.so101_yaw_collection_report.so101_episode_evidence_errors",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        "farpoint.so101_yaw_collection_report.audit_yaw_mass_episodes",
        lambda *_args: ([{"verified": True}] * expected_count, []),
    )

    report = build_so101_yaw_collection_report(plan, manifest, tmp_path)

    assert report["status"] == "PASS"
    assert report["success_count"] == expected_count
    assert report["attempted_count"] == expected_count
    assert report["yaw_mass_audit_count"] == expected_count
    assert report["yaw_degrees"] == expected_yaw
    assert report["balance"]["splits"] == expected_splits
    assert report["evidence_errors"] == []
