import copy
from pathlib import Path

import pytest

from farpoint.object_variation import load_variation_config
from farpoint.so101_collection import (
    abort_collection_manifest,
    create_manifest,
    next_attempt,
    record_attempt,
)
from farpoint.so101_yaw_collection import (
    build_yaw_collection_plan,
    load_yaw_collection_config,
)
from farpoint.so101_yaw_recovery import (
    build_yaw_completion_report,
    build_yaw_completion_selection,
    build_yaw_recovery_plan,
)


ROOT = Path(__file__).resolve().parents[1]
MISSING = {
    "cube_r01_c00_s0_k1_yaw30000_m040g",
    "cube_r01_c04_s0_k0_yaw30000_m030g",
}


def yaw30_plan():
    return build_yaw_collection_plan(
        load_variation_config(ROOT / "configs/variations/so101_cube_pick_place_v1.json"),
        load_yaw_collection_config(ROOT / "configs/collections/so101_cube_yaw30_30mm_v0_0_3.json"),
    )


def parent_with_two_gaps(tmp_path):
    plan = yaw30_plan()
    manifest = create_manifest(
        plan,
        collection_id="yaw30_parent",
        git_commit="a" * 40,
    )
    roots = tmp_path / "parent"
    for _ in range(30):
        attempt = next_attempt(manifest, plan)
        success = attempt["variation_id"] not in MISSING
        episode_id = f"parent__{attempt['attempt_id']}"
        record_attempt(
            manifest,
            plan,
            attempt,
            episode_id=episode_id,
            success=success,
            dataset_valid=True,
            failure_reason=None if success else "grasp_phase_timeout:test",
        )
        (roots / episode_id).mkdir(parents=True)
    abort_collection_manifest(manifest, "watchdog:test")
    return plan, manifest, roots


def complete_recovery(plan, parent_manifest, tmp_path):
    recovery = build_yaw_recovery_plan(
        plan,
        [(plan, parent_manifest)],
        recovery_id="yaw30_recovery2",
        maximum_attempts=18,
    )
    manifest = create_manifest(
        recovery,
        collection_id="yaw30_recovery2",
        git_commit="b" * 40,
    )
    root = tmp_path / "recovery"
    while manifest["execution_status"] == "RUNNING":
        attempt = next_attempt(manifest, recovery)
        episode_id = f"recovery__{attempt['attempt_id']}"
        record_attempt(
            manifest,
            recovery,
            attempt,
            episode_id=episode_id,
            success=True,
            dataset_valid=True,
        )
        (root / episode_id).mkdir(parents=True)
    return recovery, manifest, root


def fake_analysis(paths, verify_images=True):
    episodes = []
    for index, path in enumerate(paths):
        episodes.append(
            {
                "episode_dir": str(path),
                "metadata_sha256": f"{path.name}:metadata:{index}",
                "observations_sha256": f"{path.name}:observations:{index}",
                "observation_count": 20,
                "camera_frame_counts": {"front": 20},
                "state_dimensions": [6],
                "action_dimensions": [6],
                "timestamps_strictly_increasing": True,
                "camera_frame_integrity": {
                    "front": {
                        "referenced_frames": 20,
                        "existing_frames": 20,
                        "decodable_frames": 20,
                        "resolutions": [[640, 480]],
                        "modes": ["RGB"],
                        "unsafe_paths": [],
                    }
                },
                "success": True,
                "dataset_valid": True,
                "terminal_phase": "retreat",
                "terminal_grasp_phase": "validated",
                "proof_lift_tracking": {"actual_max_m": 0.006},
                "phase_ranges": [{"phase": "settle", "frame_count": 15}],
            }
        )
    return {
        "episode_count": len(episodes),
        "duplicate_observation_groups": [],
        "episodes": episodes,
    }


def test_recovery_freezes_only_two_gaps_and_remaining_budget(tmp_path):
    plan, parent, _ = parent_with_two_gaps(tmp_path)

    recovery = build_yaw_recovery_plan(
        plan,
        [(plan, parent)],
        recovery_id="yaw30_recovery2",
        maximum_attempts=18,
    )

    assert {trial["variation_id"] for trial in recovery["trials"]} == MISSING
    assert recovery["collection"]["required_successes"] == 2
    assert recovery["collection"]["maximum_attempts"] == 18
    assert recovery["collection"]["reference_collection"] == {
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "required_successes": 30,
    }
    assert recovery["collection"]["source_collections"][0]["selected_successes"] == 28


def test_recovery_rejects_budget_below_missing_count(tmp_path):
    plan, parent, _ = parent_with_two_gaps(tmp_path)
    with pytest.raises(ValueError, match="budget cannot cover"):
        build_yaw_recovery_plan(
            plan,
            [(plan, parent)],
            recovery_id="bad",
            maximum_attempts=1,
        )


def test_completion_proves_28_plus_2_exact_balance(tmp_path, monkeypatch):
    plan, parent, parent_root = parent_with_two_gaps(tmp_path)
    recovery, recovery_manifest, recovery_root = complete_recovery(plan, parent, tmp_path)
    monkeypatch.setattr("farpoint.so101_yaw_recovery.analyze_so101_episodes", fake_analysis)
    monkeypatch.setattr(
        "farpoint.so101_yaw_recovery.audit_yaw_mass_episodes",
        lambda _plan, rows, *_args: ([{} for _ in rows], []),
    )

    manifest, selection, report = build_yaw_completion_selection(
        plan,
        [(plan, parent, parent_root)],
        recovery,
        recovery_manifest,
        recovery_episodes_root=recovery_root,
        collection_id="yaw30_completion30",
    )

    assert report["status"] == "PASS"
    assert report["selected_successes"] == 30
    assert report["yaw_mass_audit_count"] == 30
    assert [source["selected_successes"] for source in report["sources"]] == [
        28,
        2,
    ]
    assert report["balance"] == plan["collection"]["balance_contract"]
    assert len(manifest["selected_variations"]) == 30
    assert len(selection["episodes"]) == 30
    assert len({row["episode_dir"] for row in selection["episodes"]}) == 30


def test_completion_rejects_changed_parent_manifest(tmp_path, monkeypatch):
    plan, parent, parent_root = parent_with_two_gaps(tmp_path)
    recovery, recovery_manifest, recovery_root = complete_recovery(plan, parent, tmp_path)
    changed = copy.deepcopy(parent)
    changed["abort_reason"] = "mutated"
    monkeypatch.setattr("farpoint.so101_yaw_recovery.analyze_so101_episodes", fake_analysis)
    monkeypatch.setattr(
        "farpoint.so101_yaw_recovery.audit_yaw_mass_episodes",
        lambda _plan, rows, *_args: ([{} for _ in rows], []),
    )

    report = build_yaw_completion_report(
        plan,
        [(plan, changed, parent_root)],
        recovery,
        recovery_manifest,
        recovery_episodes_root=recovery_root,
    )

    assert report["status"] == "INVALID_EVIDENCE"
    assert "recovery_source_binding_mismatch:yaw30_parent" in report["evidence_errors"]


def test_completion_rejects_duplicate_artifacts_across_sources(tmp_path, monkeypatch):
    plan, parent, parent_root = parent_with_two_gaps(tmp_path)
    recovery, recovery_manifest, recovery_root = complete_recovery(plan, parent, tmp_path)

    def overlapping_analysis(paths, verify_images=True):
        result = fake_analysis(paths, verify_images=verify_images)
        for index, episode in enumerate(result["episodes"]):
            episode["metadata_sha256"] = f"metadata:{index}"
            episode["observations_sha256"] = f"observations:{index}"
        return result

    monkeypatch.setattr("farpoint.so101_yaw_recovery.analyze_so101_episodes", overlapping_analysis)
    monkeypatch.setattr(
        "farpoint.so101_yaw_recovery.audit_yaw_mass_episodes",
        lambda _plan, rows, *_args: ([{} for _ in rows], []),
    )

    report = build_yaw_completion_report(
        plan,
        [(plan, parent, parent_root)],
        recovery,
        recovery_manifest,
        recovery_episodes_root=recovery_root,
    )

    assert report["status"] == "INVALID_EVIDENCE"
    assert "duplicate_episode_identity_across_sources" in report["evidence_errors"]
    assert "duplicate_observation_artifacts_across_sources" in report["evidence_errors"]
