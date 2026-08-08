import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from farpoint.object_variation import load_variation_config
from farpoint.so101_collection import (
    abort_collection_manifest,
    create_gate_manifest,
    next_attempt,
    record_attempt,
)
from farpoint.so101_gate import (
    build_cube_workspace_matrix_plan,
    build_fixed_cube_gate_plan,
)
from farpoint.so101_watchdog import (
    evaluate_so101_collection,
    load_watchdog_policy,
    validate_watchdog_policy,
)


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)


def policy():
    return load_watchdog_policy(
        ROOT / "configs/workflows/so101_watchdog_p0.json"
    )


def fixed_plan(repetitions=5):
    config = load_variation_config(
        ROOT / "configs/variations/so101_cube_pick_place_v1.json"
    )
    return build_fixed_cube_gate_plan(
        config,
        gate_id="fixed_watchdog",
        edge_m=0.04,
        position_xy_m=(0.20, -0.095),
        repetitions=repetitions,
    )


def workspace_plan():
    config = load_variation_config(
        ROOT / "configs/variations/so101_cube_pick_place_v1.json"
    )
    return build_cube_workspace_matrix_plan(
        config,
        gate_id="workspace_watchdog",
        positions_xy_m=[
            (0.15, -0.11),
            (0.25, -0.11),
            (0.20, -0.095),
            (0.15, -0.08),
            (0.25, -0.08),
        ],
    )


def record(manifest, plan, *, success, reason=None, category="oracle"):
    attempt = next_attempt(manifest, plan)
    record_attempt(
        manifest,
        plan,
        attempt,
        episode_id=f"episode_{attempt['attempt_id']}",
        success=success,
        dataset_valid=True,
        failure_category=None if success else category,
        failure_reason=None if success else reason,
    )


def set_recent(manifest):
    manifest["updated_at"] = NOW.isoformat()


def test_watchdog_continues_a_healthy_running_gate():
    plan = workspace_plan()
    manifest = create_gate_manifest(
        plan, collection_id="healthy", git_commit="a" * 40
    )
    for _ in range(3):
        record(manifest, plan, success=True)
    set_recent(manifest)

    report = evaluate_so101_collection(plan, manifest, policy(), now=NOW)

    assert report["decision"] == "CONTINUE"
    assert report["reasons"] == []
    assert report["progress"]["maximum_possible_successes"] == 10


def test_watchdog_stops_strict_gate_when_target_is_unreachable():
    plan = fixed_plan()
    manifest = create_gate_manifest(
        plan, collection_id="unreachable", git_commit="a" * 40
    )
    record(
        manifest,
        plan,
        success=False,
        reason="bilateral_contact_lost:static_hold",
    )
    set_recent(manifest)

    report = evaluate_so101_collection(plan, manifest, policy(), now=NOW)

    assert report["decision"] == "STOP"
    assert report["reasons"] == ["success_target_unreachable"]
    assert report["progress"] == {
        "attempted_count": 1,
        "maximum_attempts": 5,
        "remaining_attempts": 4,
        "selected_successes": 0,
        "required_successes": 5,
        "maximum_possible_successes": 4,
    }


def test_watchdog_allows_five_then_stops_on_six_consecutive_structural_failures():
    plan = workspace_plan()
    manifest = create_gate_manifest(
        plan, collection_id="structural", git_commit="b" * 40
    )
    # Exercise the structural guard independently of this gate's strict
    # success target. Formal collections retain a much larger retry budget.
    manifest["required_successes"] = 1
    manifest["maximum_attempts"] = 10
    for _ in range(5):
        record(
            manifest,
            plan,
            success=False,
            reason="grasp_phase_timeout:slow_close",
        )
    set_recent(manifest)

    report = evaluate_so101_collection(plan, manifest, policy(), now=NOW)
    assert report["decision"] == "CONTINUE"
    assert report["reasons"] == []

    record(
        manifest,
        plan,
        success=False,
        reason="grasp_phase_timeout:slow_close",
    )
    set_recent(manifest)
    report = evaluate_so101_collection(plan, manifest, policy(), now=NOW)

    assert report["decision"] == "STOP"
    assert report["reasons"] == [
        "consecutive_structural_failure:phase_timeout:6"
    ]


def test_default_watchdog_reports_but_does_not_stop_on_eight_of_last_ten():
    plan = workspace_plan()
    manifest = create_gate_manifest(
        plan, collection_id="recent_structural", git_commit="b" * 40
    )
    manifest["required_successes"] = 10
    manifest["maximum_attempts"] = 20
    for success in (False, False, False, False, True) * 2:
        record(
            manifest,
            plan,
            success=success,
            reason=None if success else "grasp_phase_timeout:slow_close",
        )
    set_recent(manifest)

    report = evaluate_so101_collection(plan, manifest, policy(), now=NOW)

    assert report["decision"] == "CONTINUE"
    assert report["reasons"] == []
    assert report["recent_window"]["stop_enabled"] is False
    assert report["recent_window"]["failure_class_counts"] == {
        "phase_timeout": 8
    }


def test_optional_recent_structural_limit_remains_backward_compatible():
    plan = workspace_plan()
    manifest = create_gate_manifest(
        plan, collection_id="recent_structural_legacy", git_commit="b" * 40
    )
    manifest["required_successes"] = 10
    manifest["maximum_attempts"] = 20
    for success in (False, False, False, False, True) * 2:
        record(
            manifest,
            plan,
            success=success,
            reason=None if success else "grasp_phase_timeout:slow_close",
        )
    set_recent(manifest)
    legacy_policy = policy()
    legacy_policy["recent_failure_fraction"] = 0.8

    report = evaluate_so101_collection(
        plan, manifest, legacy_policy, now=NOW
    )

    assert report["decision"] == "STOP"
    assert report["reasons"] == [
        "recent_structural_failure:phase_timeout:8/10"
    ]
    assert report["recent_window"]["stop_enabled"] is True


def test_watchdog_stops_on_a_stale_live_attempt(tmp_path):
    plan = workspace_plan()
    manifest = create_gate_manifest(
        plan, collection_id="stale_live", git_commit="c" * 40
    )
    set_recent(manifest)
    state_path = tmp_path / "episode_live" / "run-state.json"
    state_path.parent.mkdir()
    state_path.write_text(
        json.dumps(
            {
                "execution_status": "RUNNING",
                "identity": {"episode_id": "episode_live"},
                "provenance": {"collection_id": "stale_live"},
                "recording": {"frame_count": 12},
            }
        ),
        encoding="utf-8",
    )
    stale = (NOW - timedelta(seconds=1801)).timestamp()
    state_path.touch()
    import os

    os.utime(state_path, (stale, stale))

    report = evaluate_so101_collection(
        plan, manifest, policy(), episodes_root=tmp_path, now=NOW
    )

    assert report["decision"] == "STOP"
    assert report["reasons"] == ["stale_live_attempt:episode_live:1801s"]


def test_watchdog_reports_terminal_pass_and_abort_separately():
    plan = fixed_plan(repetitions=1)
    passed = create_gate_manifest(
        plan, collection_id="passed", git_commit="d" * 40
    )
    record(passed, plan, success=True)
    assert evaluate_so101_collection(plan, passed, policy(), now=NOW)[
        "decision"
    ] == "COMPLETE"

    aborted = create_gate_manifest(
        plan, collection_id="aborted", git_commit="e" * 40
    )
    abort_collection_manifest(aborted, "operator_stop")
    report = evaluate_so101_collection(plan, aborted, policy(), now=NOW)
    assert report["decision"] == "STOP"
    assert report["reasons"] == ["collection_aborted:operator_stop"]


def test_watchdog_rejects_invalid_policy_and_manifest():
    invalid_policy = policy()
    invalid_policy["minimum_recent_attempts"] = 6
    invalid_policy["recent_window_attempts"] = 5
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_watchdog_policy(invalid_policy)

    plan = fixed_plan(repetitions=1)
    manifest = create_gate_manifest(
        plan, collection_id="invalid", git_commit="f" * 40
    )
    manifest["selected_variations"] = {"unknown": "missing"}
    report = evaluate_so101_collection(plan, manifest, policy(), now=NOW)
    assert report["decision"] == "INVALID"
    assert report["errors"][0].startswith("invalid_input:ValueError:")

    malformed = create_gate_manifest(
        plan, collection_id="malformed", git_commit="f" * 40
    )
    malformed["maximum_attempts"] = "not-an-integer"
    report = evaluate_so101_collection(plan, malformed, policy(), now=NOW)
    assert report["decision"] == "INVALID"
