import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_oracle_repair import (  # noqa: E402
    validate_changed_files_on_disk,
    validate_evidence,
    validate_paths,
)


POLICY = json.loads(
    (ROOT / "configs/governance/so101_oracle_repair_v1.json").read_text()
)
COMMIT = "a" * 40


def _evidence():
    return {
        "schema_version": "farpoint.oracle-repair-evidence.v1",
        "policy_id": "so101-oracle-repair-v1",
        "git_commit": COMMIT,
        "campaign_id": "so101-v010",
        "parent_segment_id": "segment-000",
        "parent_manifest_sha256": "b" * 64,
        "scene_contract_unchanged": True,
        "success_criteria_unchanged": True,
        "diagnostics": [
            {
                "failure_class": "bilateral_contact_lost",
                "seeds": [
                    {"seed": seed, "attempts": 1 + seed % 3, "success": True, "dataset_valid": True}
                    for seed in range(3)
                ],
            }
        ],
        "canaries": [
            {"seed": seed + 100, "attempts": 1, "success": True, "dataset_valid": True}
            for seed in range(10)
        ],
    }


def test_path_policy_accepts_only_runtime_profile_and_test_changes():
    assert validate_paths(
        POLICY,
        ["src/farpoint/grasp_oracle.py", "configs/oracle-profiles/contact-v2.json", "tests/test_grasp_oracle.py"],
    ) == []


def test_path_policy_rejects_mixed_scene_schema_and_release_changes():
    errors = validate_paths(
        POLICY,
        ["src/farpoint/oracle.py", "src/farpoint/schemas/farpoint_episode_v4.schema.json", "configs/datasets/farpoint-so101.toml"],
    )
    assert "forbidden path: src/farpoint/schemas/farpoint_episode_v4.schema.json" in errors
    assert "forbidden path: configs/datasets/farpoint-so101.toml" in errors
    assert "Oracle repair must change an allowed test path" in errors


def test_path_policy_rejects_deletions_renames_and_duplicate_paths():
    errors = validate_paths(
        POLICY,
        [
            "D\tsrc/farpoint/oracle.py",
            "R100\tsrc/farpoint/control.py\tsrc/farpoint/oracle.py",
            "M\ttests/test_oracle.py",
            "M\ttests/test_oracle.py",
        ],
    )
    assert any("status 'D'" in error for error in errors)
    assert any("renames and copies" in error for error in errors)
    assert "changed file list contains duplicates" in errors
    assert "Oracle repair must change an allowed runtime/profile path" in errors


def test_disk_policy_rejects_symlinks_and_non_json_profiles(tmp_path):
    profile = tmp_path / "configs/oracle-profiles/profile.txt"
    profile.parent.mkdir(parents=True)
    profile.write_text("not json")
    linked_test = tmp_path / "tests/test_oracle.py"
    linked_test.parent.mkdir(parents=True)
    linked_test.symlink_to(profile)
    errors = validate_changed_files_on_disk(
        POLICY,
        ["A\tconfigs/oracle-profiles/profile.txt", "M\ttests/test_oracle.py"],
        repo_root=tmp_path,
    )
    assert "Oracle profile must be a JSON file: configs/oracle-profiles/profile.txt" in errors
    assert "changed path must not be a symlink: tests/test_oracle.py" in errors


def test_evidence_requires_exact_head_three_diagnostics_and_ten_canaries():
    assert validate_evidence(
        POLICY,
        _evidence(),
        expected_commit=COMMIT,
        expected_campaign_id="so101-v010",
        expected_parent_manifest_sha256="b" * 64,
    ) == []
    evidence = _evidence()
    evidence["git_commit"] = "b" * 40
    evidence["diagnostics"][0]["seeds"][0]["success"] = False
    evidence["canaries"].pop()
    errors = validate_evidence(POLICY, evidence, expected_commit=COMMIT)
    assert "evidence git_commit does not match the PR head commit" in errors
    assert "diagnostic bilateral_contact_lost seed did not pass" in errors
    assert "evidence must contain exactly 10 canary seeds" in errors


def test_evidence_rejects_attempt_four_and_scene_or_success_criteria_changes():
    evidence = _evidence()
    evidence["scene_contract_unchanged"] = False
    evidence["success_criteria_unchanged"] = False
    evidence["canaries"][0]["attempts"] = 4
    errors = validate_evidence(POLICY, evidence, expected_commit=COMMIT)
    assert "evidence must prove scene contract unchanged" in errors
    assert "evidence must prove success criteria unchanged" in errors
    assert "canary attempts must be within 1..3" in errors


def test_evidence_rejects_campaign_parent_and_holdout_seed_reuse():
    evidence = _evidence()
    evidence["campaign_id"] = "other"
    evidence["parent_manifest_sha256"] = "c" * 64
    evidence["canaries"][0]["seed"] = evidence["diagnostics"][0]["seeds"][0]["seed"]
    errors = validate_evidence(
        POLICY,
        evidence,
        expected_commit=COMMIT,
        expected_campaign_id="so101-v010",
        expected_parent_manifest_sha256="b" * 64,
    )
    assert "evidence campaign_id does not match the campaign grant" in errors
    assert "evidence parent manifest does not match the campaign segment" in errors
    assert "diagnostic and canary seeds must be disjoint" in errors
