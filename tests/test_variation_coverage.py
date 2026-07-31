import json

from farpoint.variation_coverage import audit_variation_coverage


def _episode(root, name, variation_id, seed, success=True, complete=True):
    episode = root / name
    episode.mkdir()
    (episode / "metadata.json").write_text(json.dumps({
        "episode_id": name,
        "variation_id": variation_id,
        "episode_seed": seed,
        "variation": {"variation_id": variation_id, "object_type": "cube", "seed": seed},
    }))
    (episode / "metrics.json").write_text(json.dumps({"success": success, "dataset_valid": success}))
    if complete:
        for filename in ("trajectory.jsonl", "observations.jsonl"):
            (episode / filename).write_text("x")
        (episode / "preview").mkdir()
        (episode / "preview" / "preview_0001.png").write_text("x")


def test_coverage_requires_profile_pass_and_artifacts(tmp_path):
    _episode(tmp_path, "pass_a", "left", 0)
    _episode(tmp_path, "fail_b", "right", 0, success=False)
    report = audit_variation_coverage(tmp_path, ["left", "right"], expected_seeds=[0], min_passes=2)
    assert report["passing_episode_count"] == 1
    assert report["missing_passing_profiles"] == ["right"]
    assert report["gate_passed"] is False


def test_coverage_accepts_complete_passing_profiles(tmp_path):
    _episode(tmp_path, "pass_a", "left", 0)
    _episode(tmp_path, "pass_b", "right", 0)
    report = audit_variation_coverage(tmp_path, ["left", "right"], expected_seeds=[0], min_passes=2)
    assert report["gate_passed"] is True
