import json
from pathlib import Path

import pytest

from farpoint.contracts import load_schema, validate_contract
from farpoint.policy_training import (
    create_training_view,
    evenly_spaced_indices,
    load_training_spec,
    parse_episode_slice,
    select_validation_checkpoint,
    training_arguments,
    unflatten_episode_stats,
    validate_dataset_info,
    validation_profile,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "training" / "so101_act_v0_0_3_baseline.json"
PILOT_CONFIG = ROOT / "configs" / "training" / "so101_act_v0_0_3_pilot.json"
BASELINE_20K_CONFIG = (
    ROOT / "configs" / "training" / "so101_act_v0_0_3_baseline_20k.json"
)


def test_frozen_act_contract_is_valid_and_partitions_all_episodes():
    spec = load_training_spec(CONFIG)
    assert validate_contract(spec) == []
    assert parse_episode_slice(spec["dataset"]["splits"]["train"]) == list(range(128))
    assert parse_episode_slice(spec["dataset"]["splits"]["validation"]) == list(
        range(128, 142)
    )
    assert parse_episode_slice(spec["dataset"]["splits"]["test"]) == list(range(142, 160))
    assert load_schema("farpoint.policy-training.v1")["title"].endswith("v1")


@pytest.mark.parametrize("expression", ["train", "0:0", "4:2", "-1:2", "0:2:1", "1:"])
def test_episode_slice_rejects_ambiguous_or_empty_ranges(expression):
    with pytest.raises(ValueError):
        parse_episode_slice(expression)


def test_dataset_metadata_must_match_pinned_release():
    spec = load_training_spec(CONFIG)
    dataset = spec["dataset"]
    info = {
        "codebase_version": dataset["codebase_version"],
        "total_episodes": dataset["expected"]["total_episodes"],
        "total_frames": dataset["expected"]["total_frames"],
        "fps": dataset["expected"]["fps"],
        "splits": dataset["splits"],
        "features": dataset["required_features"],
    }
    validate_dataset_info(spec, info)
    info["total_frames"] += 1
    with pytest.raises(ValueError, match="total_frames"):
        validate_dataset_info(spec, info)


def test_training_view_replaces_only_stats_and_preserves_source(tmp_path):
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "videos").mkdir()
    original_stats = {"action": {"mean": [100.0]}}
    (source / "meta" / "stats.json").write_text(json.dumps(original_stats), encoding="utf-8")
    (source / "meta" / "info.json").write_text("{}", encoding="utf-8")

    def write_stats(stats, root):
        (root / "meta" / "stats.json").write_text(json.dumps(stats), encoding="utf-8")

    source_hash, view_hash = create_training_view(
        source, tmp_path / "view", {"action": {"mean": [1.0]}}, write_stats
    )
    assert source_hash != view_hash
    assert json.loads((source / "meta" / "stats.json").read_text()) == original_stats
    assert (tmp_path / "view" / "data").is_symlink()
    assert (tmp_path / "view" / "videos").is_symlink()
    with pytest.raises(FileExistsError):
        create_training_view(source, tmp_path / "view", {}, write_stats)


def test_episode_stats_unflatten_and_smoke_arguments_are_split_safe(tmp_path):
    stats = unflatten_episode_stats(
        {"episode_index": 0, "stats/action/mean": [1.0], "stats/action/count": [2]}
    )
    assert stats["action"]["mean"].tolist() == [1.0]
    assert stats["action"]["count"].tolist() == [2]
    spec = load_training_spec(CONFIG)
    arguments = training_arguments(spec, tmp_path / "view", tmp_path / "out", "smoke")
    episode_arg = next(arg for arg in arguments if arg.startswith("--dataset.episodes="))
    assert "--steps=1" in arguments
    assert "--save_checkpoint=false" in arguments
    assert "--policy.push_to_hub=false" in arguments
    assert episode_arg.startswith("--dataset.episodes=[0,1,2")
    assert episode_arg.endswith(",127]")
    assert ",128" not in episode_arg


def test_pilot_contract_and_arguments_are_frozen_and_split_safe(tmp_path):
    spec = load_training_spec(PILOT_CONFIG)
    assert spec["pilot"]["steps"] == 1000
    assert spec["pilot"]["save_freq"] == 250
    assert spec["validation"]["split"] == "validation"
    arguments = training_arguments(spec, tmp_path / "view", tmp_path / "out", "pilot")
    assert "--steps=1000" in arguments
    assert "--save_freq=250" in arguments
    assert "--policy.push_to_hub=false" in arguments
    episode_arg = next(arg for arg in arguments if arg.startswith("--dataset.episodes="))
    assert episode_arg.endswith(",127]")
    assert ",128" not in episode_arg


def test_20k_baseline_contract_selects_training_checkpoints(tmp_path):
    spec = load_training_spec(BASELINE_20K_CONFIG)
    assert validation_profile(spec) == "training"
    assert spec["training"]["steps"] == 20000
    assert spec["training"]["save_freq"] == 5000
    arguments = training_arguments(spec, tmp_path / "view", tmp_path / "out", "training")
    assert "--steps=20000" in arguments
    assert "--save_freq=5000" in arguments
    assert "--policy.push_to_hub=false" in arguments
    assert spec["dataset"]["splits"]["test"] == "142:160"
    runner = (ROOT / "scripts" / "run_so101_act_baseline_20k.sh").read_text()
    assert "--profile training" in runner
    assert "evaluate_act_checkpoints.py" in runner


def test_evenly_spaced_validation_indices_cover_boundaries_without_duplicates():
    indices = evenly_spaced_indices(10164, 128)
    assert len(indices) == len(set(indices)) == 128
    assert indices[0] == 0
    assert indices[-1] == 10163
    assert indices == sorted(indices)
    with pytest.raises(ValueError):
        evenly_spaced_indices(3, 4)


def test_validation_selection_requires_a_later_meaningful_improvement():
    results = [
        {"step": 250, "mean_loss": 10.0},
        {"step": 500, "mean_loss": 8.0},
        {"step": 750, "mean_loss": 8.5},
    ]
    best, improvement = select_validation_checkpoint(results, 0.05)
    assert best["step"] == 500
    assert improvement == pytest.approx(0.2)
    with pytest.raises(ValueError, match="no later checkpoint"):
        select_validation_checkpoint(
            [{"step": 250, "mean_loss": 8.0}, {"step": 500, "mean_loss": 9.0}], 0
        )
    with pytest.raises(ValueError, match="below"):
        select_validation_checkpoint(
            [{"step": 250, "mean_loss": 10.0}, {"step": 500, "mean_loss": 9.8}], 0.05
        )


def test_training_image_preserves_ngc_cuda_torch_builds():
    dockerfile = (ROOT / "docker" / "so101-lerobot-training" / "Dockerfile").read_text()

    assert "FROM nvcr.io/nvidia/pytorch:26.01-py3@sha256:" in dockerfile
    assert "--constraint /opt/farpoint-training-constraints.txt" in dockerfile
    assert "torch.version.cuda is not None" in dockerfile
    assert "m.version('torch')" in dockerfile
    assert "m.version('torchvision')" in dockerfile
    assert "Version(m.version('torch')) < Version('2.11.0')" in dockerfile
