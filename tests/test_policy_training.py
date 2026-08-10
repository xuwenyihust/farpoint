import json
from pathlib import Path

import pytest

from farpoint.contracts import load_schema, validate_contract
from farpoint.policy_training import (
    create_training_view,
    load_training_spec,
    parse_episode_slice,
    training_arguments,
    unflatten_episode_stats,
    validate_dataset_info,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "training" / "so101_act_v0_0_3_baseline.json"


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
