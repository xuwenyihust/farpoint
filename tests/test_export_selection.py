import hashlib
import json

import pytest

from farpoint.export_selection import (
    compose_export_selections,
    repartition_export_selection,
)


def _write_selection(path, dataset_id, episode_dirs, *, split="train"):
    payload = {
        "schema_version": "farpoint.export-selection.v1",
        "dataset_id": dataset_id,
        "episodes": [
            {
                "episode_dir": episode_dir,
                "trial_id": f"trial-{index}",
                "split": split,
            }
            for index, episode_dir in enumerate(episode_dirs)
        ],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compose_export_selections_binds_sources_and_episode_roles(tmp_path):
    base = tmp_path / "base.json"
    increment = tmp_path / "increment.json"
    base_sha = _write_selection(base, "base-dataset", [str(tmp_path / "episode-a")])
    increment_sha = _write_selection(
        increment,
        "increment-dataset",
        [str(tmp_path / "episode-b")],
        split="validation",
    )

    result = compose_export_selections(
        [("base", base), ("recovery_increment", increment)],
        dataset_id="composed-dataset",
        selection_policy="immutable_base_plus_recovery",
    )

    assert result["selection_sources"] == [
        {
            "role": "base",
            "dataset_id": "base-dataset",
            "selection_sha256": base_sha,
            "episode_count": 1,
        },
        {
            "role": "recovery_increment",
            "dataset_id": "increment-dataset",
            "selection_sha256": increment_sha,
            "episode_count": 1,
        },
    ]
    assert [row["selection_source_role"] for row in result["episodes"]] == [
        "base",
        "recovery_increment",
    ]


def test_compose_export_selections_rejects_duplicate_episode_directories(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    episode = str(tmp_path / "episode")
    _write_selection(first, "first", [episode])
    _write_selection(second, "second", [episode])

    with pytest.raises(ValueError, match="duplicate episode directory"):
        compose_export_selections(
            [("first", first), ("second", second)],
            dataset_id="composed",
            selection_policy="test",
        )


def test_compose_export_selections_rejects_invalid_split(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_selection(first, "first", [str(tmp_path / "episode-a")])
    _write_selection(
        second, "second", [str(tmp_path / "episode-b")], split="holdout"
    )

    with pytest.raises(ValueError, match="invalid split"):
        compose_export_selections(
            [("first", first), ("second", second)],
            dataset_id="composed",
            selection_policy="test",
        )


def test_repartition_export_selection_is_exact_deterministic_and_source_bound(tmp_path):
    source = tmp_path / "source.json"
    source_sha = _write_selection(
        source,
        "prelift-recovery20",
        [str(tmp_path / f"episode-{index:02d}") for index in range(20)],
    )

    first = repartition_export_selection(
        source,
        dataset_id="prelift-recovery20-repartitioned",
        selection_policy="deterministic_18_train_2_validation",
        split_counts={"train": 18, "validation": 2},
        seed=20260814,
    )
    second = repartition_export_selection(
        source,
        dataset_id="prelift-recovery20-repartitioned",
        selection_policy="deterministic_18_train_2_validation",
        split_counts={"train": 18, "validation": 2},
        seed=20260814,
    )

    assert first == second
    assert first["selection_sources"] == [
        {
            "role": "repartition_source",
            "dataset_id": "prelift-recovery20",
            "selection_sha256": source_sha,
            "episode_count": 20,
        }
    ]
    assert first["split_assignment"] == {
        "algorithm": "sha256_rank_v1",
        "seed": 20260814,
        "targets": {"train": 18, "validation": 2, "test": 0},
    }
    assert sum(row["split"] == "train" for row in first["episodes"]) == 18
    assert sum(row["split"] == "validation" for row in first["episodes"]) == 2
    assert {row["source_split"] for row in first["episodes"]} == {"train"}
    assert sorted(row["split_assignment_rank"] for row in first["episodes"]) == list(
        range(20)
    )
    assert json.loads(source.read_text())["episodes"][0]["split"] == "train"


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ({"train": 19}, "sum to the source episode count"),
        ({"train": 19, "holdout": 1}, "invalid split targets"),
        ({"train": 19, "validation": -1, "test": 2}, "non-negative integers"),
        ({"train": 18.0, "validation": 2}, "non-negative integers"),
        ({"train": 18, "validation": True, "test": 1}, "non-negative integers"),
    ],
)
def test_repartition_export_selection_rejects_invalid_targets(tmp_path, counts, message):
    source = tmp_path / "source.json"
    _write_selection(
        source,
        "source",
        [str(tmp_path / f"episode-{index:02d}") for index in range(20)],
    )
    with pytest.raises(ValueError, match=message):
        repartition_export_selection(
            source,
            dataset_id="output",
            selection_policy="test",
            split_counts=counts,
            seed=1,
        )
