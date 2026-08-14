import hashlib
import json

import pytest

from farpoint.export_selection import compose_export_selections


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
