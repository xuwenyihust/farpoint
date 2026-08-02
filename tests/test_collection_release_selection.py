from pathlib import Path
import sys

import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from create_collection_release_selection import (  # noqa: E402
    REQUIRED_EPISODE_DIRECTORIES,
    REQUIRED_EPISODE_FILES,
    resolve_complete_episode,
    stage_selection_episodes,
)


def complete_episode(root: Path, episode_id: str) -> Path:
    episode = root / episode_id
    episode.mkdir(parents=True)
    for name in REQUIRED_EPISODE_FILES:
        path = episode / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    for name in REQUIRED_EPISODE_DIRECTORIES:
        (episode / name).mkdir(parents=True, exist_ok=True)
    return episode


def test_episode_resolver_skips_lightweight_dashboard_copy(tmp_path):
    episode_id = "episode-0000"
    lightweight = tmp_path / "dashboard"
    (lightweight / episode_id).mkdir(parents=True)
    (lightweight / episode_id / "metadata.json").write_text("{}\n", encoding="utf-8")
    raw = tmp_path / "raw"
    complete = complete_episode(raw, episode_id)

    assert resolve_complete_episode(episode_id, [lightweight, raw]) == complete.resolve()


def test_staging_materializes_every_selection_as_a_symlink(tmp_path):
    raw = tmp_path / "raw"
    for episode_id in ("episode-0000", "episode-0001"):
        complete_episode(raw, episode_id)
    selection = {
        "episodes": [
            {"episode_dir": f"outputs/staging/{episode_id}"}
            for episode_id in ("episode-0000", "episode-0001")
        ]
    }
    staging = tmp_path / "staging"

    stage_selection_episodes(selection, [raw], staging)

    assert all(
        (staging / episode_id).is_symlink() for episode_id in ("episode-0000", "episode-0001")
    )


def test_staging_fails_before_creating_output_when_source_is_incomplete(tmp_path):
    selection = {"episodes": [{"episode_dir": "outputs/staging/episode-missing"}]}
    staging = tmp_path / "staging"

    with pytest.raises(ValueError, match="complete source episode artifacts are missing"):
        stage_selection_episodes(selection, [tmp_path / "raw"], staging)

    assert not staging.exists()
