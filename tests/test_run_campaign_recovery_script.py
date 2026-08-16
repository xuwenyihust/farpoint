from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_campaign_recovery", ROOT / "scripts/run_campaign_recovery.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_root_episode_alias_resolves_to_segment_directory(tmp_path):
    module = _load_script()
    manifest = tmp_path / "segments/segment-000/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    expected = manifest.parent / "episodes"
    expected.mkdir()

    resolved = module._resolve_episodes_root(
        tmp_path,
        {
            "manifest": "segments/segment-000/manifest.json",
            "episodes_root": "episodes",
        },
        manifest,
    )

    assert resolved == expected.resolve()


def test_explicit_episode_root_is_not_rewritten(tmp_path):
    module = _load_script()
    manifest = tmp_path / "segments/segment-000/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")

    resolved = module._resolve_episodes_root(
        tmp_path,
        {
            "manifest": "segments/segment-000/manifest.json",
            "episodes_root": "custom/episodes",
        },
        manifest,
    )

    assert resolved == (tmp_path / "custom/episodes").resolve()
