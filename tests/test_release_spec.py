import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_release_version  # noqa: E402
from check_release_version import check_versions  # noqa: E402
from farpoint.release_spec import load_release_spec


def test_current_release_spec_is_consistent():
    spec = load_release_spec()
    assert spec["dataset_tag"] == f"v{spec['dataset_version']}"
    project_root = Path(__file__).resolve().parents[1]
    assert project_root.joinpath(spec["variation_config"]).is_file()
    assert check_versions() == []


def test_current_dataset_card_separates_frames_from_metadata_in_viewer():
    spec = load_release_spec()
    project_root = Path(__file__).resolve().parents[1]
    card = project_root.joinpath(spec["dataset_card"]).read_text()

    assert 'path: "data/**/*.parquet"' in card
    assert "config_name: episode_metadata" in card
    assert 'path: "meta/episode_metadata.parquet"' in card


def test_release_spec_rejects_prefixed_version(tmp_path):
    path = tmp_path / "dataset-release.toml"
    path.write_text(
        "\n".join(
            [
                'schema_version = "farpoint.dataset-release.v1"',
                'dataset_version = "v1.2.0"',
                'dataset_id = "dataset"',
                'hf_repo_id = "owner/dataset"',
                'dataset_schema = "farpoint.dataset.v1"',
                'variation_schema = "farpoint.variation.v1"',
                'lerobot_format = "v3"',
                'variation_config = "config.json"',
                'dataset_card = "README.md"',
            ]
        )
    )
    with pytest.raises(ValueError, match="without a v prefix"):
        load_release_spec(path)


def test_code_and_dataset_versions_are_independent(tmp_path):
    path = tmp_path / "dataset-release.toml"
    path.write_text(
        "\n".join(
            [
                'schema_version = "farpoint.dataset-release.v1"',
                'dataset_version = "9.4.1"',
                'dataset_id = "another_dataset"',
                'hf_repo_id = "owner/another-dataset"',
                'dataset_schema = "farpoint.dataset.v2"',
                'variation_schema = "farpoint.variation.v2"',
                'lerobot_format = "v3"',
                'variation_config = "config.json"',
                'dataset_card = "README.md"',
            ]
        )
    )

    spec = load_release_spec(path)

    assert spec["dataset_version"] == "9.4.1"
    assert spec["dataset_tag"] == "v9.4.1"


def test_version_check_does_not_require_code_to_match_dataset(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "2.0.0"\n')
    (tmp_path / "dataset-card.md").write_text("Published dataset: `v9.4.1`\n")
    spec_path = tmp_path / "dataset-release.toml"
    spec_path.write_text(
        "\n".join(
            [
                'schema_version = "farpoint.dataset-release.v1"',
                'dataset_version = "9.4.1"',
                'dataset_id = "another_dataset"',
                'hf_repo_id = "owner/another-dataset"',
                'dataset_schema = "farpoint.dataset.v2"',
                'variation_schema = "farpoint.variation.v2"',
                'lerobot_format = "v3"',
                'variation_config = "config.json"',
                'dataset_card = "dataset-card.md"',
            ]
        )
    )
    monkeypatch.setattr(check_release_version, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(check_release_version, "__version__", "2.0.0")

    assert check_versions(spec_path) == []
