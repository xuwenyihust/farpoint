import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_release_version  # noqa: E402
from check_release_version import check_versions  # noqa: E402
from farpoint.release_spec import load_release_spec


SO101_RELEASE_SPEC = (
    Path(__file__).resolve().parents[1] / "configs" / "datasets" / "farpoint-so101.toml"
)


def test_current_release_spec_is_consistent():
    spec = load_release_spec()
    assert spec["dataset_tag"] == f"v{spec['dataset_version']}"
    project_root = Path(__file__).resolve().parents[1]
    assert project_root.joinpath(spec["variation_config"]).is_file()
    assert check_versions() == []


def test_so101_release_spec_uses_extensible_repository_and_v3_contracts():
    spec = load_release_spec(SO101_RELEASE_SPEC)

    assert spec["dataset_id"] == "farpoint_so101"
    assert spec["hf_repo_id"] == "wenyixu101/farpoint-so101"
    assert spec["dataset_tag"] == "v0.0.3"
    assert spec["dataset_schema"] == "farpoint.dataset.v3"
    assert spec["variation_schema"] == "farpoint.variation.v3"
    assert check_versions(SO101_RELEASE_SPEC) == []


def test_so101_changelog_keeps_published_version_history():
    project_root = Path(__file__).resolve().parents[1]
    changelog = project_root.joinpath("docs/dataset-v3/farpoint-so101-changelog.md").read_text(
        encoding="utf-8"
    )

    assert "## v0.0.3" in changelog
    assert "## v0.0.2" in changelog
    assert "## v0.0.1" in changelog
    assert "## v0.0.0" in changelog


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
            ]
        )
    )

    spec = load_release_spec(path)

    assert spec["dataset_version"] == "9.4.1"
    assert spec["dataset_tag"] == "v9.4.1"


def test_version_check_does_not_require_code_to_match_dataset(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "2.0.0"\n')
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
            ]
        )
    )
    monkeypatch.setattr(check_release_version, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(check_release_version, "__version__", "2.0.0")

    assert check_versions(spec_path) == []
