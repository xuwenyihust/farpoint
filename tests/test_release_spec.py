import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_release_version import check_versions  # noqa: E402
from farpoint.release_spec import load_release_spec


def test_current_release_spec_is_consistent():
    spec = load_release_spec()
    assert spec["tag"] == f"v{spec['version']}"
    assert Path(spec["path"]).parent.joinpath(spec["variation_config"]).is_file()
    assert check_versions() == []


def test_release_spec_rejects_prefixed_version(tmp_path):
    path = tmp_path / "release.toml"
    path.write_text(
        "\n".join(
            [
                'schema_version = "farpoint.release.v1"',
                'version = "v1.2.0"',
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
