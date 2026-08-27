"""Load and validate a dataset-specific Farpoint release specification."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_SPEC = PROJECT_ROOT / "configs" / "datasets" / "ur10e-robotiq-2f85.toml"
SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
REQUIRED_FIELDS = {
    "schema_version",
    "dataset_version",
    "dataset_id",
    "hf_repo_id",
    "dataset_schema",
    "variation_schema",
    "lerobot_format",
    "variation_config",
}


def load_release_spec(path: Path | str = DEFAULT_RELEASE_SPEC) -> dict:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        spec = tomllib.load(handle)
    missing = sorted(REQUIRED_FIELDS.difference(spec))
    if missing:
        raise ValueError(f"release spec is missing fields: {', '.join(missing)}")
    if spec["schema_version"] != "farpoint.dataset-release.v1":
        raise ValueError("release spec schema_version must be farpoint.dataset-release.v1")
    if not SEMVER.fullmatch(str(spec["dataset_version"])):
        raise ValueError("dataset version must use MAJOR.MINOR.PATCH without a v prefix")
    source_configs = spec.get("source_configs", [spec["variation_config"]])
    if (
        not isinstance(source_configs, list)
        or not source_configs
        or any(not isinstance(value, str) or not value.strip() for value in source_configs)
    ):
        raise ValueError("source_configs must be a non-empty list of paths")
    if len(set(source_configs)) != len(source_configs):
        raise ValueError("source_configs must not contain duplicate paths")
    if spec["variation_config"] not in source_configs:
        raise ValueError("variation_config must be included in source_configs")
    spec["source_configs"] = source_configs
    spec["path"] = str(path)
    spec["dataset_tag"] = f"v{spec['dataset_version']}"
    return spec
