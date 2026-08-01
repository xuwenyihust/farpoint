"""Load and validate Farpoint's authoritative release specification."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_SPEC = PROJECT_ROOT / "release.toml"
SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
REQUIRED_FIELDS = {
    "schema_version",
    "version",
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
    if spec["schema_version"] != "farpoint.release.v1":
        raise ValueError("release spec schema_version must be farpoint.release.v1")
    if not SEMVER.fullmatch(str(spec["version"])):
        raise ValueError("release version must use MAJOR.MINOR.PATCH without a v prefix")
    spec["path"] = str(path)
    spec["tag"] = f"v{spec['version']}"
    return spec
