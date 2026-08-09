"""Load and validate a dataset-specific Farpoint release specification."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_SPEC = PROJECT_ROOT / "configs" / "datasets" / "farpoint-ur10e-robotiq-2f85.toml"
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
    card_mode = spec.get("dataset_card_mode", "file")
    if card_mode not in {"file", "generated"}:
        raise ValueError("dataset_card_mode must be 'file' or 'generated'")
    if card_mode == "file" and not spec.get("dataset_card"):
        raise ValueError("file Dataset Card mode requires dataset_card")
    if card_mode == "generated":
        card = spec.get("card")
        if not isinstance(card, dict):
            raise ValueError("generated Dataset Card mode requires a [card] table")
        missing_card_fields = sorted(
            {"pretty_name", "license", "description", "tags"}.difference(card)
        )
        if missing_card_fields:
            raise ValueError(
                "generated Dataset Card metadata is missing fields: "
                + ", ".join(missing_card_fields)
            )
        if not isinstance(card["tags"], list) or not card["tags"]:
            raise ValueError("generated Dataset Card tags must be a non-empty list")
    spec["dataset_card_mode"] = card_mode
    spec["path"] = str(path)
    spec["dataset_tag"] = f"v{spec['dataset_version']}"
    return spec
