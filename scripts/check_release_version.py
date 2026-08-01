#!/usr/bin/env python3
"""Fail when release-facing versions drift from release.toml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from farpoint import __version__  # noqa: E402
from farpoint.release_spec import DEFAULT_RELEASE_SPEC, load_release_spec  # noqa: E402


def check_versions(spec_path: Path = DEFAULT_RELEASE_SPEC) -> list[str]:
    spec = load_release_spec(spec_path)
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        package_version = tomllib.load(handle)["project"]["version"]
    expected = spec["version"]
    errors = []
    if package_version != expected:
        errors.append(f"pyproject.toml version is {package_version}, expected {expected}")
    if __version__ != expected:
        errors.append(f"farpoint.__version__ is {__version__}, expected {expected}")
    card = (PROJECT_ROOT / "docs" / "dataset-v1" / "huggingface-dataset-card.md").read_text(
        encoding="utf-8"
    )
    if f"`{spec['tag']}`" not in card:
        errors.append(f"Dataset Card does not document {spec['tag']}")
    if (PROJECT_ROOT / "scripts" / f"release_farpoint_v{expected.replace('.', '_')}.py").exists():
        errors.append("release script filename must not contain the release version")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-spec", type=Path, default=DEFAULT_RELEASE_SPEC)
    args = parser.parse_args()
    errors = check_versions(args.release_spec)
    if errors:
        for error in errors:
            print(f"VERSION_ERROR: {error}")
        return 1
    spec = load_release_spec(args.release_spec)
    print(f"VERSION_OK: {spec['tag']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
