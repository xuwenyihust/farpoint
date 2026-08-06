#!/usr/bin/env python3
"""Fetch and verify the pinned NVIDIA SO-101 USD asset."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


COMMIT = "ce807d99724cb65671abec01f908a2fcb4a6eab7"
SHA256 = "11f5f0bb5f2fae3eefebbcd07dfafc6b14602f6c4e5dae8f21a4a46892991006"
URL = (
    "https://media.githubusercontent.com/media/isaac-sim/"
    f"Sim-to-Real-SO-101-Workshop/{COMMIT}/"
    "source/sim_to_real_so101/assets/usd/SO-ARM101-USD.usd"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(destination: Path, *, source_url: str = URL) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == SHA256:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(source_url, headers={"User-Agent": "Farpoint/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual = sha256_file(temporary)
        if actual != SHA256:
            raise ValueError(f"SO-101 USD checksum mismatch: expected {SHA256}, got {actual}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(".cache/farpoint/assets/so101") / COMMIT / "SO-ARM101-USD.usd",
    )
    args = parser.parse_args()
    print(fetch(args.destination.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
