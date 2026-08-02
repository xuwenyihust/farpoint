#!/usr/bin/env python3
"""Create coverage-first release evidence and select every successful episode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.position_collection import (  # noqa: E402
    build_collection_selection,
    build_coverage_release_manifest,
)
from farpoint.release_spec import load_release_spec  # noqa: E402

REQUIRED_EPISODE_FILES = (
    "metadata.json",
    "metrics.json",
    "observations.jsonl",
    "trajectory.jsonl",
    "labels.jsonl",
    "phase_events.jsonl",
)
REQUIRED_EPISODE_DIRECTORIES = ("preview", "observations/rgb", "observations/depth")


def resolve_complete_episode(episode_id: str, roots: list[Path]) -> Path:
    for root in roots:
        episode = root / episode_id
        if (
            episode.is_dir()
            and all((episode / name).is_file() for name in REQUIRED_EPISODE_FILES)
            and all((episode / name).is_dir() for name in REQUIRED_EPISODE_DIRECTORIES)
        ):
            return episode.resolve()
    raise ValueError(f"complete source episode artifacts are missing: {episode_id}")


def stage_selection_episodes(selection: dict, source_roots: list[Path], staging_root: Path) -> None:
    resolved = [
        (
            Path(entry["episode_dir"]).name,
            resolve_complete_episode(Path(entry["episode_dir"]).name, source_roots),
        )
        for entry in selection["episodes"]
    ]
    staging_root.mkdir(parents=True, exist_ok=False)
    for episode_id, source in resolved:
        destination = staging_root / episode_id
        destination.symlink_to(source, target_is_directory=True)


def write_new_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_manifest", type=Path)
    parser.add_argument("--release-manifest-output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--source-episode-root", type=Path, action="append", required=True)
    parser.add_argument("--staging-episode-root", type=Path, required=True)
    parser.add_argument("--dataset-id")
    args = parser.parse_args()

    collection = json.loads(args.collection_manifest.read_text(encoding="utf-8"))
    release = build_coverage_release_manifest(collection)
    if release["release_acceptance"]["accepted"] is not True:
        observed = release["release_acceptance"]["observed_covered_cells"]
        required = release["release_acceptance"]["required_cells"]
        raise ValueError(f"collection coverage is incomplete: {observed}/{required} cells")
    try:
        relative_staging_root = args.staging_episode_root.resolve().relative_to(
            PROJECT_ROOT.resolve()
        )
    except ValueError as error:
        raise ValueError("staging episode root must be repository-relative") from error
    selection = build_collection_selection(
        release,
        dataset_id=args.dataset_id or load_release_spec()["dataset_id"],
        episode_root=relative_staging_root.as_posix(),
    )
    stage_selection_episodes(selection, args.source_episode_root, args.staging_episode_root)
    write_new_json(args.release_manifest_output, release)
    write_new_json(args.selection_output, selection)
    print(
        "COVERAGE_SELECTION_OK: "
        f"{len(selection['episodes'])} successful episodes -> {args.selection_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
