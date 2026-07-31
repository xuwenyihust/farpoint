#!/usr/bin/env python3
"""Build and optionally publish one reproducible Farpoint v1.2 release."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from create_benchmark_from_episodes import build_manifest  # noqa: E402
from farpoint.release import audit_viewer_package, prepare_viewer_package  # noqa: E402
from validate_lerobot_dataset import validate_dataset  # noqa: E402


def parse_episode_ids(args: argparse.Namespace) -> list[str]:
    if args.episode_ids_file:
        values = json.loads(args.episode_ids_file.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ValueError("episode ids file must contain a JSON array")
        return [str(value) for value in values]
    if args.episode_ids:
        return args.episode_ids
    raise ValueError("provide --episode-ids-file or at least one --episode-id")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def publish_huggingface(package: Path, repo_id: str, version: str, commit_message: str) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(package),
        commit_message=commit_message,
    )
    tag = api.create_tag(
        repo_id=repo_id,
        repo_type="dataset",
        tag=version,
        tag_message=f"Farpoint dataset release {version}",
    )
    return {"commit": getattr(commit, "oid", str(commit)), "tag": str(tag)}


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    benchmark_dir = args.output_dir / "benchmark"
    canonical_dir = args.output_dir / "canonical"
    public_dir = args.output_dir / "public"

    if args.source_dataset:
        canonical_source = args.source_dataset.resolve()
        source_validation = validate_dataset(canonical_source)
        if not source_validation["valid"]:
            raise ValueError("source dataset failed validation: " + "; ".join(source_validation["errors"]))
        shutil.copytree(canonical_source, canonical_dir)
        benchmark = {"source": "existing_dataset", "dataset_root": str(canonical_source)}
    else:
        from export_lerobot_v1_mini import export_mini

        episode_ids = parse_episode_ids(args)
        episode_dirs = [args.episode_root / episode_id for episode_id in episode_ids]
        benchmark_id = args.benchmark_id or f"farpoint_{args.release_version.replace('.', '_')}_release"
        manifest = build_manifest(
            args.episode_root,
            benchmark_id,
            episode_ids,
            args.task_name,
            args.task_type,
            args.min_success_rate,
        )
        write_json(benchmark_dir / "manifest.json", manifest)
        benchmark = {"benchmark_id": benchmark_id, **manifest}
        selected = [path for path in episode_dirs if (path / "metadata.json").is_file() and (path / "metrics.json").is_file()]
        if not selected:
            raise ValueError("no complete episodes available for export")
        export_mini(selected, canonical_dir, args.dataset_id)

    canonical_validation = validate_dataset(canonical_dir)
    if not canonical_validation["valid"]:
        raise ValueError("canonical dataset failed validation: " + "; ".join(canonical_validation["errors"]))
    package_result = prepare_viewer_package(canonical_dir, public_dir)
    viewer_audit = audit_viewer_package(public_dir)
    release = {
        "release_version": args.release_version,
        "dataset_id": args.dataset_id,
        "benchmark": benchmark,
        "canonical_validation": canonical_validation,
        "viewer_package": package_result,
        "viewer_audit": viewer_audit,
        "published": None,
    }
    if args.publish:
        if not args.hf_repo_id:
            raise ValueError("--hf-repo-id is required with --publish")
        release["published"] = publish_huggingface(
            public_dir,
            args.hf_repo_id,
            args.release_version,
            f"Release Farpoint {args.release_version}",
        )
    write_json(args.output_dir / "release.json", release)
    return release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path)
    parser.add_argument("--episode-root", type=Path)
    parser.add_argument("--episode-id", dest="episode_ids", action="append")
    parser.add_argument("--episode-ids-file", type=Path)
    parser.add_argument("--dataset-id", default="farpoint_ur10e_robotiq_2f85")
    parser.add_argument("--release-version", default="v1.2.0")
    parser.add_argument("--benchmark-id")
    parser.add_argument("--task-name", default="isaac_perception_contact_scene")
    parser.add_argument("--task-type", default="variation_expansion_v1")
    parser.add_argument("--min-success-rate", type=float, default=0.90)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--hf-repo-id")
    args = parser.parse_args()
    if not args.source_dataset and not args.episode_root:
        parser.error("provide --source-dataset or --episode-root")
    result = run(args)
    print(json.dumps({"release_version": result["release_version"], "viewer_audit": result["viewer_audit"], "published": result["published"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
