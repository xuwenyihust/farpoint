#!/usr/bin/env python3
"""Build, validate, stage, or publish a reproducible Farpoint dataset release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from create_benchmark_from_episodes import build_manifest  # noqa: E402
from farpoint.release import audit_viewer_package, prepare_viewer_package  # noqa: E402
from farpoint.release_spec import DEFAULT_RELEASE_SPEC, load_release_spec  # noqa: E402
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
        delete_patterns=["meta/*.json", "meta/*.jsonl"],
    )
    api.create_tag(
        repo_id=repo_id,
        repo_type="dataset",
        tag=version,
        tag_message=f"Farpoint dataset release {version}",
    )
    return {"commit": getattr(commit, "oid", str(commit)), "tag": version}


def git_revision() -> str:
    override = os.environ.get("FARPOINT_CODE_REVISION")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_accepted(evidence: dict | None) -> bool:
    if not isinstance(evidence, dict):
        return False
    if evidence.get("schema_version") in {
        "farpoint.benchmark.v2",
        "farpoint.collection.v1",
    }:
        if evidence.get("release_policy") == "coverage_first_all_successful":
            return (evidence.get("release_acceptance") or {}).get("accepted") is True
        return (evidence.get("acceptance") or {}).get("accepted") is True
    return evidence.get("accepted") is True


def build_release(args: argparse.Namespace, spec: dict) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    canonical_dir = args.output_dir / "canonical"
    public_dir = args.output_dir / "public"
    evidence_validation_path = None
    collection_manifest = getattr(args, "collection_manifest", None)
    benchmark_manifest = getattr(args, "benchmark_manifest", None)
    if collection_manifest and benchmark_manifest:
        raise ValueError("provide only one collection or benchmark manifest")
    supplied_evidence = collection_manifest or benchmark_manifest
    evidence_kind = "collection" if collection_manifest else "benchmark"
    evidence_dir = args.output_dir / evidence_kind

    if args.source_dataset:
        canonical_source = args.source_dataset.resolve()
        source_validation = validate_dataset(canonical_source)
        if not source_validation["valid"]:
            raise ValueError(
                "source dataset failed validation: " + "; ".join(source_validation["errors"])
            )
        is_v2 = source_validation.get("schema_version") == "farpoint.dataset.v2"
        if is_v2 and not supplied_evidence:
            raise ValueError(
                "Farpoint v2 releases require --collection-manifest or --benchmark-manifest"
            )
        benchmark_id = args.benchmark_id or f"farpoint_{spec['tag'].replace('.', '_')}_release"
        if supplied_evidence:
            manifest = json.loads(supplied_evidence.read_text(encoding="utf-8"))
        else:
            source_metadata = canonical_source / "meta" / "episode_metadata.jsonl"
            records = (
                [
                    json.loads(line)
                    for line in source_metadata.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if source_metadata.is_file()
                else []
            )
            passed = sum(1 for record in records if record.get("success"))
            manifest = {
                "schema_version": "benchmark.v1",
                "benchmark_id": benchmark_id,
                "task_name": args.task_name,
                "task_type": args.task_type,
                "planned_trials": len(records),
                "completed_trials": len(records),
                "passed_trials": passed,
                "success_rate": passed / len(records) if records else 0.0,
                "accepted": bool(records) and passed / len(records) >= args.min_success_rate,
                "provenance": {"type": "release_source_dataset", "source": str(canonical_source)},
                "trials": [
                    {"episode_id": record.get("episode_id"), "success": bool(record.get("success"))}
                    for record in records
                ],
            }
        evidence_validation_path = evidence_dir / "manifest.json"
        write_json(evidence_validation_path, manifest)
        if is_v2:
            source_validation = validate_dataset(canonical_source, evidence_validation_path)
            if not source_validation["valid"]:
                raise ValueError(
                    "source dataset failed release evidence validation: "
                    + "; ".join(source_validation["errors"])
                )
        shutil.copytree(canonical_source, canonical_dir)
        evidence = manifest
    else:
        from export_lerobot_v1_mini import export_mini

        episode_ids = parse_episode_ids(args)
        episode_dirs = [args.episode_root / episode_id for episode_id in episode_ids]
        benchmark_id = args.benchmark_id or f"farpoint_{spec['tag'].replace('.', '_')}_release"
        manifest = build_manifest(
            args.episode_root,
            benchmark_id,
            episode_ids,
            args.task_name,
            args.task_type,
            args.min_success_rate,
        )
        write_json(evidence_dir / "manifest.json", manifest)
        evidence_validation_path = evidence_dir / "manifest.json"
        evidence = {"benchmark_id": benchmark_id, **manifest}
        selected = [
            path
            for path in episode_dirs
            if (path / "metadata.json").is_file() and (path / "metrics.json").is_file()
        ]
        if not selected:
            raise ValueError("no complete episodes available for export")
        export_mini(selected, canonical_dir, spec["dataset_id"])

    canonical_validation = (
        validate_dataset(canonical_dir, evidence_validation_path)
        if evidence_validation_path
        else validate_dataset(canonical_dir)
    )
    if not canonical_validation["valid"]:
        raise ValueError(
            "canonical dataset failed validation: " + "; ".join(canonical_validation["errors"])
        )
    package_result = prepare_viewer_package(canonical_dir, public_dir)
    viewer_audit = audit_viewer_package(public_dir)
    release = {
        "schema_version": "farpoint.release-manifest.v1",
        "release_version": spec["tag"],
        "dataset_id": spec["dataset_id"],
        "hf_repo_id": spec["hf_repo_id"],
        "code_revision": git_revision(),
        "release_spec_sha256": file_sha256(Path(spec["path"])),
        evidence_kind: evidence,
        "canonical_validation": canonical_validation,
        "viewer_package": package_result,
        "viewer_audit": viewer_audit,
        "published": None,
    }
    write_json(args.output_dir / "release.json", release)
    return release


def validate_release(release_dir: Path, spec: dict) -> dict:
    release_dir = release_dir.resolve()
    manifest = json.loads((release_dir / "release.json").read_text(encoding="utf-8"))
    benchmark_path = release_dir / "benchmark" / "manifest.json"
    collection_path = release_dir / "collection" / "manifest.json"
    v2_dataset = (release_dir / "canonical" / "meta" / "farpoint_v2.json").is_file()
    preflight_errors = []
    evidence_paths = [path for path in (benchmark_path, collection_path) if path.is_file()]
    if v2_dataset and len(evidence_paths) != 1:
        if manifest.get("collection"):
            preflight_errors.append("Farpoint v2 release is missing collection/manifest.json")
        elif manifest.get("benchmark"):
            preflight_errors.append("Farpoint v2 release is missing benchmark/manifest.json")
        else:
            preflight_errors.append(
                "Farpoint v2 release requires exactly one benchmark or collection manifest"
            )
    evidence_path = evidence_paths[0] if len(evidence_paths) == 1 else None
    canonical = (
        validate_dataset(release_dir / "canonical", evidence_path)
        if v2_dataset and evidence_path
        else validate_dataset(release_dir / "canonical")
    )
    public = audit_viewer_package(release_dir / "public")
    errors = preflight_errors
    if manifest.get("release_version") != spec["tag"]:
        errors.append("release manifest version does not match release.toml")
    if manifest.get("dataset_id") != spec["dataset_id"]:
        errors.append("release manifest dataset_id does not match release.toml")
    if manifest.get("hf_repo_id") != spec["hf_repo_id"]:
        errors.append("release manifest hf_repo_id does not match release.toml")
    if manifest.get("release_spec_sha256") != file_sha256(Path(spec["path"])):
        errors.append("release.toml changed after the release was built")
    evidence = manifest.get("collection") or manifest.get("benchmark")
    if not evidence_accepted(evidence):
        evidence_name = "collection" if manifest.get("collection") else "benchmark"
        errors.append(f"release {evidence_name} has not passed acceptance")
    errors.extend(canonical.get("errors", []))
    errors.extend(public.get("errors", []))
    return {
        "valid": not errors and canonical["valid"] and public["valid"],
        "release_version": spec["tag"],
        "errors": errors,
        "canonical": canonical,
        "viewer_package": public,
    }


def stage_release(release_dir: Path, spec: dict) -> dict:
    validation = validate_release(release_dir, spec)
    if not validation["valid"]:
        raise ValueError("release validation failed: " + "; ".join(validation["errors"]))
    manifest = json.loads((release_dir / "release.json").read_text(encoding="utf-8"))
    if manifest.get("code_revision") == "unknown":
        raise ValueError("release code revision is unknown")
    staged = {
        "schema_version": "farpoint.release-stage.v1",
        "status": "READY",
        "release_version": spec["tag"],
        "code_revision": manifest["code_revision"],
        "release_spec_sha256": manifest["release_spec_sha256"],
        "validation": validation,
    }
    write_json(release_dir / "stage.json", staged)
    return staged


def publish_staged_release(release_dir: Path, spec: dict, confirmation: str) -> dict:
    if confirmation != spec["tag"]:
        raise ValueError(f"confirmation must exactly match {spec['tag']}")
    stage = json.loads((release_dir / "stage.json").read_text(encoding="utf-8"))
    if stage.get("status") != "READY" or stage.get("release_version") != spec["tag"]:
        raise ValueError("release has not passed the staging gate")
    validation = validate_release(release_dir, spec)
    if not validation["valid"]:
        raise ValueError("staged release is no longer valid: " + "; ".join(validation["errors"]))
    manifest_path = release_dir / "release.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stage.get("code_revision") != manifest.get("code_revision"):
        raise ValueError("staged code revision does not match the release manifest")
    if stage.get("release_spec_sha256") != manifest.get("release_spec_sha256"):
        raise ValueError("staged release specification does not match the release manifest")
    published = publish_huggingface(
        release_dir / "public",
        spec["hf_repo_id"],
        spec["tag"],
        f"Release Farpoint {spec['tag']}",
    )
    manifest["published"] = published
    write_json(manifest_path, manifest)
    return published


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-spec", type=Path, default=DEFAULT_RELEASE_SPEC)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--source-dataset", type=Path)
    build.add_argument("--episode-root", type=Path)
    build.add_argument("--episode-id", dest="episode_ids", action="append")
    build.add_argument("--episode-ids-file", type=Path)
    build.add_argument("--benchmark-id")
    evidence = build.add_mutually_exclusive_group()
    evidence.add_argument("--benchmark-manifest", type=Path)
    evidence.add_argument("--collection-manifest", type=Path)
    build.add_argument("--task-name", default="isaac_perception_contact_scene")
    build.add_argument("--task-type", default="variation_expansion_v1")
    build.add_argument("--min-success-rate", type=float, default=0.90)
    for name in ("validate", "stage"):
        command = commands.add_parser(name)
        command.add_argument("release_dir", type=Path)
    publish = commands.add_parser("publish")
    publish.add_argument("release_dir", type=Path)
    publish.add_argument("--confirm-version", required=True)
    args = parser.parse_args()
    spec = load_release_spec(args.release_spec)
    if args.command == "build":
        if not args.source_dataset and not args.episode_root:
            parser.error("build requires --source-dataset or --episode-root")
        result = build_release(args, spec)
        exit_code = 0
    elif args.command == "validate":
        result = validate_release(args.release_dir, spec)
        exit_code = 0 if result["valid"] else 1
    elif args.command == "stage":
        result = stage_release(args.release_dir, spec)
        exit_code = 0
    else:
        result = publish_staged_release(args.release_dir, spec, args.confirm_version)
        exit_code = 0
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
