import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

import release_dataset  # noqa: E402


def release_fixture(tmp_path: Path, spec: dict, *, accepted: bool = True, revision: str = "abc123"):
    (tmp_path / "canonical").mkdir()
    (tmp_path / "public").mkdir()
    manifest = {
        "release_version": spec["tag"],
        "dataset_id": spec["dataset_id"],
        "hf_repo_id": spec["hf_repo_id"],
        "code_revision": revision,
        "release_spec_sha256": release_dataset.file_sha256(Path(spec["path"])),
        "benchmark": {"accepted": accepted},
    }
    release_dataset.write_json(tmp_path / "release.json", manifest)
    return manifest


@pytest.fixture
def successful_audits(monkeypatch):
    result = {"valid": True, "errors": []}
    monkeypatch.setattr(release_dataset, "validate_dataset", lambda _: result)
    monkeypatch.setattr(release_dataset, "audit_viewer_package", lambda _: result)


def test_validate_release_rejects_unaccepted_benchmark(tmp_path, successful_audits):
    spec = release_dataset.load_release_spec()
    release_fixture(tmp_path, spec, accepted=False)
    result = release_dataset.validate_release(tmp_path, spec)
    assert result["valid"] is False
    assert "release benchmark has not passed acceptance" in result["errors"]


def test_coverage_first_collection_release_is_accepted_without_yield_acceptance():
    evidence = {
        "schema_version": "farpoint.collection.v1",
        "acceptance": {"accepted": False},
        "release_policy": "coverage_first_all_successful",
        "release_acceptance": {"accepted": True},
    }

    assert release_dataset.evidence_accepted(evidence) is True


def test_stage_rejects_unknown_code_revision(tmp_path, successful_audits):
    spec = release_dataset.load_release_spec()
    release_fixture(tmp_path, spec, revision="unknown")
    with pytest.raises(ValueError, match="code revision is unknown"):
        release_dataset.stage_release(tmp_path, spec)


def test_publish_requires_exact_version_confirmation(tmp_path):
    spec = release_dataset.load_release_spec()
    with pytest.raises(ValueError, match="confirmation must exactly match"):
        release_dataset.publish_staged_release(tmp_path, spec, "v0.0.0")


def test_publish_revalidates_staged_release(tmp_path, successful_audits, monkeypatch):
    spec = release_dataset.load_release_spec()
    manifest = release_fixture(tmp_path, spec)
    release_dataset.write_json(
        tmp_path / "stage.json",
        {
            "status": "READY",
            "release_version": spec["tag"],
            "code_revision": manifest["code_revision"],
            "release_spec_sha256": manifest["release_spec_sha256"],
        },
    )
    monkeypatch.setattr(
        release_dataset,
        "validate_release",
        lambda *_: {"valid": False, "errors": ["viewer failed"]},
    )
    with pytest.raises(ValueError, match="staged release is no longer valid"):
        release_dataset.publish_staged_release(tmp_path, spec, spec["tag"])


def test_v2_release_requires_collection_or_benchmark_manifest(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "release"
    monkeypatch.setattr(
        release_dataset,
        "validate_dataset",
        lambda *_: {
            "valid": True,
            "errors": [],
            "schema_version": "farpoint.dataset.v2",
        },
    )
    args = Namespace(
        output_dir=output,
        source_dataset=source,
        benchmark_manifest=None,
        benchmark_id=None,
    )
    with pytest.raises(ValueError, match="require --collection-manifest or --benchmark-manifest"):
        release_dataset.build_release(args, release_dataset.load_release_spec())


def test_validate_release_passes_v2_benchmark_to_dataset_validator(tmp_path, monkeypatch):
    spec = release_dataset.load_release_spec()
    release_fixture(tmp_path, spec)
    (tmp_path / "canonical" / "meta").mkdir(parents=True)
    (tmp_path / "canonical" / "meta" / "farpoint_v2.json").write_text("{}")
    benchmark = {
        "schema_version": "farpoint.benchmark.v2",
        "acceptance": {"accepted": True},
    }
    release_dataset.write_json(tmp_path / "benchmark" / "manifest.json", benchmark)
    manifest = json.loads((tmp_path / "release.json").read_text())
    manifest["benchmark"] = benchmark
    release_dataset.write_json(tmp_path / "release.json", manifest)
    calls = []

    def validate_dataset(path, benchmark_path=None):
        calls.append((Path(path), benchmark_path))
        return {"valid": True, "errors": []}

    monkeypatch.setattr(release_dataset, "validate_dataset", validate_dataset)
    monkeypatch.setattr(
        release_dataset, "audit_viewer_package", lambda _: {"valid": True, "errors": []}
    )
    result = release_dataset.validate_release(tmp_path, spec)
    assert result["valid"] is True
    assert calls == [(tmp_path / "canonical", tmp_path / "benchmark" / "manifest.json")]


def test_validate_release_rejects_mismatched_v2_benchmark(tmp_path, monkeypatch):
    from test_lerobot_dataset_validator import make_valid_v2_dataset

    spec = release_dataset.load_release_spec()
    benchmark_path = make_valid_v2_dataset(tmp_path / "canonical")
    benchmark = json.loads(benchmark_path.read_text())
    benchmark["trials"][0]["episode_id"] = "wrong-episode"
    target_benchmark = tmp_path / "benchmark" / "manifest.json"
    release_dataset.write_json(target_benchmark, benchmark)
    (tmp_path / "public").mkdir()
    release_dataset.write_json(
        tmp_path / "release.json",
        {
            "release_version": spec["tag"],
            "dataset_id": spec["dataset_id"],
            "hf_repo_id": spec["hf_repo_id"],
            "code_revision": "abc123",
            "release_spec_sha256": release_dataset.file_sha256(Path(spec["path"])),
            "benchmark": benchmark,
        },
    )
    monkeypatch.setattr(
        release_dataset, "audit_viewer_package", lambda _: {"valid": True, "errors": []}
    )
    result = release_dataset.validate_release(tmp_path, spec)
    assert result["valid"] is False
    assert any("benchmark episode_id mismatch" in error for error in result["errors"])


def test_validate_release_rejects_missing_v2_benchmark_file(tmp_path, monkeypatch):
    spec = release_dataset.load_release_spec()
    release_fixture(tmp_path, spec)
    (tmp_path / "canonical" / "meta").mkdir(parents=True)
    (tmp_path / "canonical" / "meta" / "farpoint_v2.json").write_text("{}")
    monkeypatch.setattr(
        release_dataset,
        "validate_dataset",
        lambda *_: {"valid": True, "errors": []},
    )
    monkeypatch.setattr(
        release_dataset, "audit_viewer_package", lambda _: {"valid": True, "errors": []}
    )
    result = release_dataset.validate_release(tmp_path, spec)
    assert result["valid"] is False
    assert "Farpoint v2 release is missing benchmark/manifest.json" in result["errors"]


def test_validate_release_accepts_collection_evidence(tmp_path, monkeypatch):
    spec = release_dataset.load_release_spec()
    (tmp_path / "canonical" / "meta").mkdir(parents=True)
    (tmp_path / "canonical" / "meta" / "farpoint_v2.json").write_text("{}")
    (tmp_path / "public").mkdir()
    collection = {
        "schema_version": "farpoint.collection.v1",
        "acceptance": {"accepted": True},
    }
    release_dataset.write_json(tmp_path / "collection" / "manifest.json", collection)
    release_dataset.write_json(
        tmp_path / "release.json",
        {
            "release_version": spec["tag"],
            "dataset_id": spec["dataset_id"],
            "hf_repo_id": spec["hf_repo_id"],
            "code_revision": "abc123",
            "release_spec_sha256": release_dataset.file_sha256(Path(spec["path"])),
            "collection": collection,
        },
    )
    calls = []

    def validate_dataset(path, evidence_path=None):
        calls.append((Path(path), evidence_path))
        return {"valid": True, "errors": []}

    monkeypatch.setattr(release_dataset, "validate_dataset", validate_dataset)
    monkeypatch.setattr(
        release_dataset, "audit_viewer_package", lambda _: {"valid": True, "errors": []}
    )

    result = release_dataset.validate_release(tmp_path, spec)

    assert result["valid"] is True
    assert calls == [(tmp_path / "canonical", tmp_path / "collection" / "manifest.json")]
