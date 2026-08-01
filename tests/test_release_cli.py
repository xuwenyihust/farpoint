import sys
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
