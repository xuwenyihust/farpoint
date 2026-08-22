import json

import pytest

from farpoint.cleanup_manifest import build_cleanup_manifest, disposable_paths


def test_cleanup_manifest_only_exposes_explicit_disposable_paths(tmp_path):
    keep = tmp_path / "release"
    drop = tmp_path / "failed-smoke"
    keep.mkdir()
    drop.mkdir()
    (keep / "data").write_text("published")
    (drop / "data").write_text("failed")
    manifest = build_cleanup_manifest(
        [
            {"path": keep, "disposition": "retain", "reason": "release evidence"},
            {"path": drop, "disposition": "disposable", "reason": "zero-scene failed smoke"},
        ],
        protected_roots=[tmp_path],
    )
    assert disposable_paths(manifest) == [drop]
    assert keep.exists() and drop.exists()
    altered = json.loads(json.dumps(manifest))
    altered["entries"][0]["reason"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        disposable_paths(altered)


def test_cleanup_manifest_rejects_deleting_a_protected_root(tmp_path):
    with pytest.raises(ValueError, match="protected root"):
        build_cleanup_manifest(
            [{"path": tmp_path, "disposition": "disposable", "reason": "too broad"}],
            protected_roots=[tmp_path],
        )
