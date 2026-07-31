import json
from pathlib import Path

from farpoint.release import audit_viewer_package, prepare_viewer_package


def make_package(root: Path):
    for relative in (
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "data/chunk-000/file-000.parquet",
        "videos/observation.images.front/chunk-000/file-000.mp4",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    (root / "meta/info.json").write_text("{}")
    (root / "meta/stats.json").write_text("{}")
    (root / "meta/episode_metadata.jsonl").write_text(
        json.dumps({"episode_id": "episode_1", "seed": 2**80, "success": True}) + "\n"
    )
    (root / "meta/farpoint_v1.json").write_text("{}")


def test_viewer_audit_rejects_nonstandard_json(tmp_path):
    make_package(tmp_path)
    result = audit_viewer_package(tmp_path)
    assert result["valid"] is False
    assert any("farpoint_v1.json" in error for error in result["errors"])


def test_prepare_viewer_package_removes_sidecars_and_writes_metadata(tmp_path, monkeypatch):
    pytest = __import__("pytest")
    pyarrow = pytest.importorskip("pyarrow")
    source = tmp_path / "source"
    destination = tmp_path / "public"
    make_package(source)
    result = prepare_viewer_package(source, destination)
    assert result["metadata_rows"] == 1
    assert not (destination / "meta/farpoint_v1.json").exists()
    assert not (destination / "meta/episode_metadata.jsonl").exists()
    assert (destination / "meta/episode_metadata.parquet").is_file()
    assert audit_viewer_package(destination)["valid"] is True
    table = pyarrow.parquet.read_table(destination / "meta/episode_metadata.parquet")
    assert table.column("seed")[0].as_py() == str(2**80)
