import json
from pathlib import Path

from farpoint.release import audit_viewer_package, prepare_viewer_package

from v2_fixtures import episode_metadata_v2


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


def test_public_metadata_keeps_normal_integers_typed(tmp_path):
    pytest = __import__("pytest")
    pyarrow = pytest.importorskip("pyarrow")
    source = tmp_path / "source"
    destination = tmp_path / "public"
    make_package(source)
    (source / "meta/episode_metadata.jsonl").write_text(
        json.dumps({"episode_id": "episode_1", "seed": 7, "success": True}) + "\n"
    )
    prepare_viewer_package(source, destination)
    table = pyarrow.parquet.read_table(destination / "meta/episode_metadata.parquet")
    assert table.column("seed")[0].as_py() == 7


def test_viewer_package_preserves_nested_v2_metadata_as_typed_parquet(tmp_path):
    pytest = __import__("pytest")
    pyarrow = pytest.importorskip("pyarrow")
    source = tmp_path / "source"
    destination = tmp_path / "public"
    make_package(source)
    (source / "meta/episode_metadata.jsonl").write_text(
        json.dumps(episode_metadata_v2()) + "\n", encoding="utf-8"
    )
    prepare_viewer_package(source, destination)
    table = pyarrow.parquet.read_table(destination / "meta/episode_metadata.parquet")
    row = table.to_pylist()[0]
    assert row["identity"]["dataset_episode_index"] == 0
    assert row["scene"]["object"]["shape"] == "cube"
    assert row["variation"]["resolved"]["object_position_m"] == [0.5, 0.0, 0.05]
