from __future__ import annotations

import json

import pytest

from farpoint.dataset_quality import canonical_sha256
from farpoint.dataset_quality_space import audit_quality_space, stage_quality_space


def report():
    value = {
        "schema_version": "farpoint.dataset-quality-report.v1",
        "identity": {
            "dataset_tag": "v0.0.3",
            "resolved_dataset_commit": "a" * 40,
            "generator_commit": "b" * 40,
        },
        "overview": {"title": "Farpoint SO-101 Dataset"},
        "integrity": {"status": "PASS"},
    }
    value["report_sha256"] = canonical_sha256(value)
    return value


def test_stage_quality_space_copies_one_version(tmp_path):
    template = tmp_path / "template"
    template.mkdir()
    for name in ("README.md", "styles.css", "app.js"):
        (template / name).write_text(name, encoding="utf-8")
    (template / "index.html").write_text("quality", encoding="utf-8")
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "report.json").write_text(json.dumps(report()), encoding="utf-8")
    (report_dir / "asset.jpg").write_bytes(b"jpeg")

    index = stage_quality_space(template, report_dir, tmp_path / "output")

    assert index["default_version"] == "v0.0.3"
    assert (tmp_path / "output/index.html").read_text() == "quality"
    assert (tmp_path / "output/reports/v0.0.3/asset.jpg").read_bytes() == b"jpeg"
    assert audit_quality_space(tmp_path / "output")["valid"]


def test_stage_rejects_mutated_report(tmp_path):
    template = tmp_path / "template"
    template.mkdir()
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    value = report()
    value["overview"]["title"] = "mutated"
    (report_dir / "report.json").write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        stage_quality_space(template, report_dir, tmp_path / "output")
