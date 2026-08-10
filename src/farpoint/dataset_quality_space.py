"""Stage a static Hugging Face Space from a precomputed quality report."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from farpoint.dataset_quality import canonical_sha256


def audit_quality_space(root: str | Path) -> dict:
    space = Path(root).resolve()
    errors = []
    for relative in ("README.md", "index.html", "styles.css", "app.js", "reports/index.json"):
        if not (space / relative).is_file():
            errors.append(f"missing required Space file: {relative}")
    index_path = space / "reports" / "index.json"
    if not index_path.is_file():
        return {"valid": False, "errors": errors}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    versions = index.get("versions") or []
    if len(versions) != 1:
        errors.append("first Space release must contain exactly one report version")
    for version in versions:
        report_path = space / str(version.get("report") or "")
        if not report_path.is_file():
            errors.append(f"missing version report: {version.get('report')}")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("report_sha256") != version.get("report_sha256"):
            errors.append(f"report hash mismatch: {version.get('version')}")
        if report.get("integrity", {}).get("status") != "PASS":
            errors.append(f"report integrity is not PASS: {version.get('version')}")
        generator_commit = str(report.get("identity", {}).get("generator_commit") or "")
        if not re.fullmatch(r"[0-9a-f]{40}", generator_commit):
            errors.append(f"report generator commit is not immutable: {version.get('version')}")
        report_root = report_path.parent.resolve()
        for sample in report.get("visual_quality", {}).get("samples", []):
            if len(sample.get("frames") or []) != 4:
                errors.append(
                    f"visual sample {sample.get('episode_index')} does not contain four frames"
                )
            for frame in sample.get("frames") or []:
                asset = (report_root / str(frame.get("path") or "")).resolve()
                if not asset.is_relative_to(report_root) or not asset.is_file():
                    errors.append(f"missing or unsafe visual asset: {frame.get('path')}")
    return {"valid": not errors, "errors": errors}


def stage_quality_space(
    template_dir: str | Path,
    report_dir: str | Path,
    destination: str | Path,
) -> dict:
    template = Path(template_dir).resolve()
    report_root = Path(report_dir).resolve()
    output = Path(destination).resolve()
    report_path = report_root / "report.json"
    if not template.is_dir():
        raise FileNotFoundError(f"Space template does not exist: {template}")
    if not report_path.is_file():
        raise FileNotFoundError(f"quality report does not exist: {report_path}")
    if output.exists():
        raise FileExistsError(f"Space destination already exists: {output}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_hash = report.get("report_sha256")
    unhashed = {key: value for key, value in report.items() if key != "report_sha256"}
    if expected_hash != canonical_sha256(unhashed):
        raise ValueError("quality report hash does not match its contents")
    if report.get("integrity", {}).get("status") != "PASS":
        raise ValueError("only PASS quality reports may be staged")
    version = str(report["identity"]["dataset_tag"])

    shutil.copytree(template, output, ignore=shutil.ignore_patterns("reports"))
    version_dir = output / "reports" / version
    shutil.copytree(report_root, version_dir)
    index = {
        "schema_version": "farpoint.dataset-quality-space-index.v1",
        "title": report["overview"]["title"],
        "default_version": version,
        "versions": [
            {
                "version": version,
                "report": f"reports/{version}/report.json",
                "dataset_commit": report["identity"]["resolved_dataset_commit"],
                "report_sha256": expected_hash,
            }
        ],
    }
    (output / "reports" / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = audit_quality_space(output)
    if not audit["valid"]:
        raise ValueError("staged quality Space failed audit: " + "; ".join(audit["errors"]))
    return index
