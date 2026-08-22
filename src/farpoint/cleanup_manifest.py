"""Conservative cleanup manifests for remote experiment storage."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "farpoint.cleanup-manifest.v1"
DISPOSITIONS = {"retain", "disposable"}


def _size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(row.stat().st_size for row in path.rglob("*") if row.is_file())


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    rows = [path] if path.is_file() else sorted(row for row in path.rglob("*") if row.is_file())
    for row in rows:
        relative = row.name if path.is_file() else row.relative_to(path).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        with row.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_cleanup_manifest(
    candidates: Iterable[dict[str, Any]],
    *,
    protected_roots: Iterable[str | Path],
) -> dict[str, Any]:
    protected = [Path(value).resolve() for value in protected_roots]
    entries = []
    for candidate in candidates:
        path = Path(candidate["path"]).resolve()
        disposition = str(candidate.get("disposition", "retain"))
        if disposition not in DISPOSITIONS:
            raise ValueError(f"unsupported cleanup disposition: {disposition}")
        if not path.exists():
            raise FileNotFoundError(path)
        if any(path == root for root in protected) and disposition == "disposable":
            raise ValueError(f"protected root cannot be disposable: {path}")
        reason = str(candidate.get("reason") or "")
        if not reason:
            raise ValueError(f"cleanup entry requires a reason: {path}")
        entries.append(
            {
                "path": str(path),
                "artifact_type": str(candidate.get("artifact_type") or "unknown"),
                "associated_version": candidate.get("associated_version"),
                "size_bytes": _size_bytes(path),
                "tree_sha256": _tree_sha256(path),
                "git_commit": candidate.get("git_commit"),
                "dashboard_references": list(candidate.get("dashboard_references") or []),
                "hugging_face_references": list(candidate.get("hugging_face_references") or []),
                "disposition": disposition,
                "reason": reason,
            }
        )
    if len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("cleanup manifest paths must be unique")
    manifest = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protected_roots": [str(value) for value in protected],
        "entries": sorted(entries, key=lambda entry: entry["path"]),
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest


def disposable_paths(manifest: dict[str, Any]) -> list[Path]:
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("cleanup manifest schema mismatch")
    digest = manifest.get("manifest_sha256")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != expected:
        raise ValueError("cleanup manifest hash mismatch")
    return [Path(row["path"]) for row in manifest["entries"] if row["disposition"] == "disposable"]
