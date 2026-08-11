#!/usr/bin/env python3
"""Validate SO-101 Oracle repair paths and controlled DGX evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def normalize_path(value: str) -> str:
    path = PurePosixPath(value.strip())
    if not value.strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid repository-relative path: {value!r}")
    return path.as_posix()


def _matches(path: str, files: set[str], prefixes: tuple[str, ...]) -> bool:
    return path in files or any(path.startswith(prefix) for prefix in prefixes)


def validate_paths(policy: dict[str, Any], changed_files: list[str]) -> list[str]:
    errors = []
    normalized = []
    for value in changed_files:
        fields = value.split("\t")
        if len(fields) == 1:
            status, raw_path = "M", fields[0]
        elif len(fields) == 2:
            status, raw_path = fields
        else:
            errors.append(f"renames and copies are not allowed in Oracle repairs: {value}")
            continue
        if status not in {"A", "M"}:
            errors.append(f"change status {status!r} is not allowed for Oracle repairs: {raw_path}")
            continue
        try:
            normalized.append(normalize_path(raw_path))
        except ValueError as error:
            errors.append(str(error))
    if len(set(normalized)) != len(normalized):
        errors.append("changed file list contains duplicates")
    runtime_files = set(policy["allowed_runtime_files"])
    profile_prefixes = tuple(policy["allowed_profile_prefixes"])
    test_files = set(policy["allowed_test_files"])
    test_prefixes = tuple(policy["allowed_test_prefixes"])
    forbidden_prefixes = tuple(policy["forbidden_prefixes"])
    runtime_changes = []
    test_changes = []
    for path in normalized:
        if any(path.startswith(prefix) for prefix in forbidden_prefixes):
            errors.append(f"forbidden path: {path}")
            continue
        if _matches(path, runtime_files, profile_prefixes):
            runtime_changes.append(path)
        elif _matches(path, test_files, test_prefixes):
            test_changes.append(path)
        else:
            errors.append(f"path is outside the Oracle repair allowlist: {path}")
    if policy.get("required_runtime_change") and not runtime_changes:
        errors.append("Oracle repair must change an allowed runtime/profile path")
    if policy.get("required_test_change") and not test_changes:
        errors.append("Oracle repair must change an allowed test path")
    return errors


def validate_evidence(
    policy: dict[str, Any], evidence: dict[str, Any], *, expected_commit: str
) -> list[str]:
    errors = []
    if evidence.get("schema_version") != "farpoint.oracle-repair-evidence.v1":
        errors.append("unsupported Oracle repair evidence schema")
    if not COMMIT_PATTERN.fullmatch(expected_commit):
        errors.append("expected commit must be a full lowercase Git SHA")
    if evidence.get("git_commit") != expected_commit:
        errors.append("evidence git_commit does not match the PR head commit")
    if evidence.get("policy_id") != policy.get("policy_id"):
        errors.append("evidence policy_id does not match governance policy")
    if evidence.get("scene_contract_unchanged") is not True:
        errors.append("evidence must prove scene contract unchanged")
    if evidence.get("success_criteria_unchanged") is not True:
        errors.append("evidence must prove success criteria unchanged")

    diagnostics = evidence.get("diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        errors.append("evidence must contain diagnostic failure classes")
        diagnostics = []
    seen_classes = set()
    required_seeds = int(policy["diagnostic_seeds_per_failure_class"])
    maximum_attempts = int(policy["maximum_attempts_per_seed"])
    for diagnostic in diagnostics:
        failure_class = diagnostic.get("failure_class")
        if not isinstance(failure_class, str) or not failure_class:
            errors.append("diagnostic failure_class must be non-empty")
            continue
        if failure_class in seen_classes:
            errors.append(f"duplicate diagnostic failure class: {failure_class}")
        seen_classes.add(failure_class)
        seeds = diagnostic.get("seeds") or []
        if len(seeds) != required_seeds:
            errors.append(f"diagnostic {failure_class} must contain {required_seeds} seeds")
        if len({row.get("seed") for row in seeds}) != len(seeds):
            errors.append(f"diagnostic {failure_class} seed identities must be unique")
        for row in seeds:
            attempts = row.get("attempts")
            if not isinstance(attempts, int) or not 1 <= attempts <= maximum_attempts:
                errors.append(f"diagnostic {failure_class} attempts must be within 1..{maximum_attempts}")
            if row.get("success") is not True or row.get("dataset_valid") is not True:
                errors.append(f"diagnostic {failure_class} seed did not pass")

    canaries = evidence.get("canaries") or []
    required_canaries = int(policy["canary_seed_count"])
    if len(canaries) != required_canaries:
        errors.append(f"evidence must contain exactly {required_canaries} canary seeds")
    if len({row.get("seed") for row in canaries}) != len(canaries):
        errors.append("canary seed identities must be unique")
    successes = 0
    for row in canaries:
        attempts = row.get("attempts")
        if not isinstance(attempts, int) or not 1 <= attempts <= maximum_attempts:
            errors.append(f"canary attempts must be within 1..{maximum_attempts}")
        if row.get("success") is True and row.get("dataset_valid") is True:
            successes += 1
        else:
            errors.append(f"canary seed did not pass: {row.get('seed')}")
    if successes != int(policy["required_canary_successes"]):
        errors.append("canary success count does not meet policy")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--changed-files", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    policy = load_json(args.policy)
    changed_files = [line for line in args.changed_files.read_text().splitlines() if line.strip()]
    errors = validate_paths(policy, changed_files)
    if args.evidence or args.expected_commit:
        if not args.evidence or not args.expected_commit:
            errors.append("--evidence and --expected-commit must be supplied together")
        else:
            errors.extend(
                validate_evidence(
                    policy, load_json(args.evidence), expected_commit=args.expected_commit
                )
            )
    print(json.dumps({"accepted": not errors, "errors": errors}, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
