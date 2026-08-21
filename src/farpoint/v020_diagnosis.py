"""Stratified failure diagnosis for frozen SO-101 v0.2.0 plans."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from farpoint.so101_watchdog import classify_so101_failure
from farpoint.v020_plan import PLAN_SCHEMA, canonical_sha256


def _counter(rows: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(rows).items()))


def build_v020_failure_diagnosis(
    plan: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Describe failures across every frozen v0.2.0 variation axis."""
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("failure diagnosis requires a v0.2.0 plan")
    if manifest.get("plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("manifest does not bind the diagnosed plan")
    trials = {row["variation_id"]: row for row in plan.get("trials") or []}
    attempts = manifest.get("attempts") or []
    if not isinstance(attempts, list):
        raise ValueError("manifest attempts must be a list")

    failures = []
    for attempt in attempts:
        if attempt.get("success"):
            continue
        variation_id = str(attempt.get("variation_id") or "")
        trial = trials.get(variation_id)
        if trial is None:
            raise ValueError(f"attempt references an unknown variation: {variation_id}")
        strata = ((trial.get("sampler") or {}).get("resolved") or {}).get("strata") or {}
        values = ((trial.get("sampler") or {}).get("resolved") or {}).get("values") or {}
        failures.append(
            {
                "attempt_id": attempt.get("attempt_id"),
                "episode_id": attempt.get("episode_id"),
                "variation_id": variation_id,
                "failure_class": classify_so101_failure(
                    attempt.get("failure_reason"), attempt.get("failure_category")
                ),
                "failure_category": attempt.get("failure_category"),
                "failure_reason": attempt.get("failure_reason"),
                "dataset_valid": bool(attempt.get("dataset_valid")),
                "object_variant_id": trial["object_variant_id"],
                "target_profile_id": trial["target_profile_id"],
                "camera_profile_id": trial["camera_profile_id"],
                "cell_id": "::".join(
                    (
                        trial["object_variant_id"],
                        trial["target_profile_id"],
                        trial["camera_profile_id"],
                    )
                ),
                "region_band": trial["region_band"],
                "lhs_strata": deepcopy(strata),
                "cube_pose": {
                    "x_m": values.get("x_m"),
                    "y_m": values.get("y_m"),
                    "yaw_degrees": values.get("yaw_degrees"),
                },
            }
        )

    report = {
        "schema_version": "farpoint.so101-v020-failure-diagnosis.v1",
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "manifest_sha256": canonical_sha256(manifest),
        "git_commit": manifest.get("git_commit"),
        "execution_status": manifest.get("execution_status"),
        "attempt_count": len(attempts),
        "selected_success_count": len(manifest.get("selected_variations") or {}),
        "failure_count": len(failures),
        "failure_counts": {
            "failure_class": _counter([row["failure_class"] for row in failures]),
            "cell": _counter([row["cell_id"] for row in failures]),
            "object_variant": _counter([row["object_variant_id"] for row in failures]),
            "target_profile": _counter([row["target_profile_id"] for row in failures]),
            "camera_profile": _counter([row["camera_profile_id"] for row in failures]),
            "region_band": _counter([row["region_band"] for row in failures]),
            "x_stratum": _counter([str(row["lhs_strata"].get("x_m")) for row in failures]),
            "y_stratum": _counter([str(row["lhs_strata"].get("y_m")) for row in failures]),
            "yaw_stratum": _counter(
                [str(row["lhs_strata"].get("yaw_degrees")) for row in failures]
            ),
        },
        "failures": failures,
    }
    report["diagnosis_sha256"] = canonical_sha256(report)
    return report
