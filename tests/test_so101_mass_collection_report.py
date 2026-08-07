import json
from pathlib import Path

from farpoint.object_variation import load_variation_config
from farpoint.so101_collection import create_manifest, next_attempt, record_attempt
from farpoint.so101_mass_candidate import build_mass_dataset_candidate
from farpoint.so101_mass_collection import (
    build_mirrored_mass_collection_plan,
    load_mass_collection_config,
)
from farpoint.so101_mass_collection_report import (
    build_so101_mass_collection_report,
)


ROOT = Path(__file__).resolve().parents[1]


def plan():
    return build_mirrored_mass_collection_plan(
        load_variation_config(
            ROOT / "configs/variations/so101_cube_pick_place_v1.json"
        ),
        load_mass_collection_config(
            ROOT / "configs/collections/so101_cube_mass_003_v0_0_1.json"
        ),
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def complete_collection(tmp_path):
    collection_plan = plan()
    manifest = create_manifest(
        collection_plan,
        collection_id="mass_003_formal",
        git_commit="a" * 40,
        maximum_attempts=150,
    )
    root = tmp_path / "episodes"
    audit = {
        "requested_mass_kg": 0.03,
        "resolved_mass_kg": 0.03,
        "physx_actual_mass_kg": 0.0300000001,
        "tolerance_kg": 1e-6,
        "verified": True,
    }
    for _ in range(50):
        attempt = next_attempt(manifest, collection_plan)
        episode_id = f"episode_{attempt['attempt_id']}"
        record_attempt(
            manifest,
            collection_plan,
            attempt,
            episode_id=episode_id,
            success=True,
            dataset_valid=True,
        )
        episode = root / episode_id
        write_json(episode / "metadata.json", {"scene": {"object": {"mass_audit": audit}}})
        write_json(
            episode / "metrics.json",
            {"success": True, "dataset_valid": True, "physics_audit": {"mass": audit}},
        )
        write_json(episode / "run-state.json", {"execution_status": "FINISHED"})
        (episode / "observations.jsonl").write_text("{}\n", encoding="utf-8")
    return collection_plan, manifest, root


def baseline_manifest(collection_plan):
    attempts = []
    selected = {}
    for trial in collection_plan["trials"]:
        source_id = trial["source_trial_id"]
        attempt_id = f"{source_id}__attempt00"
        row = {
            "attempt_id": attempt_id,
            "trial_id": source_id,
            "variation_id": source_id,
            "split": trial["split"],
            "episode_id": f"episode_baseline__{attempt_id}",
            "success": True,
            "dataset_valid": True,
            "selected_for_dataset": True,
        }
        attempts.append(row)
        selected[source_id] = attempt_id
    return {
        "schema_version": "farpoint.collection-selection.v1",
        "collection_id": collection_plan["collection"]["source_selection_id"],
        "task_id": "so101_cube_pick_place",
        "quality_status": "PASS",
        "attempts": attempts,
        "selected_variations": selected,
    }


def test_formal_report_accepts_complete_balanced_mass_audited_collection(tmp_path):
    collection_plan, manifest, root = complete_collection(tmp_path)

    report = build_so101_mass_collection_report(collection_plan, manifest, root)

    assert report["status"] == "PASS"
    assert report["success_count"] == 50
    assert report["attempted_count"] == 50
    assert report["maximum_attempts"] == 150
    assert report["mass_audit_count"] == 50
    assert report["balance"]["splits"] == {
        "test": 5,
        "train": 40,
        "validation": 5,
    }
    assert report["evidence_errors"] == []


def test_formal_report_rejects_mass_mismatch(tmp_path):
    collection_plan, manifest, root = complete_collection(tmp_path)
    first = root / manifest["attempts"][0]["episode_id"] / "metrics.json"
    metrics = json.loads(first.read_text())
    metrics["physics_audit"]["mass"]["physx_actual_mass_kg"] = 0.04
    first.write_text(json.dumps(metrics), encoding="utf-8")

    report = build_so101_mass_collection_report(collection_plan, manifest, root)

    assert report["status"] == "INVALID_EVIDENCE"
    assert any("mass_audit" in error for error in report["evidence_errors"])


def test_combined_candidate_is_exactly_balanced_across_mass_and_split(tmp_path):
    collection_plan, manifest, root = complete_collection(tmp_path)
    baseline = baseline_manifest(collection_plan)

    combined, selection = build_mass_dataset_candidate(
        baseline,
        manifest,
        collection_plan,
        collection_id="farpoint_so101_v0_0_1_candidate",
        baseline_episodes_root="artifacts/baseline",
        candidate_episodes_root=root,
    )

    assert combined["required_successes"] == 100
    assert combined["balance"]["mass_kg"] == {"0.03": 50, "0.04": 50}
    assert combined["balance"]["splits"] == {
        "train": 80,
        "validation": 10,
        "test": 10,
    }
    assert combined["balance"]["mirrored_trial_pairs"] == 50
    assert len(selection["episodes"]) == 100
    assert sum(row["split"] == "train" for row in selection["episodes"]) == 80
