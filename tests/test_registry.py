import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from farpoint.registry import EpisodeRegistry
from farpoint.retention import RetentionManager
from farpoint.so101_collection import episode_id_for_attempt


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_episode(root, episode_id, success=None, benchmark_id=None, run_id=None):
    episode = root / "episodes" / episode_id
    episode.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    write_json(
        episode / "metadata.json",
        {
            "episode_id": episode_id,
            "task_name": "test_task",
            "episode_seed": 7,
            "benchmark_id": benchmark_id,
            "run_id": run_id,
            "started_at": (now - timedelta(minutes=1)).isoformat(),
            "finished_at": now.isoformat(),
        },
    )
    if success is not None:
        write_json(
            episode / "metrics.json",
            {
                "success": success,
                "elapsed_seconds": 60,
                "dataset_valid": True,
                "dataset_observation_count": 10,
            },
        )
    return episode


def make_v3_episode(root, collection_id, episode_id, success=True, metrics=True):
    collection = root / "gates" / collection_id
    write_json(
        collection / "manifest.json",
        {"schema_version": "farpoint.so101-gate-manifest.v1", "gate_id": collection_id},
    )
    episode = collection / "episodes" / episode_id
    write_json(
        episode / "metadata.json",
        {
            "schema_version": "farpoint.episode.v3",
            "identity": {
                "episode_id": episode_id,
                "trial_id": "trial_001",
                "task_id": "so101_cube_pick_place",
                "split": "validation",
                "episode_seed": 42,
            },
            "provenance": {"created_at": "2026-08-06T00:00:00+00:00"},
            "task": {"task_id": "so101_cube_pick_place"},
            "variation": {"variation_id": "cube_30mm_position_01", "split": "validation"},
            "recording": {
                "frame_count": 2,
                "cameras": ["observation.images.front"],
            },
            "outcome": {
                "success": success,
                "dataset_valid": success,
                "failure_category": None if success else "grasp",
                "failure_reason": None if success else "bilateral contact not established",
            },
        },
    )
    if metrics:
        write_json(
            episode / "metrics.json",
            {
                "success": success,
                "dataset_valid": success,
                "observation_count": 2,
                "failure_category": None if success else "grasp",
                "failure_reason": None if success else "bilateral contact not established",
            },
        )
    rgb = episode / "rgb"
    rgb.mkdir()
    (rgb / "front_000000.png").write_bytes(b"png")
    (rgb / "front_000001.png").write_bytes(b"png")
    return episode


def make_running_v3_episode(root, collection_id, episode_id, status="RUNNING"):
    collection = root / "pilots" / collection_id
    write_json(
        collection / "manifest.json",
        {"schema_version": "farpoint.collection.v2", "collection_id": collection_id},
    )
    episode = collection / "episodes" / episode_id
    write_json(
        episode / "run-state.json",
        {
            "schema_version": "farpoint.episode-run.v1",
            "execution_status": status,
            "identity": {
                "episode_id": episode_id,
                "trial_id": "trial_001",
                "task_id": "so101_cube_pick_place",
                "split": "test",
                "episode_seed": 42,
            },
            "provenance": {"collection_id": collection_id},
            "variation": {"variation_id": "cube_40mm_position_03", "split": "test"},
            "recording": {
                "frame_count": 0,
                "cameras": ["observation.images.front"],
            },
            "outcome": {
                "success": False if status == "FAILED" else None,
                "dataset_valid": False,
                "failure_category": "runner" if status == "FAILED" else None,
                "failure_reason": "RuntimeError: simulation stopped"
                if status == "FAILED"
                else None,
            },
        },
    )
    rgb = episode / "rgb"
    rgb.mkdir()
    (rgb / "front_000000.png").write_bytes(b"png")
    return episode


class EpisodeRegistryTests(unittest.TestCase):
    def test_running_and_runner_failed_episode_sidecars_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            external = root / "farpoint-so101"
            running = make_running_v3_episode(
                external, "pilot_live", "episode_pilot_live__trial_001"
            )
            failed = make_running_v3_episode(
                external,
                "pilot_failed",
                "episode_pilot_failed__trial_001",
                status="FAILED",
            )

            registry = EpisodeRegistry(outputs, episode_roots=[external])
            registry.scan()
            rows = {row["episode_id"]: row for row in registry.list_episodes()}

            self.assertEqual(rows[running.name]["status"], "RUNNING")
            self.assertEqual(rows[running.name]["health"], "OK")
            self.assertEqual(rows[running.name]["observation_count"], 1)
            self.assertEqual(rows[running.name]["variation_id"], "cube_40mm_position_03")
            self.assertEqual(rows[failed.name]["status"], "FAIL")
            self.assertEqual(rows[failed.name]["failure_category"], "runner")
            self.assertEqual(
                rows[failed.name]["failure_reason"],
                "RuntimeError: simulation stopped",
            )
            self.assertEqual(rows[failed.name]["dataset_valid"], 0)

    def test_collection_scoped_episode_ids_keep_repeated_attempts_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            external = root / "farpoint-so101"
            attempt_id = "cube_r00_c00_s0_k0__attempt00"
            first_id = episode_id_for_attempt("pilot_v1", attempt_id)
            second_id = episode_id_for_attempt("pilot_v2", attempt_id)
            make_v3_episode(external, "pilot_v1", first_id)
            make_v3_episode(external, "pilot_v2", second_id)

            registry = EpisodeRegistry(outputs, episode_roots=[external])
            registry.scan()
            rows = {row["episode_id"]: row for row in registry.list_episodes()}

            self.assertEqual(rows[first_id]["collection_id"], "pilot_v1")
            self.assertEqual(rows[second_id]["collection_id"], "pilot_v2")
            self.assertEqual(len(rows), 2)

    def test_scans_external_v3_episode_with_collection_and_front_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            external = root / "farpoint-so101"
            episode = make_v3_episode(
                external,
                "so101_30mm_gate",
                "episode_so101_30mm_001",
            )

            registry = EpisodeRegistry(outputs, episode_roots=[external])
            registry.scan()
            row = registry.get_episode("episode_so101_30mm_001")

            self.assertEqual(row["status"], "PASS")
            self.assertEqual(row["health"], "OK")
            self.assertEqual(row["task_name"], "so101_cube_pick_place")
            self.assertEqual(row["schema_version"], "farpoint.episode.v3")
            self.assertEqual(row["variation_id"], "cube_30mm_position_01")
            self.assertEqual(row["split"], "validation")
            self.assertEqual(row["collection_id"], "so101_30mm_gate")
            self.assertEqual(row["managed"], 0)
            self.assertEqual(row["protected_reason"], "external-read-only")
            self.assertEqual(Path(row["artifact_path"]), episode.resolve())
            self.assertEqual(
                Path(row["preview_path"]).name,
                "front_000000.png",
            )
            self.assertEqual(row["observation_count"], 2)

    def test_external_scan_discovers_new_and_old_incomplete_v3_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            external = root / "farpoint-so101"
            registry = EpisodeRegistry(
                outputs,
                incomplete_timeout_seconds=60,
                episode_roots=[external],
            )
            registry.scan()
            self.assertEqual(registry.list_episodes(), [])

            complete = make_v3_episode(
                external, "pilot", "episode_new_success", success=True
            )
            incomplete = make_v3_episode(
                external, "pilot", "episode_old_incomplete", metrics=False
            )
            old = datetime.now().timestamp() - 3600
            os.utime(incomplete, (old, old))
            registry.scan()
            rows = {row["episode_id"]: row for row in registry.list_episodes()}

            self.assertEqual(rows[complete.name]["status"], "PASS")
            self.assertEqual(rows[incomplete.name]["status"], "INCOMPLETE")

    def test_scans_running_benchmark_state_before_final_manifest_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / "outputs"
            state_path = outputs / "benchmarks" / "formal" / "run-state.json"
            write_json(
                state_path,
                {
                    "schema_version": "farpoint.benchmark-run.v1",
                    "benchmark_id": "formal",
                    "task_id": "task",
                    "task_type": "cube_position_formal",
                    "execution_status": "RUNNING",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "planned_trials": 75,
                    "completed_trials": 2,
                    "passed_trials": 2,
                    "success_rate": 1.0,
                    "accepted": False,
                    "trials": [{"success": True}, {"success": True}],
                },
            )
            registry = EpisodeRegistry(outputs)
            registry.scan()
            row = registry.list_benchmarks()[0]
            self.assertEqual(row["status"], "RUNNING")
            self.assertEqual(row["completed_trials"], 2)
            self.assertEqual(Path(row["manifest_path"]).name, "run-state.json")

    def test_canonicalizes_legacy_benchmark_aliases_for_ui_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / "outputs"
            make_episode(
                outputs,
                "episode_legacy_benchmark",
                True,
                benchmark_id="robotsim_v1_release_candidate",
            )
            registry = EpisodeRegistry(outputs)
            registry.scan()
            row = registry.list_episodes()[0]
            self.assertEqual(row["benchmark_id"], "farpoint_v1_release_candidate")

    def test_collection_import_pilot_has_collection_type_and_terminal_pilot_status(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / "outputs"
            write_json(
                outputs / "benchmarks" / "collection-pilot" / "run-state.json",
                {
                    "schema_version": "farpoint.collection-run.v1",
                    "collection_id": "collection-pilot",
                    "task_id": "task",
                    "execution_status": "PILOT_COMPLETE",
                    "task_attempts": 27,
                    "task_successes": 23,
                    "task_yield": 23 / 27,
                    "attempts": [{"outcome_success": True}],
                },
            )

            registry = EpisodeRegistry(outputs)
            registry.scan()
            row = registry.list_benchmarks()[0]

            self.assertEqual(row["record_type"], "COLLECTION")
            self.assertEqual(row["status"], "PILOT")

    def test_local_display_name_is_exposed_without_changing_collection_id(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / "outputs"
            write_json(
                outputs / "benchmarks" / "collection-id" / "run-state.json",
                {
                    "schema_version": "farpoint.collection-run.v1",
                    "collection_id": "collection-id",
                    "task_id": "task",
                    "execution_status": "RUNNING",
                    "attempts": [],
                },
            )
            write_json(
                outputs / ".data-platform" / "display-names.json",
                {
                    "schema_version": "farpoint.display-names.v1",
                    "records": {"collection-id": "UR10e Cube Position Collection"},
                },
            )

            registry = EpisodeRegistry(outputs)
            registry.scan()
            row = registry.list_benchmarks()[0]

            self.assertEqual(row["benchmark_id"], "collection-id")
            self.assertEqual(row["display_name"], "UR10e Cube Position Collection")

    def test_scans_terminal_incomplete_corrupt_and_running_records(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / "outputs"
            make_episode(outputs, "episode_pass", True)
            make_episode(outputs, "episode_fail", False)
            incomplete = make_episode(outputs, "episode_incomplete")
            old = datetime.now().timestamp() - 3600
            os.utime(incomplete, (old, old))
            corrupt = make_episode(outputs, "episode_corrupt")
            (corrupt / "metrics.json").write_text("{", encoding="utf-8")
            os.utime(corrupt, (old, old))
            registry = EpisodeRegistry(outputs, incomplete_timeout_seconds=60)
            write_json(
                registry.layout.runs / "active.json",
                {
                    "run_id": "active",
                    "task_name": "test_task",
                    "status": "RUNNING",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            result = registry.scan()
            rows = {row["episode_id"]: row for row in registry.list_episodes()}

            self.assertEqual(result["episodes"], 5)
            self.assertEqual(rows["episode_pass"]["status"], "PASS")
            self.assertEqual(rows["episode_fail"]["status"], "FAIL")
            self.assertEqual(rows["episode_incomplete"]["status"], "INCOMPLETE")
            self.assertEqual(rows["episode_corrupt"]["health"], "CORRUPT")
            self.assertEqual(rows["run:active"]["status"], "RUNNING")

    def test_filters_and_rebuilds_database_from_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / "outputs"
            make_episode(outputs, "episode_pass", True, benchmark_id="formal")
            registry = EpisodeRegistry(outputs)
            registry.scan()
            rows = registry.list_episodes(
                {"status": "PASS", "task": "test_task", "seed": 7}
            )
            self.assertEqual([row["episode_id"] for row in rows], ["episode_pass"])
            registry.layout.database.write_bytes(b"not sqlite")
            result = registry.scan()
            self.assertEqual(result["statuses"], {"PASS": 1})
            self.assertEqual(
                len(list(registry.layout.state.glob("registry.sqlite3.corrupt-*"))),
                1,
            )


class RetentionManagerTests(unittest.TestCase):
    def test_external_episode_is_never_a_retention_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            external = root / "external"
            episode = make_v3_episode(
                external, "failed_gate", "episode_external_fail", success=False
            )
            old = datetime.now().timestamp() - 172800
            os.utime(episode, (old, old))
            registry = EpisodeRegistry(outputs, episode_roots=[external])
            registry.scan()
            manager = RetentionManager(registry)

            policy = manager.load_policy()
            policy["minimum_age_hours"] = 0
            preview = manager.preview(policy)
            result = manager.quarantine([episode.name])

            self.assertEqual(preview["candidate_count"], 0)
            retained = next(
                item for item in preview["retained"] if item["episode_id"] == episode.name
            )
            self.assertIn("external-read-only", retained["reasons"])
            self.assertEqual(result[0]["status"], "RETAINED")
            self.assertTrue(episode.exists())

    def test_preview_protects_benchmark_and_pin_then_quarantines_and_restores(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / "outputs"
            candidate = make_episode(outputs, "episode_candidate", False)
            benchmark = make_episode(
                outputs,
                "episode_benchmark",
                False,
                benchmark_id="formal",
            )
            pinned = make_episode(outputs, "episode_pinned", False)
            old = datetime.now().timestamp() - 172800
            for path in (candidate, benchmark, pinned):
                os.utime(path, (old, old))
            registry = EpisodeRegistry(outputs)
            manager = RetentionManager(registry)
            registry.scan()
            manager.pin("episode_pinned", "important")

            policy = manager.load_policy()
            policy["minimum_age_hours"] = 0
            preview = manager.preview(policy)
            self.assertEqual(
                [row["episode_id"] for row in preview["candidates"]],
                ["episode_candidate"],
            )

            result = manager.quarantine(["episode_candidate"])
            quarantine_id = result[0]["quarantine_id"]
            self.assertFalse(candidate.exists())
            self.assertTrue(
                (outputs / "reports" / "diagnostics" / "episode_candidate.json").exists()
            )
            manager.restore(quarantine_id)
            self.assertTrue(candidate.exists())
            self.assertIn("QUARANTINE", registry.layout.audit_log.read_text())
            self.assertIn("RESTORE", registry.layout.audit_log.read_text())


if __name__ == "__main__":
    unittest.main()
