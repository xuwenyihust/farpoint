import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from farpoint.registry import EpisodeRegistry
from farpoint.retention import RetentionManager


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


class EpisodeRegistryTests(unittest.TestCase):
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

    def test_collection_v2_uses_manifest_display_name_and_shape_task_type(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / "outputs"
            write_json(
                outputs / "benchmarks" / "cylinder" / "run-state.json",
                {
                    "schema_version": "farpoint.collection-state.v2",
                    "collection_id": "cylinder",
                    "display_name": "UR10e Cylinder Position Collection",
                    "task_id": "cylinder_task",
                    "object_shape": "cylinder",
                    "execution_status": "RUNNING",
                    "attempts": [],
                },
            )
            registry = EpisodeRegistry(outputs)
            registry.scan()
            row = registry.list_benchmarks()[0]
            self.assertEqual(row["record_type"], "COLLECTION")
            self.assertEqual(row["display_name"], "UR10e Cylinder Position Collection")
            self.assertEqual(row["task_type"], "cylinder_position_collection")

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
