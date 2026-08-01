import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_benchmark_report
from build_benchmark_report import (
    build_report,
    normalize_manifest,
    reproducibility_summary,
    summarize,
)


class BenchmarkReportTests(unittest.TestCase):
    def test_summary_enforces_batch_acceptance(self):
        manifest = {
            "benchmark_id": "test",
            "task_name": "task",
            "planned_trials": 2,
            "acceptance": {
                "min_success_rate": 0.80,
                "max_final_target_xy_distance": 0.04,
                "min_object_lift_height": 0.15,
                "min_release_settle_frames": 120,
            },
        }
        trials = [
            {
                "seed": 0,
                "success": True,
                "final_target_xy_distance": 0.01,
                "object_lift_height": 0.17,
                "release_settle_frames": 150,
            },
            {
                "seed": 1,
                "success": True,
                "final_target_xy_distance": 0.02,
                "object_lift_height": 0.18,
                "release_settle_frames": 150,
            },
        ]

        result = summarize(manifest, trials)

        self.assertTrue(result["accepted"])
        self.assertEqual(result["success_rate"], 1.0)
        self.assertAlmostEqual(result["mean_target_xy_error"], 0.015)

    def test_reproducibility_detects_inconsistent_outcome(self):
        result = reproducibility_summary(
            [
                {"seed": 3, "success": True, "final_target_xy_distance": 0.01},
                {"seed": 3, "success": False, "final_target_xy_distance": 0.03},
            ]
        )

        self.assertTrue(result["evaluated"])
        self.assertFalse(result["consistent"])

    def test_reproducibility_ignores_incomplete_trials_without_seed(self):
        result = reproducibility_summary(
            [{"trial_id": "missing", "success": False}]
        )

        self.assertFalse(result["evaluated"])
        self.assertEqual(result["repeated_seed_count"], 0)

    def test_position_pilot_manifest_is_normalized_to_benchmark_contract(self):
        result = normalize_manifest(
            {
                "pilot_id": "cube_position_pilot_1",
                "task_id": "isaac_perception_contact_scene",
                "planned_trials": 9,
                "acceptance": {
                    "required_successes": 9,
                    "contact_only": True,
                    "max_perception_xy_error_m": 0.02,
                    "min_lift_height_m": 0.15,
                    "max_final_target_xy_error_m": 0.05,
                    "min_settle_frames": 120,
                },
            }
        )

        self.assertEqual(result["benchmark_id"], "cube_position_pilot_1")
        self.assertEqual(result["task_name"], "isaac_perception_contact_scene")
        self.assertEqual(result["acceptance"]["min_success_rate"], 1.0)
        self.assertTrue(result["acceptance"]["require_contact_only"])
        self.assertEqual(result["acceptance"]["min_object_lift_height"], 0.15)

    def test_v2_manifest_uses_nested_acceptance_and_formal_defaults(self):
        result = normalize_manifest(
            {
                "schema_version": "farpoint.benchmark.v2",
                "benchmark_id": "formal",
                "task_id": "isaac_perception_contact_scene",
                "trials": [{"success": True}, {"success": False}],
                "acceptance": {
                    "accepted": True,
                    "required_success_rate": 0.5,
                    "observed_success_rate": 0.5,
                    "required_successes": 1,
                    "observed_successes": 1,
                },
            }
        )

        self.assertEqual(result["planned_trials"], 2)
        self.assertEqual(result["passed_trials"], 1)
        self.assertEqual(result["acceptance"]["min_success_rate"], 0.5)
        self.assertTrue(result["acceptance"]["require_dataset"])

    def test_position_pilot_checks_are_required_for_report_acceptance(self):
        manifest = {
            "pilot_id": "pilot",
            "task_id": "task",
            "planned_trials": 1,
            "acceptance": {
                "required_successes": 1,
                "max_final_target_xy_error_m": 0.05,
                "min_lift_height_m": 0.15,
                "min_settle_frames": 120,
            },
        }
        trial = {
            "seed": 1,
            "success": True,
            "final_target_xy_distance": 0.01,
            "object_lift_height": 0.20,
            "release_settle_frames": 120,
            "checks": {"required_files": False},
        }

        result = summarize(manifest, [trial])

        self.assertFalse(result["accepted"])
        self.assertFalse(result["acceptance_checks"]["pilot_episode_checks"])

    def test_workspace_feasibility_requires_measured_xy_span(self):
        manifest = {
            "benchmark_id": "workspace",
            "task_name": "task",
            "planned_trials": 2,
            "acceptance": {
                "min_success_rate": 1.0,
                "max_final_target_xy_distance": 0.05,
                "min_object_lift_height": 0.15,
                "min_release_settle_frames": 120,
                "min_selected_x_span_m": 0.20,
                "min_selected_y_span_m": 0.16,
            },
        }
        base = {
            "success": True,
            "final_target_xy_distance": 0.01,
            "object_lift_height": 0.20,
            "release_settle_frames": 120,
        }
        result = summarize(
            manifest,
            [
                {**base, "seed": 1, "pick_object_xy": [0.86, 0.19]},
                {**base, "seed": 2, "pick_object_xy": [1.09, 0.37]},
            ],
        )

        self.assertTrue(result["accepted"])
        self.assertTrue(result["acceptance_checks"]["workspace_x_span"])
        self.assertTrue(result["acceptance_checks"]["workspace_y_span"])
        self.assertAlmostEqual(result["workspace_coverage"]["x_span_m"], 0.23)

    def test_workspace_report_renders_position_columns_and_coverage(self):
        manifest = {
            "benchmark_id": "workspace",
            "task_name": "task",
            "planned_trials": 1,
            "acceptance": {
                "min_success_rate": 1.0,
                "max_final_target_xy_distance": 0.05,
                "min_object_lift_height": 0.15,
                "min_release_settle_frames": 120,
            },
            "trials": [
                {
                    "trial_id": "corner",
                    "seed": 1,
                    "object_position_xy_m": [0.86, 0.19],
                    "success": True,
                    "final_target_xy_distance": 0.01,
                    "object_lift_height": 0.20,
                    "release_settle_frames": 120,
                }
            ],
        }
        with self.subTest("isolated report roots"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as directory:
                root = Path(directory)
                old_episodes = build_benchmark_report.EPISODES_ROOT
                old_reports = build_benchmark_report.REPORTS_ROOT
                build_benchmark_report.EPISODES_ROOT = root / "episodes"
                build_benchmark_report.REPORTS_ROOT = root / "reports"
                try:
                    manifest_path = root / "benchmarks" / "workspace" / "manifest.json"
                    manifest_path.parent.mkdir(parents=True)
                    manifest_path.write_text(json.dumps(manifest))
                    report_path = build_report(manifest_path)
                    rendered = report_path.read_text()
                finally:
                    build_benchmark_report.EPISODES_ROOT = old_episodes
                    build_benchmark_report.REPORTS_ROOT = old_reports

        self.assertIn("Object XY (m)", rendered)
        self.assertIn("0.860, 0.190", rendered)
        self.assertIn("Position X Span", rendered)


if __name__ == "__main__":
    unittest.main()
