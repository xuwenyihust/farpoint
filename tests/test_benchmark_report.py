import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_benchmark_report import normalize_manifest, reproducibility_summary, summarize


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


if __name__ == "__main__":
    unittest.main()
