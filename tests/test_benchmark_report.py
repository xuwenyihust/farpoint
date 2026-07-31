import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_benchmark_report import reproducibility_summary, summarize


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


if __name__ == "__main__":
    unittest.main()
