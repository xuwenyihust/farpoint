import unittest

from farpoint.benchmark import classify_failure, randomize_task


TASK = {
    "scene": {
        "pick_object": {"position": [0.98, 0.25, 0.50]},
        "target_zone": {"position": [0.76, -0.06, 0.42]},
    },
    "randomization": {
        "enabled": True,
        "pick_object": {"x": [0.94, 1.00], "y": [0.22, 0.28]},
        "target_zone": {"x": [0.73, 0.79], "y": [-0.09, -0.03]},
        "min_pick_target_separation": 0.20,
        "max_sampling_attempts": 100,
    },
}


class BenchmarkTests(unittest.TestCase):
    def test_randomization_is_deterministic_and_preserves_input(self):
        first, first_record = randomize_task(TASK, 7)
        second, second_record = randomize_task(TASK, 7)

        self.assertEqual(first_record, second_record)
        self.assertEqual(first, second)
        self.assertEqual(TASK["scene"]["pick_object"]["position"], [0.98, 0.25, 0.50])

    def test_different_seeds_change_the_scene_within_bounds(self):
        first, _ = randomize_task(TASK, 7)
        second, record = randomize_task(TASK, 8)

        self.assertNotEqual(
            first["scene"]["pick_object"]["position"][:2],
            second["scene"]["pick_object"]["position"][:2],
        )
        self.assertGreaterEqual(record["pick_target_separation"], 0.20)
        self.assertTrue(0.94 <= record["pick_object_xy"][0] <= 1.00)
        self.assertTrue(-0.09 <= record["target_zone_xy"][1] <= -0.03)

    def test_failure_taxonomy_uses_stage_order(self):
        result = classify_failure(
            {
                "joint_motion": True,
                "grasp_created": False,
                "lift_height": False,
                "final_target_xy_distance": False,
            }
        )

        self.assertEqual(result["failure_category"], "grasp")
        self.assertEqual(result["failure_reason"], "gripper_did_not_form_a_valid_grasp")
        self.assertEqual(
            result["failed_checks"],
            ["final_target_xy_distance", "grasp_created", "lift_height"],
        )

    def test_success_has_no_failure_reason(self):
        result = classify_failure({"grasp_created": True, "final_target_xy_distance": True})

        self.assertIsNone(result["failure_category"])
        self.assertIsNone(result["failure_reason"])
        self.assertEqual(result["failed_checks"], [])


if __name__ == "__main__":
    unittest.main()
