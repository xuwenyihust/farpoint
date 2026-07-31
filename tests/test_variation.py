import json
import unittest
from pathlib import Path

from farpoint.variation import (
    CONFIG_VERSION,
    load_variation_config,
    plan_variations,
    resolve_variation,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "farpoint_v1_1_variations.json"
)


class VariationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_variation_config(CONFIG_PATH)

    def test_config_is_versioned_and_has_six_profiles(self):
        self.assertEqual(self.config["schema_version"], CONFIG_VERSION)
        self.assertEqual(len(self.config["profiles"]), 6)

    def test_resolution_is_deterministic(self):
        first = resolve_variation(self.config, "cube_position_center", 17)
        second = resolve_variation(self.config, "cube_position_center", 17)
        self.assertEqual(first, second)
        self.assertEqual(first["object_type"], "cube")
        self.assertEqual(first["object_position_bin"], "center")

    def test_cylinder_profiles_select_the_cylinder_grasp_profile(self):
        variation = resolve_variation(self.config, "cylinder_position_right", 0)
        self.assertEqual(variation["object_type"], "cylinder")
        self.assertEqual(variation["grasp_profile"], "cylinder_grip_v1")

    def test_different_seeds_are_planned_without_simulation(self):
        plan = plan_variations(self.config, [0, 1])
        self.assertEqual(len(plan), 12)
        self.assertEqual(
            [item["variation_id"] for item in plan[:2]],
            ["cube_position_left", "cube_position_left"],
        )
        self.assertNotEqual(plan[0]["derived_seed"], plan[1]["derived_seed"])
        self.assertTrue(
            0.94 <= plan[0]["object_position_xy"][0] <= 0.96
        )

    def test_json_config_round_trips(self):
        self.assertEqual(json.loads(CONFIG_PATH.read_text())["schema_version"], CONFIG_VERSION)


if __name__ == "__main__":
    unittest.main()
