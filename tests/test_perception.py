import unittest

import numpy as np

from farpoint.perception import (
    PerceptionError,
    backproject_pixels,
    estimate_dominant_color_pose,
    look_at_calibration,
)


class PerceptionTests(unittest.TestCase):
    def test_backprojection_uses_depth_and_calibration(self):
        intrinsics = np.asarray(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
        )
        camera_to_world = np.eye(4)
        camera_to_world[:3, 3] = [1.0, 2.0, 3.0]
        result = backproject_pixels(
            np.asarray([[50.0, 40.0], [60.0, 40.0]]),
            np.asarray([2.0, 2.0]),
            intrinsics,
            camera_to_world,
        )
        np.testing.assert_allclose(result[0], [1.0, 2.0, 5.0])
        np.testing.assert_allclose(result[1], [1.2, 2.0, 5.0])

    def test_color_pose_estimator_ignores_invalid_depth(self):
        rgb = np.zeros((20, 30, 3), dtype=np.uint8)
        rgb[6:14, 10:20, 0] = 220
        rgb[6:14, 10:20, 1] = 30
        depth = np.full((20, 30), 2.0, dtype=np.float32)
        depth[6, 10] = np.inf
        intrinsics = np.asarray(
            [[100.0, 0.0, 15.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]]
        )
        result = estimate_dominant_color_pose(
            rgb,
            depth,
            intrinsics,
            np.eye(4),
            "red",
            min_pixels=20,
        )
        self.assertGreater(result["valid_pixels"], 70)
        self.assertAlmostEqual(result["position"][0], 0.0, delta=0.02)
        self.assertAlmostEqual(result["position"][1], 0.0, delta=0.02)
        self.assertAlmostEqual(result["position"][2], 2.0, places=4)

    def test_missing_color_raises_perception_error(self):
        with self.assertRaises(PerceptionError):
            estimate_dominant_color_pose(
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.ones((8, 8), dtype=np.float32),
                np.eye(3),
                np.eye(4),
                "green",
                min_pixels=2,
            )

    def test_bounds_center_handles_asymmetric_visible_points(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb[:8, 1, 0] = 220
        rgb[:2, 5, 0] = 220
        depth = np.ones((10, 10), dtype=np.float32)

        median_result = estimate_dominant_color_pose(
            rgb,
            depth,
            np.eye(3),
            np.eye(4),
            "red",
            min_pixels=5,
        )
        bounds_result = estimate_dominant_color_pose(
            rgb,
            depth,
            np.eye(3),
            np.eye(4),
            "red",
            min_pixels=5,
            xy_center_method="bounds",
        )

        self.assertEqual(median_result["position"][0], 1.0)
        self.assertEqual(bounds_result["position"][0], 3.0)
        self.assertEqual(bounds_result["xy_center_method"], "bounds")

    def test_rejects_unknown_xy_center_method(self):
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[:, :, 0] = 220
        with self.assertRaisesRegex(ValueError, "xy_center_method"):
            estimate_dominant_color_pose(
                rgb,
                np.ones((4, 4), dtype=np.float32),
                np.eye(3),
                np.eye(4),
                "red",
                min_pixels=2,
                xy_center_method="unknown",
            )

    def test_look_at_calibration_points_forward_at_target(self):
        _, camera_to_world = look_at_calibration(
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            [320, 240],
        )
        np.testing.assert_allclose(camera_to_world[:3, 2], [0.0, 0.0, -1.0])


if __name__ == "__main__":
    unittest.main()
