import json
import tempfile
import unittest
from pathlib import Path

from farpoint.dataset import validate_episode_dataset


class DatasetTests(unittest.TestCase):
    def test_validates_multimodal_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory)
            (episode / "observations" / "rgb").mkdir(parents=True)
            (episode / "observations" / "depth").mkdir(parents=True)
            (episode / "observations" / "rgb" / "000000.png").write_bytes(b"rgb")
            (episode / "observations" / "depth" / "000000.npy").write_bytes(b"depth")
            observation = {
                "frame": 0,
                "timestamp_seconds": 0.0,
                "phase": "perception",
                "rgb_path": "observations/rgb/000000.png",
                "depth_path": "observations/depth/000000.npy",
                "joint_positions": [0.0, 0.1],
                "joint_velocities": [0.0, 0.0],
                "action_joint_positions": [0.0, 0.1],
                "contact_forces_newtons": {
                    "left_finger": 0.0,
                    "right_finger": 0.0,
                },
                "object_pose_estimate": [0.8, 0.2, 0.4],
            }
            (episode / "observations.jsonl").write_text(
                json.dumps(observation) + "\n",
                encoding="utf-8",
            )
            (episode / "labels.jsonl").write_text(
                json.dumps({"frame": 0, "object_pose_ground_truth": [0.8, 0.2, 0.4]})
                + "\n",
                encoding="utf-8",
            )
            result = validate_episode_dataset(episode)
            self.assertTrue(result["valid"], result["errors"])
            self.assertEqual(result["observation_count"], 1)

    def test_rejects_missing_artifacts_and_mismatched_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory)
            observation = {
                "frame": 0,
                "timestamp_seconds": 0.0,
                "phase": "perception",
                "rgb_path": "missing.png",
                "depth_path": "missing.npy",
                "joint_positions": [0.0],
                "joint_velocities": [],
                "action_joint_positions": [0.0],
                "contact_forces_newtons": {},
                "object_pose_estimate": None,
            }
            (episode / "observations.jsonl").write_text(
                json.dumps(observation) + "\n",
                encoding="utf-8",
            )
            (episode / "labels.jsonl").write_text("", encoding="utf-8")
            result = validate_episode_dataset(episode)
            self.assertFalse(result["valid"])
            self.assertGreaterEqual(len(result["errors"]), 4)


if __name__ == "__main__":
    unittest.main()
