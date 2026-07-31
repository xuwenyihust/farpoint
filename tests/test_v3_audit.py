import tempfile
import unittest
from pathlib import Path

from farpoint.v3_audit import (
    audit_contact_only_source,
    audit_episode_runtime,
)


class V3AuditTests(unittest.TestCase):
    def test_runtime_audit_requires_rgbd_contact_only_and_no_joint(self):
        result = audit_episode_runtime(
            {
                "grasp_constraint": "contact_only",
                "control_pose_source": "rgbd_estimate",
                "temporary_grasp_joint_created": False,
                "grasp_joint_path": None,
                "dataset_valid": True,
            }
        )
        self.assertTrue(result["valid"])

    def test_source_audit_reports_forbidden_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "scene.py"
            scene.write_text(
                "create_fixed_grasp_joint(stage, path, a, b)\n",
                encoding="utf-8",
            )
            result = audit_contact_only_source(scene)
            self.assertFalse(result["valid"])
            self.assertEqual(
                result["forbidden_calls"][0]["call"],
                "create_fixed_grasp_joint",
            )


if __name__ == "__main__":
    unittest.main()
