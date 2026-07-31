import base64
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from data_platform_server import build_preview_manifest, valid_basic_auth


class DataPlatformAuthenticationTests(unittest.TestCase):
    def test_accepts_only_expected_user_and_token(self):
        token = base64.b64encode(b"farpoint:secret-token").decode()
        self.assertTrue(valid_basic_auth(f"Basic {token}", "secret-token"))

        wrong_user = base64.b64encode(b"admin:secret-token").decode()
        wrong_token = base64.b64encode(b"farpoint:wrong").decode()
        self.assertFalse(valid_basic_auth(f"Basic {wrong_user}", "secret-token"))
        self.assertFalse(valid_basic_auth(f"Basic {wrong_token}", "secret-token"))
        self.assertFalse(valid_basic_auth("", "secret-token"))


class PreviewManifestTests(unittest.TestCase):
    def test_lists_preview_frames_in_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            episodes = Path(temporary)
            preview = episodes / "episode_001" / "preview"
            preview.mkdir(parents=True)
            (preview / "rgb_0010.png").write_bytes(b"png")
            (preview / "rgb_0002.png").write_bytes(b"png")

            manifest = build_preview_manifest(episodes, "episode_001")

            self.assertEqual(manifest["episode_id"], "episode_001")
            self.assertEqual(manifest["frame_count"], 2)
            self.assertEqual(
                manifest["frames"],
                [
                    "/files/episodes/episode_001/preview/rgb_0002.png",
                    "/files/episodes/episode_001/preview/rgb_0010.png",
                ],
            )

    def test_rejects_paths_outside_episode_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            episodes = Path(temporary)
            with self.assertRaises(ValueError):
                build_preview_manifest(episodes, "../outside")
            with self.assertRaises(ValueError):
                build_preview_manifest(episodes, "nested/episode")


if __name__ == "__main__":
    unittest.main()
