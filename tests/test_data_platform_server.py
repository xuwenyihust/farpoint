import base64
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from data_platform_server import (  # noqa: E402
    build_preview_manifest,
    resolve_episode_asset,
    resolve_registered_episode_asset,
    valid_basic_auth,
)
from data_platform_cli import supports_legacy_episode_report  # noqa: E402


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
    def test_lists_v3_front_rgb_frames_without_a_preview_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary) / "episode_so101_001"
            rgb = episode / "rgb"
            rgb.mkdir(parents=True)
            (episode / "metadata.json").write_text(
                '{"schema_version":"farpoint.episode.v3","identity":'
                '{"episode_id":"episode_so101_001"}}',
                encoding="utf-8",
            )
            (rgb / "front_000001.png").write_bytes(b"png")
            (rgb / "front_000000.png").write_bytes(b"png")

            manifest = build_preview_manifest(
                Path(temporary),
                episode.name,
                episode_dir=episode,
            )
            resolved = resolve_registered_episode_asset(
                episode, episode.name, "rgb/front_000001.png"
            )

            self.assertEqual(manifest["frame_count"], 2)
            self.assertEqual(
                manifest["frames"],
                [
                    "/files/episodes/episode_so101_001/rgb/front_000000.png",
                    "/files/episodes/episode_so101_001/rgb/front_000001.png",
                ],
            )
            self.assertEqual(resolved, (rgb / "front_000001.png").resolve())

    def test_registered_v3_asset_rejects_escape_and_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary) / "episode_so101_001"
            episode.mkdir()
            (episode / "metadata.json").write_text(
                '{"identity":{"episode_id":"different"}}', encoding="utf-8"
            )

            with self.assertRaises(ValueError):
                resolve_registered_episode_asset(
                    episode, "episode_so101_001", "metadata.json"
                )
            with self.assertRaises(ValueError):
                resolve_registered_episode_asset(episode, "different", "../outside")


class EpisodeReportCompatibilityTests(unittest.TestCase):
    def test_v3_rgb_episode_does_not_enter_legacy_report_builder(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary) / "episode_so101_001"
            episode.mkdir()
            (episode / "metadata.json").write_text(
                '{"schema_version":"farpoint.episode.v3"}', encoding="utf-8"
            )
            for name in ("metrics.json", "trajectory.jsonl", "phase_events.jsonl"):
                (episode / name).write_text("{}", encoding="utf-8")
            preview = episode / "preview"
            preview.mkdir()
            (preview / "frame.png").write_bytes(b"png")

            self.assertFalse(supports_legacy_episode_report(episode))

    def test_complete_legacy_episode_can_still_build_a_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary) / "episode_legacy_001"
            episode.mkdir()
            for name in (
                "metadata.json",
                "metrics.json",
                "trajectory.jsonl",
                "phase_events.jsonl",
            ):
                (episode / name).write_text("{}", encoding="utf-8")
            preview = episode / "preview"
            preview.mkdir()
            (preview / "frame.png").write_bytes(b"png")

            self.assertTrue(supports_legacy_episode_report(episode))

    def test_lists_preview_frames_in_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            episodes = Path(temporary)
            preview = episodes / "episode_001" / "preview"
            preview.mkdir(parents=True)
            (preview.parent / "metadata.json").write_text(
                '{"episode_id":"episode_001"}', encoding="utf-8"
            )
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

    def test_resolves_assets_from_an_identity_checked_symlink_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episodes = root / "episodes"
            source = root / "source" / "episode_001"
            preview = source / "preview"
            preview.mkdir(parents=True)
            (source / "metadata.json").write_text(
                '{"episode_id":"episode_001"}', encoding="utf-8"
            )
            frame = preview / "rgb_0001.png"
            frame.write_bytes(b"png")
            episodes.mkdir()
            (episodes / "episode_001").symlink_to(source, target_is_directory=True)

            resolved = resolve_episode_asset(
                episodes, "episode_001/preview/rgb_0001.png"
            )
            manifest = build_preview_manifest(episodes, "episode_001")

            self.assertEqual(resolved, frame.resolve())
            self.assertEqual(manifest["frame_count"], 1)

    def test_rejects_episode_asset_traversal_or_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episodes = root / "episodes"
            source = root / "source" / "episode_001"
            source.mkdir(parents=True)
            episodes.mkdir()
            (source / "metadata.json").write_text(
                '{"episode_id":"different"}', encoding="utf-8"
            )
            (episodes / "episode_001").symlink_to(source, target_is_directory=True)

            with self.assertRaises(ValueError):
                resolve_episode_asset(episodes, "../outside")
            with self.assertRaises(ValueError):
                resolve_episode_asset(episodes, "episode_001/metadata.json")


if __name__ == "__main__":
    unittest.main()
