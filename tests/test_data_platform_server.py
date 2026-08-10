import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from data_platform_server import (  # noqa: E402
    build_episode_detail,
    build_preview_manifest,
    resolve_episode_asset,
    resolve_registered_episode_asset,
    valid_basic_auth,
)
from data_platform_cli import build_reports, supports_legacy_episode_report  # noqa: E402


class DataPlatformAuthenticationTests(unittest.TestCase):
    def test_accepts_only_expected_user_and_token(self):
        token = base64.b64encode(b"farpoint:secret-token").decode()
        self.assertTrue(valid_basic_auth(f"Basic {token}", "secret-token"))

        wrong_user = base64.b64encode(b"admin:secret-token").decode()
        wrong_token = base64.b64encode(b"farpoint:wrong").decode()
        self.assertFalse(valid_basic_auth(f"Basic {wrong_user}", "secret-token"))
        self.assertFalse(valid_basic_auth(f"Basic {wrong_token}", "secret-token"))
        self.assertFalse(valid_basic_auth("", "secret-token"))


class DataPlatformReportBuildTests(unittest.TestCase):
    def test_benchmark_builder_receives_registry_episode_and_report_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "benchmarks" / "selection" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}", encoding="utf-8")
            layout = SimpleNamespace(
                episodes=root / "registered-episodes",
                reports=root / "registered-reports",
                display_names=root / ".data-platform" / "display-names.json",
            )

            class Registry:
                def __init__(self):
                    self.layout = layout

                def list_episodes(self, limit):
                    self.limit = limit
                    return []

                def list_benchmarks(self):
                    return [
                        {
                            "benchmark_id": "selection",
                            "manifest_path": str(manifest),
                            "display_name": None,
                        }
                    ]

                def scan(self):
                    self.scanned = True

            registry = Registry()
            with mock.patch("data_platform_cli.subprocess.run") as run:
                run.return_value.returncode = 0

                result = build_reports(registry)

        command = run.call_args.args[0]
        self.assertEqual(result["benchmark_reports"], 1)
        self.assertEqual(command[command.index("--episodes-root") + 1], str(layout.episodes))
        self.assertEqual(command[command.index("--reports-root") + 1], str(layout.reports))


class PreviewManifestTests(unittest.TestCase):
    def test_running_episode_sidecar_supports_detail_and_front_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary) / "episode_pilot_live__trial_001"
            rgb = episode / "rgb"
            rgb.mkdir(parents=True)
            requested = {"pick_object": {"entity_type": "cube"}}
            resolved = {"pick_object": {"entity_type": "cube"}}
            (episode / "run-state.json").write_text(
                json.dumps(
                    {
                        "schema_version": "farpoint.episode-run.v1",
                        "execution_status": "RUNNING",
                        "identity": {
                            "episode_id": episode.name,
                            "task_id": "so101_cube_pick_place",
                            "split": "train",
                        },
                        "variation": {
                            "variation_id": "cube_live",
                            "requested": {"entities": requested},
                            "resolved": {"entities": resolved},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (rgb / "front_000000.png").write_bytes(b"png")

            detail = build_episode_detail(
                episode,
                episode.name,
                registry_row={"collection_id": "pilot_live", "managed": 0},
            )
            preview = build_preview_manifest(Path(temporary), episode.name, episode_dir=episode)

            self.assertEqual(detail["variation"]["variation_id"], "cube_live")
            self.assertEqual(detail["task"]["task_id"], "so101_cube_pick_place")
            self.assertEqual(detail["requested_entities"], requested)
            self.assertEqual(detail["resolved_entities"], resolved)
            self.assertEqual(preview["frame_count"], 1)

    def test_builds_v3_entity_detail_from_registered_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary) / "episode_so101_entities"
            episode.mkdir()
            entity = {
                "entity_id": "placement_target",
                "role": "placement_target",
                "entity_type": "box",
                "asset_id": "open_box_v1",
            }
            requested = {"placement_target": {**entity, "requested": True}}
            resolved = {"placement_target": {**entity, "requested": False}}
            (episode / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": "farpoint.episode.v3",
                        "identity": {
                            "episode_id": episode.name,
                            "split": "validation",
                        },
                        "task": {
                            "task_id": "pick_place_generic",
                            "target_entity_id": "placement_target",
                            "acceptance_region_id": "placement_region",
                        },
                        "scene": {"entities": [entity]},
                        "variation": {
                            "variation_id": "box_pose_01",
                            "varied_axes": ["entities.placement_target.pose.position_m"],
                            "requested": {"entities": requested},
                            "resolved": {"entities": resolved},
                        },
                    }
                ),
                encoding="utf-8",
            )

            detail = build_episode_detail(
                episode,
                episode.name,
                registry_row={"collection_id": "box_gate", "managed": 0},
            )

            self.assertEqual(detail["scene_entities"], [entity])
            self.assertEqual(detail["requested_entities"], requested)
            self.assertEqual(detail["resolved_entities"], resolved)
            self.assertEqual(detail["source"]["collection_id"], "box_gate")
            self.assertFalse(detail["source"]["managed"])

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
                resolve_registered_episode_asset(episode, "episode_so101_001", "metadata.json")
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
            (source / "metadata.json").write_text('{"episode_id":"episode_001"}', encoding="utf-8")
            frame = preview / "rgb_0001.png"
            frame.write_bytes(b"png")
            episodes.mkdir()
            (episodes / "episode_001").symlink_to(source, target_is_directory=True)

            resolved = resolve_episode_asset(episodes, "episode_001/preview/rgb_0001.png")
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
            (source / "metadata.json").write_text('{"episode_id":"different"}', encoding="utf-8")
            (episodes / "episode_001").symlink_to(source, target_is_directory=True)

            with self.assertRaises(ValueError):
                resolve_episode_asset(episodes, "../outside")
            with self.assertRaises(ValueError):
                resolve_episode_asset(episodes, "episode_001/metadata.json")


if __name__ == "__main__":
    unittest.main()
