import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_benchmark_report
from build_benchmark_report import (
    build_report,
    load_trial,
    normalize_manifest,
    reproducibility_summary,
    summarize,
)


class BenchmarkReportTests(unittest.TestCase):
    def test_v3_trial_links_back_to_searchable_dashboard_episode(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            episode_id = "episode_so101__cube 01"
            episode = root / "episodes" / episode_id
            episode.mkdir(parents=True)
            (episode / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": "farpoint.episode.v3",
                        "identity": {"episode_id": episode_id},
                    }
                ),
                encoding="utf-8",
            )
            (episode / "metrics.json").write_text(
                json.dumps({"success": True, "dataset_valid": True}),
                encoding="utf-8",
            )
            previous_episodes = build_benchmark_report.EPISODES_ROOT
            previous_reports = build_benchmark_report.REPORTS_ROOT
            build_benchmark_report.EPISODES_ROOT = root / "episodes"
            build_benchmark_report.REPORTS_ROOT = root / "reports"
            try:
                result = load_trial(
                    {"episode_id": episode_id, "success": True},
                    root / "reports" / "benchmarks" / "collection",
                )
            finally:
                build_benchmark_report.EPISODES_ROOT = previous_episodes
                build_benchmark_report.REPORTS_ROOT = previous_reports

        self.assertEqual(
            result["report_href"],
            "/?view=episodes&search=episode_so101__cube%2001",
        )

    def test_v3_trial_infers_middle_rgb_frame_from_recording_metadata(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            episode_id = "episode_so101__cube 01"
            episode = root / "episodes" / episode_id
            episode.mkdir(parents=True)
            (episode / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": "farpoint.episode.v3",
                        "identity": {"episode_id": episode_id},
                        "recording": {
                            "cameras": ["observation.images.front"],
                            "frame_count": 3,
                        },
                    }
                ),
                encoding="utf-8",
            )
            previous_episodes = build_benchmark_report.EPISODES_ROOT
            previous_reports = build_benchmark_report.REPORTS_ROOT
            build_benchmark_report.EPISODES_ROOT = root / "episodes"
            build_benchmark_report.REPORTS_ROOT = root / "reports"
            try:
                result = load_trial(
                    {"episode_id": episode_id, "success": True},
                    root / "reports" / "benchmarks" / "collection",
                )
            finally:
                build_benchmark_report.EPISODES_ROOT = previous_episodes
                build_benchmark_report.REPORTS_ROOT = previous_reports

        self.assertEqual(
            result["preview_href"],
            "/files/episodes/episode_so101__cube%2001/rgb/front_000001.png",
        )

    def test_balanced_so101_selection_normalizes_as_collection(self):
        balance = {
            "total": 50,
            "splits": {"train": 40, "validation": 5, "test": 5},
            "workspace_cells": {
                f"r{row:02d}_c{column:02d}": 2 for row in range(5) for column in range(5)
            },
            "workspace_rows": {f"r{row:02d}": 10 for row in range(5)},
            "workspace_columns": {f"c{column:02d}": 10 for column in range(5)},
            "sizes": {"size_0": 25, "size_1": 25},
            "colors": {"color_0": 25, "color_1": 25},
            "size_color": {"a": 12, "b": 13, "c": 13, "d": 12},
        }
        result = normalize_manifest(
            {
                "schema_version": "farpoint.collection-selection.v1",
                "collection_id": "balanced50",
                "task_id": "so101_cube_pick_place",
                "execution_status": "FINISHED",
                "quality_status": "PASS",
                "required_successes": 50,
                "attempts": [
                    {
                        "attempt_id": str(index),
                        "trial_id": str(index),
                        "episode_id": f"episode_{index}",
                        "split": "train" if index < 40 else "validation" if index < 45 else "test",
                        "success": True,
                        "dataset_valid": True,
                    }
                    for index in range(50)
                ],
                "balance": balance,
            }
        )

        self.assertEqual(result["report_kind"], "collection")
        self.assertEqual(result["benchmark_id"], "balanced50")
        self.assertEqual(result["planned_trials"], 50)
        self.assertEqual(result["passed_trials"], 50)
        self.assertEqual(result["acceptance"]["selection_balance"], balance)

        trials = [
            {
                **trial,
                "dataset_observation_count": 2,
                "final_target_xy_distance": None,
                "object_lift_height": None,
                "release_settle_frames": None,
            }
            for trial in result["trials"]
        ]
        summary = summarize(result, trials)
        self.assertTrue(summary["accepted"])
        self.assertTrue(summary["acceptance_checks"]["successful_trials_meet_task_thresholds"])
        self.assertTrue(summary["acceptance_checks"]["dataset_valid"])

    def test_selection_acceptance_uses_its_frozen_30_episode_balance(self):
        attempts = [
            {
                "attempt_id": str(index),
                "trial_id": f"cube_r{index // 5:02d}_c{index % 5:02d}",
                "episode_id": f"episode_{index}",
                "split": "train" if index < 24 else "validation" if index < 27 else "test",
                "success": True,
                "dataset_valid": True,
            }
            for index in range(30)
        ]
        balance = {
            "total": 30,
            "splits": {"train": 24, "validation": 3, "test": 3},
            "workspace_cells": {
                f"r{row:02d}_c{column:02d}": 2 if row == 0 else 1
                for row in range(5)
                for column in range(5)
            },
            "sizes": {"size_0": 30},
            "colors": {"color_0": 15, "color_1": 15},
            "masses_kg": {"0.03": 15, "0.04": 15},
            "yaw_degrees": {"30.0": 30},
        }
        manifest = normalize_manifest(
            {
                "schema_version": "farpoint.collection-selection.v1",
                "collection_id": "yaw30",
                "task_id": "so101_cube_pick_place",
                "execution_status": "FINISHED",
                "quality_status": "PASS",
                "required_successes": 30,
                "attempts": attempts,
                "balance": balance,
            }
        )
        trials = [{**trial, "dataset_observation_count": 10} for trial in manifest["trials"]]

        summary = summarize(manifest, trials)

        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["collection"]["required_cells"], 25)
        self.assertEqual(summary["collection"]["selected_episodes"], 30)
        self.assertTrue(summary["acceptance_checks"]["selection_balance_totals"])

    def test_report_uses_configured_dashboard_roots_and_v3_preview(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            episodes_root = root / "dashboard" / "episodes"
            reports_root = root / "dashboard" / "reports"
            episode_id = "episode_so101_custom_root"
            episode = episodes_root / episode_id
            rgb = episode / "rgb"
            rgb.mkdir(parents=True)
            (episode / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": "farpoint.episode.v3",
                        "variation": {"resolved": {"position_m": [0.15, -0.10, 0.047]}},
                    }
                ),
                encoding="utf-8",
            )
            (episode / "metrics.json").write_text(
                json.dumps({"success": True, "dataset_valid": True, "observation_count": 3}),
                encoding="utf-8",
            )
            (rgb / "front_000001.png").write_bytes(b"png")
            manifest_path = root / "source" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "farpoint.collection-selection.v1",
                        "collection_id": "custom-root-selection",
                        "task_id": "so101_cube_pick_place",
                        "execution_status": "FINISHED",
                        "quality_status": "PASS",
                        "required_successes": 1,
                        "attempts": [
                            {
                                "attempt_id": "attempt_0",
                                "trial_id": "cube_r00_c00",
                                "variation_id": "cube_r00_c00",
                                "episode_id": episode_id,
                                "split": "train",
                                "success": True,
                                "dataset_valid": True,
                            }
                        ],
                        "balance": {
                            "total": 1,
                            "splits": {"train": 1},
                            "workspace_cells": {"r00_c00": 1},
                            "sizes": {"size_0": 1},
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = build_report(
                manifest_path,
                episodes_root=episodes_root,
                reports_root=reports_root,
            )
            rendered = report.read_text(encoding="utf-8")

        self.assertEqual(
            report,
            reports_root / "benchmarks" / "custom-root-selection" / "index.html",
        )
        self.assertIn(
            "/files/episodes/episode_so101_custom_root/rgb/front_000001.png",
            rendered,
        )
        self.assertIn("/?view=episodes&amp;search=episode_so101_custom_root", rendered)
        self.assertNotIn("No preview", rendered)

    def test_report_links_do_not_follow_snapshot_symlinks(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            external = root / "external" / "episode-report"
            external.mkdir(parents=True)
            local_report = reports / "episode-imported"
            local_report.parent.mkdir(parents=True)
            local_report.symlink_to(external, target_is_directory=True)

            href = build_benchmark_report.relative_path(
                local_report / "index.html",
                reports / "benchmarks" / "collection-id",
            )

        self.assertEqual(href.as_posix(), "../../episode-imported/index.html")

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
        result = reproducibility_summary([{"trial_id": "missing", "success": False}])

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

    def test_v2_manifest_uses_nested_acceptance_and_formal_defaults(self):
        result = normalize_manifest(
            {
                "schema_version": "farpoint.benchmark.v2",
                "benchmark_id": "formal",
                "task_id": "isaac_perception_contact_scene",
                "trials": [{"success": True}, {"success": False}],
                "acceptance": {
                    "accepted": True,
                    "required_success_rate": 0.5,
                    "observed_success_rate": 0.5,
                    "required_successes": 1,
                    "observed_successes": 1,
                },
            }
        )

        self.assertEqual(result["planned_trials"], 2)
        self.assertEqual(result["passed_trials"], 1)
        self.assertEqual(result["acceptance"]["min_success_rate"], 0.5)
        self.assertTrue(result["acceptance"]["require_dataset"])

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

    def test_workspace_feasibility_requires_measured_xy_span(self):
        manifest = {
            "benchmark_id": "workspace",
            "task_name": "task",
            "planned_trials": 2,
            "acceptance": {
                "min_success_rate": 1.0,
                "max_final_target_xy_distance": 0.05,
                "min_object_lift_height": 0.15,
                "min_release_settle_frames": 120,
                "min_selected_x_span_m": 0.20,
                "min_selected_y_span_m": 0.16,
            },
        }
        base = {
            "success": True,
            "final_target_xy_distance": 0.01,
            "object_lift_height": 0.20,
            "release_settle_frames": 120,
        }
        result = summarize(
            manifest,
            [
                {**base, "seed": 1, "pick_object_xy": [0.86, 0.19]},
                {**base, "seed": 2, "pick_object_xy": [1.09, 0.37]},
            ],
        )

        self.assertTrue(result["accepted"])
        self.assertTrue(result["acceptance_checks"]["workspace_x_span"])
        self.assertTrue(result["acceptance_checks"]["workspace_y_span"])
        self.assertAlmostEqual(result["workspace_coverage"]["x_span_m"], 0.23)

    def test_workspace_report_renders_position_columns_and_coverage(self):
        manifest = {
            "benchmark_id": "workspace",
            "task_name": "task",
            "planned_trials": 1,
            "acceptance": {
                "min_success_rate": 1.0,
                "max_final_target_xy_distance": 0.05,
                "min_object_lift_height": 0.15,
                "min_release_settle_frames": 120,
            },
            "trials": [
                {
                    "trial_id": "corner",
                    "seed": 1,
                    "object_position_xy_m": [0.86, 0.19],
                    "success": True,
                    "final_target_xy_distance": 0.01,
                    "object_lift_height": 0.20,
                    "release_settle_frames": 120,
                }
            ],
        }
        with self.subTest("isolated report roots"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as directory:
                root = Path(directory)
                old_episodes = build_benchmark_report.EPISODES_ROOT
                old_reports = build_benchmark_report.REPORTS_ROOT
                build_benchmark_report.EPISODES_ROOT = root / "episodes"
                build_benchmark_report.REPORTS_ROOT = root / "reports"
                try:
                    manifest_path = root / "benchmarks" / "workspace" / "manifest.json"
                    manifest_path.parent.mkdir(parents=True)
                    manifest_path.write_text(json.dumps(manifest))
                    report_path = build_report(manifest_path)
                    rendered = report_path.read_text()
                finally:
                    build_benchmark_report.EPISODES_ROOT = old_episodes
                    build_benchmark_report.REPORTS_ROOT = old_reports

        self.assertIn("Object XY (m)", rendered)
        self.assertIn("0.860, 0.190", rendered)
        self.assertIn("Position X Span", rendered)

    def test_collection_report_renders_collection_metrics_and_cell_grid(self):
        attempts = [
            {
                "trial_id": "trial-0",
                "episode_id": "episode-0",
                "variation_id": "position_r00_c00_s00",
                "cell_id": "r00_c00",
                "slot": 0,
                "source_split": "train",
                "dataset_split": "train",
                "selection_rank": 1,
                "origin": "imported",
                "source_run_id": "source",
                "source_git_commit": "a" * 40,
                "outcome_success": True,
                "dataset_valid": True,
                "selected_for_dataset": True,
                "failure_category": None,
                "failure_reason": None,
            }
        ]
        manifest = {
            "schema_version": "farpoint.collection-run.v1",
            "collection_id": "collection-report",
            "task_id": "task",
            "execution_status": "PILOT_COMPLETE",
            "attempts": attempts,
            "acceptance": {
                "accepted": False,
                "required_task_yield": 0.75,
                "observed_task_yield": 1.0,
                "maximum_task_attempts": 73,
                "observed_task_attempts": 1,
                "observed_task_successes": 1,
                "required_selected_episodes": 50,
                "observed_selected_episodes": 1,
                "required_cells": 25,
                "observed_covered_cells": 1,
                "required_selected_per_cell": 2,
                "selected_per_cell": {"r00_c00": 1},
                "required_splits": {"train": 34, "validation": 8, "test": 8},
                "observed_splits": {"train": 1, "validation": 0, "test": 0},
            },
        }
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            old_episodes = build_benchmark_report.EPISODES_ROOT
            old_reports = build_benchmark_report.REPORTS_ROOT
            build_benchmark_report.EPISODES_ROOT = root / "episodes"
            build_benchmark_report.REPORTS_ROOT = root / "reports"
            try:
                path = root / "benchmarks" / "collection-report" / "run-state.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(manifest))
                rendered = build_report(
                    path, display_name="UR10e Cube Position Balanced Collection"
                ).read_text()
            finally:
                build_benchmark_report.EPISODES_ROOT = old_episodes
                build_benchmark_report.REPORTS_ROOT = old_reports

        self.assertIn("Collection Status", rendered)
        self.assertIn("UR10e Cube Position Balanced Collection", rendered)
        self.assertIn("Collection ID: <code>collection-report</code>", rendered)
        self.assertIn("PILOT COMPLETE", rendered)
        self.assertIn('All planned trials completed</td><td class="pending">NOT RUN', rendered)
        self.assertIn("Selected Episodes", rendered)
        self.assertIn("Grid Cell Coverage", rendered)
        self.assertIn("r04_c04", rendered)


if __name__ == "__main__":
    unittest.main()
