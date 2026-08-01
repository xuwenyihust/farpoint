from farpoint.contracts import (
    validate_benchmark_episode_links,
    validate_benchmark_semantics,
    validate_contract,
    validate_episode_semantics,
)

from v2_fixtures import GIT_COMMIT, SHA, dataset_sidecar_v2, episode_metadata_v2


def complete_variation_plan():
    from pathlib import Path
    from farpoint.position_plan import generate_position_plan, load_position_config

    config = Path(__file__).resolve().parents[1] / "configs" / "variations" / "farpoint_v1_3_cube_position.json"
    return generate_position_plan(load_position_config(config))


def test_all_v2_contracts_accept_complete_typed_metadata():
    episode = episode_metadata_v2()
    dataset = dataset_sidecar_v2([episode])
    variation = complete_variation_plan()
    benchmark = {
        "schema_version": "farpoint.benchmark.v2",
        "benchmark_id": "farpoint_v1_3_cube_position",
        "task_id": episode["task"]["task_id"],
        "git_commit": GIT_COMMIT,
        "config_sha256": SHA,
        "simulator_image_digest": f"sha256:{SHA}",
        "trials": [{
            "trial_id": "trial-0000",
            "episode_id": "episode-0000",
            "variation_id": "position_r00_c00_s00",
            "split": "train",
            "status": "completed",
            "success": True,
            "dataset_valid": True,
        }],
        "acceptance": {
            "accepted": True,
            "required_success_rate": 0.9,
            "observed_success_rate": 1.0,
            "required_successes": 1,
            "observed_successes": 1,
        },
    }
    assert validate_contract(dataset) == []
    assert validate_contract(episode) == []
    assert validate_contract(variation) == []
    assert validate_contract(benchmark) == []
    assert validate_benchmark_semantics(benchmark) == []
    assert validate_episode_semantics(episode) == []
    assert validate_benchmark_episode_links(benchmark, [episode]) == []


def test_episode_semantics_reject_shape_and_resolved_pose_drift():
    episode = episode_metadata_v2()
    episode["scene"]["object"]["shape"] = "cylinder"
    episode["variation"]["resolved"]["object_position_m"] = [0.0, 0.0, 0.0]
    errors = validate_episode_semantics(episode)
    assert "task.object_shape does not match scene.object.shape" in errors
    assert any("object_position_m" in error for error in errors)


def test_contract_rejects_untyped_extension_fields():
    episode = episode_metadata_v2()
    episode["scene"]["extra_json"] = "{\"untyped\": true}"
    errors = validate_contract(episode)
    assert any("Additional properties" in error for error in errors)


def test_benchmark_link_detects_trial_identity_mismatch():
    episode = episode_metadata_v2()
    benchmark = {
        "task_id": episode["task"]["task_id"],
        "git_commit": GIT_COMMIT,
        "config_sha256": SHA,
        "simulator_image_digest": f"sha256:{SHA}",
        "trials": [{
            "trial_id": "trial-0000",
            "episode_id": "wrong",
            "variation_id": "position_r00_c00_s00",
            "split": "train",
            "success": True,
            "dataset_valid": True,
        }]
    }
    assert validate_benchmark_episode_links(benchmark, [episode]) == [
        "benchmark episode_id mismatch for trial: trial-0000"
    ]


def test_benchmark_acceptance_cannot_disagree_with_trials():
    benchmark = {
        "trials": [{"success": False}],
        "acceptance": {
            "accepted": True,
            "required_success_rate": 1.0,
            "observed_success_rate": 1.0,
            "required_successes": 1,
            "observed_successes": 1,
        },
    }
    errors = validate_benchmark_semantics(benchmark)
    assert "benchmark observed_successes does not match its trials" in errors
    assert "benchmark accepted does not match its acceptance thresholds" in errors
