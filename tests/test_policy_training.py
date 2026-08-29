import json
import importlib.util
import hashlib
from pathlib import Path

import pytest

from farpoint.contracts import load_schema, validate_contract
from farpoint.policy_training import (
    create_training_view,
    directory_tree_sha256,
    evenly_spaced_indices,
    load_training_spec,
    parse_episode_slice,
    select_validation_checkpoint,
    training_arguments,
    unflatten_episode_stats,
    validate_dataset_info,
    validation_profile,
)
from farpoint.training_sampler import expected_group_sample_counts, parse_episode_slices


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_SPEC = importlib.util.spec_from_file_location(
    "preflight_policy_training", ROOT / "scripts" / "preflight_policy_training.py"
)
PREFLIGHT_MODULE = importlib.util.module_from_spec(PREFLIGHT_SPEC)
PREFLIGHT_SPEC.loader.exec_module(PREFLIGHT_MODULE)
CONFIG = ROOT / "configs" / "training" / "so101_act_v0_0_3_baseline.json"
PILOT_CONFIG = ROOT / "configs" / "training" / "so101_act_v0_0_3_pilot.json"
BASELINE_20K_CONFIG = (
    ROOT / "configs" / "training" / "so101_act_v0_0_3_baseline_20k.json"
)
V010_PILOT_CONFIG = ROOT / "configs" / "training" / "so101_act_v0_1_0_pilot.json"
V010_BASELINE_20K_CONFIG = (
    ROOT / "configs" / "training" / "so101_act_v0_1_0_baseline_20k.json"
)
V011_RECOVERY20_20K_CONFIG = (
    ROOT / "configs" / "training" / "so101_act_v0_1_1_recovery20_20k.json"
)
V012_GRASP20_20K_CONFIG = (
    ROOT / "configs" / "training" / "so101_act_v0_1_2_grasp20_20k.json"
)
V012_NOMINAL_ONLY_20K_CONFIG = (
    ROOT / "configs" / "training" / "so101_act_v0_1_2_nominal_only_20k.json"
)
V012_BALANCED_MIX_20K_CONFIG = (
    ROOT / "configs" / "training" / "so101_act_v0_1_2_balanced_mix_20k.json"
)
V013_BALANCED_MIX_20K_CONFIG = (
    ROOT / "configs" / "training" / "so101_act_v0_1_3_balanced_mix_20k.json"
)
V013_BALANCED_MIX_200K_CONFIG = (
    ROOT / "configs" / "training" / "so101_act_v0_1_3_balanced_mix_200k.json"
)
V014_BALANCED_MIX_200K_CONFIG = (
    ROOT / "configs" / "training" / "so101_act_v0_1_4_balanced_mix_200k.json"
)
V014_BALANCED_MIX_300K_CONTINUATION_CONFIG = (
    ROOT
    / "configs"
    / "training"
    / "so101_act_v0_1_4_balanced_mix_300k_continuation.json"
)
V020_CELL_BALANCED_200K_CONFIG = (
    ROOT / "configs" / "training" / "so101_act_v0_2_0_cell_balanced_200k.json"
)
V020_SMOLVLA_UNIFORM_20K_CONFIG = (
    ROOT / "configs" / "training" / "so101_smolvla_v0_2_0_uniform_20k.json"
)


def test_local_snapshot_tree_hash_binds_paths_sizes_and_contents(tmp_path):
    snapshot = tmp_path / "snapshot"
    (snapshot / "meta").mkdir(parents=True)
    (snapshot / "meta" / "info.json").write_text("{}\n", encoding="utf-8")
    first, count = directory_tree_sha256(snapshot)
    assert len(first) == 64
    assert count == 1

    (snapshot / "meta" / "info.json").write_text('{"changed": true}\n', encoding="utf-8")
    second, second_count = directory_tree_sha256(snapshot)
    assert second != first
    assert second_count == 1

    (snapshot / "alias").symlink_to(snapshot / "meta", target_is_directory=True)
    with pytest.raises(ValueError, match="must not contain symlinks"):
        directory_tree_sha256(snapshot)


def test_local_snapshot_contract_does_not_require_a_hub_commit():
    spec = load_training_spec(V012_BALANCED_MIX_20K_CONFIG)
    dataset = spec["dataset"]
    dataset.pop("resolved_commit")
    dataset["source"] = {"kind": "local_snapshot", "tree_sha256": "a" * 64}
    assert validate_contract(spec) == []

    dataset["source"] = {"kind": "hub"}
    assert any("resolved_commit" in error for error in validate_contract(spec))
    dataset["resolved_commit"] = "b" * 40
    dataset["source"]["tree_sha256"] = "a" * 64
    assert any("tree_sha256" in error for error in validate_contract(spec))


def test_frozen_act_contract_is_valid_and_partitions_all_episodes():
    spec = load_training_spec(CONFIG)
    assert validate_contract(spec) == []
    assert parse_episode_slice(spec["dataset"]["splits"]["train"]) == list(range(128))
    assert parse_episode_slice(spec["dataset"]["splits"]["validation"]) == list(
        range(128, 142)
    )
    assert parse_episode_slice(spec["dataset"]["splits"]["test"]) == list(range(142, 160))
    assert load_schema("farpoint.policy-training.v1")["title"].endswith("v1")


@pytest.mark.parametrize("expression", ["train", "0:0", "4:2", "-1:2", "0:2:1", "1:"])
def test_episode_slice_rejects_ambiguous_or_empty_ranges(expression):
    with pytest.raises(ValueError):
        parse_episode_slice(expression)


def test_dataset_metadata_must_match_pinned_release():
    spec = load_training_spec(CONFIG)
    dataset = spec["dataset"]
    info = {
        "codebase_version": dataset["codebase_version"],
        "total_episodes": dataset["expected"]["total_episodes"],
        "total_frames": dataset["expected"]["total_frames"],
        "fps": dataset["expected"]["fps"],
        "splits": dataset["splits"],
        "features": dataset["required_features"],
    }
    validate_dataset_info(spec, info)
    info["total_frames"] += 1
    with pytest.raises(ValueError, match="total_frames"):
        validate_dataset_info(spec, info)


def test_v010_dual_camera_contract_has_no_logical_test_demonstrations(tmp_path):
    spec = load_training_spec(V010_PILOT_CONFIG)
    dataset = spec["dataset"]
    assert dataset["splits"] == {"train": "0:180", "validation": "180:200"}
    assert dataset["metadata_splits"]["test"] == "200:200"
    assert parse_episode_slice(dataset["metadata_splits"]["test"], allow_empty=True) == []
    assert dataset["expected"]["selected_frames"] == {
        "train": 135147,
        "validation": 14801,
    }
    assert "observation.images.wrist" in dataset["required_features"]

    info = {
        "codebase_version": dataset["codebase_version"],
        "total_episodes": dataset["expected"]["total_episodes"],
        "total_frames": dataset["expected"]["total_frames"],
        "fps": dataset["expected"]["fps"],
        "splits": dataset["metadata_splits"],
        "features": dataset["required_features"],
    }
    validate_dataset_info(spec, info)
    arguments = training_arguments(spec, tmp_path / "view", tmp_path / "out", "pilot")
    assert "--policy.vision_backbone=resnet18" in arguments
    assert "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1" in arguments
    assert "--policy.chunk_size=100" in arguments
    assert "--policy.n_action_steps=100" in arguments
    assert next(arg for arg in arguments if arg.startswith("--dataset.episodes=")).endswith(
        ",179]"
    )


def test_v010_formal_contract_selects_validation_without_dataset_test_split():
    spec = load_training_spec(V010_BASELINE_20K_CONFIG)
    assert validation_profile(spec) == "training"
    assert set(spec["dataset"]["splits"]) == {"train", "validation"}
    assert spec["training"]["steps"] == 20000
    runner = (ROOT / "scripts" / "run_so101_act_experiment.sh").read_text()
    assert "CONFIG_NAME must be a basename" in runner
    assert "--profile \"${PROFILE}\"" in runner
    assert "evaluate_act_checkpoints.py" in runner
    evaluator = (ROOT / "scripts" / "evaluate_act_checkpoints.py").read_text()
    assert 'rename_map=spec.get("rename_map")' in evaluator
    assert "--allow-evaluator-commit-mismatch" in evaluator
    assert '"evaluation_git_commit": git_commit' in evaluator
    assert '"${SOURCE_ROOT}" != "/workspace/source-dataset"' in runner
    container_runner = (ROOT / "scripts" / "run_so101_training.sh").read_text()
    assert '"${IMMUTABLE_SOURCE_ROOT}:/workspace/source-dataset:ro"' in container_runner
    assert '"${RESUME_CHECKPOINT_ROOT}:/workspace/resume-checkpoint:ro"' in container_runner


def test_v011_recovery20_contract_is_a_dataset_only_baseline_delta():
    baseline = load_training_spec(V010_BASELINE_20K_CONFIG)
    recovery = load_training_spec(V011_RECOVERY20_20K_CONFIG)

    for field in ("environment", "policy", "training", "validation", "smoke"):
        assert recovery[field] == baseline[field]

    dataset = recovery["dataset"]
    assert dataset["revision"] == "v0.1.1"
    assert dataset["resolved_commit"] == "ff1a812584b677b02998a722ac2a446ce1003e55"
    assert dataset["splits"] == {"train": "0:200", "validation": "200:220"}
    assert dataset["metadata_splits"]["test"] == "220:220"
    assert dataset["expected"] == {
        "total_episodes": 220,
        "total_frames": 166605,
        "fps": 30,
        "selected_frames": {"train": 151804, "validation": 14801},
    }
    assert dataset["required_features"] == baseline["dataset"]["required_features"]
    assert parse_episode_slice(dataset["splits"]["train"]) == list(range(200))
    assert parse_episode_slice(dataset["splits"]["validation"]) == list(range(200, 220))


def test_v012_grasp20_contract_is_a_dataset_only_v011_delta():
    recovery20 = load_training_spec(V011_RECOVERY20_20K_CONFIG)
    grasp20 = load_training_spec(V012_GRASP20_20K_CONFIG)

    for field in ("environment", "policy", "training", "validation", "smoke"):
        assert grasp20[field] == recovery20[field]

    dataset = grasp20["dataset"]
    assert dataset["revision"] == "v0.1.2"
    assert dataset["resolved_commit"] == "5458c9b17e8fe85774f73aa03515cd6fc2127fda"
    assert dataset["splits"] == {"train": "0:220", "validation": "220:240"}
    assert dataset["metadata_splits"]["test"] == "240:240"
    assert dataset["expected"] == {
        "total_episodes": 240,
        "total_frames": 183914,
        "fps": 30,
        "selected_frames": {"train": 169113, "validation": 14801},
    }
    assert dataset["required_features"] == recovery20["dataset"]["required_features"]
    assert parse_episode_slice(dataset["splits"]["train"]) == list(range(220))
    assert parse_episode_slice(dataset["splits"]["validation"]) == list(range(220, 240))


def test_v012_nominal_only_ablation_keeps_runtime_policy_identical(tmp_path):
    full = load_training_spec(V012_GRASP20_20K_CONFIG)
    nominal = load_training_spec(V012_NOMINAL_ONLY_20K_CONFIG)

    for field in ("dataset", "environment", "policy", "training", "validation", "smoke"):
        assert nominal[field] == full[field]
    assert nominal["sampling"] == {
        "kind": "uniform_frames",
        "episode_slices": ["0:180"],
        "expected_episode_count": 180,
        "expected_frame_count": 136814,
    }
    arguments = training_arguments(nominal, tmp_path / "view", tmp_path / "out", "training")
    assert arguments[0] == "lerobot-train"
    episode_arg = next(arg for arg in arguments if arg.startswith("--dataset.episodes="))
    assert episode_arg.endswith(",179]")


def test_v012_balanced_mix_ablation_uses_grouped_entrypoint(tmp_path):
    full = load_training_spec(V012_GRASP20_20K_CONFIG)
    balanced = load_training_spec(V012_BALANCED_MIX_20K_CONFIG)

    for field in ("dataset", "environment", "policy", "training", "validation", "smoke"):
        assert balanced[field] == full[field]
    assert balanced["sampling"]["episode_slices"] == ["0:220"]
    arguments = training_arguments(
        balanced, tmp_path / "balanced-view", tmp_path / "out", "training"
    )
    assert arguments[:2] == [
        "python",
        "/workspace/project/scripts/train_so101_act_grouped.py",
    ]
    assert arguments[2].endswith("/meta/farpoint_sampler.json")
    smoke_arguments = training_arguments(
        balanced, tmp_path / "balanced-view", tmp_path / "out", "smoke"
    )
    assert smoke_arguments[0] == "lerobot-train"


def test_v013_balanced_mix_binds_local_candidate_and_exact_group_draws():
    previous = load_training_spec(V012_BALANCED_MIX_20K_CONFIG)
    candidate = load_training_spec(V013_BALANCED_MIX_20K_CONFIG)

    for field in ("environment", "policy", "training", "validation", "smoke"):
        assert candidate[field] == previous[field]
    dataset = candidate["dataset"]
    assert "resolved_commit" not in dataset
    assert dataset["source"] == {
        "kind": "local_snapshot",
        "tree_sha256": "631031a0fe36969d2be6f23e243ac9fd55bfe79b465472147bcb65461c240ce2",
    }
    assert dataset["expected"] == {
        "total_episodes": 260,
        "total_frames": 198571,
        "fps": 30,
        "selected_frames": {"train": 183770, "validation": 14801},
    }
    sampling = candidate["sampling"]
    group_counts = {
        group["group_id"]: len(parse_episode_slices(group["episode_slices"]))
        for group in sampling["groups"]
    }
    assert group_counts == {
        "nominal_blue": 90,
        "nominal_red": 90,
        "approach_blue": 20,
        "approach_red": 20,
        "grasp_blue": 10,
        "grasp_red": 10,
    }
    plan = {
        "kind": sampling["kind"],
        "steps": candidate["training"]["steps"],
        "groups": sampling["groups"],
        "batch_cycle": sampling["batch_cycle"],
    }
    assert expected_group_sample_counts(plan) == {
        "nominal_blue": 64000,
        "nominal_red": 64000,
        "approach_blue": 8000,
        "approach_red": 8000,
        "grasp_blue": 8000,
        "grasp_red": 8000,
    }


def test_v013_balanced_mix_200k_is_fresh_and_reuses_exact_grouping(tmp_path):
    baseline = load_training_spec(V013_BALANCED_MIX_20K_CONFIG)
    curve = load_training_spec(V013_BALANCED_MIX_200K_CONFIG)

    for field in ("dataset", "environment", "policy", "sampling", "smoke"):
        assert curve[field] == baseline[field] or field == "smoke"
    assert curve["training"] == {
        **baseline["training"],
        "steps": 200000,
        "save_freq": 20000,
        "resume": False,
    }
    assert curve["smoke"] == {**baseline["smoke"], "resume": False}
    assert curve["validation"] == {
        **baseline["validation"],
        "minimum_relative_improvement": 0,
        "require_later_improvement": False,
    }
    arguments = training_arguments(curve, tmp_path / "view", tmp_path / "out", "training")
    assert "--steps=200000" in arguments
    assert "--save_freq=20000" in arguments
    assert "--resume=false" in arguments
    plan = {
        "kind": curve["sampling"]["kind"],
        "steps": curve["training"]["steps"],
        "groups": curve["sampling"]["groups"],
        "batch_cycle": curve["sampling"]["batch_cycle"],
    }
    assert expected_group_sample_counts(plan) == {
        "nominal_blue": 640000,
        "nominal_red": 640000,
        "approach_blue": 80000,
        "approach_red": 80000,
        "grasp_blue": 80000,
        "grasp_red": 80000,
    }


def test_v014_balanced_mix_200k_adds_transport_without_changing_recovery_share(tmp_path):
    baseline = load_training_spec(V013_BALANCED_MIX_200K_CONFIG)
    candidate = load_training_spec(V014_BALANCED_MIX_200K_CONFIG)

    for field in ("environment", "policy", "training", "validation", "smoke"):
        assert candidate[field] == baseline[field]
    assert candidate["dataset"]["revision"] == "v0.1.4"
    assert candidate["dataset"]["source"] == {
        "kind": "local_snapshot",
        "tree_sha256": "466cf33b2dab02122751466226c3a8d7518bebba93fa6e18bde9d9709499373b",
    }
    assert candidate["dataset"]["expected"] == {
        "total_episodes": 280,
        "total_frames": 212606,
        "fps": 30,
        "selected_frames": {"train": 197805, "validation": 14801},
    }
    sampling = candidate["sampling"]
    groups = {
        group["group_id"]: parse_episode_slices(group["episode_slices"])
        for group in sampling["groups"]
    }
    assert {group: len(episodes) for group, episodes in groups.items()} == {
        "nominal_blue": 90,
        "nominal_red": 90,
        "approach_blue": 20,
        "approach_red": 20,
        "grasp_blue": 10,
        "grasp_red": 10,
        "transport_blue": 8,
        "transport_red": 12,
    }
    assert sorted(episode for episodes in groups.values() for episode in episodes) == list(
        range(260)
    )
    plan = {
        "kind": sampling["kind"],
        "steps": candidate["training"]["steps"],
        "groups": sampling["groups"],
        "batch_cycle": sampling["batch_cycle"],
    }
    assert expected_group_sample_counts(plan) == {
        "nominal_blue": 639999,
        "nominal_red": 639999,
        "approach_blue": 53334,
        "approach_red": 53334,
        "grasp_blue": 53334,
        "grasp_red": 53334,
        "transport_blue": 53333,
        "transport_red": 53333,
    }
    arguments = training_arguments(
        candidate, tmp_path / "view", tmp_path / "out", "training"
    )
    assert "--steps=200000" in arguments
    assert "--save_freq=20000" in arguments
    assert "--resume=false" in arguments


def test_v020_cell_balanced_200k_binds_candidate_and_all_thirty_cells(tmp_path):
    baseline = load_training_spec(V014_BALANCED_MIX_200K_CONFIG)
    candidate = load_training_spec(V020_CELL_BALANCED_200K_CONFIG)

    for field in ("environment", "policy", "training", "validation", "smoke"):
        assert candidate[field] == baseline[field]
    assert candidate["experiment_id"] == "so101_act_v0_2_0_cell_balanced_200k"
    assert candidate["dataset"] == {
        "repo_id": "wenyixu101/so101-sim-oracle-pick-and-place",
        "revision": "v0.2.0",
        "source": {
            "kind": "local_snapshot",
            "tree_sha256": (
                "893bf831cc4b44d8a5606c7e7a5118bc2dca1e0e5889e4d9931093bf99540e96"
            ),
        },
        "codebase_version": "v3.0",
        "splits": {"train": "0:270", "validation": "270:300"},
        "metadata_splits": {
            "train": "0:270",
            "validation": "270:300",
            "test": "300:300",
        },
        "expected": {
            "total_episodes": 300,
            "total_frames": 255043,
            "fps": 30,
            "selected_frames": {"train": 230101, "validation": 24942},
        },
        "video_backend": "pyav",
        "required_features": baseline["dataset"]["required_features"],
    }

    sampling = candidate["sampling"]
    groups = {
        group["group_id"]: parse_episode_slices(group["episode_slices"])
        for group in sampling["groups"]
    }
    assert len(groups) == 30
    assert {len(episodes) for episodes in groups.values()} == {9}
    assert sorted(episode for episodes in groups.values() for episode in episodes) == list(
        range(270)
    )
    for template in sampling["batch_cycle"]:
        assert sum(template.values()) == 8
        assert sum(group.startswith("blue__") for group in template) == 4
        assert sum(group.startswith("red__") for group in template) == 4

    plan = {
        "kind": sampling["kind"],
        "steps": candidate["training"]["steps"],
        "groups": sampling["groups"],
        "batch_cycle": sampling["batch_cycle"],
    }
    draws = expected_group_sample_counts(plan)
    assert sum(draws.values()) == 1_600_000
    assert min(draws.values()) == 53333
    assert max(draws.values()) == 53334
    assert sum(value == 53334 for value in draws.values()) == 10

    by_axis = {"object": {}, "target": {}, "camera": {}}
    for group_id, count in draws.items():
        object_id, target_id, camera_id = group_id.split("__")
        for axis, value in (
            ("object", object_id),
            ("target", target_id),
            ("camera", camera_id),
        ):
            by_axis[axis][value] = by_axis[axis].get(value, 0) + count
    assert by_axis["object"] == {"blue": 800000, "red": 800000}
    assert by_axis["target"] == {
        "target-a": 533335,
        "target-b": 533333,
        "target-c": 533332,
    }
    assert by_axis["camera"] == {
        "front-nominal": 320000,
        "front-x-negative": 320000,
        "front-x-positive": 320000,
        "front-yz-negative": 320000,
        "front-yz-positive": 320000,
    }

    arguments = training_arguments(
        candidate, tmp_path / "view", tmp_path / "out", "training"
    )
    assert arguments[:2] == [
        "python",
        "/workspace/project/scripts/train_so101_act_grouped.py",
    ]
    assert "--steps=200000" in arguments
    assert "--save_freq=20000" in arguments
    assert "--resume=false" in arguments


def test_v020_smolvla_binds_base_revision_and_camera_mapping(tmp_path):
    spec = load_training_spec(V020_SMOLVLA_UNIFORM_20K_CONFIG)
    assert spec["policy"] == {
        "type": "smolvla",
        "device": "cuda",
        "pretrained_path": "lerobot/smolvla_base",
        "pretrained_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
    }
    assert spec["rename_map"] == {
        "observation.images.front": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }
    assert spec["smoke"]["batch_size"] == spec["training"]["batch_size"] == 8
    arguments = training_arguments(spec, tmp_path / "view", tmp_path / "out", "smoke")
    assert "--policy.path=lerobot/smolvla_base" in arguments
    assert (
        "--rename_map={\"observation.images.front\":\"observation.images.camera1\","
        "\"observation.images.wrist\":\"observation.images.camera2\"}" in arguments
    )


def test_v014_300k_continuation_resumes_exact_200k_sampler_suffix(tmp_path):
    baseline = load_training_spec(V014_BALANCED_MIX_200K_CONFIG)
    continuation = load_training_spec(V014_BALANCED_MIX_300K_CONTINUATION_CONFIG)

    for field in ("dataset", "environment", "policy", "sampling", "validation", "smoke"):
        assert continuation[field] == baseline[field]
    assert continuation["continuation"] == {
        "source_experiment_id": "so101_act_v0_1_4_balanced_mix_200k",
        "source_step": 200000,
        "source_model_sha256": (
            "fc444f76dd61bd4cf9b982c4e93ff406800e713be59542499e6e18fe85474a83"
        ),
    }
    assert continuation["training"] == {
        **baseline["training"],
        "steps": 300000,
        "resume": True,
    }
    checkpoint = tmp_path / "checkpoints" / "200000"
    arguments = training_arguments(
        continuation,
        tmp_path / "view",
        tmp_path / "out",
        "training",
        resume_checkpoint=checkpoint,
    )
    assert "--steps=300000" in arguments
    assert "--resume=true" in arguments
    assert (
        f"--config_path={checkpoint / 'pretrained_model' / 'train_config.json'}" in arguments
    )
    with pytest.raises(ValueError, match="resume checkpoint is required"):
        training_arguments(
            continuation, tmp_path / "view", tmp_path / "out", "training"
        )
    plan = {
        "kind": continuation["sampling"]["kind"],
        "start_step": continuation["continuation"]["source_step"],
        "steps": continuation["training"]["steps"],
        "groups": continuation["sampling"]["groups"],
        "batch_cycle": continuation["sampling"]["batch_cycle"],
    }
    assert expected_group_sample_counts(plan) == {
        "nominal_blue": 320001,
        "nominal_red": 320001,
        "approach_blue": 26666,
        "approach_red": 26666,
        "grasp_blue": 26666,
        "grasp_red": 26666,
        "transport_blue": 26667,
        "transport_red": 26667,
    }


def test_continuation_checkpoint_binding_requires_optimizer_rng_step_and_model(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    pretrained = checkpoint / "pretrained_model"
    state = checkpoint / "training_state"
    pretrained.mkdir(parents=True)
    state.mkdir()
    files = {
        pretrained / "model.safetensors": b"model",
        state / "optimizer_state.safetensors": b"optimizer",
        state / "optimizer_param_groups.json": b"{}\n",
        state / "rng_state.safetensors": b"rng",
    }
    for path, content in files.items():
        path.write_bytes(content)
    (state / "training_step.json").write_text('{"step": 200000}\n', encoding="utf-8")
    (pretrained / "train_config.json").write_text(
        json.dumps(
            {
                "job_name": "source_experiment_training",
                "steps": 200000,
                "seed": 1010,
                "batch_size": 8,
            }
        ),
        encoding="utf-8",
    )
    spec = {
        "continuation": {
            "source_experiment_id": "source_experiment",
            "source_step": 200000,
            "source_model_sha256": hashlib.sha256(b"model").hexdigest(),
        },
        "training": {"seed": 1010, "batch_size": 8},
    }
    binding = PREFLIGHT_MODULE.validate_resume_checkpoint(spec, checkpoint)
    assert binding["source_step"] == 200000
    assert binding["model_sha256"] == hashlib.sha256(b"model").hexdigest()
    assert binding["optimizer_state_sha256"] == hashlib.sha256(b"optimizer").hexdigest()

    spec["continuation"]["source_model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="model hash mismatch"):
        PREFLIGHT_MODULE.validate_resume_checkpoint(spec, checkpoint)


def test_training_view_replaces_only_stats_and_preserves_source(tmp_path):
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "videos").mkdir()
    original_stats = {"action": {"mean": [100.0]}}
    (source / "meta" / "stats.json").write_text(json.dumps(original_stats), encoding="utf-8")
    (source / "meta" / "info.json").write_text("{}", encoding="utf-8")

    def write_stats(stats, root):
        (root / "meta" / "stats.json").write_text(json.dumps(stats), encoding="utf-8")

    source_hash, view_hash = create_training_view(
        source, tmp_path / "view", {"action": {"mean": [1.0]}}, write_stats
    )
    assert source_hash != view_hash
    assert json.loads((source / "meta" / "stats.json").read_text()) == original_stats
    assert (tmp_path / "view" / "data").is_symlink()
    assert (tmp_path / "view" / "videos").is_symlink()
    with pytest.raises(FileExistsError):
        create_training_view(source, tmp_path / "view", {}, write_stats)


def test_episode_stats_unflatten_and_smoke_arguments_are_split_safe(tmp_path):
    stats = unflatten_episode_stats(
        {"episode_index": 0, "stats/action/mean": [1.0], "stats/action/count": [2]}
    )
    assert stats["action"]["mean"].tolist() == [1.0]
    assert stats["action"]["count"].tolist() == [2]
    spec = load_training_spec(CONFIG)
    arguments = training_arguments(spec, tmp_path / "view", tmp_path / "out", "smoke")
    episode_arg = next(arg for arg in arguments if arg.startswith("--dataset.episodes="))
    assert "--steps=1" in arguments
    assert "--save_checkpoint=false" in arguments
    assert "--policy.push_to_hub=false" in arguments
    assert episode_arg.startswith("--dataset.episodes=[0,1,2")
    assert episode_arg.endswith(",127]")
    assert ",128" not in episode_arg


def test_pilot_contract_and_arguments_are_frozen_and_split_safe(tmp_path):
    spec = load_training_spec(PILOT_CONFIG)
    assert spec["pilot"]["steps"] == 1000
    assert spec["pilot"]["save_freq"] == 250
    assert spec["validation"]["split"] == "validation"
    arguments = training_arguments(spec, tmp_path / "view", tmp_path / "out", "pilot")
    assert "--steps=1000" in arguments
    assert "--save_freq=250" in arguments
    assert "--policy.push_to_hub=false" in arguments
    episode_arg = next(arg for arg in arguments if arg.startswith("--dataset.episodes="))
    assert episode_arg.endswith(",127]")
    assert ",128" not in episode_arg


def test_20k_baseline_contract_selects_training_checkpoints(tmp_path):
    spec = load_training_spec(BASELINE_20K_CONFIG)
    assert validation_profile(spec) == "training"
    assert spec["training"]["steps"] == 20000
    assert spec["training"]["save_freq"] == 5000
    arguments = training_arguments(spec, tmp_path / "view", tmp_path / "out", "training")
    assert "--steps=20000" in arguments
    assert "--save_freq=5000" in arguments
    assert "--policy.push_to_hub=false" in arguments
    assert spec["dataset"]["splits"]["test"] == "142:160"
    runner = (ROOT / "scripts" / "run_so101_act_baseline_20k.sh").read_text()
    assert "--profile training" in runner
    assert "evaluate_act_checkpoints.py" in runner


def test_evenly_spaced_validation_indices_cover_boundaries_without_duplicates():
    indices = evenly_spaced_indices(10164, 128)
    assert len(indices) == len(set(indices)) == 128
    assert indices[0] == 0
    assert indices[-1] == 10163
    assert indices == sorted(indices)
    with pytest.raises(ValueError):
        evenly_spaced_indices(3, 4)


def test_validation_selection_requires_a_later_meaningful_improvement():
    results = [
        {"step": 250, "mean_loss": 10.0},
        {"step": 500, "mean_loss": 8.0},
        {"step": 750, "mean_loss": 8.5},
    ]
    best, improvement = select_validation_checkpoint(results, 0.05)
    assert best["step"] == 500
    assert improvement == pytest.approx(0.2)
    with pytest.raises(ValueError, match="no later checkpoint"):
        select_validation_checkpoint(
            [{"step": 250, "mean_loss": 8.0}, {"step": 500, "mean_loss": 9.0}], 0
        )
    with pytest.raises(ValueError, match="below"):
        select_validation_checkpoint(
            [{"step": 250, "mean_loss": 10.0}, {"step": 500, "mean_loss": 9.8}], 0.05
        )


def test_validation_curve_can_report_overfitting_without_failing_execution():
    first = {"step": 20000, "mean_loss": 8.0}
    best, improvement = select_validation_checkpoint(
        [first, {"step": 40000, "mean_loss": 8.5}],
        0,
        require_later_improvement=False,
    )
    assert best == first
    assert improvement == 0


def test_training_image_preserves_ngc_cuda_torch_builds():
    dockerfile = (ROOT / "docker" / "so101-lerobot-training" / "Dockerfile").read_text()

    assert "FROM nvcr.io/nvidia/pytorch:26.01-py3@sha256:" in dockerfile
    assert "--constraint /opt/farpoint-training-constraints.txt" in dockerfile
    assert "torch.version.cuda is not None" in dockerfile
    assert "m.version('torch')" in dockerfile
    assert "m.version('torchvision')" in dockerfile
    assert "Version(m.version('torch')) < Version('2.11.0')" in dockerfile


def test_v020_smolvla_contract_pins_base_model_and_uses_official_path(tmp_path):
    path = ROOT / "configs/training/so101_smolvla_v0_2_0_uniform_20k.json"
    spec = load_training_spec(path)

    assert validate_contract(spec) == []
    assert spec["policy"] == {
        "type": "smolvla",
        "device": "cuda",
        "pretrained_path": "lerobot/smolvla_base",
        "pretrained_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
    }
    assert spec["dataset"]["splits"] == {"train": "0:270", "validation": "270:300"}
    assert spec["sampling"] == {
        "kind": "uniform_frames",
        "episode_slices": ["0:270"],
        "expected_episode_count": 270,
        "expected_frame_count": 230101,
    }
    arguments = training_arguments(spec, tmp_path / "view", tmp_path / "out", "training")
    assert "--policy.path=lerobot/smolvla_base" in arguments
    assert not any(argument.startswith("--policy.type=") for argument in arguments)
    assert "--steps=20000" in arguments
    assert "--save_freq=5000" in arguments
    dockerfile = (ROOT / "docker" / "so101-lerobot-training" / "Dockerfile").read_text()
    assert "lerobot${LEROBOT_EXTRAS}==${LEROBOT_VERSION}" in dockerfile
