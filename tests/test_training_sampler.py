import copy

import pytest

from farpoint.training_sampler import (
    DeterministicGroupedBatchSampler,
    build_sampler_plan,
    expected_group_sample_counts,
    parse_episode_slices,
    selected_training_episodes,
    validate_sampling_contract,
)


def grouped_spec():
    return {
        "dataset": {
            "splits": {"train": "0:8", "validation": "8:10"},
            "expected": {"selected_frames": {"train": 40, "validation": 10}},
        },
        "training": {"seed": 17, "batch_size": 4, "steps": 5},
        "sampling": {
            "kind": "deterministic_grouped_batches",
            "episode_slices": ["0:8"],
            "expected_episode_count": 8,
            "expected_frame_count": 40,
            "groups": [
                {"group_id": "nominal_blue", "episode_slices": ["0:2"]},
                {"group_id": "nominal_red", "episode_slices": ["2:4"]},
                {"group_id": "recovery_blue", "episode_slices": ["4:6"]},
                {"group_id": "recovery_red", "episode_slices": ["6:8"]},
            ],
            "batch_cycle": [
                {"nominal_blue": 2, "nominal_red": 2},
                {"nominal_blue": 1, "nominal_red": 1, "recovery_blue": 1, "recovery_red": 1},
            ],
        },
    }


def test_episode_slice_selection_rejects_overlap_and_out_of_train():
    assert parse_episode_slices(["0:2", "3:5"]) == [0, 1, 3, 4]
    with pytest.raises(ValueError, match="overlaps"):
        parse_episode_slices(["0:3", "2:4"])
    broken = grouped_spec()
    broken["sampling"]["episode_slices"] = ["0:9"]
    broken["sampling"]["expected_episode_count"] = 9
    with pytest.raises(ValueError, match="subset"):
        validate_sampling_contract(broken)


def test_grouped_contract_requires_an_exact_partition_and_batch_size():
    validate_sampling_contract(grouped_spec())
    broken = grouped_spec()
    broken["sampling"]["groups"][0]["episode_slices"] = ["0:3"]
    with pytest.raises(ValueError, match="partition"):
        validate_sampling_contract(broken)
    broken = grouped_spec()
    broken["sampling"]["batch_cycle"][0] = {"nominal_blue": 1}
    with pytest.raises(ValueError, match="batch size"):
        validate_sampling_contract(broken)


def test_grouped_sampler_is_deterministic_and_honors_each_template():
    spec = grouped_spec()
    lengths = {episode: 5 for episode in range(10)}
    plan = build_sampler_plan(spec, lengths)
    first = list(DeterministicGroupedBatchSampler(plan, list(range(8)), lengths))
    second = list(DeterministicGroupedBatchSampler(plan, list(range(8)), lengths))
    assert first == second
    assert len(first) == 5
    assert all(len(batch) == 4 for batch in first)

    def group_for_frame(frame):
        episode = frame // 5
        return (
            "nominal_blue" if episode < 2 else
            "nominal_red" if episode < 4 else
            "recovery_blue" if episode < 6 else
            "recovery_red"
        )

    assert [group_for_frame(frame) for frame in first[0]].count("nominal_blue") == 2
    assert [group_for_frame(frame) for frame in first[1]].count("recovery_blue") == 1
    assert expected_group_sample_counts(plan) == {
        "nominal_blue": 8,
        "nominal_red": 8,
        "recovery_blue": 2,
        "recovery_red": 2,
    }


def test_sampler_plan_binds_selected_frame_count_and_dataset_order():
    spec = grouped_spec()
    lengths = {episode: 5 for episode in range(10)}
    plan = build_sampler_plan(spec, lengths)
    assert selected_training_episodes(spec) == list(range(8))
    assert plan["selected_frame_count"] == 40
    with pytest.raises(ValueError, match="frame count"):
        build_sampler_plan(spec, {episode: 4 for episode in range(10)})
    with pytest.raises(ValueError, match="episode order"):
        DeterministicGroupedBatchSampler(plan, list(reversed(range(8))), lengths)

    uniform = copy.deepcopy(spec)
    uniform["sampling"] = {
        "kind": "uniform_frames",
        "episode_slices": ["0:4"],
        "expected_episode_count": 4,
        "expected_frame_count": 20,
    }
    validate_sampling_contract(uniform)
    assert build_sampler_plan(uniform, lengths)["kind"] == "uniform_frames"
