import copy

import pytest

from farpoint.balanced_selection import (
    candidate_rows,
    select_balanced,
    selection_stats,
    validate_balance,
    validate_selection_policy,
)


POLICY = {
    "schema_version": "farpoint.balanced-selection-policy.v1",
    "policy_id": "generic_shape_mass_v1",
    "target_count": 4,
    "seed": 9,
    "iterations": 1000,
    "split_targets": {"train": 2, "validation": 1, "test": 1},
    "labels": {
        "shapes": {"path": "resolved.shape"},
        "masses": {"path": "resolved.mass_kg", "number_format": ".2f"},
    },
    "joints": {"shape_mass": ["shapes", "masses"]},
    "constraints": [
        {
            "kind": "balanced",
            "key": "shapes",
            "categories": ["cube", "cylinder"],
            "max_difference": 0,
        },
        {
            "kind": "balanced",
            "key": "masses",
            "categories": ["0.03", "0.04"],
            "max_difference": 0,
        },
    ],
    "objectives": [
        {
            "key": "shapes",
            "targets": {"cube": 2, "cylinder": 2},
            "weight": 100,
        },
        {
            "key": "masses",
            "targets": {"0.03": 2, "0.04": 2},
            "weight": 100,
        },
    ],
}


def _fixture():
    trials = []
    attempts = []
    index = 0
    for split, count in (("train", 4), ("validation", 2), ("test", 2)):
        for local in range(count):
            shape = "cube" if local % 2 == 0 else "cylinder"
            mass = 0.03 if (local // 2) % 2 == 0 else 0.04
            variation_id = f"variation-{index}"
            trials.append(
                {
                    "trial_id": variation_id,
                    "variation_id": variation_id,
                    "split": split,
                    "resolved": {"shape": shape, "mass_kg": mass},
                }
            )
            attempts.append(
                {
                    "attempt_id": f"attempt-{index}",
                    "trial_id": variation_id,
                    "variation_id": variation_id,
                    "split": split,
                    "success": True,
                    "dataset_valid": True,
                }
            )
            index += 1
    return {"trials": trials}, {"attempts": attempts}


def test_generic_selector_uses_policy_paths_not_so101_labels():
    plan, manifest = _fixture()

    selected, stats = select_balanced(manifest, plan, POLICY)

    assert len(selected) == 4
    assert validate_balance(stats, POLICY) == []
    assert stats["shapes"] == {"cube": 2, "cylinder": 2}
    assert stats["masses"] == {"0.03": 2, "0.04": 2}


def test_candidate_filter_and_stats_do_not_mutate_source_attempts():
    plan, manifest = _fixture()
    original = copy.deepcopy(manifest)

    rows = candidate_rows(manifest, plan, POLICY)
    stats = selection_stats(rows[:4], POLICY)

    assert manifest == original
    assert set(stats) >= {"total", "splits", "shapes", "masses", "shape_mass"}


def test_policy_rejects_unknown_constraint_labels():
    policy = copy.deepcopy(POLICY)
    policy["constraints"] = [{"kind": "coverage", "key": "undeclared"}]

    with pytest.raises(ValueError, match="unknown key"):
        validate_selection_policy(policy)
