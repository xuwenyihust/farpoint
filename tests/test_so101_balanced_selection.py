from farpoint.so101_balanced_selection import build_artifacts, validate_balance


def balanced_stats():
    cells = {f"r{row:02d}_c{column:02d}": 2 for row in range(5) for column in range(5)}
    return {
        "total": 50,
        "splits": {"train": 40, "validation": 5, "test": 5},
        "workspace_cells": cells,
        "workspace_rows": {f"r{row:02d}": 10 for row in range(5)},
        "workspace_columns": {f"c{column:02d}": 10 for column in range(5)},
        "sizes": {"size_0": 25, "size_1": 25},
        "colors": {"color_0": 25, "color_1": 25},
        "size_color": {"size_0__color_0": 12, "size_0__color_1": 13, "size_1__color_0": 13, "size_1__color_1": 12},
    }


def test_balance_contract_accepts_full_stratification():
    assert validate_balance(balanced_stats()) == []


def test_balance_contract_rejects_missing_cell_and_skew():
    stats = balanced_stats()
    stats["workspace_cells"].pop("r00_c00")
    stats["sizes"] = {"size_0": 26, "size_1": 24}
    errors = validate_balance(stats)
    assert any("sizes" in error for error in errors)
    assert any("25 workspace cells" in error for error in errors)


def test_build_artifacts_preserves_source_lineage_and_absolute_episode_paths(tmp_path):
    source = {
        "collection_id": "formal-v0",
        "task_id": "so101_cube_pick_place",
    }
    plan = {"plan_id": "plan", "plan_sha256": "a" * 64}
    selected = [
        {
            "attempt_id": "trial__attempt00",
            "trial_id": "trial",
            "variation_id": "trial",
            "episode_id": "episode_trial",
            "split": "train",
            "success": True,
            "dataset_valid": True,
            "cell_id": "r00_c00",
            "size_label": "size_0",
            "color_label": "color_0",
        }
    ]
    manifest, selection = build_artifacts(
        source,
        plan,
        selected,
        balanced_stats(),
        collection_id="balanced50",
        dataset_id="dataset",
        episodes_root=tmp_path,
        git_commit="b" * 40,
    )

    assert manifest["source_collection"]["collection_id"] == "formal-v0"
    assert manifest["selection_policy"] == "so101_balanced_stratified_subset_v1"
    assert manifest["attempts"][0]["selected_for_dataset"] is True
    assert selection["collection_id"] == "balanced50"
    assert selection["episodes"][0]["episode_dir"] == str(tmp_path / "episode_trial")
