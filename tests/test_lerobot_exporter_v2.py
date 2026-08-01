import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_lerobot_dataset import (
    build_dataset_sidecar,
    export_dataset,
    lerobot_split_ranges,
    order_selection,
)

from v2_fixtures import episode_metadata_v2


def test_selection_is_ordered_by_split_then_trial():
    entries = [
        {"split": "test", "trial_id": "t3"},
        {"split": "train", "trial_id": "t2"},
        {"split": "validation", "trial_id": "t1"},
        {"split": "train", "trial_id": "t0"},
    ]
    assert [item["trial_id"] for item in order_selection(entries)] == ["t0", "t2", "t1", "t3"]


def test_split_ranges_include_empty_splits():
    assert lerobot_split_ranges({"train": 2, "validation": 1, "test": 1}) == {
        "train": "0:2",
        "validation": "2:3",
        "test": "3:4",
    }


def test_sidecar_derives_multiple_tasks_and_splits_from_episodes():
    train = episode_metadata_v2()
    validation = episode_metadata_v2(
        episode_id="episode-0001",
        trial_id="trial-0001",
        split="validation",
        dataset_episode_index=1,
        shape="cylinder",
    )
    sidecar = build_dataset_sidecar(
        "farpoint-test",
        [train, validation],
        fps=20,
        image_width=640,
        image_height=360,
        selected_names=["joint_0"],
    )
    assert sidecar["splits"] == {"train": 1, "validation": 1, "test": 0}
    assert [task["object_shape"] for task in sidecar["tasks"]] == ["cube", "cylinder"]


def test_sidecar_rejects_ambiguous_task_ids_for_one_instruction():
    first = episode_metadata_v2()
    second = deepcopy(first)
    second["identity"]["episode_id"] = "episode-0001"
    second["identity"]["trial_id"] = "trial-0001"
    second["identity"]["dataset_episode_index"] = 1
    second["identity"]["task_id"] = "another-task-id"
    second["task"]["task_id"] = "another-task-id"
    with pytest.raises(ValueError, match="cannot share one LeRobot instruction"):
        build_dataset_sidecar(
            "farpoint-test",
            [first, second],
            fps=20,
            image_width=640,
            image_height=360,
            selected_names=["joint_0"],
        )


class FakeLeRobotDataset:
    last_instance = None

    def __init__(self, root):
        self.root = Path(root)
        self.frames = []
        self.saved_tasks = []
        self.current_tasks = []

    @classmethod
    def create(cls, *, root, **_kwargs):
        cls.last_instance = cls(root)
        return cls.last_instance

    def add_frame(self, frame):
        self.frames.append(frame)
        self.current_tasks.append(frame["task"])

    def save_episode(self, **_kwargs):
        self.saved_tasks.append(self.current_tasks[-1])
        self.current_tasks = []

    def finalize(self):
        meta = self.root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "info.json").write_text(json.dumps({"features": {}}), encoding="utf-8")


def _write_source_episode(root, record):
    pytest = __import__("pytest")
    image_module = pytest.importorskip("PIL.Image")
    root.mkdir(parents=True)
    raw = deepcopy(record)
    raw["episode_id"] = raw["identity"]["episode_id"]
    raw["trial_id"] = raw["identity"]["trial_id"]
    raw.pop("identity")
    raw.pop("schema_version")
    (root / "metadata.json").write_text(json.dumps(raw), encoding="utf-8")
    (root / "metrics.json").write_text(
        json.dumps({"success": True, "dataset_valid": True}), encoding="utf-8"
    )
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
        "finger_joint",
    ]
    rows = []
    for frame in range(2):
        image_module.new("RGB", (8, 6), color=(frame * 10, 0, 0)).save(root / f"frame-{frame}.png")
        rows.append({
            "frame": frame,
            "timestamp_seconds": frame * 0.05,
            "rgb_path": f"frame-{frame}.png",
            "joint_names": joint_names,
            "joint_positions": [0.0] * 7,
            "action_joint_positions": [0.1] * 7,
        })
    (root / "observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_export_fixture_writes_dynamic_tasks_and_three_splits(tmp_path):
    episodes = [
        episode_metadata_v2(),
        episode_metadata_v2(
            episode_id="episode-0001", trial_id="trial-0001", split="validation",
            dataset_episode_index=1, shape="cylinder",
        ),
        episode_metadata_v2(
            episode_id="episode-0002", trial_id="trial-0002", split="test",
            dataset_episode_index=2,
        ),
    ]
    selection = []
    for episode in episodes:
        source = tmp_path / episode["identity"]["episode_id"]
        _write_source_episode(source, episode)
        selection.append({
            "episode_dir": str(source),
            "trial_id": episode["identity"]["trial_id"],
            "split": episode["identity"]["split"],
        })
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps({
        "schema_version": "farpoint.export-selection.v1",
        "dataset_id": "farpoint-fixture",
        "episodes": selection,
    }), encoding="utf-8")

    output = export_dataset(
        manifest, tmp_path / "export", dataset_class=FakeLeRobotDataset
    )
    sidecar = json.loads((output / "meta/farpoint_v2.json").read_text())
    info = json.loads((output / "meta/info.json").read_text())
    records = [
        json.loads(line)
        for line in (output / "meta/episode_metadata.jsonl").read_text().splitlines()
    ]
    assert sidecar["splits"] == {"train": 1, "validation": 1, "test": 1}
    assert info["splits"] == {"train": "0:1", "validation": "1:2", "test": "2:3"}
    assert [record["identity"]["split"] for record in records] == [
        "train", "validation", "test"
    ]
    assert FakeLeRobotDataset.last_instance.saved_tasks == [
        "Pick up the cube and place it in the target zone.",
        "Pick up the cylinder and place it in the target zone.",
        "Pick up the cube and place it in the target zone.",
    ]
