import json

import numpy as np
import pytest

from farpoint.action_replay import ExpertActionReplay
from farpoint.so101 import radians_to_lerobot


def _manifest(*, exact=True):
    scene = {
        "scene_id": "scene-1",
        "actions_calibrated": [[1.0] * 6, [2.0] * 6],
    }
    payload = {
        "schema_version": "farpoint.expert-action-replay.v1",
        "camera_features": ["observation.images.front"],
        "scenes": [scene],
    }
    if exact:
        payload["physics_replay"] = {
            "mode": "exact_trace",
            "unit": "radian",
            "maximum_targets_per_policy_step": 4,
        }
        scene["physics_action_groups_radians"] = [
            [[1.0] * 6, [1.25] * 6, [1.5] * 6, [1.75] * 6],
            [[2.0] * 6, [2.1] * 6],
        ]
    return payload


def test_exact_action_replay_serves_each_physics_target_group(tmp_path):
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(_manifest()))
    replay = ExpertActionReplay(path)
    replay.reset("scene-1")
    action, execution = replay.next_action()
    assert action.tolist() == [1.0] * 6
    assert len(execution["physics_actions_radians"]) == 4
    action, execution = replay.next_action()
    assert action.tolist() == [2.0] * 6
    assert execution["physics_actions_radians"] == [[2.0] * 6, [2.1] * 6]
    action, execution = replay.next_action()
    assert execution["source_exhausted"] is True
    assert action == pytest.approx(
        radians_to_lerobot(np.asarray([2.1] * 6), clip=True)
    )
    assert execution["physics_actions_radians"] == [[2.1] * 6] * 4


def test_legacy_action_replay_retains_policy_rate_endpoint_mode(tmp_path):
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(_manifest(exact=False)))
    replay = ExpertActionReplay(path)
    replay.reset("scene-1")
    _, execution = replay.next_action()
    assert "physics_actions_radians" not in execution


def test_exact_action_replay_rejects_misaligned_physics_group(tmp_path):
    payload = _manifest()
    payload["scenes"][0]["physics_action_groups_radians"][0] = []
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid physics action group"):
        ExpertActionReplay(path)


def test_exact_action_replay_rejects_non_radian_physics_unit(tmp_path):
    payload = _manifest()
    payload["physics_replay"]["unit"] = "calibrated"
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unit must be radian"):
        ExpertActionReplay(path)
