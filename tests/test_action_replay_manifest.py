import importlib.util
import json
from pathlib import Path

from farpoint.policy_training import file_sha256


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_so101_action_replay_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_action_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_replay_manifest_binds_source_and_converts_actions(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    metadata = episode / "metadata.json"
    metadata.write_text('{"identity":{"episode_id":"episode-1"}}')
    observations = episode / "observations.jsonl"
    observations.write_text(
        json.dumps(
            {
                "action_joint_positions": [0, 0, 0, 0, 0, 0],
                "phase": "home",
            }
        )
        + "\n"
    )
    source = {
        "dataset_tag": "v0.1.0",
        "scenes": [
            {
                "scene_id": "seen_1",
                "source_training_episode_id": "episode-1",
                "source_training_episode_path": str(episode),
                "source_metadata_sha256": file_sha256(metadata),
            }
        ],
    }
    manifest = MODULE.build_manifest(source)
    assert manifest["schema_version"] == "farpoint.expert-action-replay.v1"
    assert manifest["scenes"][0]["phases"] == ["home"]
    assert manifest["action_conversion"]["clip_to_calibrated_range"] is True
    assert manifest["scenes"][0]["source_values_clipped_by_exporter"] == 0
    assert len(manifest["scenes"][0]["actions_calibrated"][0]) == 6
    assert manifest["scenes"][0]["source_observations_sha256"] == file_sha256(
        observations
    )
