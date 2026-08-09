import json

from farpoint.dataset_card import generate_dataset_card, variation_coverage


def _record(index, split, cell, mass, yaw_quaternion):
    entity = {
        "entity_type": "cube",
        "geometry": {"dimensions_m": [0.03, 0.03, 0.03]},
        "pose": {
            "position_m": [0.15 + index * 0.1, -0.1 + index * 0.05, 0.047],
            "orientation_xyzw": yaw_quaternion,
        },
        "appearance": {"rgba": [0.85, 0.08, 0.06, 1.0]},
        "physics": {"mass_kg": mass},
    }
    target = {
        "entity_type": "pad",
        "geometry": {"dimensions_m": [0.16, 0.14, 0.01]},
    }
    return {
        "identity": {"split": split},
        "variation": {
            "variation_id": f"cube_{cell}_s0_k0",
            "resolved": {
                "entities": {
                    "pick_object": entity,
                    "placement_target": target,
                }
            },
        },
    }


def test_generated_card_uses_exported_counts_features_and_variations(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    records = [
        _record(0, "train", "r00_c00", 0.03, [0, 0, 0, 1]),
        _record(1, "test", "r00_c01", 0.04, [0, 0, 0.3826834324, 0.9238795325]),
    ]
    (meta / "episode_metadata.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    (meta / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "total_episodes": 2,
                "total_frames": 180,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [6]},
                    "observation.images.front": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                    },
                    "action": {"dtype": "float32", "shape": [6]},
                    "timestamp": {"dtype": "float32", "shape": []},
                },
            }
        ),
        encoding="utf-8",
    )
    (meta / "farpoint_v3.json").write_text(
        json.dumps(
            {
                "splits": {"train": 1, "validation": 0, "test": 1},
                "robot": {"name": "so101"},
                "simulation": {"simulator": "Isaac Sim", "physics": "PhysX"},
                "recording": {
                    "fps": 30,
                    "cameras": ["observation.images.front"],
                },
            }
        ),
        encoding="utf-8",
    )
    spec = {
        "dataset_card_mode": "generated",
        "dataset_id": "farpoint_so101",
        "dataset_tag": "v9.9.9",
        "hf_repo_id": "owner/dataset",
        "card": {
            "pretty_name": "Farpoint SO-101",
            "license": "cc-by-4.0",
            "description": "Generated test dataset.",
            "tags": ["robotics"],
        },
    }

    card = generate_dataset_card(tmp_path, spec)

    assert "| **Episodes** | 2 successful demonstrations |" in card
    assert "| Train | 1 | 50.0% |" in card
    assert "| Object position | 2 stratified cells;" in card
    assert "| Object mass | 0.03 kg, 0.04 kg |" in card
    assert "| Object yaw | 0°, 45° |" in card
    assert card.index("### Variation coverage") < card.index("### Policy features")
    assert "Distribution" not in card
    assert "timestamp" not in card


def test_variation_coverage_accepts_non_cube_entity_types():
    record = _record(0, "train", "r00_c00", 0.03, [0, 0, 0, 1])
    record["variation"]["resolved"]["entities"]["pick_object"]["entity_type"] = "cylinder"

    rows = dict(variation_coverage([record]))

    assert rows["Object type"] == "cylinder"
