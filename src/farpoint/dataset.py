import json
from pathlib import Path


REQUIRED_OBSERVATION_FIELDS = {
    "frame",
    "timestamp_seconds",
    "phase",
    "rgb_path",
    "depth_path",
    "joint_positions",
    "joint_velocities",
    "action_joint_positions",
    "contact_forces_newtons",
    "object_pose_estimate",
}


def validate_observation(row):
    missing = sorted(REQUIRED_OBSERVATION_FIELDS.difference(row))
    if missing:
        return [f"missing field: {field}" for field in missing]
    errors = []
    if int(row["frame"]) < 0:
        errors.append("frame must be non-negative")
    if float(row["timestamp_seconds"]) < 0.0:
        errors.append("timestamp_seconds must be non-negative")
    joint_positions = row["joint_positions"]
    joint_velocities = row["joint_velocities"]
    action = row["action_joint_positions"]
    if not (len(joint_positions) == len(joint_velocities) == len(action)):
        errors.append("joint state and action vectors must have matching lengths")
    contacts = row["contact_forces_newtons"]
    if not {"left_finger", "right_finger"}.issubset(contacts):
        errors.append("contact forces must include left_finger and right_finger")
    return errors


def validate_episode_dataset(episode_dir):
    episode_dir = Path(episode_dir)
    observations_path = episode_dir / "observations.jsonl"
    labels_path = episode_dir / "labels.jsonl"
    errors = []
    observation_count = 0
    if not observations_path.exists():
        return {
            "valid": False,
            "observation_count": 0,
            "label_count": 0,
            "errors": ["observations.jsonl is missing"],
        }

    with observations_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"observations line {line_number}: {error}")
                continue
            observation_count += 1
            errors.extend(
                f"observations line {line_number}: {message}"
                for message in validate_observation(row)
            )
            for field in ("rgb_path", "depth_path"):
                artifact = episode_dir / row.get(field, "")
                if not artifact.is_file():
                    errors.append(
                        f"observations line {line_number}: {field} does not exist"
                    )

    label_count = 0
    if not labels_path.exists():
        errors.append("labels.jsonl is missing")
    else:
        with labels_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    json.loads(line)
                except json.JSONDecodeError as error:
                    errors.append(f"labels line {line_number}: {error}")
                    continue
                label_count += 1
        if label_count != observation_count:
            errors.append(
                f"label count {label_count} does not match observation count "
                f"{observation_count}"
            )

    return {
        "valid": not errors,
        "observation_count": observation_count,
        "label_count": label_count,
        "errors": errors,
    }
