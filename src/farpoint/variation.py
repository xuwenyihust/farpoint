"""Versioned, deterministic variation planning for Farpoint task episodes."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path


CONFIG_VERSION = "farpoint.variation.v1"
SUPPORTED_OBJECT_TYPES = frozenset({"cube", "cylinder"})
REQUIRED_PROFILE_FIELDS = frozenset(
    {"variation_id", "object_type", "object_position_bin", "position_xy"}
)


def load_variation_config(path: str | Path) -> dict:
    """Load and validate a versioned variation configuration."""
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_variation_config(config)
    return config


def validate_variation_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("variation config must be an object")
    if config.get("schema_version") != CONFIG_VERSION:
        raise ValueError(
            f"variation config schema_version must be {CONFIG_VERSION!r}"
        )
    profiles = config.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("variation config must contain a non-empty profiles list")

    variation_ids = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("each variation profile must be an object")
        missing = REQUIRED_PROFILE_FIELDS - profile.keys()
        if missing:
            raise ValueError(
                f"variation profile is missing fields: {sorted(missing)}"
            )
        variation_id = profile["variation_id"]
        if variation_id in variation_ids:
            raise ValueError(f"duplicate variation_id: {variation_id}")
        variation_ids.add(variation_id)
        if profile["object_type"] not in SUPPORTED_OBJECT_TYPES:
            raise ValueError(f"unsupported object_type: {profile['object_type']}")
        bounds = profile["position_xy"]
        if (
            not isinstance(bounds, dict)
            or set(bounds) != {"x", "y"}
            or any(
                not isinstance(bounds[axis], list)
                or len(bounds[axis]) != 2
                or float(bounds[axis][0]) > float(bounds[axis][1])
                for axis in ("x", "y")
            )
        ):
            raise ValueError(
                f"position_xy for {variation_id} must contain x/y bounds"
            )


def _derived_seed(schema_version: str, variation_id: str, seed: int) -> int:
    material = f"{schema_version}:{variation_id}:{int(seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def resolve_variation(config: dict, variation_id: str, seed: int) -> dict:
    """Resolve one profile and seed into an immutable episode configuration."""
    validate_variation_config(config)
    profile = next(
        (item for item in config["profiles"] if item["variation_id"] == variation_id),
        None,
    )
    if profile is None:
        known = ", ".join(item["variation_id"] for item in config["profiles"])
        raise ValueError(f"unknown variation_id {variation_id!r}; expected one of {known}")

    resolved_seed = _derived_seed(config["schema_version"], variation_id, seed)
    rng = random.Random(resolved_seed)
    x_bounds = profile["position_xy"]["x"]
    y_bounds = profile["position_xy"]["y"]
    position_xy = [
        round(rng.uniform(float(x_bounds[0]), float(x_bounds[1])), 6),
        round(rng.uniform(float(y_bounds[0]), float(y_bounds[1])), 6),
    ]
    return {
        "schema_version": config["schema_version"],
        "variation_id": profile["variation_id"],
        "object_type": profile["object_type"],
        "object_position_bin": profile["object_position_bin"],
        "object_position_xy": position_xy,
        "object_yaw_degrees": float(profile.get("object_yaw_degrees", 0.0)),
        "appearance_profile": profile.get("appearance_profile", "orange_default"),
        "grasp_profile": profile.get("grasp_profile", "default"),
        "seed": int(seed),
        "derived_seed": resolved_seed,
        "config": copy.deepcopy(config.get("fixed_parameters", {})),
    }


def plan_variations(config: dict, seeds: list[int]) -> list[dict]:
    """Return a stable profile-major plan for dry-runs and batch generation."""
    return [
        resolve_variation(config, profile["variation_id"], seed)
        for profile in config["profiles"]
        for seed in seeds
    ]
