"""Generic, deterministic object variation plans for manipulation tasks."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from farpoint.scene_entities import bind_scene_entities, placement_target_entity


SCHEMA_VERSION = "farpoint.variation.v3"
SPLITS = ("train", "validation", "test")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class ObjectSpec:
    """Resolved physical and visual properties for one manipulation object."""

    shape: str
    asset_id: str
    dimensions_m: tuple[float, float, float]
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    rgba: tuple[float, float, float, float]
    mass_kg: float
    static_friction: float
    dynamic_friction: float
    restitution: float = 0.0

    def validate(self) -> None:
        if not isinstance(self.shape, str) or not self.shape:
            raise ValueError("shape must be non-empty")
        if not self.asset_id:
            raise ValueError("asset_id must be non-empty")
        if len(self.dimensions_m) != 3 or any(value <= 0 for value in self.dimensions_m):
            raise ValueError("dimensions_m must contain three positive values")
        if len(self.position_m) != 3 or not all(math.isfinite(v) for v in self.position_m):
            raise ValueError("position_m must contain three finite values")
        if len(self.orientation_xyzw) != 4:
            raise ValueError("orientation_xyzw must contain four values")
        norm = math.sqrt(sum(value * value for value in self.orientation_xyzw))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("orientation_xyzw must be normalized")
        if len(self.rgba) != 4 or any(value < 0.0 or value > 1.0 for value in self.rgba):
            raise ValueError("rgba values must be in [0, 1]")
        if self.mass_kg <= 0:
            raise ValueError("mass_kg must be positive")
        if self.static_friction < 0 or self.dynamic_friction < 0:
            raise ValueError("friction values must be non-negative")
        if self.dynamic_friction > self.static_friction:
            raise ValueError("dynamic_friction cannot exceed static_friction")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        for key in ("dimensions_m", "position_m", "orientation_xyzw", "rgba"):
            payload[key] = list(payload[key])
        return payload


def load_variation_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_variation_config(config)
    return config


def validate_variation_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    for key in (
        "plan_id",
        "task_id",
        "config_revision",
        "workspace",
        "object",
        "target",
        "recording",
    ):
        if key not in config:
            raise ValueError(f"variation config is missing {key}")
    workspace = config["workspace"]
    if int(workspace.get("rows", 0)) != 5 or int(workspace.get("columns", 0)) != 5:
        raise ValueError("SO-101 cube MVP requires a 5x5 workspace grid")
    for key in ("x_bounds_m", "y_bounds_m"):
        bounds = workspace.get(key)
        if not isinstance(bounds, list) or len(bounds) != 2 or bounds[0] >= bounds[1]:
            raise ValueError(f"workspace.{key} must contain increasing bounds")
    base = config["object"]
    dimensions = _dimension_profiles(base)
    colors = config["object"].get("colors")
    if len(dimensions) != 2:
        raise ValueError("object must define exactly two dimension profiles")
    if not isinstance(colors, list) or len(colors) != 2:
        raise ValueError("object.colors must contain two color profiles")
    for color in colors:
        ObjectSpec(
            shape=str(base.get("shape") or "cube"),
            asset_id=str(base.get("asset_id", "procedural_cube")),
            dimensions_m=tuple(dimensions[0]),
            position_m=(0.0, 0.0, dimensions[0][2] / 2),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            rgba=tuple(color["rgba"]),
            mass_kg=float(base["mass_kg"]),
            static_friction=float(base["static_friction"]),
            dynamic_friction=float(base["dynamic_friction"]),
            restitution=float(base.get("restitution", 0.0)),
        ).validate()
    placement_target_entity(
        config["target"],
        entity_type=str(config["target"].get("entity_type", "pad")),
        relation=str(config["target"].get("relation", "on")),
    )


def _dimension_profiles(base: dict[str, Any]) -> list[list[float]]:
    """Resolve generic XYZ profiles while retaining cube edge-size configs."""
    profiles = base.get("dimension_profiles_m")
    if profiles is None:
        sizes = base.get("edge_sizes_m")
        if not isinstance(sizes, list) or len(sizes) != 2:
            raise ValueError(
                "object.edge_sizes_m or object.dimension_profiles_m must define two sizes"
            )
        profiles = [[value, value, value] for value in sizes]
    if not isinstance(profiles, list) or len(profiles) != 2:
        raise ValueError("object.dimension_profiles_m must contain two XYZ profiles")
    resolved = []
    for profile in profiles:
        if (
            not isinstance(profile, list)
            or len(profile) != 3
            or any(not isinstance(value, (int, float)) or value <= 0 for value in profile)
        ):
            raise ValueError(
                "object.dimension_profiles_m entries must contain three positive values"
            )
        resolved.append([float(value) for value in profile])
    return resolved


def _split_for(index: int) -> str:
    if index < 80:
        return "train"
    if index < 90:
        return "validation"
    return "test"


def _position(config: dict[str, Any], row: int, column: int, seed: int) -> list[float]:
    grid = config["workspace"]
    x_min, x_max = (float(value) for value in grid["x_bounds_m"])
    y_min, y_max = (float(value) for value in grid["y_bounds_m"])
    x_width = (x_max - x_min) / 5
    y_width = (y_max - y_min) / 5
    low, high = (float(value) for value in grid.get("interior_fraction", [0.2, 0.8]))
    rng = random.Random(seed)
    return [
        round(x_min + column * x_width + x_width * rng.uniform(low, high), 9),
        round(y_min + row * y_width + y_width * rng.uniform(low, high), 9),
    ]


def generate_variation_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Create exactly 100 stratified cube trials and deterministic reserves."""
    validate_variation_config(config)
    trials = []
    base = config["object"]
    dimensions = _dimension_profiles(base)
    colors = config["object"]["colors"]
    ordinal = 0
    for row in range(5):
        for column in range(5):
            for size_index, profile in enumerate(dimensions):
                for color_index, color in enumerate(colors):
                    material = {
                        "schema_version": SCHEMA_VERSION,
                        "plan_id": config["plan_id"],
                        "row": row,
                        "column": column,
                        "size_index": size_index,
                        "color_index": color_index,
                    }
                    seed = int.from_bytes(hashlib.sha256(_canonical_json(material)).digest()[:8], "big")
                    xy = _position(config, row, column, seed)
                    requested = ObjectSpec(
                        shape=str(base.get("shape") or "cube"),
                        asset_id=str(base.get("asset_id", "procedural_cube")),
                        dimensions_m=tuple(profile),
                        position_m=(xy[0], xy[1], float(config["workspace"]["table_z_m"]) + profile[2] / 2),
                        orientation_xyzw=tuple(
                            float(value)
                            for value in base.get(
                                "orientation_xyzw", (0.0, 0.0, 0.0, 1.0)
                            )
                        ),
                        rgba=tuple(float(value) for value in color["rgba"]),
                        mass_kg=float(base["mass_kg"]),
                        static_friction=float(base["static_friction"]),
                        dynamic_friction=float(base["dynamic_friction"]),
                        restitution=float(base.get("restitution", 0.0)),
                    ).to_dict()
                    requested = bind_scene_entities(requested, config["target"])
                    trial_id = f"cube_r{row:02d}_c{column:02d}_s{size_index}_k{color_index}"
                    trials.append(
                        {
                            "trial_id": trial_id,
                            "variation_id": trial_id,
                            "cell_id": f"r{row:02d}_c{column:02d}",
                            "split": _split_for(ordinal),
                            "seed": seed,
                            "seed_material": material,
                            "requested": requested,
                            "resolved": copy.deepcopy(requested),
                        }
                    )
                    ordinal += 1
    plan = {
        "schema_version": SCHEMA_VERSION,
        "plan_id": config["plan_id"],
        "task_id": config["task_id"],
        "config_revision": str(config["config_revision"]),
        "config_sha256": _sha256(config),
        "varied_axes": [
            "entities.pick_object.pose.position_m.x",
            "entities.pick_object.pose.position_m.y",
            "entities.pick_object.geometry.dimensions_m",
            "entities.pick_object.appearance.rgba",
        ],
        "frozen_axes": [
            "entities.pick_object.entity_type",
            "entities.pick_object.physics",
            "entities.placement_target.pose",
            "entities.placement_target.geometry",
            "lighting.profile",
        ],
        "dimensions": [
            {"name": "workspace_cell", "kind": "categorical", "values": 25},
            {
                "name": "object_dimensions_m",
                "kind": "categorical",
                "values": dimensions,
            },
            {"name": "cube_color", "kind": "categorical", "values": [row["id"] for row in colors]},
        ],
        "target": copy.deepcopy(config["target"]),
        "recording": copy.deepcopy(config["recording"]),
        "trials": trials,
    }
    plan["plan_sha256"] = _sha256(plan)
    return plan
