"""Generic scene-entity metadata for extensible manipulation datasets."""

from __future__ import annotations

import copy
import math
from typing import Any, Iterable


ENTITY_SCHEMA_VERSION = "farpoint.scene-entity.v1"
ENTITY_ROLES = {
    "manipulated_object",
    "placement_target",
    "support_surface",
    "fixture",
    "distractor",
}
BODY_TYPES = {"dynamic", "kinematic", "static"}
PHYSX_COMBINE_MODES = {"average", "min", "multiply", "max"}


def _finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _validate_pose(pose: Any, name: str) -> None:
    if not isinstance(pose, dict):
        raise ValueError(f"{name} must be an object")
    _finite_vector(pose.get("position_m"), 3, f"{name}.position_m")
    quaternion = _finite_vector(
        pose.get("orientation_xyzw"), 4, f"{name}.orientation_xyzw"
    )
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError(f"{name}.orientation_xyzw must be normalized")
    if not isinstance(pose.get("frame_id"), str) or not pose["frame_id"]:
        raise ValueError(f"{name}.frame_id must be non-empty")


def _validate_geometry(geometry: Any, name: str) -> None:
    if not isinstance(geometry, dict):
        raise ValueError(f"{name} must be an object")
    if not isinstance(geometry.get("representation"), str) or not geometry[
        "representation"
    ]:
        raise ValueError(f"{name}.representation must be non-empty")
    dimensions = geometry.get("dimensions_m")
    if dimensions is not None:
        values = _finite_vector(dimensions, 3, f"{name}.dimensions_m")
        if any(value <= 0.0 for value in values):
            raise ValueError(f"{name}.dimensions_m must be positive")
    scale = geometry.get("scale_xyz")
    if scale is not None:
        values = _finite_vector(scale, 3, f"{name}.scale_xyz")
        if any(value <= 0.0 for value in values):
            raise ValueError(f"{name}.scale_xyz must be positive")


def _validate_physics(physics: Any, name: str) -> None:
    if not isinstance(physics, dict):
        raise ValueError(f"{name} must be an object")
    body_type = physics.get("body_type")
    if body_type not in BODY_TYPES:
        raise ValueError(f"{name}.body_type must be one of {sorted(BODY_TYPES)}")
    mass = physics.get("mass_kg")
    if body_type == "dynamic" and (
        not isinstance(mass, (int, float))
        or not math.isfinite(float(mass))
        or float(mass) <= 0.0
    ):
        raise ValueError(f"{name}.mass_kg must be positive for dynamic entities")
    material = physics.get("material")
    if material is None:
        return
    if not isinstance(material, dict):
        raise ValueError(f"{name}.material must be an object")
    static = float(material.get("static_friction", 0.0))
    dynamic = float(material.get("dynamic_friction", 0.0))
    restitution = float(material.get("restitution", 0.0))
    if not all(math.isfinite(value) for value in (static, dynamic, restitution)):
        raise ValueError(f"{name}.material values must be finite")
    if static < 0.0 or dynamic < 0.0:
        raise ValueError(f"{name}.material friction must be non-negative")
    if dynamic > static:
        raise ValueError(
            f"{name}.material dynamic_friction cannot exceed static_friction"
        )
    if not 0.0 <= restitution <= 1.0:
        raise ValueError(f"{name}.material restitution must be in [0, 1]")
    for field in ("friction_combine_mode", "restitution_combine_mode"):
        mode = material.get(field)
        if mode is not None and mode not in PHYSX_COMBINE_MODES:
            raise ValueError(
                f"{name}.material.{field} must be one of {sorted(PHYSX_COMBINE_MODES)}"
            )


def validate_scene_entity(entity: dict[str, Any]) -> None:
    """Validate one open-ended entity without restricting asset taxonomy."""
    if entity.get("schema_version") != ENTITY_SCHEMA_VERSION:
        raise ValueError(f"entity.schema_version must be {ENTITY_SCHEMA_VERSION!r}")
    for key in ("entity_id", "role", "entity_type", "asset_id"):
        if not isinstance(entity.get(key), str) or not entity[key]:
            raise ValueError(f"entity.{key} must be non-empty")
    if entity["role"] not in ENTITY_ROLES:
        raise ValueError(f"entity.role must be one of {sorted(ENTITY_ROLES)}")
    _validate_pose(entity.get("pose"), "entity.pose")
    _validate_geometry(entity.get("geometry"), "entity.geometry")
    _validate_physics(entity.get("physics"), "entity.physics")
    appearance = entity.get("appearance")
    if appearance is not None:
        if not isinstance(appearance, dict):
            raise ValueError("entity.appearance must be an object")
        rgba = appearance.get("rgba")
        if rgba is not None:
            values = _finite_vector(rgba, 4, "entity.appearance.rgba")
            if any(value < 0.0 or value > 1.0 for value in values):
                raise ValueError("entity.appearance.rgba values must be in [0, 1]")
    region_ids = []
    for index, region in enumerate(entity.get("regions") or ()):
        name = f"entity.regions[{index}]"
        if not isinstance(region, dict):
            raise ValueError(f"{name} must be an object")
        if not isinstance(region.get("region_id"), str) or not region["region_id"]:
            raise ValueError(f"{name}.region_id must be non-empty")
        if not isinstance(region.get("relation"), str) or not region["relation"]:
            raise ValueError(f"{name}.relation must be non-empty")
        _validate_pose(region.get("pose"), f"{name}.pose")
        _validate_geometry(region.get("geometry"), f"{name}.geometry")
        margin = float(region.get("margin_m", 0.0))
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError(f"{name}.margin_m must be non-negative")
        region_ids.append(region["region_id"])
    if len(set(region_ids)) != len(region_ids):
        raise ValueError("entity region_id values must be unique")


def validate_scene_entities(entities: Iterable[dict[str, Any]]) -> None:
    entities = list(entities)
    if not entities:
        raise ValueError("scene.entities must not be empty")
    for entity in entities:
        validate_scene_entity(entity)
    ids = [entity["entity_id"] for entity in entities]
    if len(set(ids)) != len(ids):
        raise ValueError("scene entity_id values must be unique")


def legacy_object_entity(
    object_spec: dict[str, Any],
    *,
    entity_id: str = "pick_object",
    frame_id: str = "isaac_world",
) -> dict[str, Any]:
    """Create the canonical entity view of a legacy scene.object record."""
    pose = object_spec.get("initial_pose") or {
        "position_m": object_spec.get("position_m"),
        "orientation_xyzw": object_spec.get("orientation_xyzw"),
    }
    asset_id = str(object_spec.get("asset_id") or "unspecified_asset")
    entity = {
        "schema_version": ENTITY_SCHEMA_VERSION,
        "entity_id": entity_id,
        "role": "manipulated_object",
        "entity_type": str(object_spec.get("shape") or "unspecified"),
        "asset_id": asset_id,
        "pose": {
            "frame_id": frame_id,
            "position_m": copy.deepcopy(pose.get("position_m")),
            "orientation_xyzw": copy.deepcopy(
                pose.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0])
            ),
        },
        "geometry": {
            "representation": (
                "procedural" if asset_id.startswith("procedural_") else "asset"
            ),
            "shape": str(object_spec.get("shape") or "unspecified"),
            "dimensions_m": copy.deepcopy(object_spec.get("dimensions_m")),
            "scale_xyz": copy.deepcopy(object_spec.get("scale_xyz")),
            "collision_geometry_id": object_spec.get("collision_geometry_id"),
        },
        "physics": {
            "body_type": "dynamic",
            "mass_kg": object_spec.get("mass_kg"),
            "collision_enabled": True,
            "material": {
                "static_friction": object_spec.get("static_friction", 0.0),
                "dynamic_friction": object_spec.get("dynamic_friction", 0.0),
                "restitution": object_spec.get("restitution", 0.0),
                "friction_combine_mode": object_spec.get("friction_combine_mode"),
                "restitution_combine_mode": object_spec.get(
                    "restitution_combine_mode"
                ),
            },
        },
        "appearance": {"rgba": copy.deepcopy(object_spec.get("rgba"))},
    }
    if entity["geometry"]["scale_xyz"] is None:
        entity["geometry"].pop("scale_xyz")
    if entity["geometry"]["collision_geometry_id"] is None:
        entity["geometry"].pop("collision_geometry_id")
    entity["physics"]["material"] = {
        key: value
        for key, value in entity["physics"]["material"].items()
        if value is not None
    }
    validate_scene_entity(entity)
    return entity


def placement_target_entity(
    target_spec: dict[str, Any],
    *,
    entity_id: str = "placement_target",
    entity_type: str = "pad",
    relation: str = "on",
    region_id: str = "placement_region",
    frame_id: str = "isaac_world",
) -> dict[str, Any]:
    """Create a target entity whose physical geometry and success region differ."""
    pose = target_spec.get("pose") or {
        "position_m": target_spec.get("position_m"),
        "orientation_xyzw": target_spec.get(
            "orientation_xyzw", [0.0, 0.0, 0.0, 1.0]
        ),
    }
    dimensions = copy.deepcopy(target_spec.get("dimensions_m"))
    canonical_pose = {
        "frame_id": frame_id,
        "position_m": copy.deepcopy(pose.get("position_m")),
        "orientation_xyzw": copy.deepcopy(pose.get("orientation_xyzw")),
    }
    region_pose = {
        "frame_id": str(target_spec.get("region_frame_id", frame_id)),
        "position_m": copy.deepcopy(
            target_spec.get("region_position_m", canonical_pose["position_m"])
        ),
        "orientation_xyzw": copy.deepcopy(
            target_spec.get(
                "region_orientation_xyzw", canonical_pose["orientation_xyzw"]
            )
        ),
    }
    entity = {
        "schema_version": ENTITY_SCHEMA_VERSION,
        "entity_id": entity_id,
        "role": "placement_target",
        "entity_type": entity_type,
        "asset_id": str(
            target_spec.get("asset_id")
            or target_spec.get("target_id")
            or "unspecified_target"
        ),
        "pose": canonical_pose,
        "geometry": {
            "representation": str(target_spec.get("representation", "procedural")),
            "shape": str(target_spec.get("shape", "cuboid")),
            "dimensions_m": dimensions,
            "scale_xyz": copy.deepcopy(target_spec.get("scale_xyz")),
            "collision_geometry_id": target_spec.get("collision_geometry_id"),
        },
        "physics": {
            "body_type": str(target_spec.get("body_type", "static")),
            "collision_enabled": bool(target_spec.get("collision_enabled", True)),
            "material": copy.deepcopy(target_spec.get("material", {})),
        },
        "appearance": {"rgba": copy.deepcopy(target_spec.get("rgba"))},
        "regions": [
            {
                "region_id": region_id,
                "relation": relation,
                "pose": region_pose,
                "geometry": {
                    "representation": "analytic",
                    "shape": str(target_spec.get("region_shape", "cuboid")),
                    "dimensions_m": copy.deepcopy(
                        target_spec.get("region_dimensions_m", dimensions)
                    ),
                },
                "margin_m": float(target_spec.get("footprint_margin_m", 0.0)),
            }
        ],
    }
    if entity["geometry"]["scale_xyz"] is None:
        entity["geometry"].pop("scale_xyz")
    if entity["geometry"]["collision_geometry_id"] is None:
        entity["geometry"].pop("collision_geometry_id")
    validate_scene_entity(entity)
    return entity


def bind_scene_entities(
    object_state: dict[str, Any],
    target_state: dict[str, Any],
    *,
    object_entity_id: str = "pick_object",
    target_entity_id: str = "placement_target",
) -> dict[str, Any]:
    """Return a legacy-compatible variation state with canonical entities."""
    state = copy.deepcopy(object_state)
    state["entities"] = {
        object_entity_id: legacy_object_entity(
            state, entity_id=object_entity_id
        ),
        target_entity_id: placement_target_entity(
            target_state,
            entity_id=target_entity_id,
            entity_type=str(target_state.get("entity_type", "pad")),
            relation=str(target_state.get("relation", "on")),
        ),
    }
    return state
