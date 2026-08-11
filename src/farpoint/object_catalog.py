"""Versioned archetypes and variants for extensible manipulation objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

from farpoint.scene_entities import PHYSX_COMBINE_MODES


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MaterialSpec:
    static_friction: float
    dynamic_friction: float
    restitution: float
    friction_combine_mode: str
    restitution_combine_mode: str

    def validate(self) -> None:
        values = (self.static_friction, self.dynamic_friction, self.restitution)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("material coefficients must be finite")
        if self.static_friction < 0.0 or self.dynamic_friction < 0.0:
            raise ValueError("friction coefficients must be non-negative")
        if self.dynamic_friction > self.static_friction:
            raise ValueError("dynamic friction cannot exceed static friction")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must be in [0, 1]")
        if self.friction_combine_mode not in PHYSX_COMBINE_MODES:
            raise ValueError("invalid PhysX friction combine mode")
        if self.restitution_combine_mode not in PHYSX_COMBINE_MODES:
            raise ValueError("invalid PhysX restitution combine mode")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ObjectArchetype:
    """Geometry semantics shared by variants such as cube, cylinder, or doll."""

    archetype_id: str
    version: str
    semantic_type: str
    geometry_representation: str
    anchor: str
    default_orientation_xyzw: tuple[float, float, float, float]

    def validate(self) -> None:
        if any(not value for value in (self.archetype_id, self.version, self.semantic_type)):
            raise ValueError("object archetype identity must be non-empty")
        if self.geometry_representation not in {"procedural", "asset", "mesh"}:
            raise ValueError("unsupported geometry representation")
        if self.anchor not in {"center", "bottom_center", "asset_origin"}:
            raise ValueError("unsupported object anchor")
        if len(self.default_orientation_xyzw) != 4:
            raise ValueError("default orientation must be a quaternion")
        norm = math.sqrt(sum(value * value for value in self.default_orientation_xyzw))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("default orientation quaternion must be normalized")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["default_orientation_xyzw"] = list(self.default_orientation_xyzw)
        payload["config_sha256"] = canonical_sha256(payload)
        return payload


@dataclass(frozen=True)
class ObjectVariant:
    """A frozen geometry/appearance/physics bundle, not automatically an axis."""

    variant_id: str
    version: str
    archetype_id: str
    asset_id: str
    dimensions_m: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    mass_kg: float
    object_material: MaterialSpec
    table_material: MaterialSpec
    gripper_material: MaterialSpec

    def validate(self) -> None:
        if any(not value for value in (self.variant_id, self.version, self.archetype_id, self.asset_id)):
            raise ValueError("object variant identity must be non-empty")
        if len(self.dimensions_m) != 3 or any(
            value <= 0.0 or not math.isfinite(value) for value in self.dimensions_m
        ):
            raise ValueError("variant dimensions must be three positive finite values")
        if len(self.rgba) != 4 or any(
            value < 0.0 or value > 1.0 or not math.isfinite(value) for value in self.rgba
        ):
            raise ValueError("variant RGBA must contain four values in [0, 1]")
        if self.mass_kg <= 0.0 or not math.isfinite(self.mass_kg):
            raise ValueError("variant mass must be positive and finite")
        self.object_material.validate()
        self.table_material.validate()
        self.gripper_material.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["dimensions_m"] = list(self.dimensions_m)
        payload["rgba"] = list(self.rgba)
        payload["config_sha256"] = canonical_sha256(payload)
        return payload


def varied_variant_fields(variants: list[ObjectVariant]) -> list[str]:
    """Report variant fields as axes only when values differ in the frozen plan."""
    if not variants:
        raise ValueError("at least one object variant is required")
    records = [variant.to_dict() for variant in variants]
    fields = (
        "archetype_id",
        "asset_id",
        "dimensions_m",
        "rgba",
        "mass_kg",
        "object_material",
        "table_material",
        "gripper_material",
    )
    return [field for field in fields if len({canonical_sha256(record[field]) for record in records}) > 1]
