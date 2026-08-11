import pytest

from farpoint.object_catalog import (
    MaterialSpec,
    ObjectArchetype,
    ObjectVariant,
    varied_variant_fields,
)


def _material(static=0.8, dynamic=0.6):
    return MaterialSpec(static, dynamic, 0.0, "average", "max")


def _variant(variant_id, dimensions, rgba, mass):
    return ObjectVariant(
        variant_id=variant_id,
        version="1",
        archetype_id="cube-v1",
        asset_id="procedural_cube",
        dimensions_m=dimensions,
        rgba=rgba,
        mass_kg=mass,
        object_material=_material(),
        table_material=_material(0.7, 0.5),
        gripper_material=_material(1.0, 0.8),
    )


def test_archetype_separates_geometry_semantics_from_variant_physics():
    archetype = ObjectArchetype(
        archetype_id="cube-v1",
        version="1",
        semantic_type="cube",
        geometry_representation="procedural",
        anchor="bottom_center",
        default_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    ).to_dict()
    assert archetype["semantic_type"] == "cube"
    assert len(archetype["config_sha256"]) == 64


def test_variant_records_object_table_and_gripper_materials_with_combine_modes():
    variant = _variant("red-40mm", (0.04, 0.04, 0.04), (0.9, 0.1, 0.1, 1.0), 0.04)
    payload = variant.to_dict()
    assert payload["mass_kg"] == 0.04
    assert payload["object_material"]["friction_combine_mode"] == "average"
    assert payload["table_material"]["dynamic_friction"] == 0.5
    assert payload["gripper_material"]["static_friction"] == 1.0


def test_mass_and_friction_are_axes_only_when_variants_actually_differ():
    red = _variant("red-40mm", (0.04, 0.04, 0.04), (0.9, 0.1, 0.1, 1.0), 0.04)
    blue = _variant("blue-30mm", (0.03, 0.03, 0.03), (0.1, 0.2, 0.9, 1.0), 0.03)
    axes = varied_variant_fields([red, blue])
    assert "mass_kg" in axes
    assert "dimensions_m" in axes
    assert "rgba" in axes
    assert "object_material" not in axes


def test_invalid_physx_material_is_rejected():
    with pytest.raises(ValueError, match="dynamic friction"):
        _material(0.2, 0.3).validate()
    with pytest.raises(ValueError, match="combine"):
        MaterialSpec(0.8, 0.6, 0.0, "invalid", "max").validate()
