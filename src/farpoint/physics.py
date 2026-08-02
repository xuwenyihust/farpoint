def enable_enhanced_determinism(
    stage,
    *,
    physx_schema=None,
    usd_physics=None,
):
    if physx_schema is None or usd_physics is None:
        from pxr import PhysxSchema, UsdPhysics

        physx_schema = PhysxSchema
        usd_physics = UsdPhysics

    physics_scenes = [
        prim for prim in stage.Traverse() if prim.IsA(usd_physics.Scene)
    ]
    if len(physics_scenes) != 1:
        raise ValueError(
            "expected exactly one physics scene, "
            f"found {len(physics_scenes)}"
        )
    physics_scene = physics_scenes[0]
    scene_api = physx_schema.PhysxSceneAPI.Apply(physics_scene)
    scene_api.CreateEnableEnhancedDeterminismAttr().Set(True)
    return str(physics_scene.GetPath())
