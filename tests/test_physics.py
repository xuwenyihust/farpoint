from farpoint.physics import enable_enhanced_determinism


class FakeAttribute:
    def __init__(self):
        self.value = None

    def Set(self, value):
        self.value = value


class FakeSceneApi:
    def __init__(self):
        self.attribute = FakeAttribute()

    def CreateEnableEnhancedDeterminismAttr(self):
        return self.attribute


class FakePhysxSceneApi:
    applied_prim = None
    instance = None

    @classmethod
    def Apply(cls, prim):
        cls.applied_prim = prim
        cls.instance = FakeSceneApi()
        return cls.instance


class FakePhysxSchema:
    PhysxSceneAPI = FakePhysxSceneApi


class FakeUsdPhysics:
    class Scene:
        pass


class FakePrim:
    def __init__(self, path, is_physics_scene):
        self.path = path
        self.is_physics_scene = is_physics_scene

    def IsA(self, schema):
        return schema is FakeUsdPhysics.Scene and self.is_physics_scene

    def GetPath(self):
        return self.path


class FakeStage:
    def __init__(self, prims):
        self.prims = prims

    def Traverse(self):
        return iter(self.prims)


def test_enhanced_determinism_is_authored_on_the_only_physics_scene():
    visual = FakePrim("/World/Cube", False)
    physics = FakePrim("/World/physicsScene", True)

    path = enable_enhanced_determinism(
        FakeStage([visual, physics]),
        physx_schema=FakePhysxSchema,
        usd_physics=FakeUsdPhysics,
    )

    assert path == "/World/physicsScene"
    assert FakePhysxSceneApi.applied_prim is physics
    assert FakePhysxSceneApi.instance.attribute.value is True


def test_enhanced_determinism_requires_one_physics_scene():
    try:
        enable_enhanced_determinism(
            FakeStage([]),
            physx_schema=FakePhysxSchema,
            usd_physics=FakeUsdPhysics,
        )
    except ValueError as error:
        assert "found 0" in str(error)
    else:
        raise AssertionError("missing physics scene should fail")
