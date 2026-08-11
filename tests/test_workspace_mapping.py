import pytest

from farpoint.workspace_mapping import (
    REQUIRED_CONSTRAINTS,
    WorkspaceProbe,
    derive_feasible_region,
    feasible_region_record,
)


def _probe(probe_id, x, y, *, failed_check=None):
    return WorkspaceProbe(
        x_m=x,
        y_m=y,
        checks=tuple((name, name != failed_check) for name in REQUIRED_CONSTRAINTS),
        probe_id=probe_id,
    )


def test_workspace_mapping_requires_every_constraint_and_derives_region():
    probes = [
        _probe("a", 0.10, -0.12),
        _probe("b", 0.27, -0.12),
        _probe("c", 0.27, 0.02),
        _probe("d", 0.10, 0.02),
        _probe("outside-fail", 0.30, 0.04, failed_check="full_path_ik"),
    ]
    region = derive_feasible_region(
        probes,
        region_id="red-40mm-v1",
        version="1",
        frame_id="isaac_world",
        object_anchor="bottom_center",
        footprint_xy_m=(0.04, 0.04),
        generator_identity={"git_commit": "a" * 40, "profile": "mapping-v1"},
    )
    assert region.contains((0.20, -0.05))
    assert region.max_clearance_m > 0
    assert region.band((0.20, -0.05)) in {"middle", "core"}
    assert len(region.generator_sha256) == 64
    assert len(region.constraints_sha256) == 64
    assert feasible_region_record(region)["schema_version"] == "farpoint.feasible-region.v1"


def test_workspace_mapping_rejects_failed_probe_hidden_by_convex_hull():
    probes = [
        _probe("a", 0.0, 0.0),
        _probe("b", 1.0, 0.0),
        _probe("c", 1.0, 1.0),
        _probe("d", 0.0, 1.0),
        _probe("hole", 0.5, 0.5, failed_check="wrist_camera_visible"),
    ]
    with pytest.raises(ValueError, match="hole"):
        derive_feasible_region(
            probes,
            region_id="unsafe",
            version="1",
            frame_id="world",
            object_anchor="center",
            footprint_xy_m=(0.04, 0.04),
            generator_identity={"version": 1},
        )


def test_workspace_probe_rejects_partial_checks():
    with pytest.raises(ValueError, match="required constraints"):
        WorkspaceProbe(0.0, 0.0, (("full_path_ik", True),), "partial")
