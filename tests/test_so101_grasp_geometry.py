import numpy as np
import pytest

from farpoint.so101_grasp_geometry import (
    SO101_APERTURE_REFERENCE_IN_GRIPPER_M,
    SO101_RUNTIME_QUATERNION_ORDER,
    aabb_corners,
    posture_geometry_diagnostics,
    transform_points_xyzw,
)


def test_aabb_corners_are_complete_and_validate_order():
    corners = aabb_corners(((-1.0, -2.0, -3.0), (1.0, 2.0, 3.0)))

    assert corners.shape == (8, 3)
    assert {tuple(row) for row in corners} == {
        (x, y, z)
        for x in (-1.0, 1.0)
        for y in (-2.0, 2.0)
        for z in (-3.0, 3.0)
    }
    with pytest.raises(ValueError, match="must not exceed"):
        aabb_corners(((1.0, 0.0, 0.0), (0.0, 1.0, 1.0)))


def test_transform_points_uses_pinned_isaac_lab_3_xyzw_order():
    quarter_turn_z_xyzw = [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]
    pose = [1.0, 2.0, 3.0, *quarter_turn_z_xyzw]

    transformed = transform_points_xyzw(pose, [[1.0, 0.0, 0.0]])

    assert SO101_RUNTIME_QUATERNION_ORDER == "xyzw"
    np.testing.assert_allclose(transformed, [[1.0, 3.0, 3.0]], atol=1e-7)


def test_posture_diagnostic_aligns_aperture_without_using_link_origin():
    identity_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    object_center = np.asarray([0.15, -0.10, 0.052])

    result = posture_geometry_diagnostics(
        identity_pose,
        identity_pose,
        object_center,
        object_half_height_m=0.020,
    )

    expected_target = object_center - SO101_APERTURE_REFERENCE_IN_GRIPPER_M
    np.testing.assert_allclose(result["aligned_gripper_target_m"], expected_target)
    np.testing.assert_allclose(
        np.asarray(result["aperture_center_world_m"])
        + np.asarray(result["alignment_translation_m"]),
        object_center,
    )
    assert result["inferred_table_height_m"] == pytest.approx(0.032)
    # Identity points the long fixed finger down.  The conservative screen
    # must reject this posture for a table-resting 40 mm cube.
    assert result["conservative_table_clearance_m"] < 0.0


@pytest.mark.parametrize(
    ("half_height", "message"),
    [(0.0, "positive"), (float("nan"), "positive")],
)
def test_posture_diagnostic_rejects_invalid_object_height(half_height, message):
    identity_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match=message):
        posture_geometry_diagnostics(
            identity_pose,
            identity_pose,
            [0.15, -0.10, 0.052],
            object_half_height_m=half_height,
        )
