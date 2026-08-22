import numpy as np
import pytest

from farpoint.grasp_oracle import quaternion_rotation_matrix_xyzw
from farpoint.so101_grasp_geometry import (
    SO101_APERTURE_REFERENCE_IN_GRIPPER_M,
    SO101_CAPTURE_APERTURE_CALIBRATION,
    SO101_CAPTURE_CLOSING_AXIS_LOCAL,
    SO101_RUNTIME_QUATERNION_ORDER,
    aabb_corners,
    posture_geometry_diagnostics,
    so101_capture_aperture_reference,
    so101_capture_channel_direction_world,
    so101_level_capture_orientation_xyzw,
    so101_pre_capture_recenter_aperture_reference,
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


def test_capture_aperture_reference_matches_exact_mesh_anchors():
    for jaw_position, expected_reference in SO101_CAPTURE_APERTURE_CALIBRATION:
        np.testing.assert_allclose(
            so101_capture_aperture_reference(jaw_position),
            expected_reference,
            atol=1e-8,
        )


def test_capture_aperture_reference_preserves_30mm_path_and_interpolates():
    reference_12 = SO101_CAPTURE_APERTURE_CALIBRATION[0][1]
    reference_14 = SO101_CAPTURE_APERTURE_CALIBRATION[1][1]

    # The validated 30 mm approach uses 0.9 rad but the production aperture
    # reference was calibrated at 1.2 rad; preserve that path exactly.
    np.testing.assert_allclose(so101_capture_aperture_reference(0.9), reference_12)
    np.testing.assert_allclose(
        so101_capture_aperture_reference(1.3),
        (reference_12 + reference_14) / 2.0,
        atol=1e-8,
    )


def test_capture_aperture_reference_returns_copy_and_validates_joint_limit():
    first = so101_capture_aperture_reference(1.7)
    first[:] = 0.0
    assert np.linalg.norm(so101_capture_aperture_reference(1.7)) > 0.0

    for invalid in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            so101_capture_aperture_reference(invalid)
    for invalid in (-0.1747, 1.7454):
        with pytest.raises(ValueError, match="pinned USD limits"):
            so101_capture_aperture_reference(invalid)


def test_pre_capture_recenter_reference_preserves_small_and_biases_large_cube():
    base = so101_capture_aperture_reference(1.4)
    small = so101_pre_capture_recenter_aperture_reference(1.4, 0.030)
    middle = so101_pre_capture_recenter_aperture_reference(1.4, 0.035)
    large = so101_pre_capture_recenter_aperture_reference(1.4, 0.040)

    np.testing.assert_allclose(small, base)
    assert middle[0] == pytest.approx((float(base[0]) + 0.016) / 2.0)
    assert large[0] == pytest.approx(0.016)
    np.testing.assert_allclose(middle[1:], base[1:])
    np.testing.assert_allclose(large[1:], base[1:])


@pytest.mark.parametrize("width", (0.0, float("nan"), float("inf")))
def test_pre_capture_recenter_reference_validates_width(width):
    with pytest.raises(ValueError, match="object_width_m"):
        so101_pre_capture_recenter_aperture_reference(1.4, width)


def test_capture_channel_is_horizontal_normalized_and_perpendicular():
    orientation = np.asarray(
        (-0.6417146921, 0.1408973038, 0.0826371983, -0.7493473291)
    )
    channel = so101_capture_channel_direction_world(orientation)
    closing_world = (
        quaternion_rotation_matrix_xyzw(orientation)
        @ SO101_CAPTURE_CLOSING_AXIS_LOCAL
    )

    assert channel[2] == 0.0
    assert np.linalg.norm(channel) == pytest.approx(1.0)
    assert np.dot(channel, closing_world) == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_allclose(channel[:2], [-0.6689, -0.7434], atol=1e-4)


@pytest.mark.parametrize(
    "invalid",
    ([0.0, 0.0, 0.0], [0.0, 0.0, float("nan"), 1.0]),
)
def test_capture_channel_validates_orientation(invalid):
    with pytest.raises(ValueError):
        so101_capture_channel_direction_world(invalid)


def test_level_capture_orientation_flattens_axis_and_preserves_channel():
    orientation = np.asarray(
        (-0.6417146921, 0.1408973038, 0.0826371983, -0.7493473291)
    )
    original_channel = so101_capture_channel_direction_world(orientation)

    levelled = so101_level_capture_orientation_xyzw(orientation)
    levelled_closing = (
        quaternion_rotation_matrix_xyzw(levelled)
        @ SO101_CAPTURE_CLOSING_AXIS_LOCAL
    )

    assert np.linalg.norm(levelled) == pytest.approx(1.0)
    assert levelled_closing[2] == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_allclose(
        so101_capture_channel_direction_world(levelled),
        original_channel,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "invalid",
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
)
def test_level_capture_orientation_validates_input(invalid):
    with pytest.raises(ValueError):
        so101_level_capture_orientation_xyzw(invalid)


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
