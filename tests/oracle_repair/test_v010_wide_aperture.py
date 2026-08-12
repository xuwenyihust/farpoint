"""Regression contract for the v0.1.0 wide-aperture Oracle repair."""

import pytest

from farpoint.control import (
    advance_so101_slow_close_target,
    so101_approach_jaw_target,
    so101_capture_admission_ready,
)


def test_repair_changes_only_the_large_cube_approach_endpoint():
    assert so101_approach_jaw_target(0.03) == pytest.approx(0.90)
    assert so101_approach_jaw_target(0.04) == pytest.approx(1.40)


def test_supported_widths_interpolate_monotonically_within_mesh_anchors():
    widths = (0.030, 0.0325, 0.035, 0.0375, 0.040)
    targets = tuple(so101_approach_jaw_target(width) for width in widths)

    assert targets == pytest.approx((0.90, 1.025, 1.15, 1.275, 1.40))
    assert all(left < right for left, right in zip(targets, targets[1:]))


def test_wide_corner_contact_cannot_arm_capture_before_enclosure():
    assert not so101_capture_admission_ready(1.36, 0.04)
    assert so101_capture_admission_ready(0.78, 0.04)
    assert so101_capture_admission_ready(0.53, 0.04)


def test_wide_corner_contact_keeps_closing_until_admission_is_ready():
    admissible = so101_capture_admission_ready(1.36, 0.04)
    update = advance_so101_slow_close_target(
        1.36,
        1.36,
        3.2,
        2.9,
        open_position=1.7453,
        closed_position=-0.1745,
        capture_admissible=admissible,
    )

    assert update == {"position": pytest.approx(1.359), "action": "close"}
