import pytest

from farpoint.cylinder_control import hold_pregrasp_hover


def test_pregrasp_hover_clamps_only_the_vertical_axis_before_release():
    assert hold_pregrasp_hover(
        [0.9, 0.3, 0.4],
        motion_frame=99,
        release_frame=100,
        hover_height=0.6,
    ) == [0.9, 0.3, 0.6]


def test_pregrasp_hover_preserves_higher_targets_and_releases_on_time():
    assert hold_pregrasp_hover(
        [0.9, 0.3, 0.7],
        motion_frame=99,
        release_frame=100,
        hover_height=0.6,
    ) == [0.9, 0.3, 0.7]
    assert hold_pregrasp_hover(
        [0.9, 0.3, 0.4],
        motion_frame=100,
        release_frame=100,
        hover_height=0.6,
    ) == [0.9, 0.3, 0.4]


def test_pregrasp_hover_rejects_invalid_targets():
    with pytest.raises(ValueError, match="three coordinates"):
        hold_pregrasp_hover(
            [0.9, 0.3],
            motion_frame=0,
            release_frame=1,
            hover_height=0.6,
        )


def test_repeated_calls_at_same_motion_frame_do_not_release_hover():
    for _ in range(16):
        assert hold_pregrasp_hover(
            [0.9, 0.3, 0.4],
            motion_frame=539,
            release_frame=540,
            hover_height=0.6,
        ) == [0.9, 0.3, 0.6]
