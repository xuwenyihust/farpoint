import pytest

from farpoint.so101_runtime import resolve_headless_mode


@pytest.mark.parametrize(
    ("mode", "livestream", "livestream_env", "expected"),
    [
        ("headless", 0, None, True),
        ("viewer", 0, None, False),
        ("viewer", 1, None, True),
        ("viewer", 2, None, True),
        ("viewer", -1, "2", True),
        ("viewer", -1, "0", False),
        ("viewer", -1, None, False),
    ],
)
def test_resolve_headless_mode_preserves_livestream_semantics(
    mode, livestream, livestream_env, expected
):
    assert (
        resolve_headless_mode(
            mode, livestream, livestream_env=livestream_env
        )
        is expected
    )


def test_resolve_headless_mode_rejects_unknown_collector_mode():
    with pytest.raises(ValueError, match="unsupported SO-101 runtime mode"):
        resolve_headless_mode("invalid", 0)
