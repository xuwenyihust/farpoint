"""Runtime launch decisions shared by the SO-101 Isaac collector."""

from __future__ import annotations


def resolve_headless_mode(
    mode: str, livestream: int, *, livestream_env: str | None = None
) -> bool:
    """Keep WebRTC launches headless even when the collector is in viewer mode.

    Isaac Lab's AppLauncher treats both livestream modes as headless.  The
    collector still uses ``mode=viewer`` to distinguish an interactive
    inspection run from a batch collection, but that label must not re-enable
    an unavailable X11 window after AppLauncher has resolved WebRTC.
    """
    if mode not in {"headless", "viewer"}:
        raise ValueError(f"unsupported SO-101 runtime mode: {mode}")
    effective_livestream = livestream
    if livestream == -1 and livestream_env in {"0", "1", "2"}:
        effective_livestream = int(livestream_env)
    return mode == "headless" or effective_livestream in {1, 2}
