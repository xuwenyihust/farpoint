"""Atomically seal and verify raw episode RGB streams as MP4 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _run(command: list[str], *, runner: CommandRunner) -> subprocess.CompletedProcess[str]:
    return runner(command, check=True, capture_output=True, text=True)


def seal_rgb_video(
    episode_root: str | Path,
    *,
    camera_id: str,
    frame_count: int,
    fps: int = 30,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Encode one contiguous PNG stream, fully decode it, then publish atomically."""
    if camera_id not in {"front", "wrist"}:
        raise ValueError("camera_id must be front or wrist")
    if frame_count <= 0 or fps <= 0:
        raise ValueError("frame_count and fps must be positive")
    root = Path(episode_root)
    missing = [
        index
        for index in range(frame_count)
        if not (root / "rgb" / f"{camera_id}_{index:06d}.png").is_file()
    ]
    if missing:
        raise ValueError(
            f"{camera_id} RGB stream is not contiguous; first missing frame={missing[0]}"
        )

    video_dir = root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    destination = video_dir / f"{camera_id}.mp4"
    temporary = video_dir / f".{camera_id}.mp4.partial"
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"video artifact already exists for {camera_id}")
    _run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-start_number",
            "0",
            "-i",
            str(root / "rgb" / f"{camera_id}_%06d.png"),
            "-frames:v",
            str(frame_count),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(temporary),
        ],
        runner=runner,
    )
    probe = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            str(temporary),
        ],
        runner=runner,
    )
    streams = json.loads(probe.stdout).get("streams") or []
    if len(streams) != 1:
        raise ValueError(f"{camera_id} MP4 must contain exactly one video stream")
    stream = streams[0]
    decoded_frames = int(stream.get("nb_read_frames", -1))
    width = int(stream.get("width", -1))
    height = int(stream.get("height", -1))
    if (decoded_frames, width, height) != (frame_count, 640, 480):
        raise ValueError(
            f"{camera_id} MP4 audit mismatch: "
            f"frames={decoded_frames}, width={width}, height={height}"
        )
    _run(
        ["ffmpeg", "-loglevel", "error", "-i", str(temporary), "-f", "null", "-"],
        runner=runner,
    )
    temporary.replace(destination)
    payload = destination.read_bytes()
    return {
        "path": str(destination.relative_to(root)),
        "container": "mp4",
        "codec": "h264",
        "frame_count": decoded_frames,
        "width": width,
        "height": height,
        "fps": fps,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "decode_verified": True,
    }
