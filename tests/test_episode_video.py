from pathlib import Path
import subprocess

import pytest

from farpoint.episode_video import seal_rgb_video


def test_episode_video_rejects_non_contiguous_frames(tmp_path: Path):
    (tmp_path / "rgb").mkdir()
    (tmp_path / "rgb" / "front_000000.png").write_bytes(b"png")
    with pytest.raises(ValueError, match="first missing frame=1"):
        seal_rgb_video(tmp_path, camera_id="front", frame_count=2)


def test_episode_video_encodes_probes_decodes_and_publishes_atomically(tmp_path: Path):
    rgb = tmp_path / "rgb"
    rgb.mkdir()
    for index in range(2):
        (rgb / f"wrist_{index:06d}.png").write_bytes(b"png")
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        if command[0] == "ffmpeg" and "-frames:v" in command:
            Path(command[-1]).write_bytes(b"verified-mp4")
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(
                command,
                0,
                '{"streams":[{"width":640,"height":480,"nb_read_frames":"2"}]}',
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    artifact = seal_rgb_video(
        tmp_path,
        camera_id="wrist",
        frame_count=2,
        runner=runner,
    )
    assert artifact["path"] == "videos/wrist.mp4"
    assert artifact["frame_count"] == 2
    assert artifact["decode_verified"] is True
    assert (tmp_path / artifact["path"]).read_bytes() == b"verified-mp4"
    assert not (tmp_path / "videos" / ".wrist.mp4.partial").exists()
    assert [command[0] for command in commands] == ["ffmpeg", "ffprobe", "ffmpeg"]
