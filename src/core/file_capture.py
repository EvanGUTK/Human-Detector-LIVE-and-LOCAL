"""Threaded video file playback with latest-frame buffer."""

from __future__ import annotations

from src.core.file_playback import VideoFilePlayback


class FileCapture(VideoFilePlayback):
    """Frame source adapter for Monitor pipeline."""

    def __init__(
        self,
        file_path: str,
        loop: bool = True,
        playback_speed: float = 1.0,
    ) -> None:
        super().__init__(file_path, loop=loop, playback_speed=playback_speed)
