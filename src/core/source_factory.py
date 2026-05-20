"""Create frame source from config."""

from __future__ import annotations

from typing import Any

from src.core.capture import WebcamCapture
from src.core.file_capture import FileCapture
from src.core.frame_source import FrameSource
from src.core.screen_capture import ScreenCapture


def create_frame_source(cfg: dict[str, Any]) -> FrameSource:
    if cfg.get("input_source") == "screen":
        region = cfg.get("screen_region")
        reg_tuple = tuple(region) if region and len(region) == 4 else None
        return ScreenCapture(
            monitor_index=int(cfg.get("screen_monitor", 1)),
            region=reg_tuple,
            fps_cap=float(cfg.get("screen_fps_cap", 30)),
        )
    if cfg.get("input_source") == "file" and cfg.get("file_path"):
        return FileCapture(
            str(cfg["file_path"]),
            loop=bool(cfg.get("file_loop", True)),
            playback_speed=float(cfg.get("file_playback_speed", 1.0)),
        )
    return WebcamCapture(
        camera_index=int(cfg.get("camera_index", 0)),
        width=int(cfg.get("capture_width", 1280)),
        height=int(cfg.get("capture_height", 720)),
        mirror=bool(cfg.get("mirror", False)),
    )
