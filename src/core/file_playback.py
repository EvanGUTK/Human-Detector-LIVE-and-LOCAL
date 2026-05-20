"""Shared video file playback (pause, seek, speed) for FileCapture and Compare."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoFilePlayback:
    """Thread-safe OpenCV video reader with transport controls."""

    def __init__(
        self,
        file_path: str,
        *,
        loop: bool = True,
        playback_speed: float = 1.0,
    ) -> None:
        self.file_path = str(file_path)
        self.loop = loop
        self.playback_speed = max(0.25, min(playback_speed, 4.0))
        self._name = Path(self.file_path).name
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._playback_pause = threading.Event()
        self._stop = threading.Event()
        self._latest: np.ndarray | None = None
        self._frame_id = 0
        self._frame_index = 0
        self._frame_count = 0
        self._fps = 30.0
        self._running = False
        self._error: str | None = None
        self._thread: threading.Thread | None = None
        self._seek_request: int | None = None

    @property
    def supports_playback(self) -> bool:
        return True

    @property
    def source_type(self) -> str:
        return "file"

    @property
    def source_label(self) -> str:
        return self._name

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def video_fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def current_frame_index(self) -> int:
        return self._frame_index

    @property
    def is_playback_paused(self) -> bool:
        return self._playback_pause.is_set()

    def open(self) -> bool:
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.file_path)
        if not self._cap.isOpened():
            self._error = f"Cannot open video: {self.file_path}"
            return False
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 1 else 30.0
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self._error = None
        return True

    def start(self) -> bool:
        if self._running:
            return True
        self._stop.clear()
        if self._cap is None and not self.open():
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._latest = None

    def get_latest(self, *, copy: bool = True) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self._latest is None:
                return None, self._frame_id
            if copy:
                return self._latest.copy(), self._frame_id
            return self._latest, self._frame_id

    def pause_playback(self, paused: bool) -> None:
        if paused:
            self._playback_pause.set()
        else:
            self._playback_pause.clear()

    def set_playback_speed(self, speed: float) -> None:
        self.playback_speed = max(0.25, min(speed, 4.0))

    def set_loop(self, loop: bool) -> None:
        self.loop = loop

    def seek_frame(self, index: int) -> None:
        self._seek_request = max(0, index)

    def _apply_seek(self, target: int) -> None:
        if self._cap is None:
            return
        if self._frame_count > 0:
            target = min(target, max(0, self._frame_count - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = self._cap.read()
        if ok:
            with self._lock:
                self._latest = frame
                self._frame_index = target
                self._frame_id += 1
        self._seek_request = None

    def _loop(self) -> None:
        import time

        delay = (1.0 / self._fps) / self.playback_speed if self._fps > 0 else 0.033
        while not self._stop.is_set() and self._cap is not None:
            if self._seek_request is not None:
                self._apply_seek(self._seek_request)
                continue
            if self._playback_pause.is_set():
                time.sleep(0.02)
                continue
            t0 = time.perf_counter()
            ok, frame = self._cap.read()
            if not ok:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._frame_index = 0
                    continue
                self._error = "End of video file"
                break
            with self._lock:
                self._latest = frame
                self._frame_id += 1
                self._frame_index = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if self._frame_index < 0:
                    self._frame_index = 0
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, delay - elapsed))
