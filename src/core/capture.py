"""Threaded webcam capture with latest-frame buffer."""

from __future__ import annotations

import logging
import threading
import time
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class WebcamCapture:
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        mirror: bool = False,
        max_reconnect_attempts: int = 3,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.mirror = mirror
        self.max_reconnect_attempts = max_reconnect_attempts

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._frame_id = 0
        self._running = False
        self._error: str | None = None
        self._fail_count = 0

    @property
    def source_type(self) -> str:
        return "webcam"

    @property
    def source_label(self) -> str:
        return f"Camera {self.camera_index}"

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> bool:
        if self._running:
            return True
        self._stop.clear()
        self._error = None
        if not self._open_capture():
            self._error = f"Cannot open camera index {self.camera_index}"
            logger.error(self._error)
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

    def _open_capture(self) -> bool:
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def _try_reconnect(self) -> bool:
        for attempt in range(self.max_reconnect_attempts):
            logger.warning("Webcam reconnect attempt %s", attempt + 1)
            time.sleep(0.5 * (attempt + 1))
            if self._open_capture():
                self._fail_count = 0
                self._error = None
                return True
        self._error = f"Camera {self.camera_index} lost — reconnect failed"
        return False

    def _loop(self) -> None:
        while not self._stop.is_set() and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                self._fail_count += 1
                if self._fail_count >= 30:
                    if not self._try_reconnect():
                        break
                time.sleep(0.01)
                continue
            self._fail_count = 0
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._latest = frame
                self._frame_id += 1
