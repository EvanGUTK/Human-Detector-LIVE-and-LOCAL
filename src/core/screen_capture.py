"""Desktop screen capture via mss."""

from __future__ import annotations

import logging
import threading
import time

import cv2
import mss
import numpy as np

logger = logging.getLogger(__name__)


def list_monitors() -> list[dict]:
    with mss.mss() as sct:
        return [
            {
                "index": i + 1,
                "left": m["left"],
                "top": m["top"],
                "width": m["width"],
                "height": m["height"],
            }
            for i, m in enumerate(sct.monitors[1:])
        ]


class ScreenCapture:
    def __init__(
        self,
        monitor_index: int = 1,
        region: tuple[int, int, int, int] | None = None,
        fps_cap: float = 0.0,
    ) -> None:
        self.monitor_index = max(1, monitor_index)
        self.region = region
        # 0 = uncapped; otherwise clamp 5–144
        if fps_cap <= 0:
            self.fps_cap = 0.0
        else:
            self.fps_cap = max(5.0, min(float(fps_cap), 144.0))

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._frame_id = 0
        self._running = False
        self._error: str | None = None
        self._grab_rect: dict | None = None

    @property
    def source_type(self) -> str:
        return "screen"

    @property
    def source_label(self) -> str:
        if self.region:
            x, y, w, h = self.region
            return f"Screen M{self.monitor_index} ({w}x{h})"
        return f"Screen monitor {self.monitor_index}"

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def error(self) -> str | None:
        return self._error

    def _resolve_grab_rect(self, sct: mss.mss) -> dict:
        monitors = sct.monitors
        idx = min(self.monitor_index, len(monitors) - 1)
        mon = monitors[idx]
        if self.region:
            x, y, w, h = self.region
            return {"left": mon["left"] + x, "top": mon["top"] + y, "width": w, "height": h}
        return mon

    def start(self) -> bool:
        if self._running:
            return True
        self._stop.clear()
        self._error = None
        try:
            with mss.mss() as sct:
                self._grab_rect = self._resolve_grab_rect(sct)
        except Exception as exc:
            self._error = str(exc)
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
        with self._lock:
            self._latest = None

    def get_latest(self, *, copy: bool = True) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self._latest is None:
                return None, self._frame_id
            if copy:
                return self._latest.copy(), self._frame_id
            return self._latest, self._frame_id

    def _loop(self) -> None:
        delay = (1.0 / self.fps_cap) if self.fps_cap > 0 else 0.0
        with mss.mss() as sct:
            rect = self._resolve_grab_rect(sct)
            while not self._stop.is_set():
                t0 = time.perf_counter()
                try:
                    shot = sct.grab(rect)
                    frame = np.array(shot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    with self._lock:
                        self._latest = frame
                        self._frame_id += 1
                except Exception as exc:
                    logger.warning("Screen grab failed: %s", exc)
                    self._error = str(exc)
                if delay > 0:
                    elapsed = time.perf_counter() - t0
                    time.sleep(max(0.0, delay - elapsed))
