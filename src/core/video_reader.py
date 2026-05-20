"""Random-access video frame reader for annotation."""

from __future__ import annotations

import cv2
import numpy as np


class VideoReader:
    def __init__(self, path: str) -> None:
        self.path = path
        self._cap = cv2.VideoCapture(path)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap.isOpened() else 0
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def is_open(self) -> bool:
        return self._cap.isOpened()

    @property
    def frame_count(self) -> int:
        return max(0, self._frame_count)

    @property
    def fps(self) -> float:
        return float(self._fps)

    def read_frame(self, index: int) -> np.ndarray | None:
        if not self._cap.isOpened() or index < 0:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
