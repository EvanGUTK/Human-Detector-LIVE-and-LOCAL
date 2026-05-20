"""Resize frames for inference and map detection coordinates back to capture space."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.core.detector import Detection


@dataclass(frozen=True)
class InferScale:
    """Maps capture coordinates to the frame passed to the detector."""

    capture_w: int
    capture_h: int
    infer_w: int
    infer_h: int
    scale_x: float
    scale_y: float

    @property
    def active(self) -> bool:
        return self.scale_x != 1.0 or self.scale_y != 1.0


def _effective_max(max_w: int, max_h: int, cap_w: int, cap_h: int) -> tuple[int, int]:
    mw = int(max_w) if int(max_w) > 0 else cap_w
    mh = int(max_h) if int(max_h) > 0 else cap_h
    return mw, mh


def resize_for_infer(
    frame_bgr: np.ndarray,
    max_width: int = 0,
    max_height: int = 0,
) -> tuple[np.ndarray, InferScale]:
    """Downscale preserving aspect so neither side exceeds max_width/max_height (0 = no cap)."""
    h, w = frame_bgr.shape[:2]
    mw, mh = _effective_max(max_width, max_height, w, h)
    if w <= mw and h <= mh:
        return frame_bgr, InferScale(w, h, w, h, 1.0, 1.0)
    scale = min(mw / w, mh / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    sx = w / nw
    sy = h / nh
    return resized, InferScale(w, h, nw, nh, sx, sy)


def scale_detections(detections: list[Detection], scale: InferScale) -> list[Detection]:
    if not scale.active:
        return detections
    sx, sy = scale.scale_x, scale.scale_y
    out: list[Detection] = []
    for d in detections:
        out.append(
            Detection(
                x1=d.x1 * sx,
                y1=d.y1 * sy,
                x2=d.x2 * sx,
                y2=d.y2 * sy,
                confidence=d.confidence,
                class_id=d.class_id,
                display_name=d.display_name,
            )
        )
    return out


def upscale_frame_for_display(frame_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    if frame_bgr.shape[1] == target_w and frame_bgr.shape[0] == target_h:
        return frame_bgr
    return cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
