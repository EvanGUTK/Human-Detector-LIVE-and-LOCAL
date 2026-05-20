"""MediaPipe Tasks API — body pose + hand skeleton overlay."""

from __future__ import annotations

import logging
import urllib.request
from typing import Any

import cv2
import numpy as np

from src.utils.config import models_dir

logger = logging.getLogger(__name__)

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

_last_error: str | None = None


def skeleton_last_error() -> str | None:
    return _last_error


def _download_model(url: str, dest) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MediaPipe model: %s", dest.name)
    urllib.request.urlretrieve(url, dest)


def _ensure_model_file(name: str, url: str):
    path = models_dir() / name
    if not path.is_file():
        _download_model(url, path)
    return path


def _has_landmarks(data: Any) -> bool:
    return data is not None and len(data) > 0


class SkeletonOverlay:
    """Draw body pose and hand landmarks (MediaPipe 0.10+ Tasks API)."""

    def __init__(
        self,
        show_pose: bool = True,
        show_hands: bool = True,
        stride: int = 2,
    ) -> None:
        global _last_error
        self.show_pose = show_pose
        self.show_hands = show_hands
        self.stride = max(1, stride)
        self._hand_stride = 1
        self._frame_count = 0
        self._timestamp_ms = 0
        self._pose_landmarker: Any = None
        self._hand_landmarker: Any = None
        # Hands always use IMAGE mode (per-frame); pose uses VIDEO streaming.
        self._hand_image_mode = True
        self._last_pose: Any = None
        self._last_hands: Any = None
        self._enabled = False
        self.error: str | None = None

        if not (show_pose or show_hands):
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            self._mp = mp
            self._vision = vision

            if show_pose:
                pose_path = _ensure_model_file(
                    "pose_landmarker_lite.task", POSE_MODEL_URL
                )
                pose_options = vision.PoseLandmarkerOptions(
                    base_options=python.BaseOptions(
                        model_asset_path=str(pose_path)
                    ),
                    running_mode=vision.RunningMode.VIDEO,
                    min_pose_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._pose_landmarker = vision.PoseLandmarker.create_from_options(
                    pose_options
                )

            if show_hands:
                hand_path = _ensure_model_file(
                    "hand_landmarker.task", HAND_MODEL_URL
                )
                hand_options = vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(
                        model_asset_path=str(hand_path)
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.35,
                    min_hand_presence_confidence=0.35,
                    min_tracking_confidence=0.35,
                )
                self._hand_landmarker = vision.HandLandmarker.create_from_options(
                    hand_options
                )

            self._enabled = True
            _last_error = None
            logger.info("MediaPipe skeleton (pose=%s, hands=%s)", show_pose, show_hands)
        except Exception as exc:
            self.error = str(exc)
            _last_error = self.error
            logger.warning("MediaPipe init failed: %s", exc)
            self._enabled = False

    def reset(self) -> None:
        self._timestamp_ms = 0
        self._frame_count = 0
        self._last_pose = None
        self._last_hands = None

    def close(self) -> None:
        if self._pose_landmarker is not None:
            self._pose_landmarker.close()
            self._pose_landmarker = None
        if self._hand_landmarker is not None:
            self._hand_landmarker.close()
            self._hand_landmarker = None

    def apply(
        self,
        frame_bgr: np.ndarray,
        person_boxes: list[tuple[float, float, float, float]] | None = None,
    ) -> np.ndarray:
        if not self._enabled:
            return frame_bgr

        self._frame_count += 1
        run_pose = (
            self._pose_landmarker is not None
            and self._frame_count % self.stride == 0
        )
        run_hands = (
            self._hand_landmarker is not None
            and self._frame_count % self._hand_stride == 0
        )

        if not run_pose and not run_hands:
            return self._draw_cached(frame_bgr)

        rgb_full = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        h, w = rgb_full.shape[:2]

        if run_pose:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=rgb_full
            )
            self._timestamp_ms += 33
            result = self._pose_landmarker.detect_for_video(
                mp_image, self._timestamp_ms
            )
            self._last_pose = result.pose_landmarks

        if run_hands:
            self._last_hands = self._detect_hands(rgb_full, w, h, person_boxes or [])

        return self._draw_cached(frame_bgr)

    def _detect_hands(
        self,
        rgb_full: np.ndarray,
        width: int,
        height: int,
        person_boxes: list[tuple[float, float, float, float]],
    ) -> list[Any]:
        """Detect hands on full frame plus expanded YOLO person boxes."""
        all_hands: list[Any] = []
        regions: list[tuple[int, int, int, int]] = [(0, 0, width, height)]

        for x1, y1, x2, y2 in person_boxes:
            pad_x = (x2 - x1) * 0.25
            pad_y = (y2 - y1) * 0.25
            regions.append(
                (
                    int(max(0, x1 - pad_x)),
                    int(max(0, y1 - pad_y)),
                    int(min(width, x2 + pad_x)),
                    int(min(height, y2 + pad_y)),
                )
            )

        seen: set[tuple[int, int, int, int]] = set()
        for cx1, cy1, cx2, cy2 in regions:
            if (cx2 - cx1) < 48 or (cy2 - cy1) < 48:
                continue
            if (cx1, cy1, cx2, cy2) in seen:
                continue
            seen.add((cx1, cy1, cx2, cy2))

            crop = rgb_full[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(crop),
            )
            result = self._hand_landmarker.detect(mp_image)
            if not result.hand_landmarks:
                continue

            cw, ch = cx2 - cx1, cy2 - cy1
            for hand_lm in result.hand_landmarks:
                if cx1 == 0 and cy1 == 0 and cx2 == width and cy2 == height:
                    all_hands.append(hand_lm)
                else:
                    all_hands.append(
                        self._landmarks_crop_to_full(
                            hand_lm, cx1, cy1, cw, ch, width, height
                        )
                    )
        return all_hands

    @staticmethod
    def _landmarks_crop_to_full(
        hand_lm: Any,
        ox: int,
        oy: int,
        cw: int,
        ch: int,
        full_w: int,
        full_h: int,
    ) -> list:
        from mediapipe.tasks.python.components.containers import landmark as lm_mod

        out = []
        for lm in hand_lm:
            px = lm.x * cw + ox
            py = lm.y * ch + oy
            out.append(
                lm_mod.NormalizedLandmark(
                    x=px / full_w,
                    y=py / full_h,
                    z=lm.z,
                    visibility=getattr(lm, "visibility", None),
                    presence=getattr(lm, "presence", None),
                )
            )
        return out

    def _draw_cached(self, frame_bgr: np.ndarray) -> np.ndarray:
        out = frame_bgr.copy()
        if not _has_landmarks(self._last_pose) and not _has_landmarks(self._last_hands):
            return out

        try:
            from mediapipe.tasks.python.vision import drawing_utils
            from mediapipe.tasks.python.vision.hand_landmarker import (
                HandLandmarksConnections,
            )
            from mediapipe.tasks.python.vision.pose_landmarker import (
                PoseLandmarksConnections,
            )

            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

            if _has_landmarks(self._last_pose):
                for landmarks in self._last_pose:
                    drawing_utils.draw_landmarks(
                        rgb,
                        landmarks,
                        PoseLandmarksConnections.POSE_LANDMARKS,
                        drawing_utils.DrawingSpec(
                            color=(0, 255, 255), thickness=2, circle_radius=2
                        ),
                        drawing_utils.DrawingSpec(
                            color=(255, 128, 0), thickness=2, circle_radius=1
                        ),
                    )

            if _has_landmarks(self._last_hands):
                for landmarks in self._last_hands:
                    drawing_utils.draw_landmarks(
                        rgb,
                        landmarks,
                        HandLandmarksConnections.HAND_LANDMARKS,
                        drawing_utils.DrawingSpec(
                            color=(255, 0, 255), thickness=2, circle_radius=3
                        ),
                        drawing_utils.DrawingSpec(
                            color=(0, 200, 255), thickness=2, circle_radius=2
                        ),
                    )

            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            logger.warning("Skeleton draw failed: %s", exc)
            return out
