"""Multi-object tracking with ByteTrack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

from src.utils.config import CLASS_NAMES, PERSON_CLASS_ID

from src.core.detector import Detection


@dataclass
class TrackedObject:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    track_id: int
    class_id: int = PERSON_CLASS_ID
    identity: str = "Unknown"
    display_name: str | None = None

    @property
    def class_name(self) -> str:
        if self.display_name:
            return self.display_name
        return CLASS_NAMES.get(self.class_id, f"class_{self.class_id}")

    @property
    def feet_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def feet_y(self) -> float:
        return self.y2

    @property
    def is_person(self) -> bool:
        return self.class_id == PERSON_CLASS_ID


# Backward compatibility
TrackedPerson = TrackedObject


class ObjectTracker:
    def __init__(self) -> None:
        self._tracker = sv.ByteTrack()
        self.seen_ids: set[int] = set()

    def reset(self) -> None:
        self._tracker = sv.ByteTrack()
        self.seen_ids.clear()

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        if not detections:
            empty = sv.Detections.empty()
            tracked = self._tracker.update_with_detections(empty)
            detections_for_meta: list[Detection] = []
        else:
            xyxy = np.array(
                [[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32
            )
            conf = np.array([d.confidence for d in detections], dtype=np.float32)
            class_id = np.array([d.class_id for d in detections], dtype=int)
            dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)
            tracked = self._tracker.update_with_detections(dets)
            detections_for_meta = detections

        results: list[TrackedObject] = []
        if tracked.xyxy is None or len(tracked.xyxy) == 0:
            return results

        for i in range(len(tracked.xyxy)):
            tid = tracked.tracker_id[i] if tracked.tracker_id is not None else None
            if tid is None:
                continue
            tid = int(tid)
            self.seen_ids.add(tid)
            x1, y1, x2, y2 = tracked.xyxy[i]
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            cid = int(tracked.class_id[i]) if tracked.class_id is not None else PERSON_CLASS_ID
            disp = None
            if i < len(detections_for_meta):
                disp = detections_for_meta[i].display_name
            results.append(
                TrackedObject(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=conf,
                    track_id=tid,
                    class_id=cid,
                    display_name=disp,
                )
            )
        return results


PersonTracker = ObjectTracker
