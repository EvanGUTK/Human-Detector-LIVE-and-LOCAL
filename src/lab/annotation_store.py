"""Per-frame bounding box annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.lab.paths import project_dir


@dataclass
class BBox:
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    split: str = "train"

    def to_norm_dict(self, img_w: int, img_h: int) -> dict[str, Any]:
        return {
            "class": self.class_name,
            "xyxy_norm": [
                self.x1 / img_w,
                self.y1 / img_h,
                self.x2 / img_w,
                self.y2 / img_h,
            ],
            "split": self.split,
        }

    @classmethod
    def from_norm_dict(cls, data: dict[str, Any], img_w: int, img_h: int) -> BBox:
        x1, y1, x2, y2 = data["xyxy_norm"]
        return cls(
            class_name=str(data.get("class", "person")),
            x1=float(x1) * img_w,
            y1=float(y1) * img_h,
            x2=float(x2) * img_w,
            y2=float(y2) * img_h,
            split=str(data.get("split", "train")),
        )


@dataclass
class FrameAnnotation:
    frame_index: int
    boxes: list[BBox] = field(default_factory=list)


class AnnotationStore:
    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self._path = project_dir(project_id) / "annotations.json"
        self._frames: dict[int, FrameAnnotation] = {}
        self.load()

    def load(self) -> None:
        self._frames.clear()
        if not self._path.is_file():
            return
        with self._path.open(encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("frames", []):
            fi = int(entry["frame_index"])
            boxes = []
            for b in entry.get("boxes", []):
                boxes.append(
                    BBox(
                        class_name=str(b.get("class", "person")),
                        x1=float(b["x1"]),
                        y1=float(b["y1"]),
                        x2=float(b["x2"]),
                        y2=float(b["y2"]),
                        split=str(b.get("split", "train")),
                    )
                )
            self._frames[fi] = FrameAnnotation(fi, boxes)

    def save(self) -> None:
        payload = {
            "frames": [
                {
                    "frame_index": fa.frame_index,
                    "boxes": [
                        {
                            "class": b.class_name,
                            "x1": b.x1,
                            "y1": b.y1,
                            "x2": b.x2,
                            "y2": b.y2,
                            "split": b.split,
                        }
                        for b in fa.boxes
                    ],
                }
                for fa in sorted(self._frames.values(), key=lambda x: x.frame_index)
            ]
        }
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def get_frame(self, frame_index: int) -> FrameAnnotation:
        if frame_index not in self._frames:
            self._frames[frame_index] = FrameAnnotation(frame_index, [])
        return self._frames[frame_index]

    def set_boxes(self, frame_index: int, boxes: list[BBox]) -> None:
        self._frames[frame_index] = FrameAnnotation(frame_index, boxes)
        self.save()

    def annotated_frame_indices(self) -> list[int]:
        return sorted(
            fi for fi, fa in self._frames.items() if fa.boxes
        )

    def copy_to_next(self, frame_index: int) -> None:
        fa = self.get_frame(frame_index)
        if not fa.boxes:
            return
        import copy

        nxt = frame_index + 1
        self._frames[nxt] = FrameAnnotation(
            nxt, [copy.deepcopy(b) for b in fa.boxes]
        )
        self.save()
