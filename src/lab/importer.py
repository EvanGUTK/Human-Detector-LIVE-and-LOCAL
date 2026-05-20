"""Import YOLO dataset into a lab project."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.lab.annotation_store import AnnotationStore, BBox
from src.lab.project_manager import ProjectManager


def import_yolo_folder(project_id: str, dataset_root: Path) -> int:
    """Import images+labels from standard YOLO layout; returns frame count."""
    store = AnnotationStore(project_id)
    count = 0
    idx = 0
    for split in ("train", "val"):
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.glob("*.*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.is_file():
                continue
            import cv2

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            boxes: list[BBox] = []
            for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                cname = "car" if cid == 1 else "person"
                boxes.append(BBox(cname, x1, y1, x2, y2, split=split))
            store.set_boxes(idx, boxes)
            idx += 1
            count += 1
    return count
