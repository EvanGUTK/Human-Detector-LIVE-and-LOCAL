"""Batch folder evaluation and label validation."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2
import numpy as np

from src.core.detector import YoloDetector
from src.lab.annotation_store import AnnotationStore
from src.lab.yolo_exporter import export_yolo_dataset
from src.lab.paths import project_dir

logger = logging.getLogger(__name__)


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def validate_project_ultralytics(project_id: str, video_path: str) -> dict[str, float]:
    store = AnnotationStore(project_id)
    root = export_yolo_dataset(project_id, store, video_path)
    data_yaml = root / "data.yaml"
    from ultralytics import YOLO

    best = project_dir(project_id) / "runs" / "train" / "weights" / "best.pt"
    if best.is_file():
        model = YOLO(str(best))
    else:
        model = YOLO("yolo11n.pt")
    metrics = model.val(data=str(data_yaml), verbose=False)
    return {
        "map50": float(getattr(metrics.box, "map50", 0) or 0),
        "map50_95": float(getattr(metrics.box, "map", 0) or 0),
        "precision": float(getattr(metrics.box, "mp", 0) or 0),
        "recall": float(getattr(metrics.box, "mr", 0) or 0),
    }


def batch_folder_detect(
    detector: YoloDetector,
    folder: Path,
    out_csv: Path,
) -> None:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mkv"}
    rows: list[list] = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in exts:
            continue
        if path.suffix.lower() in (".mp4", ".avi", ".mkv"):
            cap = cv2.VideoCapture(str(path))
            fidx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                dets = detector.detect(frame)
                persons = sum(1 for d in dets if d.class_id == 0)
                cars = sum(1 for d in dets if d.class_id == 2)
                rows.append([str(path), fidx, persons, cars, len(dets)])
                fidx += 1
            cap.release()
        else:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            dets = detector.detect(frame)
            persons = sum(1 for d in dets if d.class_id == 0)
            cars = sum(1 for d in dets if d.class_id == 2)
            rows.append([str(path), 0, persons, cars, len(dets)])

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "frame", "person_count", "car_count", "total_detections"])
        w.writerows(rows)


def label_check_on_annotations(
    detector: YoloDetector,
    store: AnnotationStore,
    video_path: str,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    tp, fp, fn = 0, 0, 0
    for fi in store.annotated_frame_indices():
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        fa = store.get_frame(fi)
        preds = detector.detect(frame)
        gt = [(b.class_name, (b.x1, b.y1, b.x2, b.y2)) for b in fa.boxes]
        matched_gt: set[int] = set()
        for d in preds:
            cname = d.class_name
            pb = (d.x1, d.y1, d.x2, d.y2)
            best_iou, best_j = 0.0, -1
            for j, (gcn, gb) in enumerate(gt):
                if gcn != cname:
                    continue
                iou = _iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold and best_j >= 0:
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1
        fn += len(gt) - len(matched_gt)
    cap.release()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}
