"""Export annotations to YOLO dataset layout."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import yaml

from src.lab.annotation_store import AnnotationStore
from src.lab.paths import project_dir
from src.utils.config import CAR_CLASS_ID, PERSON_CLASS_ID

# Legacy two-class training labels (reindexed 0,1 in export)
CLASS_TO_YOLO = {"person": 0, "car": 1}
YOLO_TO_COCO = {0: PERSON_CLASS_ID, 1: CAR_CLASS_ID}


def _collect_class_names(store: AnnotationStore) -> list[str]:
    names: set[str] = set()
    for fi in store.annotated_frame_indices():
        fa = store.get_frame(fi)
        for b in fa.boxes:
            names.add(b.class_name.strip().lower())
    return sorted(names)


def _build_class_map(class_names: list[str]) -> dict[str, int]:
    """Map annotation class name (lower) -> contiguous YOLO id 0..K-1."""
    if not class_names:
        return {"person": 0}
    if set(class_names) <= {"person", "car"}:
        m: dict[str, int] = {}
        if "person" in class_names:
            m["person"] = 0
        if "car" in class_names:
            m["car"] = 1
        return m
    return {n: i for i, n in enumerate(class_names)}


def export_yolo_dataset(
    project_id: str,
    store: AnnotationStore,
    video_path: str,
) -> Path:
    root = project_dir(project_id) / "exports" / "yolo"
    if root.is_dir():
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    raw_names = _collect_class_names(store)
    cmap = _build_class_map(raw_names)

    indices = store.annotated_frame_indices()
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        fa = store.get_frame(fi)
        split = fa.boxes[0].split if fa.boxes else "train"
        h, w = frame.shape[:2]
        stem = f"frame_{fi:06d}"
        img_path = root / "images" / split / f"{stem}.jpg"
        lbl_path = root / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(img_path), frame)
        lines = []
        for b in fa.boxes:
            key = b.class_name.strip().lower()
            cid = cmap.get(key)
            if cid is None:
                raise KeyError(
                    f"Unknown class {key!r} in frame {fi}; known: {sorted(cmap)}"
                )
            xc = ((b.x1 + b.x2) / 2) / w
            yc = ((b.y1 + b.y2) / 2) / h
            bw = abs(b.x2 - b.x1) / w
            bh = abs(b.y2 - b.y1) / h
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    cap.release()

    inv = [""] * len(cmap)
    for name, yi in cmap.items():
        inv[yi] = name
    for i, n in enumerate(inv):
        if not n:
            inv[i] = f"class_{i}"

    data_yaml = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: inv[i] for i in range(len(inv))},
        "nc": len(inv),
    }
    with (root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, default_flow_style=False)

    return root
