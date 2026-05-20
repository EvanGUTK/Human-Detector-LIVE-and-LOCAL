"""Built-in and custom trained models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.lab.paths import registry_path
from src.lab.ngc_download import ensure_ngc_model_onnx, tao_onnx_path
from src.utils.config import models_dir
from src.utils.model_setup import ensure_onnx_model, onnx_path_for_model

TAO_CLASS_NAMES = {
    "peoplenet": ["person", "bag", "face"],
    "detectnet_v2": ["person", "bag", "face"],
}


@dataclass
class ModelEntry:
    id: str
    display_name: str
    onnx_path: str
    pt_path: str = ""
    is_builtin: bool = True
    val_map50: float | None = None
    family: str = "yolo"
    class_names: list[str] | None = None

    def resolved_onnx(self) -> Path:
        return Path(self.onnx_path)


class ModelRegistry:
    def __init__(self) -> None:
        self._custom: list[ModelEntry] = []
        self._load()

    def _load(self) -> None:
        self._custom.clear()
        path = registry_path()
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("custom", []):
            self._custom.append(
                ModelEntry(
                    id=str(item["id"]),
                    display_name=str(item.get("display_name", item["id"])),
                    onnx_path=str(item["onnx_path"]),
                    pt_path=str(item.get("pt_path", "")),
                    is_builtin=False,
                    val_map50=item.get("val_map50"),
                    family=str(item.get("family", "yolo")),
                    class_names=item.get("class_names"),
                )
            )

    def _save(self) -> None:
        payload = {
            "custom": [
                {
                    "id": e.id,
                    "display_name": e.display_name,
                    "onnx_path": e.onnx_path,
                    "pt_path": e.pt_path,
                    "val_map50": e.val_map50,
                    "family": e.family,
                    "class_names": e.class_names,
                }
                for e in self._custom
            ]
        }
        with registry_path().open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def list_models(self) -> list[ModelEntry]:
        builtin_ids = (
            "yolo11n",
            "yolo11s",
            "yolo11m",
            "yolo11l",
            "yolo11x",
            "yolov8m",
            "rtdetr-s",
            "peoplenet",
            "detectnet_v2",
        )
        labels = {
            "yolo11n": "YOLO11n (nano)",
            "yolo11s": "YOLO11s (small)",
            "yolo11m": "YOLO11m (medium, CCTV)",
            "yolo11l": "YOLO11l (large)",
            "yolo11x": "YOLO11x (extra large)",
            "yolov8m": "YOLOv8m (classic vehicles baseline)",
            "rtdetr-s": "RT-DETR-s (transformer rival)",
            "peoplenet": "PeopleNet (NVIDIA TAO)",
            "detectnet_v2": "DetectNet_v2 (NVIDIA TAO)",
        }
        builtins = [
            ModelEntry(
                bid,
                labels.get(bid, bid),
                str(
                    tao_onnx_path(bid)
                    if bid in TAO_CLASS_NAMES
                    else onnx_path_for_model(bid, 640)
                ),
                family=(
                    "rtdetr"
                    if bid.startswith("rtdetr")
                    else ("peoplenet" if bid == "peoplenet" else ("detectnet_v2" if bid == "detectnet_v2" else "yolo"))
                ),
                class_names=(
                    TAO_CLASS_NAMES.get(bid)
                    if bid in TAO_CLASS_NAMES
                    else None
                ),
            )
            for bid in builtin_ids
        ]
        return builtins + self._custom

    def get(self, model_id: str) -> ModelEntry | None:
        for m in self.list_models():
            if m.id == model_id:
                return m
        return None

    def register_custom(
        self,
        model_id: str,
        display_name: str,
        onnx_path: Path,
        pt_path: Path | None = None,
        val_map50: float | None = None,
        family: str = "yolo",
        class_names: list[str] | None = None,
    ) -> None:
        self._custom = [c for c in self._custom if c.id != model_id]
        self._custom.append(
            ModelEntry(
                id=model_id,
                display_name=display_name,
                onnx_path=str(onnx_path),
                pt_path=str(pt_path) if pt_path else "",
                is_builtin=False,
                val_map50=val_map50,
                family=family,
                class_names=class_names,
            )
        )
        self._save()

    def resolve_onnx(self, model_id: str, imgsz: int = 640) -> Path:
        if model_id in ("peoplenet", "detectnet_v2"):
            return ensure_ngc_model_onnx(model_id)
        entry = self.get(model_id)
        if entry is None:
            return ensure_onnx_model(model_id, imgsz)
        path = entry.resolved_onnx()
        if path.is_file():
            return path
        if entry.is_builtin:
            return ensure_onnx_model(model_id, imgsz)
        raise FileNotFoundError(f"Model not found: {model_id}")

    def base_model_name(self, model_id: str) -> str:
        builtins = (
            "yolo11n",
            "yolo11s",
            "yolo11m",
            "yolo11l",
            "yolo11x",
            "yolov8m",
            "rtdetr-s",
            "peoplenet",
            "detectnet_v2",
        )
        if model_id in builtins:
            return model_id
        entry = self.get(model_id)
        if entry and entry.pt_path:
            return Path(entry.pt_path).stem
        return "yolo11n"

    def class_names_for_model(self, model_id: str) -> list[str]:
        m = self.get(model_id)
        if m and m.class_names:
            return list(m.class_names)
        from src.utils.coco_classes import COCO80_NAMES

        return list(COCO80_NAMES)
