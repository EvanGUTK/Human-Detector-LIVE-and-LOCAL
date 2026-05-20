"""Background YOLO fine-tuning."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from src.lab.model_registry import ModelRegistry
from src.lab.paths import project_dir
from src.lab.yolo_exporter import export_yolo_dataset
from src.lab.annotation_store import AnnotationStore
from src.utils.model_setup import ensure_onnx_model

logger = logging.getLogger(__name__)


class TrainingWorker(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(str, str, float)
    failed = pyqtSignal(str)

    def __init__(
        self,
        project_id: str,
        project_name: str,
        video_path: str,
        base_model: str = "yolo11n",
        epochs: int = 50,
        imgsz: int = 640,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.project_id = project_id
        self.project_name = project_name
        self.video_path = video_path
        self.base_model = base_model
        self.epochs = epochs
        self.imgsz = imgsz
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            self.progress.emit("Exporting YOLO dataset…")
            store = AnnotationStore(self.project_id)
            indices = store.annotated_frame_indices()
            if len(indices) < 5:
                self.failed.emit("Annotate at least 5 frames before training.")
                return

            data_root = export_yolo_dataset(self.project_id, store, self.video_path)
            data_yaml = data_root / "data.yaml"

            self.progress.emit(f"Training {self.base_model} for {self.epochs} epochs…")
            from ultralytics import YOLO

            ensure_onnx_model(self.base_model, self.imgsz)
            pt = Path(f"{self.base_model}.pt")
            if not pt.is_file():
                from src.utils.config import models_dir

                alt = models_dir() / f"{self.base_model}.pt"
                pt = alt if alt.is_file() else pt

            runs_dir = project_dir(self.project_id) / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)

            if self._cancel:
                return

            model = YOLO(str(pt))
            results = model.train(
                data=str(data_yaml),
                epochs=self.epochs,
                imgsz=self.imgsz,
                project=str(runs_dir),
                name="train",
                exist_ok=True,
                verbose=False,
            )

            if self._cancel:
                return

            best_pt = runs_dir / "train" / "weights" / "best.pt"
            if not best_pt.is_file():
                self.failed.emit("Training finished but best.pt not found.")
                return

            self.progress.emit("Exporting ONNX…")
            yolo_best = YOLO(str(best_pt))
            export_path = yolo_best.export(format="onnx", imgsz=self.imgsz, simplify=True)
            onnx_out = project_dir(self.project_id) / "runs" / "best.onnx"
            Path(export_path).replace(onnx_out)

            map50 = 0.0
            try:
                metrics = yolo_best.val(data=str(data_yaml), verbose=False)
                map50 = float(getattr(metrics.box, "map50", 0) or 0)
            except Exception as exc:
                logger.warning("Val after train failed: %s", exc)

            model_id = f"custom_{self.project_id}"
            display = f"{self.project_name} (trained)"
            ModelRegistry().register_custom(
                model_id, display, onnx_out, best_pt, val_map50=map50
            )
            self.finished_ok.emit(model_id, str(onnx_out), map50)
        except Exception as exc:
            logger.exception("Training failed")
            self.failed.emit(str(exc))
