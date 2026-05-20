"""Evaluate tab: validation metrics and batch folder."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.detector import YoloDetector, yolo_kwargs_from_config
from src.lab.batch_evaluator import (
    batch_folder_detect,
    label_check_on_annotations,
    validate_project_ultralytics,
)
from src.lab.annotation_store import AnnotationStore
from src.lab.model_registry import ModelRegistry
from src.lab.project_manager import ProjectManager
from src.utils.config import app_data_dir, load_config


class EvaluatePanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.model_combo = QComboBox()
        self._fill_models()
        form.addRow("Evaluate model:", self.model_combo)
        layout.addLayout(form)
        layout.addWidget(QLabel("Project validation (Ultralytics mAP on val split)"))
        self.project_list = QListWidget()
        layout.addWidget(self.project_list)
        self.btn_val = QPushButton("Run validation")
        self.btn_val.clicked.connect(self._run_val)
        layout.addWidget(self.btn_val)
        self.val_result = QLabel("—")
        self.val_result.setWordWrap(True)
        layout.addWidget(self.val_result)

        layout.addWidget(QLabel("Label check (IoU vs your annotations)"))
        self.btn_label = QPushButton("Check labels on current project")
        self.btn_label.clicked.connect(self._label_check)
        layout.addWidget(self.btn_label)
        self.label_result = QLabel("—")
        layout.addWidget(self.label_result)

        layout.addWidget(QLabel("Batch folder detection"))
        self.btn_batch = QPushButton("Scan folder…")
        self.btn_batch.clicked.connect(self._batch_folder)
        layout.addWidget(self.btn_batch)
        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        layout.addWidget(self.batch_progress)
        self.batch_result = QLabel("—")
        layout.addWidget(self.batch_result)

        self._refresh_projects()

    def _fill_models(self) -> None:
        self.model_combo.clear()
        for m in ModelRegistry().list_models():
            self.model_combo.addItem(m.display_name, m.id)
        cfg = load_config()
        mid = str(cfg.get("active_model_id", "yolo11s"))
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == mid:
                self.model_combo.setCurrentIndex(i)
                break

    def _refresh_projects(self) -> None:
        self.project_list.clear()
        for p in ProjectManager().list_projects():
            self.project_list.addItem(f"{p.name} ({p.id})")

    def _selected_project(self):
        item = self.project_list.currentItem()
        if not item:
            return None
        pid = item.text().split("(")[-1].rstrip(")")
        return ProjectManager().get(pid)

    def _make_detector(self) -> YoloDetector:
        cfg = load_config()
        reg = ModelRegistry()
        mid = str(self.model_combo.currentData() or cfg.get("active_model_id", "yolo11s"))
        onnx = reg.resolve_onnx(mid, int(cfg.get("model_imgsz", 640)))
        return YoloDetector(
            onnx,
            model_name=mid,
            **yolo_kwargs_from_config(cfg),
        )

    def _run_val(self) -> None:
        proj = self._selected_project()
        if not proj or not proj.video_path:
            self.val_result.setText("Select a project with a video")
            return
        try:
            m = validate_project_ultralytics(proj.id, proj.video_path)
            self.val_result.setText(
                f"mAP50: {m['map50']:.4f} | mAP50-95: {m['map50_95']:.4f}\n"
                f"Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f}"
            )
        except Exception as exc:
            self.val_result.setText(f"Error: {exc}")

    def _label_check(self) -> None:
        proj = self._selected_project()
        if not proj:
            return
        try:
            det = self._make_detector()
            store = AnnotationStore(proj.id)
            r = label_check_on_annotations(det, store, proj.video_path)
            self.label_result.setText(
                f"P: {r['precision']:.3f} R: {r['recall']:.3f} "
                f"(TP={r['tp']} FP={r['fp']} FN={r['fn']})"
            )
        except Exception as exc:
            self.label_result.setText(f"Error: {exc}")

    def _batch_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder of videos/images")
        if not folder:
            return
        out = app_data_dir() / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            det = self._make_detector()
            self.batch_progress.setVisible(True)
            self.batch_progress.setRange(0, 0)
            batch_folder_detect(det, Path(folder), out)
            self.batch_progress.setVisible(False)
            self.batch_result.setText(f"Wrote {out}")
        except Exception as exc:
            self.batch_progress.setVisible(False)
            self.batch_result.setText(f"Error: {exc}")
