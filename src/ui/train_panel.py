"""Train tab: annotate video, export, fine-tune YOLO."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.core.video_reader import VideoReader
from src.lab.annotation_store import AnnotationStore, BBox
from src.lab.dataset_manager import assign_splits, split_counts
from src.lab.exporter import export_yolo_zip
from src.lab.importer import import_yolo_folder
from src.lab.project_manager import ProjectManager
from src.lab.training_worker import TrainingWorker
from src.utils.coco_classes import COCO80_NAMES
from src.ui.video_widget import VideoWidget


class TrainPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pm = ProjectManager()
        self._project = None
        self._store: AnnotationStore | None = None
        self._reader: VideoReader | None = None
        self._current_frame = 0
        self._active_class = "person"
        self._worker: TrainingWorker | None = None
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)

        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.addWidget(QLabel("Projects"))
        self.project_list = QListWidget()
        self.project_list.currentTextChanged.connect(self._on_project_selected)
        left_l.addWidget(self.project_list)
        row = QHBoxLayout()
        self.btn_new = QPushButton("New")
        self.btn_del = QPushButton("Delete")
        self.btn_new.clicked.connect(self._new_project)
        self.btn_del.clicked.connect(self._delete_project)
        row.addWidget(self.btn_new)
        row.addWidget(self.btn_del)
        left_l.addLayout(row)
        self.btn_import_ds = QPushButton("Import YOLO dataset…")
        self.btn_export_ds = QPushButton("Export YOLO zip…")
        self.btn_import_ds.clicked.connect(self._import_dataset)
        self.btn_export_ds.clicked.connect(self._export_zip)
        left_l.addWidget(self.btn_import_ds)
        left_l.addWidget(self.btn_export_ds)
        self.split_label = QLabel("Split: —")
        left_l.addWidget(self.split_label)
        self.btn_assign_split = QPushButton("Auto train/val split")
        self.btn_assign_split.clicked.connect(self._auto_split)
        left_l.addWidget(self.btn_assign_split)
        left_l.addStretch()
        split.addWidget(left)

        right = QWidget()
        right_l = QVBoxLayout(right)
        self.video = VideoWidget()
        self.video.bbox_drawn.connect(self._on_bbox_drawn)
        right_l.addWidget(self.video, stretch=1)

        nav = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._toggle_play)
        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        self.btn_prev10 = QPushButton("-10")
        self.btn_next10 = QPushButton("+10")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_label = QLabel("Frame 0/0")
        nav.addWidget(self.btn_play)
        nav.addWidget(self.btn_prev10)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.slider)
        nav.addWidget(self.btn_next)
        nav.addWidget(self.btn_next10)
        nav.addWidget(self.frame_label)
        right_l.addLayout(nav)

        cls_row = QHBoxLayout()
        cls_row.addWidget(QLabel("Class:"))
        self.class_combo = QComboBox()
        self.class_combo.addItems(list(COCO80_NAMES))
        self.class_combo.setCurrentText("person")
        self.class_combo.currentTextChanged.connect(self._on_class_combo)
        cls_row.addWidget(self.class_combo, stretch=1)
        self.btn_draw = QPushButton("Draw box")
        self.btn_draw.setCheckable(True)
        self.btn_draw.toggled.connect(self.video.set_bbox_mode)
        cls_row.addWidget(self.btn_draw)
        self.btn_copy = QPushButton("Copy → next frame")
        self.btn_copy.clicked.connect(self._copy_next)
        self.btn_clear = QPushButton("Clear frame")
        self.btn_clear.clicked.connect(self._clear_frame)
        cls_row.addWidget(self.btn_copy)
        cls_row.addWidget(self.btn_clear)
        right_l.addLayout(cls_row)

        self.box_list = QListWidget()
        right_l.addWidget(self.box_list)
        self.box_list.itemSelectionChanged.connect(self._refresh_overlay)

        train_row = QHBoxLayout()
        train_row.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 300)
        self.epochs_spin.setValue(50)
        train_row.addWidget(self.epochs_spin)
        self.btn_train = QPushButton("Train model")
        self.btn_train.clicked.connect(self._start_training)
        train_row.addWidget(self.btn_train)
        self.btn_open_runs = QPushButton("Open runs folder")
        self.btn_open_runs.clicked.connect(self._open_runs)
        self.btn_open_compare = QPushButton("Open in Compare")
        self.btn_open_compare.clicked.connect(self._open_in_compare)
        right_l.addWidget(self.btn_open_compare)
        right_l.addLayout(train_row)
        self.train_status = QLabel("Annotate 50+ varied frames, then train.")
        right_l.addWidget(self.train_status)

        split.addWidget(right)
        split.setSizes([220, 900])

        self.btn_prev.clicked.connect(lambda: self._seek_rel(-1))
        self.btn_next.clicked.connect(lambda: self._seek_rel(1))
        self.btn_prev10.clicked.connect(lambda: self._seek_rel(-10))
        self.btn_next10.clicked.connect(lambda: self._seek_rel(10))
        self.slider.valueChanged.connect(self._on_slider)

        self.refresh_projects()

    def refresh_projects(self) -> None:
        self.project_list.clear()
        for p in self._pm.list_projects():
            self.project_list.addItem(f"{p.name} ({p.id})")

    def _new_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Training video", "", "Video (*.mp4 *.avi *.mkv *.mov)"
        )
        if not path:
            return
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "Project name", "Name:", text=Path(path).stem)
        if not ok or not name.strip():
            return
        proj = self._pm.create(name.strip(), path)
        self.refresh_projects()
        items = [self.project_list.item(i).text() for i in range(self.project_list.count())]
        for i, t in enumerate(items):
            if proj.id in t:
                self.project_list.setCurrentRow(i)
                break

    def _delete_project(self) -> None:
        if not self._project:
            return
        self._pm.delete(self._project.id)
        self._project = None
        self._store = None
        self.refresh_projects()

    def _on_project_selected(self, text: str) -> None:
        if not text or "(" not in text:
            return
        pid = text.split("(")[-1].rstrip(")")
        proj = self._pm.get(pid)
        if not proj:
            return
        self._project = proj
        self._store = AnnotationStore(pid)
        if self._reader:
            self._reader.close()
        self._reader = VideoReader(proj.video_path)
        self.slider.setMaximum(max(0, self._reader.frame_count - 1))
        self._seek(0)
        self._update_split_label()

    def _on_class_combo(self, name: str) -> None:
        self._active_class = name or "person"

    def _seek(self, index: int) -> None:
        if not self._reader or not self._reader.is_open:
            return
        index = max(0, min(index, self._reader.frame_count - 1))
        self._current_frame = index
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        frame = self._reader.read_frame(index)
        if frame is not None:
            self.video.set_frame(frame)
        self.frame_label.setText(f"Frame {index}/{max(0, self._reader.frame_count - 1)}")
        self._refresh_box_list()
        self._refresh_overlay()

    def _seek_rel(self, delta: int) -> None:
        if self.btn_play.isChecked():
            self.btn_play.setChecked(False)
        self._seek(self._current_frame + delta)

    def _toggle_play(self, on: bool) -> None:
        self.btn_play.setText("Pause" if on else "Play")
        if not on or not self._reader:
            self._play_timer.stop()
            return
        fps = self._reader.fps if self._reader.fps > 1e-3 else 30.0
        self._play_timer.start(max(1, int(1000.0 / fps)))

    def _play_tick(self) -> None:
        if not self._reader:
            return
        nxt = self._current_frame + 1
        if nxt >= self._reader.frame_count:
            self.btn_play.setChecked(False)
            return
        self._seek(nxt)

    def _open_in_compare(self) -> None:
        if not self._project or not self._project.video_path:
            return
        mw = self.window()
        if mw and hasattr(mw, "compare_panel") and hasattr(mw, "lab_tabs"):
            mw.compare_panel.set_video_path(self._project.video_path)
            mw.lab_tabs.setCurrentWidget(mw.compare_panel)
            if hasattr(mw, "tabs"):
                for i in range(mw.tabs.count()):
                    if mw.tabs.tabText(i) == "Lab":
                        mw.tabs.setCurrentIndex(i)
                        break

    def _on_slider(self, value: int) -> None:
        if value != self._current_frame:
            self._seek(value)

    def _on_bbox_drawn(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if not self._store:
            return
        fa = self._store.get_frame(self._current_frame)
        fa.boxes.append(BBox(self._active_class, x1, y1, x2, y2, split="train"))
        self._store.set_boxes(self._current_frame, fa.boxes)
        self._refresh_box_list()
        self._refresh_overlay()
        self.btn_draw.setChecked(False)
        self.video.set_bbox_mode(False)

    def _refresh_box_list(self) -> None:
        self.box_list.clear()
        if not self._store:
            return
        fa = self._store.get_frame(self._current_frame)
        for i, b in enumerate(fa.boxes):
            self.box_list.addItem(f"{i}: {b.class_name} [{b.split}]")

    def _refresh_overlay(self) -> None:
        if not self._store:
            return
        fa = self._store.get_frame(self._current_frame)
        boxes = [(b.x1, b.y1, b.x2, b.y2, b.class_name) for b in fa.boxes]
        self.video.set_overlay_boxes(boxes)

    def _copy_next(self) -> None:
        if self._store:
            self._store.copy_to_next(self._current_frame)
            self._seek_rel(1)

    def _clear_frame(self) -> None:
        if self._store:
            self._store.set_boxes(self._current_frame, [])
            self._refresh_box_list()
            self._refresh_overlay()

    def _auto_split(self) -> None:
        if self._project and self._store:
            assign_splits(self._store, self._project.val_ratio)
            self._update_split_label()

    def _update_split_label(self) -> None:
        if not self._store:
            return
        c = split_counts(self._store)
        self.split_label.setText(
            f"Train: {c['train']['frames']} fr | Val: {c['val']['frames']} fr"
        )

    def _import_dataset(self) -> None:
        if not self._project:
            return
        folder = QFileDialog.getExistingDirectory(self, "YOLO dataset folder")
        if folder:
            n = import_yolo_folder(self._project.id, Path(folder))
            QMessageBox.information(self, "Import", f"Imported {n} frames.")
            self._store = AnnotationStore(self._project.id)
            self._update_split_label()

    def _export_zip(self) -> None:
        if not self._project or not self._store:
            return
        try:
            export_yolo_dataset(self._project.id, self._store, self._project.video_path)
        except Exception as exc:
            QMessageBox.warning(self, "Export", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save zip", "", "Zip (*.zip)")
        if path:
            export_yolo_zip(self._project.id, Path(path))
            QMessageBox.information(self, "Export", f"Saved {path}")

    def _start_training(self) -> None:
        if not self._project:
            return
        self.btn_train.setEnabled(False)
        self._worker = TrainingWorker(
            self._project.id,
            self._project.name,
            self._project.video_path,
            epochs=self.epochs_spin.value(),
        )
        self._worker.progress.connect(lambda m: self.train_status.setText(m))
        self._worker.finished_ok.connect(self._on_train_ok)
        self._worker.failed.connect(self._on_train_fail)
        self._worker.start()

    def _on_train_ok(self, model_id: str, onnx_path: str, map50: float) -> None:
        self.btn_train.setEnabled(True)
        self.train_status.setText(f"Done: {model_id} mAP50={map50:.3f}")
        QMessageBox.information(
            self,
            "Training complete",
            f"Model registered as {model_id}\nONNX: {onnx_path}\nmAP50: {map50:.3f}",
        )

    def _on_train_fail(self, msg: str) -> None:
        self.btn_train.setEnabled(True)
        self.train_status.setText(f"Failed: {msg}")
        QMessageBox.warning(self, "Training failed", msg)

    def _open_runs(self) -> None:
        if not self._project:
            return
        from src.lab.paths import project_dir

        path = project_dir(self._project.id) / "runs"
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))
