"""A/B model comparison on file or desktop source."""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.core.detector import YoloDetector, yolo_kwargs_from_config
from src.core.file_playback import VideoFilePlayback
from src.core.screen_capture import ScreenCapture
from src.lab.model_registry import ModelRegistry
from src.ui.screen_region_dialog import ScreenRegionDialog
from src.ui.video_transport import VideoTransportBar
from src.ui.video_widget import VideoWidget
from src.utils.config import app_data_dir, load_config, save_config

_FAMILY_FILTERS = {
    "All": None,
    "YOLO": "yolo",
    "TAO": ("peoplenet", "detectnet_v2"),
    "RT-DETR": "rtdetr",
    "Custom": "custom",
}


class ComparePanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cfg = load_config()
        self._registry = ModelRegistry()
        self._det_a: YoloDetector | None = None
        self._det_b: YoloDetector | None = None
        self._playback: VideoFilePlayback | None = None
        self._screen: ScreenCapture | None = None
        self._running = False
        self._source_mode = str(self._cfg.get("compare_source", "video"))
        self._desktop_counter = 0
        self._last_frame: np.ndarray | None = None
        self._last_dets_a = []
        self._last_dets_b = []
        self._rows: list[list] = []
        self._last_ms_a = 0.0
        self._last_ms_b = 0.0

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.model_filter = QComboBox()
        for label in _FAMILY_FILTERS:
            self.model_filter.addItem(label, label)
        form.addRow("Show models:", self.model_filter)
        self.model_a = QComboBox()
        self.model_b = QComboBox()
        self._fill_models()
        form.addRow("Model A:", self.model_a)
        form.addRow("Model B:", self.model_b)
        self.source_combo = QComboBox()
        self.source_combo.addItem("Video file", "video")
        self.source_combo.addItem("Desktop", "desktop")
        i = self.source_combo.findData(self._source_mode)
        if i >= 0:
            self.source_combo.setCurrentIndex(i)
        form.addRow("Source:", self.source_combo)
        layout.addLayout(form)

        src_row = QHBoxLayout()
        self.btn_open = QPushButton("Open test video…")
        self.btn_open.clicked.connect(self._open_video)
        self.btn_desktop = QPushButton("Desktop region…")
        self.btn_desktop.clicked.connect(self._set_desktop_region)
        self.btn_start = QPushButton("Compare")
        self.btn_stop = QPushButton("Stop")
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        src_row.addWidget(self.btn_open)
        src_row.addWidget(self.btn_desktop)
        src_row.addWidget(self.btn_start)
        src_row.addWidget(self.btn_stop)
        layout.addLayout(src_row)

        self.transport = VideoTransportBar()
        self.transport.setVisible(False)
        self.transport.pause_toggled.connect(self._on_transport_pause)
        self.transport.seek_requested.connect(self._on_transport_seek)
        self.transport.speed_changed.connect(self._on_transport_speed)
        self.transport.loop_changed.connect(self._on_transport_loop)
        self.transport.step_requested.connect(self._on_transport_step)
        layout.addWidget(self.transport)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.video_a = VideoWidget()
        self.video_b = VideoWidget()
        split.addWidget(self._wrap("Model A", self.video_a))
        split.addWidget(self._wrap("Model B", self.video_b))
        layout.addWidget(split, stretch=1)

        self.stats = QLabel("Open a test clip and pick two models, then Compare.")
        layout.addWidget(self.stats)

        btn_row = QHBoxLayout()
        self.btn_export = QPushButton("Export comparison CSV…")
        self.btn_export.clicked.connect(self._export_csv)
        self.btn_export_frames = QPushButton("Export overlay frames (PNG)…")
        self.btn_export_frames.clicked.connect(self._export_overlay_frames)
        btn_row.addWidget(self.btn_export)
        btn_row.addWidget(self.btn_export_frames)
        layout.addLayout(btn_row)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._video_path = ""
        self.model_filter.currentIndexChanged.connect(self._fill_models)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self._on_source_changed()

    def _wrap(self, title: str, widget: VideoWidget) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel(title))
        l.addWidget(widget)
        return w

    def refresh_models(self) -> None:
        self._fill_models()

    def _models_for_filter(self) -> list:
        key = str(self.model_filter.currentData() or "All")
        all_m = self._registry.list_models()
        spec = _FAMILY_FILTERS.get(key)
        if spec is None:
            return all_m
        if key == "TAO":
            return [m for m in all_m if m.id in spec]
        if key == "Custom":
            return [m for m in all_m if not m.is_builtin]
        return [m for m in all_m if m.family == spec or m.family.startswith(spec)]

    def _fill_models(self) -> None:
        cur_a = self.model_a.currentData()
        cur_b = self.model_b.currentData()
        models = self._models_for_filter()
        self.model_a.clear()
        self.model_b.clear()
        for m in models:
            self.model_a.addItem(m.display_name, m.id)
            self.model_b.addItem(m.display_name, m.id)
        cfg = self._cfg
        def pick(combo: QComboBox, prefer: str | None) -> None:
            if prefer:
                for i in range(combo.count()):
                    if combo.itemData(i) == prefer:
                        combo.setCurrentIndex(i)
                        return
        pick(self.model_a, cur_a or str(cfg.get("compare_model_a", "yolo11s")))
        pick(self.model_b, cur_b or str(cfg.get("compare_model_b", "yolo11m")))

    def set_video_path(self, path: str) -> None:
        self._video_path = str(path)
        self.source_combo.setCurrentIndex(0)
        self._source_mode = "video"
        self.stats.setText(f"Video: {Path(self._video_path).name}")

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Test video", "", "Video (*.mp4 *.avi *.mkv)")
        if path:
            self.set_video_path(path)

    def _load_detectors(self) -> bool:
        cfg = load_config()
        reg = ModelRegistry()
        try:
            id_a = self.model_a.currentData()
            id_b = self.model_b.currentData()
            onnx_a = reg.resolve_onnx(str(id_a), int(cfg.get("model_imgsz", 640)))
            onnx_b = reg.resolve_onnx(str(id_b), int(cfg.get("model_imgsz", 640)))
            kw = yolo_kwargs_from_config(cfg)
            self._det_a = YoloDetector(onnx_a, model_name=str(id_a), **kw)
            self._det_b = YoloDetector(onnx_b, model_name=str(id_b), **kw)
            self._cfg["compare_model_a"] = str(id_a)
            self._cfg["compare_model_b"] = str(id_b)
            save_config(self._cfg)
            return True
        except Exception as exc:
            self.stats.setText(f"Model load error: {exc}")
            return False

    def _start(self) -> None:
        if not self._load_detectors():
            return
        self._rows.clear()
        self._running = True
        if self._source_mode == "video":
            if not self._video_path:
                self.stats.setText("Open a test video first")
                self._running = False
                return
            loop = bool(self._cfg.get("file_loop", True))
            spd = float(self._cfg.get("compare_playback_speed", 1.0))
            self._playback = VideoFilePlayback(self._video_path, loop=loop, playback_speed=spd)
            if not self._playback.start():
                self.stats.setText(self._playback.error or "Failed to open test video")
                self._playback = None
                self._running = False
                return
            self.transport.setVisible(True)
            self.transport.set_metadata(
                self._playback.frame_count,
                self._playback.video_fps,
                self._playback.current_frame_index,
            )
            si = self.transport.speed_combo.findData(spd)
            if si >= 0:
                self.transport.speed_combo.setCurrentIndex(si)
            self.transport.chk_loop.setChecked(loop)
        else:
            region = self._cfg.get("screen_region")
            reg_tuple = tuple(region) if region and len(region) == 4 else None
            self._screen = ScreenCapture(
                monitor_index=int(self._cfg.get("screen_monitor", 1)),
                region=reg_tuple,
                fps_cap=float(self._cfg.get("screen_fps_cap", 30)),
            )
            if not self._screen.start():
                self.stats.setText(f"Desktop source error: {self._screen.error or 'unknown'}")
                self._screen = None
                self._running = False
                return
            self.transport.setVisible(False)
            self._desktop_counter = 0
        self._timer.start(10)
        self._apply_source_mode_ui()

    def _stop(self) -> None:
        self._running = False
        self._timer.stop()
        if self._playback:
            self._playback.stop()
            self._playback = None
        if self._screen:
            self._screen.stop()
            self._screen = None
        self.transport.setVisible(False)
        self._apply_source_mode_ui()

    def _draw_dets(self, frame: np.ndarray, dets) -> np.ndarray:
        out = frame.copy()
        for d in dets:
            color = (0, 165, 255) if d.class_id == 2 else (0, 255, 0)
            cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 2)
        return out

    def _tick(self) -> None:
        if not self._running or not self._det_a or not self._det_b:
            return
        if self._source_mode == "video":
            if self._playback is None:
                return
            frame, fidx = self._playback.get_latest()
            if frame is None:
                return
            fidx = self._playback.current_frame_index
            self.transport.update_position(fidx)
        else:
            if self._screen is None:
                return
            if self.transport.btn_play.isChecked():
                if self._last_frame is not None:
                    frame = self._last_frame.copy()
                    fidx = self._desktop_counter
                else:
                    return
            else:
                frame, fid = self._screen.get_latest()
                if frame is None:
                    return
                self._last_frame = frame.copy()
                self._desktop_counter = int(fid)
                fidx = self._desktop_counter
        t0 = time.perf_counter()
        da = self._det_a.detect(frame)
        ms_a = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        db = self._det_b.detect(frame)
        ms_b = (time.perf_counter() - t1) * 1000.0
        self._last_dets_a, self._last_dets_b = da, db
        self._last_ms_a, self._last_ms_b = ms_a, ms_b
        self._last_frame = frame.copy()
        pa, ca = sum(1 for d in da if d.class_id == 0), sum(1 for d in da if d.class_id == 2)
        pb, cb = sum(1 for d in db if d.class_id == 0), sum(1 for d in db if d.class_id == 2)
        self.video_a.set_frame(self._draw_dets(frame, da))
        self.video_b.set_frame(self._draw_dets(frame, db))
        fps_a = 1000.0 / ms_a if ms_a > 0 else 0.0
        fps_b = 1000.0 / ms_b if ms_b > 0 else 0.0
        if self._source_mode == "video" and self._playback:
            total = max(1, self._playback.frame_count)
            self.stats.setText(
                f"Frame {fidx + 1}/{total} | "
                f"A {ms_a:.1f}ms ({fps_a:.0f} FPS) B {ms_b:.1f}ms ({fps_b:.0f} FPS) | "
                f"A: {pa}p {ca}c | B: {pb}p {cb}c | Δ person {pb - pa} car {cb - ca}"
            )
        else:
            self.stats.setText(
                f"Desktop frame {fidx} | A {ms_a:.1f}ms B {ms_b:.1f}ms | "
                f"A: {pa}p {ca}c | B: {pb}p {cb}c | Δ person {pb - pa} car {cb - ca}"
            )
        self._rows.append([fidx, pa, ca, pb, cb, round(ms_a, 2), round(ms_b, 2)])

    def _on_transport_pause(self, paused: bool) -> None:
        if self._playback:
            self._playback.pause_playback(paused)

    def _on_transport_seek(self, index: int) -> None:
        if self._playback:
            self._playback.pause_playback(True)
            self.transport.set_video_paused(True)
            self._playback.seek_frame(index)

    def _on_transport_speed(self, speed: float) -> None:
        self._cfg["compare_playback_speed"] = speed
        save_config(self._cfg)
        if self._playback:
            self._playback.set_playback_speed(speed)

    def _on_transport_loop(self, loop: bool) -> None:
        self._cfg["file_loop"] = loop
        save_config(self._cfg)
        if self._playback:
            self._playback.set_loop(loop)

    def _on_transport_step(self, delta: int) -> None:
        if not self._playback:
            return
        self._playback.pause_playback(True)
        self.transport.set_video_paused(True)
        cur = self._playback.current_frame_index
        self._playback.seek_frame(cur + int(delta))

    def _on_source_changed(self) -> None:
        if self._running:
            self._stop()
        self._source_mode = str(self.source_combo.currentData() or "video")
        self._cfg["compare_source"] = self._source_mode
        save_config(self._cfg)
        self._apply_source_mode_ui()

    def _apply_source_mode_ui(self) -> None:
        is_video = self._source_mode == "video"
        self.btn_open.setEnabled(is_video)
        self.btn_desktop.setEnabled(not is_video)
        if not self._running:
            self.transport.setVisible(False)

    def _set_desktop_region(self) -> None:
        dlg = ScreenRegionDialog(self._cfg, self)
        if dlg.exec():
            self._cfg.update(dlg.result_config())
            save_config(self._cfg)
            self.stats.setText(
                f"Desktop: monitor {self._cfg.get('screen_monitor', 1)} "
                f"@ {int(self._cfg.get('screen_fps_cap', 30))} FPS cap"
            )

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export comparison",
            str(app_data_dir() / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
            "CSV (*.csv)",
        )
        if not path:
            return
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "a_person", "a_car", "b_person", "b_car", "ms_a", "ms_b"])
            w.writerows(self._rows)

    def _export_overlay_frames(self) -> None:
        if self._last_frame is None:
            self.stats.setText("Run Compare first to capture a frame")
            return
        folder = QFileDialog.getExistingDirectory(self, "Export overlay frames")
        if not folder:
            return
        base = Path(folder) / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        base.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(base / "model_a.png"), self._draw_dets(self._last_frame, self._last_dets_a))
        cv2.imwrite(str(base / "model_b.png"), self._draw_dets(self._last_frame, self._last_dets_b))
        cv2.imwrite(str(base / "source.png"), self._last_frame)
        self.stats.setText(f"Saved overlays to {base}")
