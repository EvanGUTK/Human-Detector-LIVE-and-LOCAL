"""Main application window — dashboard tabs + overlay mode."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QTimer, Qt, pyqtSlot
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QSystemTrayIcon,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.core.alerts import AlertManager
from src.core.detector import YoloDetector, yolo_kwargs_from_config
from src.lab.model_registry import ModelRegistry
from src.core.event_log import EventLog
from src.core.face_recognizer import FaceRecognizer, UNKNOWN
from src.core.metrics import MetricsStore
from src.core.performance_profiles import (
    BALANCED,
    COMPATIBILITY,
    CUSTOM,
    apply_profile,
    preset_ids,
    profile_label,
)
from src.ui.perf_hud import PerfHudPanel
from src.ui.setup_wizard import SetupWizard
from src.utils.config import project_root
from src.core.pipeline import DetectionPipeline
from src.core.profile_manager import ProfileManager
from src.core.source_factory import create_frame_source
from src.core.zones import ZoneEventType, ZoneManager
from src.ui.analytics_panel import AnalyticsPanel
from src.ui.faces_panel import FacesPanel
from src.ui.help_dialog import ZonesHelpDialog
from src.ui.settings_panel import SettingsPanel
from src.ui.video_widget import VideoWidget
from src.ui.zones_panel import ZonesPanel
from src.ui.train_panel import TrainPanel
from src.ui.compare_panel import ComparePanel
from src.ui.evaluate_panel import EvaluatePanel
from src.ui.lab_dashboard import LabDashboard
from src.ui.model_health_panel import ModelHealthPanel
from src.ui.screen_region_dialog import ScreenRegionDialog
from src.ui.video_transport import VideoTransportBar
from src.core.file_playback import VideoFilePlayback
from src.utils.config import app_data_dir, load_config, save_config

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, detector: YoloDetector, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.detector = detector
        self.setWindowTitle("Person Detector — AI Testing Lab")
        self.resize(1320, 760)

        self.profile_manager = ProfileManager()
        active = str(cfg.get("active_profile", "Default room"))
        if active:
            self.cfg = self.profile_manager.apply_to_config(self.cfg, active)

        self.zone_manager = ZoneManager()
        if self.cfg.get("zones"):
            self.zone_manager.zones_from_config(self.cfg["zones"])

        self.alert_manager = AlertManager(
            sound_enabled=bool(self.cfg.get("alert_sound", True)),
            toast_enabled=bool(self.cfg.get("alert_toast", True)),
            flash_ms=int(self.cfg.get("alert_flash_ms", 800)),
        )
        self.event_log = EventLog()
        self.metrics = MetricsStore()
        self.capture = create_frame_source(self.cfg)
        self.face_recognizer = self._build_face_recognizer()

        self.pipeline: DetectionPipeline | None = None
        self._running = False
        self._paused = False
        self._last_logged_events: set[tuple] = set()
        self._overlay_mode = str(self.cfg.get("ui_mode", "dashboard")) == "overlay"
        self.cfg["_last_model_id"] = str(cfg.get("active_model_id", cfg.get("model_name", "yolo11s")))

        self._build_ui()
        self._build_menus()
        self._build_toolbar()
        self._setup_tray()
        self._setup_shortcuts()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.setInterval(16)

        self.settings.set_backend_label(
            detector.backend_name, getattr(detector, "using_gpu", False)
        )
        self.faces_panel.set_recognizer(self.face_recognizer)
        self._apply_ui_mode()
        self._sync_video_transport()
        self.status.showMessage("Ready — Monitor tab → Start")

    def _build_face_recognizer(self) -> FaceRecognizer | None:
        if not bool(self.cfg.get("face_enabled", True)):
            return None
        gid = str(self.cfg.get("face_gallery_id", self.cfg.get("active_profile", "Default room")))
        return FaceRecognizer(
            profile_id=gid,
            match_threshold=float(self.cfg.get("face_match_threshold", 0.45)),
            enabled=True,
            face_stride=int(self.cfg.get("face_stride", 3)),
        )

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        # Monitor tab (video + perf rail)
        monitor = QWidget()
        mon_row = QHBoxLayout(monitor)
        video_col = QVBoxLayout()
        self.video = VideoWidget()
        self.video.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.video.zone_finished.connect(self._on_zone_finished)
        self.video_transport = VideoTransportBar()
        self.video_transport.setVisible(False)
        self.video_transport.seek_requested.connect(self._on_video_seek)
        self.video_transport.pause_toggled.connect(self._on_video_pause_toggled)
        self.video_transport.speed_changed.connect(self._on_video_speed_changed)
        self.video_transport.loop_changed.connect(self._on_video_loop_changed)
        self.video_transport.step_requested.connect(self._on_video_step)
        video_col.addWidget(self.video)
        video_col.addWidget(self.video_transport)
        mon_row.addLayout(video_col, stretch=4)

        rail = QVBoxLayout()
        self.perf_hud = PerfHudPanel()
        self.perf_hud.setMaximumWidth(320)
        rail.addWidget(self.perf_hud)
        rail.addWidget(QLabel("Recent events"))
        self.monitor_events = QListWidget()
        self.monitor_events.setMaximumHeight(160)
        rail.addWidget(self.monitor_events)
        rail.addStretch()
        mon_row.addLayout(rail, stretch=1)
        self.tabs.addTab(monitor, "Monitor")

        lab_host = QWidget()
        lab_layout = QVBoxLayout(lab_host)
        self.lab_tabs = QTabWidget()
        self.lab_dashboard = LabDashboard(self)
        self.lab_tabs.addTab(self.lab_dashboard, "Dashboard")
        self.compare_panel = ComparePanel()
        self.lab_tabs.addTab(self.compare_panel, "Compare")
        self.train_panel = TrainPanel()
        self.lab_tabs.addTab(self.train_panel, "Train")
        self.evaluate_panel = EvaluatePanel()
        self.lab_tabs.addTab(self.evaluate_panel, "Evaluate")
        self.health_panel = ModelHealthPanel()
        self.lab_tabs.addTab(self.health_panel, "Model Health")
        lab_layout.addWidget(self.lab_tabs)
        self.tabs.addTab(lab_host, "Lab")

        self.zones_panel = ZonesPanel(self.zone_manager)
        self.zones_panel.zone_help_requested.connect(self._show_zones_help)
        self.zones_panel.test_alert_requested.connect(self._test_alert)
        self.zones_panel.export_log_requested.connect(self._export_event_log)
        self.zones_panel.btn_delete.clicked.connect(self._delete_selected_zone)
        self.zones_panel.zone_delete_requested.connect(self._delete_selected_zone)
        self.zones_panel.btn_toggle.clicked.connect(self._toggle_selected_zone)
        self.zones_panel.btn_rename.clicked.connect(self._rename_selected_zone)
        self.tabs.addTab(self.zones_panel, "Zones")

        self.analytics_panel = AnalyticsPanel()
        self.analytics_panel.btn_export_session.clicked.connect(self._export_session_metrics)
        self.tabs.addTab(self.analytics_panel, "Analytics")

        self.faces_panel = FacesPanel()
        self.faces_panel.capture_requested.connect(self._capture_face_from_video)
        self.faces_panel.gallery_changed.connect(self._on_gallery_changed)
        self.tabs.addTab(self.faces_panel, "Faces")

        self.settings = SettingsPanel()
        self.settings.load_from_config(self.cfg)
        self.tabs.addTab(self.settings, "Settings")

        self.settings.settings_changed.connect(lambda: self._apply_settings())
        self.settings.rebuild_model_requested.connect(self._rebuild_model)
        self.settings.btn_setup_wizard.clicked.connect(self._run_setup_wizard)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.zones_panel.refresh_zones()

    def _build_menus(self) -> None:
        view = self.menuBar().addMenu("View")
        self.act_dashboard = QAction("Dashboard layout", self)
        self.act_overlay = QAction("Overlay layout", self)
        self.act_dashboard.triggered.connect(lambda: self._set_ui_mode("dashboard"))
        self.act_overlay.triggered.connect(lambda: self._set_ui_mode("overlay"))
        view.addAction(self.act_dashboard)
        view.addAction(self.act_overlay)

        help_menu = self.menuBar().addMenu("Help")
        act_zones = QAction("How zones work…", self)
        act_zones.triggered.connect(self._show_zones_help)
        help_menu.addAction(act_zones)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        self.addToolBar(tb)

        tb.addWidget(QLabel(" Profile: "))
        self.profile_combo = QComboBox()
        for name in self.profile_manager.list_profiles():
            self.profile_combo.addItem(name)
        active = str(self.cfg.get("active_profile", "Default room"))
        idx = self.profile_combo.findText(active)
        if idx >= 0:
            self.profile_combo.setCurrentIndex(idx)
        self.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        tb.addWidget(self.profile_combo)

        self.act_save_profile = QAction("Save profile", self)
        self.act_save_profile.triggered.connect(self._save_profile)
        tb.addAction(self.act_save_profile)

        tb.addSeparator()

        tb.addWidget(QLabel(" Preset: "))
        self.perf_combo = QComboBox()
        for pid in preset_ids():
            self.perf_combo.addItem(profile_label(pid), pid)
        prof = str(self.cfg.get("model_preset", self.cfg.get("performance_profile", BALANCED)))
        if prof == "performance":
            prof = BALANCED
        for i in range(self.perf_combo.count()):
            if self.perf_combo.itemData(i) == prof:
                self.perf_combo.setCurrentIndex(i)
                break
        self.perf_combo.currentIndexChanged.connect(self._on_perf_profile_changed)
        tb.addWidget(self.perf_combo)

        tb.addWidget(QLabel(" Preview: "))
        self.preview_combo = QComboBox()
        self.preview_combo.addItem("Performance", "performance")
        self.preview_combo.addItem("Full", "full")
        pm = str(self.cfg.get("preview_mode", "performance"))
        self.preview_combo.setCurrentIndex(0 if pm == "performance" else 1)
        self.preview_combo.currentIndexChanged.connect(self._on_preview_mode_changed)
        tb.addWidget(self.preview_combo)

        self.chk_perf_hud = QAction("Perf HUD", self)
        self.chk_perf_hud.setCheckable(True)
        self.chk_perf_hud.setChecked(True)
        self.chk_perf_hud.triggered.connect(self._toggle_perf_hud)
        tb.addAction(self.chk_perf_hud)

        tb.addSeparator()

        tb.addWidget(QLabel(" Detection: "))

        self.act_start = QAction("Start", self)
        self.act_stop = QAction("Stop", self)
        self.act_pause = QAction("Pause detection", self)
        self.act_snapshot = QAction("Snapshot", self)
        self.act_draw = QAction("Draw Zone", self)
        self.act_draw.setCheckable(True)
        self.act_open_file = QAction("Open video…", self)
        self.act_webcam = QAction("Use webcam", self)
        self.act_desktop = QAction("Desktop capture", self)
        self.act_screen_region = QAction("Set screen region…", self)

        self.act_start.triggered.connect(self.start_detection)
        self.act_stop.triggered.connect(self.stop_detection)
        self.act_pause.triggered.connect(self.toggle_pause)
        self.act_snapshot.triggered.connect(self._export_snapshot)
        self.act_draw.triggered.connect(self.toggle_draw_mode)
        self.act_open_file.triggered.connect(self._open_video_file)
        self.act_webcam.triggered.connect(self._use_webcam)
        self.act_desktop.triggered.connect(self._use_desktop)
        self.act_screen_region.triggered.connect(self._set_screen_region)

        for act in (
            self.act_start,
            self.act_stop,
            self.act_pause,
            self.act_snapshot,
            self.act_draw,
        ):
            tb.addAction(act)

        tb.addSeparator()
        tb.addWidget(QLabel(" Source: "))
        for act in (
            self.act_open_file,
            self.act_webcam,
            self.act_desktop,
            self.act_screen_region,
        ):
            tb.addAction(act)

    def _setup_tray(self) -> None:
        self._tray = QSystemTrayIcon(self)
        self._tray.setToolTip("Person Detector")
        if bool(self.cfg.get("minimize_to_tray", True)):
            self._tray.activated.connect(self._tray_activated)

    def _tray_activated(self, reason) -> None:
        from PyQt6.QtWidgets import QSystemTrayIcon

        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.showNormal()
            self.activateWindow()

    def _setup_shortcuts(self) -> None:
        QAction("Debug overlay", self, shortcut=QKeySequence("D")).triggered.connect(
            self._toggle_debug
        )
        QAction("Pause detection", self, shortcut=QKeySequence("Ctrl+P")).triggered.connect(
            self.toggle_pause
        )
        QAction("Video play/pause", self, shortcut=QKeySequence(Qt.Key.Key_Space)).triggered.connect(
            self._shortcut_video_play_pause
        )

    def _shortcut_video_play_pause(self) -> None:
        if self.cfg.get("input_source") != "file" or not self.video_transport.isVisible():
            return
        self.video_transport.btn_play.click()

    def _sync_video_transport(self) -> None:
        is_file = self.cfg.get("input_source") == "file" and bool(self.cfg.get("file_path"))
        self.video_transport.setVisible(is_file)
        cap = self.capture
        if is_file and isinstance(cap, VideoFilePlayback):
            self.video_transport.set_metadata(
                cap.frame_count,
                cap.video_fps,
                cap.current_frame_index,
            )
            self.video_transport.set_video_paused(cap.is_playback_paused)
            spd = float(self.cfg.get("file_playback_speed", 1.0))
            for i in range(self.video_transport.speed_combo.count()):
                if self.video_transport.speed_combo.itemData(i) == spd:
                    self.video_transport.speed_combo.setCurrentIndex(i)
                    break
            self.video_transport.chk_loop.setChecked(bool(self.cfg.get("file_loop", True)))

    def _file_capture(self) -> VideoFilePlayback | None:
        if isinstance(self.capture, VideoFilePlayback):
            return self.capture
        return None

    def _on_video_pause_toggled(self, paused: bool) -> None:
        fc = self._file_capture()
        if fc:
            fc.pause_playback(paused)

    def _on_video_seek(self, index: int) -> None:
        fc = self._file_capture()
        if fc:
            fc.seek_frame(index)

    def _on_video_speed_changed(self, speed: float) -> None:
        self.cfg["file_playback_speed"] = speed
        fc = self._file_capture()
        if fc:
            fc.set_playback_speed(speed)
        self._persist()

    def _on_video_loop_changed(self, loop: bool) -> None:
        self.cfg["file_loop"] = loop
        fc = self._file_capture()
        if fc:
            fc.set_loop(loop)
        self._persist()

    def _on_video_step(self, delta: int) -> None:
        fc = self._file_capture()
        if fc:
            fc.pause_playback(True)
            self.video_transport.set_video_paused(True)
            target = max(0, fc.current_frame_index + delta)
            if fc.frame_count > 0:
                target = min(target, fc.frame_count - 1)
            fc.seek_frame(target)
            self.video_transport.update_position(target)

    def _toggle_debug(self) -> None:
        self.cfg["show_debug_overlay"] = not bool(self.cfg.get("show_debug_overlay"))
        self.settings.show_debug_overlay.setChecked(self.cfg["show_debug_overlay"])
        if self.pipeline:
            self.pipeline.show_debug_overlay = self.cfg["show_debug_overlay"]
        self._persist()

    def _set_ui_mode(self, mode: str) -> None:
        self.cfg["ui_mode"] = mode
        self._overlay_mode = mode == "overlay"
        self._apply_ui_mode()
        self._persist()
        if self.pipeline:
            self.pipeline.ui_mode = mode

    def _apply_ui_mode(self) -> None:
        if self._overlay_mode:
            self.tabs.tabBar().hide()
            self.setWindowTitle("Person Detector — AI Testing Lab (Overlay)")
        else:
            self.tabs.tabBar().show()
            self.setWindowTitle("Person Detector — AI Testing Lab")

    def _on_profile_changed(self, name: str) -> None:
        if not name:
            return
        was_running = self._running
        if was_running:
            self.stop_detection()
        self.cfg = self.profile_manager.apply_to_config(self.cfg, name)
        self.zone_manager.zones_from_config(self.cfg.get("zones", []))
        self.settings.load_from_config(self.cfg)
        self.capture = create_frame_source(self.cfg)
        self.face_recognizer = self._build_face_recognizer()
        self.faces_panel.set_recognizer(self.face_recognizer)
        self.zones_panel.refresh_zones()
        self._persist()
        if was_running:
            self.start_detection()

    def _save_profile(self) -> None:
        name = self.profile_combo.currentText()
        self.profile_manager.save(name, self.cfg)
        self.status.showMessage(f"Saved profile '{name}'")

    def _on_perf_profile_changed(self, _index: int) -> None:
        name = self.perf_combo.currentData()
        if not name or str(name) == CUSTOM:
            self.cfg["model_preset"] = CUSTOM
            self.cfg["performance_profile"] = CUSTOM
            self._persist()
            return
        was_running = self._running
        if was_running:
            self.stop_detection()
        self.cfg = apply_profile(self.cfg, str(name))
        self.settings.load_from_config(self.cfg)
        self._recreate_detector()
        self._persist()
        if was_running:
            self.start_detection()

    def _on_preview_mode_changed(self, _index: int) -> None:
        mode = str(self.preview_combo.currentData() or "performance")
        self.cfg["preview_mode"] = mode
        if self.pipeline:
            self.pipeline.preview_mode = mode
        self._persist()

    def _toggle_perf_hud(self) -> None:
        visible = self.chk_perf_hud.isChecked()
        self.perf_hud.setVisible(visible)

    def _run_setup_wizard(self) -> None:
        dlg = SetupWizard(self.cfg, project_root(), self)
        if dlg.exec():
            was_running = self._running
            if was_running:
                self.stop_detection()
            self.cfg.update(dlg.result_config())
            self.settings.load_from_config(self.cfg)
            self.capture = create_frame_source(self.cfg)
            self._recreate_detector()
            prof = str(self.cfg.get("model_preset", BALANCED))
            for i in range(self.perf_combo.count()):
                if self.perf_combo.itemData(i) == prof:
                    self.perf_combo.setCurrentIndex(i)
                    break
            pm = str(self.cfg.get("preview_mode", "performance"))
            self.preview_combo.setCurrentIndex(0 if pm == "performance" else 1)
            self._persist()
            if was_running:
                self.start_detection()

    def _recreate_detector(self) -> None:
        model_id = str(self.cfg.get("active_model_id", self.cfg.get("model_name", "yolo11s")))
        imgsz = int(self.cfg.get("model_imgsz", 640))
        reg = ModelRegistry()
        onnx_path = reg.resolve_onnx(model_id, imgsz)
        self.detector = YoloDetector(
            onnx_path,
            model_name=model_id,
            **yolo_kwargs_from_config(self.cfg),
        )
        self.detector.debug_collect_faded = bool(self.cfg.get("show_faded_low_conf", False))
        self.settings.set_backend_label(
            self.detector.backend_name, getattr(self.detector, "using_gpu", False)
        )
        self.settings.refresh_model_list(model_id)
        if hasattr(self, "compare_panel"):
            self.compare_panel.refresh_models()

    def _open_video_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open video file",
            "",
            "Video (*.mp4 *.avi *.mkv *.mov *.wmv);;All (*.*)",
        )
        if not path:
            return
        was_running = self._running
        if was_running:
            self.stop_detection()
        self.cfg["input_source"] = "file"
        self.cfg["file_path"] = path
        self.capture = create_frame_source(self.cfg)
        self._persist()
        self._sync_video_transport()
        self.status.showMessage(f"Video: {Path(path).name}")
        if was_running:
            self.start_detection()

    def _use_webcam(self) -> None:
        was_running = self._running
        if was_running:
            self.stop_detection()
        self.cfg["input_source"] = "webcam"
        self.capture = create_frame_source(self.cfg)
        self._persist()
        self._sync_video_transport()
        self.status.showMessage("Input: webcam")
        if was_running:
            self.start_detection()

    def _use_desktop(self) -> None:
        was_running = self._running
        if was_running:
            self.stop_detection()
        if not self.cfg.get("screen_region"):
            dlg = ScreenRegionDialog(self.cfg, self)
            if dlg.exec():
                self.cfg.update(dlg.result_config())
            else:
                self.cfg["input_source"] = "screen"
                self.cfg["screen_region"] = None
        else:
            self.cfg["input_source"] = "screen"
        self.capture = create_frame_source(self.cfg)
        self._persist()
        self._sync_video_transport()
        self.status.showMessage(self.capture.source_label)
        if was_running:
            self.start_detection()

    def _set_screen_region(self) -> None:
        dlg = ScreenRegionDialog(self.cfg, self)
        if dlg.exec():
            was_running = self._running
            if was_running:
                self.stop_detection()
            self.cfg.update(dlg.result_config())
            self.capture = create_frame_source(self.cfg)
            self._persist()
            self.status.showMessage(self.capture.source_label)
            if was_running:
                self.start_detection()

    def _persist(self) -> None:
        self.cfg["zones"] = self.zone_manager.zones_to_config()
        self.cfg.update(self.settings.to_config_updates())
        self.cfg["active_profile"] = self.profile_combo.currentText()
        save_config(self.cfg)

    def _pipeline_kwargs(self) -> dict:
        ui_mode = "overlay" if self._overlay_mode else "dashboard"
        fire = self._build_fire_smoke_detector()
        return {
            "show_pose_skeleton": bool(self.cfg.get("show_pose_skeleton", False)),
            "show_hand_skeleton": bool(self.cfg.get("show_hand_skeleton", True)),
            "skeleton_stride": int(self.cfg.get("skeleton_stride", 2)),
            "show_zones_on_frame": bool(self.cfg.get("show_zones_on_frame", True)),
            "show_boxes_on_frame": bool(self.cfg.get("show_boxes_on_frame", True)),
            "show_hud_on_frame": bool(self.cfg.get("show_hud_on_frame", True)),
            "show_debug_overlay": bool(self.cfg.get("show_debug_overlay", False)),
            "ui_mode": ui_mode,
            "show_faded_low_conf": bool(self.cfg.get("show_faded_low_conf", False)),
            "fire_smoke_detector": fire,
            "zone_alert_class_ids": set(self.cfg.get("zone_alert_class_ids", [0])),
            "face_rec_class_ids": set(self.cfg.get("face_rec_class_ids", [0])),
            "infer_max_width": int(self.cfg.get("infer_max_width", 1280)),
            "infer_max_height": int(self.cfg.get("infer_max_height", 720)),
            "preview_mode": str(self.cfg.get("preview_mode", "performance")),
        }

    def _build_fire_smoke_detector(self):
        from pathlib import Path

        from src.core.fire_smoke_detector import FireSmokeDetector

        if not bool(self.cfg.get("fire_smoke_enabled", False)):
            return None
        p = str(self.cfg.get("fire_smoke_onnx_path") or "").strip()
        return FireSmokeDetector(
            Path(p) if p else None,
            enabled=True,
            force_cpu=bool(self.cfg.get("force_cpu", False)),
            full_frame=bool(self.cfg.get("fire_smoke_full_frame", True)),
            imgsz=int(self.cfg.get("model_imgsz", 640)),
        )

    @pyqtSlot()
    def start_detection(self) -> None:
        if self._running:
            return
        self._apply_settings(silent=True)
        self._last_logged_events.clear()
        self.metrics.reset()
        self.pipeline = DetectionPipeline(
            self.detector,
            self.capture,
            self.zone_manager,
            self.alert_manager,
            metrics=self.metrics,
            face_recognizer=self.face_recognizer,
            **self._pipeline_kwargs(),
        )
        if not self.pipeline.start():
            QMessageBox.critical(
                self,
                "Source error",
                getattr(self.capture, "error", None) or "Failed to open video source.",
            )
            self.pipeline = None
            return
        sk = self.pipeline._skeleton
        if sk is not None and not sk._enabled and sk.error:
            QMessageBox.warning(
                self,
                "Pose skeleton unavailable",
                f"MediaPipe could not start:\n{sk.error}",
            )
        self._running = True
        self._paused = False
        self.act_pause.setText("Pause detection")
        self._timer.start()
        self._sync_video_transport()
        src = getattr(self.capture, "source_label", "source")
        self.status.showMessage(f"Detection running — {src}")

    @pyqtSlot()
    def stop_detection(self) -> None:
        self._timer.stop()
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        self._running = False
        self._paused = False
        self.act_draw.setChecked(False)
        self.video.set_draw_mode(False)
        self.status.showMessage("Stopped")

    @pyqtSlot()
    def toggle_pause(self) -> None:
        if not self._running or not self.pipeline:
            return
        self._paused = not self._paused
        self.pipeline.pause(self._paused)
        self.act_pause.setText(
            "Resume detection" if self._paused else "Pause detection"
        )

    def _export_snapshot(self) -> None:
        import json

        if not self.pipeline:
            QMessageBox.information(self, "Snapshot", "Start detection first.")
            return
        result = self.pipeline.get_result()
        if result is None:
            return
        out_dir = app_data_dir() / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = out_dir / f"snapshot_{stamp}.jpg"
        meta_path = out_dir / f"snapshot_{stamp}.json"
        import cv2

        cv2.imwrite(str(img_path), result.frame)
        meta = {
            "fps": result.fps,
            "infer_ms": result.infer_ms,
            "total_ms": result.total_ms,
            "tracks": [
                {
                    "id": t.track_id,
                    "class_id": t.class_id,
                    "confidence": t.confidence,
                    "box": [t.x1, t.y1, t.x2, t.y2],
                }
                for t in result.tracks
            ],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        QMessageBox.information(
            self,
            "Snapshot",
            f"Saved:\n{img_path}\n{meta_path}",
        )

    @pyqtSlot(bool)
    def toggle_draw_mode(self, checked: bool) -> None:
        if checked and not self._running:
            self.act_draw.setChecked(False)
            QMessageBox.information(self, "Draw Zone", "Click Start first, then Draw Zone.")
            return
        self.video.set_draw_mode(checked)
        if checked:
            self.tabs.setCurrentIndex(0)
            self.video.setFocus()
        else:
            self.video.cancel_draft()

    @pyqtSlot(list)
    def _on_zone_finished(self, points: list) -> None:
        name, ok = QInputDialog.getText(self, "Zone name", "Name this zone:", text="Zone")
        if not ok or not name.strip():
            return
        zone = self.zone_manager.add_zone(name.strip(), [(p[0], p[1]) for p in points])
        self.zones_panel.refresh_zones()
        self._persist()
        self.act_draw.setChecked(False)
        self.video.set_draw_mode(False)
        if self.pipeline:
            self.pipeline.pulse_zone(zone.id)
        self.event_log.append("zone_created", f"Created zone '{zone.name}'")
        self.zones_panel.refresh_events(self.event_log.recent_messages(15))

    def _test_alert(self) -> None:
        zid, zname = "test", "Test"
        zid_sel = self.zones_panel.selected_zone_id()
        if zid_sel:
            zone = next((z for z in self.zone_manager.zones if z.id == zid_sel), None)
            if zone:
                zid, zname = zone.id, zone.name
        self.alert_manager.test_alert(zid, zname)
        self.event_log.append("test_alert", f"Test alert for zone '{zname}'")
        self.zones_panel.refresh_events(self.event_log.recent_messages(15))

    def _delete_selected_zone(self) -> None:
        zid = self.zones_panel.selected_zone_id()
        if zid:
            self.zone_manager.remove_zone(zid)
            self.zones_panel.refresh_zones()
            self._persist()

    def _toggle_selected_zone(self) -> None:
        zid = self.zones_panel.selected_zone_id()
        if not zid:
            return
        for z in self.zone_manager.zones:
            if z.id == zid:
                z.enabled = not z.enabled
                break
        self.zones_panel.refresh_zones()
        self._persist()

    def _rename_selected_zone(self) -> None:
        zid = self.zones_panel.selected_zone_id()
        if not zid:
            return
        zone = next((z for z in self.zone_manager.zones if z.id == zid), None)
        if not zone:
            return
        name, ok = QInputDialog.getText(self, "Rename zone", "New name:", text=zone.name)
        if ok and name.strip():
            zone.name = name.strip()
            self.zones_panel.refresh_zones()
            self._persist()

    def _export_event_log(self) -> None:
        default = app_data_dir() / f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export event log", str(default), "CSV (*.csv)")
        if path:
            self.event_log.export_csv(Path(path))
            self.status.showMessage(f"Exported log to {path}")

    def _export_session_metrics(self) -> None:
        default = app_data_dir() / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export session metrics", str(default), "CSV (*.csv)"
        )
        if path:
            self.metrics.export_session_csv(Path(path))

    def _capture_face_from_video(self) -> None:
        if not self._running or not self.pipeline:
            QMessageBox.information(self, "Faces", "Start detection first.")
            return
        result = self.pipeline.get_result()
        if result is None:
            return
        self.faces_panel.enroll_frame(result.frame)

    def _on_gallery_changed(self) -> None:
        if self.face_recognizer:
            self.face_recognizer.reload_gallery()

    @pyqtSlot()
    def _apply_settings(self, silent: bool = False) -> None:
        was_running = self._running
        if was_running:
            self.stop_detection()

        prev_model = str(self.cfg.get("_last_model_id", ""))
        prev_imgsz = int(self.cfg.get("model_imgsz", 640))
        self.cfg.update(self.settings.to_config_updates())
        self.cfg["model_preset"] = CUSTOM
        self.cfg["performance_profile"] = CUSTOM
        for i in range(self.perf_combo.count()):
            if self.perf_combo.itemData(i) == CUSTOM:
                self.perf_combo.blockSignals(True)
                self.perf_combo.setCurrentIndex(i)
                self.perf_combo.blockSignals(False)
                break
        ids = self.cfg.get("detect_class_ids", [0])
        if not isinstance(ids, list):
            ids = [0]
        pcc = self.cfg.get("confidence_per_class")
        if not isinstance(pcc, dict):
            pcc = {}
        self.detector.set_detect_class_ids(ids)
        self.detector.set_confidence_per_class(pcc)
        self.detector.debug_collect_faded = bool(self.cfg.get("show_faded_low_conf", False))
        self.detector.update_thresholds(
            confidence=self.cfg["confidence"],
            iou=self.cfg["iou"],
            max_detections=self.cfg["max_detections"],
        )
        self.alert_manager.sound_enabled = bool(self.cfg.get("alert_sound", True))
        self.alert_manager.toast_enabled = bool(self.cfg.get("alert_toast", True))
        self.alert_manager.flash_ms = int(self.cfg.get("alert_flash_ms", 800))
        self.settings.set_backend_label(
            self.detector.backend_name, getattr(self.detector, "using_gpu", False)
        )
        prev_model = str(self.cfg.get("_last_model_id", ""))
        new_model = str(self.cfg.get("active_model_id", ""))
        new_imgsz = int(self.cfg.get("model_imgsz", 640))
        new_ort = bool(self.cfg.get("ort_io_binding", True))
        prev_ort = bool(self.cfg.get("_last_ort_io_binding", new_ort))
        if new_model and (
            new_model != prev_model
            or new_imgsz != prev_imgsz
            or new_ort != prev_ort
        ):
            self._recreate_detector()
            self.cfg["_last_model_id"] = new_model
            self.cfg["_last_ort_io_binding"] = new_ort
        self.capture = create_frame_source(self.cfg)
        fe = bool(self.cfg.get("face_enabled", True))
        if fe and (self.face_recognizer is None or not self.face_recognizer.enabled):
            self.face_recognizer = self._build_face_recognizer()
        elif not fe:
            self.face_recognizer = None
        elif self.face_recognizer:
            self.face_recognizer.match_threshold = float(self.cfg.get("face_match_threshold", 0.45))
            self.face_recognizer.face_stride = int(self.cfg.get("face_stride", 3))
        self.faces_panel.set_recognizer(self.face_recognizer)
        self._persist()
        if not silent:
            self.status.showMessage("Settings applied")
        if was_running:
            self.start_detection()

    @pyqtSlot()
    def _rebuild_model(self) -> None:
        from src.utils.model_setup import onnx_path_for_model

        model = str(
            self.cfg.get("active_model_id", self.settings.model_combo.currentData() or "yolo11s")
        )
        imgsz = int(self.cfg.get("model_imgsz", 640))
        path = onnx_path_for_model(model, imgsz)
        if path.is_file():
            path.unlink()
        QMessageBox.information(
            self,
            "Rebuild model",
            f"Deleted {path.name}. Model re-exports on next start or when detection runs.",
        )

    def _format_enter_message(self, ev) -> str:
        ident = ev.identity if ev.identity and ev.identity != UNKNOWN else "Unknown person"
        return f"{ident} entered {ev.zone_name}"

    def _on_tick(self) -> None:
        if not self.pipeline:
            return
        result = self.pipeline.get_result()
        if result is None:
            return
        self.video.set_frame(result.frame)
        fc = self._file_capture()
        if fc and self.video_transport.isVisible():
            self.video_transport.update_position(fc.current_frame_index)

        for ev in result.zone_events:
            if ev.event_type == ZoneEventType.ENTER:
                key = (ev.zone_id, ev.track_id, ev.event_type.value)
                if key not in self._last_logged_events:
                    self._last_logged_events.add(key)
                    msg = self._format_enter_message(ev)
                    self.event_log.append(
                        "zone_enter",
                        msg,
                        ev.track_id,
                        ev.zone_name,
                        result.fps,
                        result.infer_ms,
                        ev.identity,
                        ev.match_score,
                    )
            elif ev.event_type == ZoneEventType.EXIT:
                key = (ev.zone_id, ev.track_id, "exit")
                if key not in self._last_logged_events:
                    self._last_logged_events.add(key)
                    ident = ev.identity if ev.identity != UNKNOWN else "Unknown"
                    msg = f"{ident} left {ev.zone_name}"
                    self.event_log.append(
                        "zone_exit",
                        msg,
                        ev.track_id,
                        ev.zone_name,
                        result.fps,
                        result.infer_ms,
                        ev.identity,
                        ev.match_score,
                    )

        self.zones_panel.refresh_events(self.event_log.recent_messages(15))
        self.monitor_events.clear()
        for msg in self.event_log.recent_messages(8):
            self.monitor_events.addItem(msg)
        self.analytics_panel.update_metrics(self.metrics)

        model_id = str(self.cfg.get("active_model_id", "yolo11s"))
        imgsz = int(self.cfg.get("model_imgsz", 640))
        self.perf_hud.update_from_result(
            result,
            backend=getattr(self.detector, "backend_name", "?"),
            using_gpu=getattr(self.detector, "using_gpu", False),
            model_id=model_id,
            imgsz=imgsz,
            screen_fps_cap=int(self.cfg.get("screen_fps_cap", 0)),
            force_cpu=bool(self.cfg.get("force_cpu", False)),
        )

        alerts = self.alert_manager.state.total_alerts
        backend = getattr(self.detector, "backend_name", "?")
        prof = str(self.cfg.get("model_preset", self.cfg.get("performance_profile", BALANCED)))
        self.status.showMessage(
            f"FPS {result.fps:.1f} | det {result.detect_ms:.1f}ms | pre {result.preprocess_ms:.1f}ms | "
            f"draw {result.draw_ms:.1f}ms | {result.capture_w}x{result.capture_h}→{result.infer_w}x{result.infer_h} | "
            f"In {result.in_frame_count} | Alerts {alerts} | {backend} | {prof} | {result.source_label}"
        )

    def _show_zones_help(self) -> None:
        ZonesHelpDialog(self).exec()

    def changeEvent(self, event) -> None:
        from PyQt6.QtCore import QEvent

        if (
            event.type() == QEvent.Type.WindowStateChange
            and self.windowState() & Qt.WindowState.WindowMinimized
            and bool(self.cfg.get("minimize_to_tray", True))
        ):
            self._tray.show()
            QTimer.singleShot(0, self.hide)
        super().changeEvent(event)

    def closeEvent(self, event) -> None:
        self.profile_manager.save(self.profile_combo.currentText(), self.cfg)
        self.stop_detection()
        self._persist()
        if self.face_recognizer:
            self.face_recognizer.close()
        super().closeEvent(event)
