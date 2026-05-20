"""Settings sidebar panel."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.utils.coco_classes import COCO80_NAMES, CCTV_TRAFFIC_CLASS_IDS
from src.utils.config import migrate_config_inplace, sync_legacy_detect_flags


class SettingsPanel(QWidget):
    settings_changed = pyqtSignal()
    rebuild_model_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._detect_class_ids: list[int] = [0]
        self._zone_alert_class_ids: list[int] = [0]
        self._face_rec_class_ids: list[int] = [0]
        root = QVBoxLayout(self)
        self._settings_tabs = QTabWidget()
        root.addWidget(self._settings_tabs)

        def _scroll_tab(inner: QWidget) -> QWidget:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(inner)
            return scroll

        def _tab_page() -> tuple[QWidget, QVBoxLayout]:
            page = QWidget()
            return page, QVBoxLayout(page)

        cam_group = QGroupBox("Input — camera")
        cam_form = QFormLayout(cam_group)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 10)
        cam_form.addRow("Index:", self.camera_index)
        self.capture_width = QSpinBox()
        self.capture_width.setRange(320, 3840)
        self.capture_width.setSingleStep(160)
        cam_form.addRow("Width:", self.capture_width)
        self.capture_height = QSpinBox()
        self.capture_height.setRange(240, 2160)
        self.capture_height.setSingleStep(120)
        cam_form.addRow("Height:", self.capture_height)
        self.mirror = QCheckBox("Mirror webcam")
        cam_form.addRow(self.mirror)

        perf_group = QGroupBox("Display & performance")
        perf_form = QFormLayout(perf_group)
        self.infer_max_width = QSpinBox()
        self.infer_max_width.setRange(0, 3840)
        self.infer_max_width.setSingleStep(160)
        self.infer_max_width.setToolTip("Max width before letterbox (0 = use full capture size)")
        perf_form.addRow("Infer max width:", self.infer_max_width)
        self.infer_max_height = QSpinBox()
        self.infer_max_height.setRange(0, 2160)
        self.infer_max_height.setSingleStep(120)
        perf_form.addRow("Infer max height:", self.infer_max_height)
        self.preview_mode = QComboBox()
        self.preview_mode.addItem("Performance", "performance")
        self.preview_mode.addItem("Full resolution", "full")
        perf_form.addRow("Preview mode:", self.preview_mode)
        self.screen_fps_cap = QSpinBox()
        self.screen_fps_cap.setRange(0, 144)
        self.screen_fps_cap.setSpecialValueText("Uncapped")
        perf_form.addRow("Screen FPS cap:", self.screen_fps_cap)
        self.ort_io_binding = QCheckBox("ONNX IO binding (CUDA)")
        self.ort_io_binding.setChecked(True)
        perf_form.addRow(self.ort_io_binding)
        self.btn_setup_wizard = QPushButton("Setup assistant…")
        perf_form.addRow(self.btn_setup_wizard)

        det_group = QGroupBox("Detection")
        det_form = QFormLayout(det_group)
        self.detect_person = QCheckBox("Quick: detect person (class 0)")
        self.detect_person.setChecked(True)
        self.detect_car = QCheckBox("Quick: detect car (class 2)")
        det_form.addRow(self.detect_person)
        det_form.addRow(self.detect_car)

        self.class_summary = QLabel("Classes: 1 selected.")
        det_form.addRow(self.class_summary)
        class_row = QHBoxLayout()
        self.btn_coco = QPushButton("COCO classes…")
        self.btn_coco.setToolTip("Search and tick any of the 80 COCO classes")
        self.btn_cctv = QPushButton("CCTV preset")
        self.btn_cctv.setToolTip(
            "Traffic-focused classes, YOLO11m, 960px, lower vehicle thresholds (slower, better recall)"
        )
        self.btn_import_onnx = QPushButton("Import ONNX…")
        self.btn_import_onnx.setToolTip("Register a bring-your-own ONNX (e.g. NGC / TAO export)")
        class_row.addWidget(self.btn_coco)
        class_row.addWidget(self.btn_cctv)
        class_row.addWidget(self.btn_import_onnx)
        det_form.addRow(class_row)
        self.zone_class_summary = QLabel("Zone alerts: person")
        zrow = QHBoxLayout()
        self.btn_zone_classes = QPushButton("Zone classes…")
        zrow.addWidget(self.btn_zone_classes)
        zrow.addWidget(self.zone_class_summary)
        det_form.addRow(zrow)
        self.face_class_summary = QLabel("Face ID classes: person")
        frow = QHBoxLayout()
        self.btn_face_classes = QPushButton("Face-ID classes…")
        frow.addWidget(self.btn_face_classes)
        frow.addWidget(self.face_class_summary)
        det_form.addRow(frow)

        self.model_combo = QComboBox()
        det_form.addRow("Model:", self.model_combo)

        self.model_name = QComboBox()
        self.model_name.addItems(["yolo11n", "yolo11s"])
        self.model_name.hide()
        det_form.addRow("Base (legacy):", self.model_name)

        self.model_imgsz = QComboBox()
        for sz in (416, 640, 960, 1280):
            self.model_imgsz.addItem(str(sz), sz)
        self.model_imgsz.setToolTip(
            "Inference resolution. Larger = better small-object recall, slower. "
            "ONNX is stored per size (e.g. yolo11m_960.onnx)."
        )
        det_form.addRow("Inference size:", self.model_imgsz)

        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.05, 0.95)
        self.confidence.setSingleStep(0.05)
        self.confidence.setDecimals(2)
        det_form.addRow("Confidence (default):", self.confidence)

        self.show_faded_low_conf = QCheckBox("Show below-threshold boxes (faded, debug)")
        self.show_faded_low_conf.setToolTip(
            "Draws detections between a low floor and the per-class threshold in gray"
        )
        det_form.addRow(self.show_faded_low_conf)

        self.iou = QDoubleSpinBox()
        self.iou.setRange(0.1, 0.9)
        self.iou.setSingleStep(0.05)
        self.iou.setDecimals(2)
        det_form.addRow("IoU:", self.iou)
        self.max_detections = QSpinBox()
        self.max_detections.setRange(1, 100)
        det_form.addRow("Max detections:", self.max_detections)
        self.backend_label = QLabel("Inference: —")
        self.backend_label.setStyleSheet("color: #8cf; font-weight: bold;")
        det_form.addRow("Backend:", self.backend_label)

        thr_group = QGroupBox("Per-class thresholds (selected classes)")
        thr_group.setToolTip("Override default confidence for specific classes. Blank row uses default.")
        thr_layout = QVBoxLayout(thr_group)
        self.per_class_table = QTableWidget(0, 2)
        self.per_class_table.setHorizontalHeaderLabels(["Class", "Min confidence"])
        self.per_class_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.per_class_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.per_class_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        thr_layout.addWidget(self.per_class_table)

        fire_group = QGroupBox("Fire / smoke (optional second ONNX)")
        fire_form = QFormLayout(fire_group)
        self.fire_enabled = QCheckBox("Enable fire/smoke model")
        self.fire_path = QLabel("(no file)")
        self.fire_path.setWordWrap(True)
        btn_fire_pick = QPushButton("ONNX file…")
        btn_fire_pick.clicked.connect(self._pick_fire_onnx)
        fire_form.addRow(self.fire_enabled)
        fire_form.addRow(btn_fire_pick)
        fire_form.addRow(self.fire_path)
        self.fire_full_frame = QCheckBox("Run on full frame (else center half crop)")
        self.fire_full_frame.setChecked(True)
        fire_form.addRow(self.fire_full_frame)

        skel_group = QGroupBox("Pose skeleton (MediaPipe)")
        skel_form = QFormLayout(skel_group)
        self.show_pose_skeleton = QCheckBox("Show body skeleton")
        skel_form.addRow(self.show_pose_skeleton)
        self.show_hand_skeleton = QCheckBox("Show hand bones")
        self.show_hand_skeleton.setToolTip(
            "Works on its own or together with body skeleton"
        )
        skel_form.addRow(self.show_hand_skeleton)
        self.skeleton_stride = QSpinBox()
        self.skeleton_stride.setRange(1, 5)
        self.skeleton_stride.setToolTip("Run pose every N frames (higher = faster)")
        skel_form.addRow("Stride:", self.skeleton_stride)

        disp_group = QGroupBox("On-frame HUD")
        disp_form = QFormLayout(disp_group)
        self.show_debug_overlay = QCheckBox("Debug overlay (feet, zone in/out)")
        disp_form.addRow(self.show_debug_overlay)
        self.show_hud_on_frame = QCheckBox("HUD on video (FPS text)")
        disp_form.addRow(self.show_hud_on_frame)

        face_group = QGroupBox("Face recognition")
        face_form = QFormLayout(face_group)
        self.face_enabled = QCheckBox("Enable face recognition")
        face_form.addRow(self.face_enabled)
        self.face_match_threshold = QDoubleSpinBox()
        self.face_match_threshold.setRange(0.2, 0.9)
        self.face_match_threshold.setSingleStep(0.05)
        self.face_match_threshold.setDecimals(2)
        face_form.addRow("Match threshold:", self.face_match_threshold)
        self.face_stride = QSpinBox()
        self.face_stride.setRange(1, 10)
        face_form.addRow("Detect every N frames:", self.face_stride)

        alert_group = QGroupBox("Alerts")
        alert_form = QFormLayout(alert_group)
        self.alert_sound = QCheckBox("Play sound")
        alert_form.addRow(self.alert_sound)
        self.alert_toast = QCheckBox("Windows toast")
        alert_form.addRow(self.alert_toast)
        self.alert_flash_ms = QSpinBox()
        self.alert_flash_ms.setRange(200, 3000)
        self.alert_flash_ms.setSingleStep(100)
        alert_form.addRow("Flash (ms):", self.alert_flash_ms)

        tab_in, lay_in = _tab_page()
        lay_in.addWidget(cam_group)
        lay_in.addStretch()
        self._settings_tabs.addTab(_scroll_tab(tab_in), "Input")

        tab_disp, lay_disp = _tab_page()
        lay_disp.addWidget(perf_group)
        lay_disp.addWidget(disp_group)
        lay_disp.addStretch()
        self._settings_tabs.addTab(_scroll_tab(tab_disp), "Display")

        tab_det, lay_det = _tab_page()
        lay_det.addWidget(det_group)
        lay_det.addWidget(thr_group)
        lay_det.addWidget(fire_group)
        lay_det.addStretch()
        self._settings_tabs.addTab(_scroll_tab(tab_det), "Detection")

        tab_alerts, lay_alerts = _tab_page()
        lay_alerts.addWidget(alert_group)
        lay_alerts.addStretch()
        self._settings_tabs.addTab(_scroll_tab(tab_alerts), "Alerts")

        tab_adv, lay_adv = _tab_page()
        lay_adv.addWidget(skel_group)
        lay_adv.addWidget(face_group)
        lay_adv.addStretch()
        self._settings_tabs.addTab(_scroll_tab(tab_adv), "Advanced")

        tab_lab, lay_lab = _tab_page()
        lab_group = QGroupBox("Lab defaults")
        lab_form = QFormLayout(lab_group)
        self.compare_model_a = QComboBox()
        self.compare_model_b = QComboBox()
        lab_form.addRow("Compare model A:", self.compare_model_a)
        lab_form.addRow("Compare model B:", self.compare_model_b)
        self.lab_clips_dir = QLabel("")
        self.lab_clips_dir.setWordWrap(True)
        lab_form.addRow("Test clips folder:", self.lab_clips_dir)
        self.ngc_status = QLabel("NGC: not checked")
        lab_form.addRow(self.ngc_status)
        self.btn_check_ngc = QPushButton("Check NGC / models")
        self.btn_check_ngc.clicked.connect(self._check_ngc_status)
        lab_form.addRow(self.btn_check_ngc)
        lay_lab.addWidget(lab_group)
        lay_lab.addStretch()
        self._settings_tabs.addTab(_scroll_tab(tab_lab), "Lab")

        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.rebuild_btn = QPushButton("Rebuild model")
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.rebuild_btn)
        root.addLayout(btn_row)
        self.apply_btn.clicked.connect(self.settings_changed.emit)
        self.rebuild_btn.clicked.connect(self.rebuild_model_requested.emit)

        self.detect_person.stateChanged.connect(lambda _: self._on_quick_class_toggle())
        self.detect_car.stateChanged.connect(lambda _: self._on_quick_class_toggle())
        self.btn_coco.clicked.connect(self._open_coco_picker)
        self.btn_cctv.clicked.connect(self._apply_cctv_preset)
        self.btn_import_onnx.clicked.connect(self._import_external_onnx)
        self.btn_zone_classes.clicked.connect(self._open_zone_class_picker)
        self.btn_face_classes.clicked.connect(self._open_face_class_picker)
        self.model_combo.currentIndexChanged.connect(lambda _: self._refresh_class_summary())

        self._fire_onnx_path = ""

    def _active_class_names(self) -> list[str]:
        from src.lab.model_registry import ModelRegistry

        mid = self.model_combo.currentData()
        if isinstance(mid, str) and mid:
            return ModelRegistry().class_names_for_model(mid)
        return list(COCO80_NAMES)

    def _refresh_class_summary(self) -> None:
        n = len(self._detect_class_ids)
        names = self._active_class_names()
        sample = ", ".join(names[i] if 0 <= i < len(names) else f"class_{i}" for i in self._detect_class_ids[:6])
        if len(self._detect_class_ids) > 6:
            sample += ", …"
        self.class_summary.setText(f"Classes: {n} selected ({sample})")
        zsample = ", ".join(names[i] if 0 <= i < len(names) else f"class_{i}" for i in self._zone_alert_class_ids[:4])
        fsample = ", ".join(names[i] if 0 <= i < len(names) else f"class_{i}" for i in self._face_rec_class_ids[:4])
        self.zone_class_summary.setText(f"Zone alerts: {zsample or 'person'}")
        self.face_class_summary.setText(f"Face ID classes: {fsample or 'person'}")

    def _on_quick_class_toggle(self) -> None:
        s = set(self._detect_class_ids)
        if self.detect_person.isChecked():
            s.add(0)
        else:
            s.discard(0)
        if self.detect_car.isChecked():
            s.add(2)
        else:
            s.discard(2)
        if not s:
            s.add(0)
            self.detect_person.blockSignals(True)
            self.detect_person.setChecked(True)
            self.detect_person.blockSignals(False)
        self._detect_class_ids = sorted(s)
        self._refresh_class_summary()
        self._populate_per_class_table()

    def _sync_quick_checkboxes(self) -> None:
        self.detect_person.blockSignals(True)
        self.detect_car.blockSignals(True)
        self.detect_person.setChecked(0 in self._detect_class_ids)
        self.detect_car.setChecked(2 in self._detect_class_ids)
        self.detect_person.blockSignals(False)
        self.detect_car.blockSignals(False)

    def _open_coco_picker(self) -> None:
        from src.ui.coco_class_picker import CocoClassPickerDialog

        dlg = CocoClassPickerDialog(list(self._detect_class_ids), self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._detect_class_ids = dlg.selected_ids()
            self._sync_quick_checkboxes()
            self._refresh_class_summary()
            self._populate_per_class_table()

    def _open_zone_class_picker(self) -> None:
        from src.ui.coco_class_picker import CocoClassPickerDialog

        dlg = CocoClassPickerDialog(list(self._zone_alert_class_ids), self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            ids = dlg.selected_ids()
            self._zone_alert_class_ids = ids if ids else [0]
            self._refresh_class_summary()

    def _open_face_class_picker(self) -> None:
        from src.ui.coco_class_picker import CocoClassPickerDialog

        dlg = CocoClassPickerDialog(list(self._face_rec_class_ids), self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            ids = dlg.selected_ids()
            self._face_rec_class_ids = ids if ids else [0]
            self._refresh_class_summary()

    def _apply_cctv_preset(self) -> None:
        self._detect_class_ids = list(CCTV_TRAFFIC_CLASS_IDS)
        self._sync_quick_checkboxes()
        self._refresh_class_summary()
        self.model_imgsz.setCurrentIndex(self.model_imgsz.findData(960))
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == "yolo11m":
                self.model_combo.setCurrentIndex(i)
                break
        self.confidence.setValue(0.45)
        vehicle_low = 0.28
        pcc = {
            "bicycle": vehicle_low,
            "car": vehicle_low,
            "motorcycle": vehicle_low,
            "bus": vehicle_low,
            "truck": vehicle_low,
            "person": 0.45,
        }
        self._populate_per_class_table(extra_defaults=pcc)
        self._refresh_class_summary()

    def _populate_per_class_table(self, extra_defaults: dict[str, float] | None = None) -> None:
        defaults = dict(extra_defaults or {})
        gconf = float(self.confidence.value())
        pcc: dict[str, float] = {}
        for row in range(self.per_class_table.rowCount()):
            name_item = self.per_class_table.item(row, 0)
            spin = self.per_class_table.cellWidget(row, 1)
            if name_item and spin and isinstance(spin, QDoubleSpinBox):
                pcc[name_item.text()] = float(spin.value())
        merged = {**defaults}
        for k, v in pcc.items():
            merged[k] = v
        names = self._active_class_names()
        self.per_class_table.setRowCount(0)
        for cid in sorted(self._detect_class_ids):
            if not (0 <= cid < len(names)):
                continue
            name = names[cid]
            row = self.per_class_table.rowCount()
            self.per_class_table.insertRow(row)
            self.per_class_table.setItem(row, 0, QTableWidgetItem(name))
            spin = QDoubleSpinBox()
            spin.setRange(0.05, 0.95)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setValue(float(merged.get(name, gconf)))
            self.per_class_table.setCellWidget(row, 1, spin)

    def _pick_fire_onnx(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Fire/smoke ONNX", "", "ONNX (*.onnx);;All (*.*)"
        )
        if path:
            self._fire_onnx_path = path
            self.fire_path.setText(path)

    def _import_external_onnx(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import ONNX (e.g. NGC / TAO)", "", "ONNX (*.onnx);;All (*.*)"
        )
        if not path:
            return
        p = Path(path)
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,
            "Display name",
            "Label for this model (shown in the model list):",
            text=f"NGC: {p.stem}",
        )
        if not ok or not name.strip():
            return
        from src.lab.model_registry import ModelRegistry
        import re
        import time

        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", p.stem).strip("_") or "custom"
        model_id = f"ngc_{slug}_{int(time.time())}"
        stem = p.stem.lower()
        family = "yolo"
        class_names = None
        if "rtdetr" in stem:
            family = "rtdetr"
        elif "peoplenet" in stem:
            family = "peoplenet"
            class_names = ["person", "bag", "face"]
        elif "detectnet" in stem:
            family = "detectnet_v2"
            class_names = ["person", "bag", "face"]
        ModelRegistry().register_custom(
            model_id,
            name.strip(),
            p,
            family=family,
            class_names=class_names,
        )
        self.refresh_model_list(model_id)
        QMessageBox.information(
            self,
            "Import ONNX",
            f"Registered as '{name.strip()}'. Select it under Model.\n\n"
            "Supported outputs: YOLO/RT-DETR style, plus TAO PeopleNet/DetectNet_v2.",
        )

    def load_from_config(self, cfg: dict) -> None:
        tmp = dict(cfg)
        migrate_config_inplace(tmp)
        ids = tmp.get("detect_class_ids", [0])
        self._detect_class_ids = list(ids) if isinstance(ids, list) else [0]
        zids = tmp.get("zone_alert_class_ids", [0])
        fids = tmp.get("face_rec_class_ids", [0])
        self._zone_alert_class_ids = list(zids) if isinstance(zids, list) and zids else [0]
        self._face_rec_class_ids = list(fids) if isinstance(fids, list) and fids else [0]

        self.camera_index.setValue(int(cfg.get("camera_index", 0)))
        self.capture_width.setValue(int(cfg.get("capture_width", 1280)))
        self.capture_height.setValue(int(cfg.get("capture_height", 720)))
        self.mirror.setChecked(bool(cfg.get("mirror", False)))
        model = str(cfg.get("model_name", "yolo11s"))
        idx = self.model_name.findText(model)
        if idx >= 0:
            self.model_name.setCurrentIndex(idx)
        self.confidence.setValue(float(cfg.get("confidence", 0.45)))
        self.iou.setValue(float(cfg.get("iou", 0.5)))
        self.max_detections.setValue(int(cfg.get("max_detections", 50)))
        self.alert_sound.setChecked(bool(cfg.get("alert_sound", True)))
        self.alert_toast.setChecked(bool(cfg.get("alert_toast", True)))
        self.alert_flash_ms.setValue(int(cfg.get("alert_flash_ms", 800)))
        self.show_pose_skeleton.setChecked(bool(cfg.get("show_pose_skeleton", False)))
        self.show_hand_skeleton.setChecked(bool(cfg.get("show_hand_skeleton", True)))
        self.skeleton_stride.setValue(int(cfg.get("skeleton_stride", 2)))
        self.show_debug_overlay.setChecked(bool(cfg.get("show_debug_overlay", False)))
        self.show_hud_on_frame.setChecked(bool(cfg.get("show_hud_on_frame", True)))
        self.face_enabled.setChecked(bool(cfg.get("face_enabled", True)))
        self.face_match_threshold.setValue(float(cfg.get("face_match_threshold", 0.45)))
        self.face_stride.setValue(int(cfg.get("face_stride", 3)))

        imsz = int(cfg.get("model_imgsz", 640))
        ix = self.model_imgsz.findData(imsz)
        if ix >= 0:
            self.model_imgsz.setCurrentIndex(ix)
        else:
            self.model_imgsz.setCurrentIndex(self.model_imgsz.findData(640))

        self.show_faded_low_conf.setChecked(bool(cfg.get("show_faded_low_conf", False)))
        self.fire_enabled.setChecked(bool(cfg.get("fire_smoke_enabled", False)))
        fp = str(cfg.get("fire_smoke_onnx_path", "") or "")
        self._fire_onnx_path = fp
        self.fire_path.setText(fp if fp else "(no file)")
        self.fire_full_frame.setChecked(bool(cfg.get("fire_smoke_full_frame", True)))

        self.infer_max_width.setValue(int(cfg.get("infer_max_width", 1280)))
        self.infer_max_height.setValue(int(cfg.get("infer_max_height", 720)))
        pm = str(cfg.get("preview_mode", "performance"))
        self.preview_mode.setCurrentIndex(0 if pm == "performance" else 1)
        self.screen_fps_cap.setValue(int(cfg.get("screen_fps_cap", 0)))
        self.ort_io_binding.setChecked(bool(cfg.get("ort_io_binding", True)))

        self._sync_quick_checkboxes()
        self._refresh_class_summary()
        self.refresh_model_list(str(cfg.get("active_model_id", cfg.get("model_name", "yolo11s"))))
        pcc = cfg.get("confidence_per_class")
        self._populate_per_class_table(
            extra_defaults=pcc if isinstance(pcc, dict) else None
        )
        self._refresh_lab_tab(cfg)

    def _refresh_lab_tab(self, cfg: dict) -> None:
        from src.ui.lab_dashboard import clips_dir

        self.compare_model_a.clear()
        self.compare_model_b.clear()
        from src.lab.model_registry import ModelRegistry

        for m in ModelRegistry().list_models():
            self.compare_model_a.addItem(m.display_name, m.id)
            self.compare_model_b.addItem(m.display_name, m.id)
        for combo, key, default in (
            (self.compare_model_a, "compare_model_a", "yolo11s"),
            (self.compare_model_b, "compare_model_b", "yolo11m"),
        ):
            mid = str(cfg.get(key, default))
            for i in range(combo.count()):
                if combo.itemData(i) == mid:
                    combo.setCurrentIndex(i)
                    break
        self.lab_clips_dir.setText(str(clips_dir()))

    def _check_ngc_status(self) -> None:
        import os
        import shutil

        key = os.environ.get("NGC_API_KEY", "").strip()
        cli = shutil.which("ngc")
        if key and cli:
            self.ngc_status.setText("NGC: API key set, CLI found")
            self.ngc_status.setStyleSheet("color: #6f6;")
        elif key:
            self.ngc_status.setText("NGC: API key set — install NGC CLI for auto-download")
            self.ngc_status.setStyleSheet("color: #ff8c66;")
        else:
            self.ngc_status.setText("NGC: set NGC_API_KEY for PeopleNet / DetectNet_v2 download")
            self.ngc_status.setStyleSheet("color: #9aa3b2;")

    def refresh_model_list(self, select_id: str | None = None) -> None:
        from src.lab.model_registry import ModelRegistry

        cur = select_id or self.model_combo.currentData()
        self.model_combo.clear()
        for m in ModelRegistry().list_models():
            self.model_combo.addItem(m.display_name, m.id)
        if cur:
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == cur:
                    self.model_combo.setCurrentIndex(i)
                    break
        self._refresh_class_summary()
        self._populate_per_class_table()
        from src.utils.config import load_config

        self._refresh_lab_tab(load_config())

    def set_backend_label(self, backend: str, using_gpu: bool = False) -> None:
        if using_gpu and backend == "onnx":
            text, color = "ONNX (CUDA)", "#6f6"
        elif using_gpu and backend == "torch":
            text, color = "PyTorch (CUDA)", "#6f6"
        else:
            text, color = "CPU (install CUDA 12 for 60+ FPS)", "#f66"
        self.backend_label.setText(text)
        self.backend_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def to_config_updates(self) -> dict:
        """Collect confidence_per_class from the table."""
        pcc: dict[str, float] = {}
        gconf = float(self.confidence.value())
        for row in range(self.per_class_table.rowCount()):
            name_item = self.per_class_table.item(row, 0)
            spin = self.per_class_table.cellWidget(row, 1)
            if name_item and spin and isinstance(spin, QDoubleSpinBox):
                v = float(spin.value())
                if abs(v - gconf) > 1e-6:
                    pcc[name_item.text()] = v

        tmp: dict = {
            "detect_class_ids": list(self._detect_class_ids),
        }
        sync_legacy_detect_flags(tmp)

        return {
            "camera_index": self.camera_index.value(),
            "capture_width": self.capture_width.value(),
            "capture_height": self.capture_height.value(),
            "mirror": self.mirror.isChecked(),
            "model_name": self.model_combo.currentData() or self.model_name.currentText(),
            "active_model_id": self.model_combo.currentData() or self.model_name.currentText(),
            "model_imgsz": int(self.model_imgsz.currentData() or 640),
            "detect_person": tmp["detect_person"],
            "detect_car": tmp["detect_car"],
            "detect_class_ids": tmp["detect_class_ids"],
            "zone_alert_class_ids": list(self._zone_alert_class_ids),
            "face_rec_class_ids": list(self._face_rec_class_ids),
            "confidence": gconf,
            "confidence_per_class": pcc,
            "show_faded_low_conf": self.show_faded_low_conf.isChecked(),
            "fire_smoke_enabled": self.fire_enabled.isChecked(),
            "fire_smoke_onnx_path": self._fire_onnx_path,
            "fire_smoke_full_frame": self.fire_full_frame.isChecked(),
            "iou": self.iou.value(),
            "max_detections": self.max_detections.value(),
            "alert_sound": self.alert_sound.isChecked(),
            "alert_toast": self.alert_toast.isChecked(),
            "alert_flash_ms": self.alert_flash_ms.value(),
            "show_pose_skeleton": self.show_pose_skeleton.isChecked(),
            "show_hand_skeleton": self.show_hand_skeleton.isChecked(),
            "skeleton_stride": self.skeleton_stride.value(),
            "show_debug_overlay": self.show_debug_overlay.isChecked(),
            "show_hud_on_frame": self.show_hud_on_frame.isChecked(),
            "face_enabled": self.face_enabled.isChecked(),
            "face_match_threshold": self.face_match_threshold.value(),
            "face_stride": self.face_stride.value(),
            "infer_max_width": self.infer_max_width.value(),
            "infer_max_height": self.infer_max_height.value(),
            "preview_mode": str(self.preview_mode.currentData() or "performance"),
            "screen_fps_cap": self.screen_fps_cap.value(),
            "ort_io_binding": self.ort_io_binding.isChecked(),
            "compare_model_a": self.compare_model_a.currentData() or "yolo11s",
            "compare_model_b": self.compare_model_b.currentData() or "yolo11m",
        }
