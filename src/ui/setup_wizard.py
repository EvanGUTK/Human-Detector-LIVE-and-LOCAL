"""Multi-step first-run / setup assistant."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.core.performance_profiles import (
    BALANCED,
    COMPATIBILITY,
    FAST,
    QUALITY,
    apply_profile,
    profile_label,
    preset_ids,
)


def _run_gpu_check(root: Path) -> tuple[bool, str]:
    script = root / "scripts" / "check_gpu.ps1"
    if not script.is_file():
        return False, "check_gpu.ps1 not found"
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(root),
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        ok = "GPU: True" in out or "GPU:True" in out.replace(" ", "")
        return ok, out[-2000:] if len(out) > 2000 else out
    except Exception as exc:
        return False, str(exc)


class SetupWizard(QDialog):
    def __init__(self, cfg: dict, project_root: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Person Detector — Setup")
        self.resize(520, 380)
        self._cfg = dict(cfg)
        self._root = project_root
        self._result_cfg = dict(cfg)

        layout = QVBoxLayout(self)
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        # Step 0 — GPU
        p0 = QWidget()
        l0 = QVBoxLayout(p0)
        l0.addWidget(QLabel("<b>Step 1 — GPU check</b>"))
        self._gpu_label = QLabel("Checking ONNX CUDA…")
        self._gpu_label.setWordWrap(True)
        self._gpu_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        l0.addWidget(self._gpu_label)
        self._stack.addWidget(p0)

        # Step 1 — Source
        p1 = QWidget()
        l1 = QVBoxLayout(p1)
        l1.addWidget(QLabel("<b>Step 2 — Input source</b>"))
        self._source_combo = QComboBox()
        self._source_combo.addItem("Webcam", "webcam")
        self._source_combo.addItem("Video file", "file")
        self._source_combo.addItem("Desktop / screen (CCTV testing)", "screen")
        src = str(cfg.get("input_source", "webcam"))
        for i in range(self._source_combo.count()):
            if self._source_combo.itemData(i) == src:
                self._source_combo.setCurrentIndex(i)
                break
        l1.addWidget(self._source_combo)
        l1.addWidget(
            QLabel(
                "For best FPS on large monitors, use screen capture with a "
                "region ROI (set after setup via Desktop capture)."
            )
        )
        self._stack.addWidget(p1)

        # Step 2 — Preset
        p2 = QWidget()
        l2 = QVBoxLayout(p2)
        l2.addWidget(QLabel("<b>Step 3 — Model preset</b>"))
        self._preset_combo = QComboBox()
        for pid in preset_ids():
            if pid == COMPATIBILITY:
                continue
            self._preset_combo.addItem(profile_label(pid), pid)
        self._preset_combo.addItem(profile_label(COMPATIBILITY), COMPATIBILITY)
        l2.addWidget(self._preset_combo)
        l2.addWidget(
            QLabel(
                "Fast = yolo11n (max FPS)\n"
                "Balanced = yolo11s (recommended)\n"
                "Quality = yolo11m @ 960 (heavier)"
            )
        )
        self._stack.addWidget(p2)

        # Step 3 — Zones
        p3 = QWidget()
        l3 = QVBoxLayout(p3)
        l3.addWidget(QLabel("<b>Step 4 — Zones (optional)</b>"))
        l3.addWidget(
            QLabel(
                "After setup, open the Monitor tab, click Start, then "
                "<b>Draw Zone</b> to define alert areas.\n\n"
                "Zones + faded debug boxes stay enabled in Balanced/Fast presets."
            )
        )
        self._stack.addWidget(p3)

        # Step 4 — Done
        p4 = QWidget()
        l4 = QVBoxLayout(p4)
        l4.addWidget(QLabel("<b>Ready</b>"))
        self._done_label = QLabel("")
        self._done_label.setWordWrap(True)
        l4.addWidget(self._done_label)
        self._stack.addWidget(p4)

        self._btn_box = QDialogButtonBox()
        self._back = self._btn_box.addButton("Back", QDialogButtonBox.ButtonRole.ActionRole)
        self._next = self._btn_box.addButton("Next", QDialogButtonBox.ButtonRole.ActionRole)
        self._finish = self._btn_box.addButton(
            QDialogButtonBox.StandardButton.Finish
        )
        self._finish.hide()
        self._back.hide()
        self._btn_box.rejected.connect(self.reject)
        self._next.clicked.connect(self._on_next)
        self._back.clicked.connect(self._on_back)
        self._finish.clicked.connect(self._on_finish)
        layout.addWidget(self._btn_box)

        self._step = 0
        self._gpu_ok = False
        self._refresh_gpu_step()

    def _refresh_gpu_step(self) -> None:
        ok, detail = _run_gpu_check(self._root)
        self._gpu_ok = ok
        status = "GPU inference: OK (ONNX CUDA)" if ok else "GPU inference: not active (CPU fallback)"
        self._gpu_label.setText(f"{status}\n\n{detail}")

    def _on_next(self) -> None:
        if self._step == 0:
            self._step = 1
            self._stack.setCurrentIndex(1)
            self._back.show()
        elif self._step == 1:
            self._step = 2
            self._stack.setCurrentIndex(2)
        elif self._step == 2:
            self._step = 3
            self._stack.setCurrentIndex(3)
            self._next.setText("Next")
        elif self._step == 3:
            self._step = 4
            self._stack.setCurrentIndex(4)
            preset = str(self._preset_combo.currentData() or BALANCED)
            src = str(self._source_combo.currentData() or "webcam")
            self._result_cfg = apply_profile(self._cfg, preset)
            self._result_cfg["input_source"] = src
            self._result_cfg["first_run_complete"] = True
            self._done_label.setText(
                f"Preset: {profile_label(preset)}\n"
                f"Source: {src}\n"
                f"GPU: {'yes' if self._gpu_ok else 'no (see scripts/check_gpu.ps1)'}\n\n"
                "Click Finish to start."
            )
            self._next.hide()
            self._finish.show()
            return
        self._stack.setCurrentIndex(self._step)

    def _on_back(self) -> None:
        if self._step <= 1:
            self._back.hide()
        if self._step > 0:
            self._step -= 1
            self._stack.setCurrentIndex(self._step)
        if self._step < 4:
            self._next.show()
            self._finish.hide()
            self._next.setText("Next")

    def _on_finish(self) -> None:
        self.accept()

    def result_config(self) -> dict:
        return self._result_cfg
