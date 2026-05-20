"""Pick monitor and screen region for desktop capture."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from src.core.screen_capture import list_monitors


class ScreenRegionDialog(QDialog):
    def __init__(self, cfg: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Desktop capture region")
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.monitor_combo = QComboBox()
        self._monitors = list_monitors()
        for m in self._monitors:
            self.monitor_combo.addItem(
                f"Monitor {m['index']} ({m['width']}x{m['height']})",
                m["index"],
            )
        idx = int(cfg.get("screen_monitor", 1))
        for i in range(self.monitor_combo.count()):
            if self.monitor_combo.itemData(i) == idx:
                self.monitor_combo.setCurrentIndex(i)
                break
        form.addRow("Monitor:", self.monitor_combo)

        region = cfg.get("screen_region") or [0, 0, 0, 0]
        self.x_spin = QSpinBox()
        self.x_spin.setRange(0, 10000)
        self.y_spin = QSpinBox()
        self.y_spin.setRange(0, 10000)
        self.w_spin = QSpinBox()
        self.w_spin.setRange(0, 10000)
        self.h_spin = QSpinBox()
        self.h_spin.setRange(0, 10000)
        self.x_spin.setValue(int(region[0]) if len(region) > 0 else 0)
        self.y_spin.setValue(int(region[1]) if len(region) > 1 else 0)
        self.w_spin.setValue(int(region[2]) if len(region) > 2 else 0)
        self.h_spin.setValue(int(region[3]) if len(region) > 3 else 0)
        form.addRow("Region X:", self.x_spin)
        form.addRow("Region Y:", self.y_spin)
        form.addRow("Region W (0=full):", self.w_spin)
        form.addRow("Region H (0=full):", self.h_spin)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(0, 144)
        self.fps_spin.setSpecialValueText("Uncapped")
        self.fps_spin.setValue(int(cfg.get("screen_fps_cap", 0)))
        form.addRow("Max FPS (0=uncapped):", self.fps_spin)

        self.infer_w_spin = QSpinBox()
        self.infer_w_spin.setRange(0, 3840)
        self.infer_w_spin.setSingleStep(160)
        self.infer_w_spin.setValue(int(cfg.get("infer_max_width", 1280)))
        form.addRow("Infer max width (0=full):", self.infer_w_spin)

        self.infer_h_spin = QSpinBox()
        self.infer_h_spin.setRange(0, 2160)
        self.infer_h_spin.setSingleStep(120)
        self.infer_h_spin.setValue(int(cfg.get("infer_max_height", 720)))
        form.addRow("Infer max height (0=full):", self.infer_h_spin)

        self.preview_combo = QComboBox()
        self.preview_combo.addItem("Performance (faster display)", "performance")
        self.preview_combo.addItem("Full resolution preview", "full")
        pm = str(cfg.get("preview_mode", "performance"))
        self.preview_combo.setCurrentIndex(0 if pm == "performance" else 1)
        form.addRow("Preview mode:", self.preview_combo)

        layout.addLayout(form)
        layout.addWidget(
            QLabel(
                "Set W and H to 0 to capture the full monitor.\n"
                "Region is relative to the monitor top-left."
            )
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result_config(self) -> dict:
        w, h = self.w_spin.value(), self.h_spin.value()
        region = None
        if w > 0 and h > 0:
            region = [self.x_spin.value(), self.y_spin.value(), w, h]
        return {
            "screen_monitor": int(self.monitor_combo.currentData()),
            "screen_region": region,
            "screen_fps_cap": self.fps_spin.value(),
            "infer_max_width": self.infer_w_spin.value(),
            "infer_max_height": self.infer_h_spin.value(),
            "preview_mode": str(self.preview_combo.currentData() or "performance"),
            "input_source": "screen",
        }
