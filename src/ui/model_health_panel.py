"""Lab — model health and verification."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.detector import YoloDetector, yolo_kwargs_from_config
from src.lab.model_registry import ModelRegistry
from src.lab.ngc_download import NGC_MODELS, ensure_ngc_model_onnx
from src.utils.config import load_config, project_root
from src.utils.model_setup import ensure_onnx_model


class ModelHealthPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Verify built-in and custom ONNX models. TAO models need NGC_API_KEY for download."
            )
        )
        row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh all")
        self.btn_verify_script = QPushButton("Run verify_models.py")
        self.btn_refresh.clicked.connect(self.refresh_table)
        self.btn_verify_script.clicked.connect(self._run_script)
        row.addWidget(self.btn_refresh)
        row.addWidget(self.btn_verify_script)
        layout.addLayout(row)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Model", "Family", "ONNX", "Status", "Details", "Actions"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self.refresh_table()

    def refresh_table(self) -> None:
        reg = ModelRegistry()
        models = reg.list_models()
        self.table.setRowCount(len(models))
        for row, m in enumerate(models):
            path = Path(m.onnx_path)
            exists = path.is_file()
            self.table.setItem(row, 0, QTableWidgetItem(m.display_name))
            self.table.setItem(row, 1, QTableWidgetItem(m.family))
            self.table.setItem(row, 2, QTableWidgetItem(path.name if exists else "missing"))
            status = "Ready" if exists else "Missing"
            self.table.setItem(row, 3, QTableWidgetItem(status))
            self.table.setItem(row, 4, QTableWidgetItem("—"))
            btn_row = QWidget()
            bl = QHBoxLayout(btn_row)
            bl.setContentsMargins(2, 2, 2, 2)
            btn_test = QPushButton("Test")
            btn_test.clicked.connect(lambda _c, mid=m.id: self._test_model(mid))
            bl.addWidget(btn_test)
            if m.id in NGC_MODELS:
                btn_dl = QPushButton("NGC")
                btn_dl.clicked.connect(lambda _c, mid=m.id: self._download_ngc(mid))
                bl.addWidget(btn_dl)
            elif m.is_builtin and m.family in ("yolo", "rtdetr"):
                btn_ex = QPushButton("Export")
                btn_ex.clicked.connect(lambda _c, mid=m.id: self._export_onnx(mid))
                bl.addWidget(btn_ex)
            bl.addStretch()
            self.table.setCellWidget(row, 5, btn_row)

    def _test_model(self, model_id: str) -> None:
        cfg = load_config()
        reg = ModelRegistry()
        row = self._row_for_id(model_id)
        try:
            onnx = reg.resolve_onnx(model_id, int(cfg.get("model_imgsz", 640)))
            det = YoloDetector(onnx, model_name=model_id, **yolo_kwargs_from_config(cfg))
            frame = np.zeros((544, 960, 3), dtype=np.uint8)
            dets = det.detect(frame)
            msg = (
                f"infer ok, {len(dets)} boxes, "
                f"backend={det.backend_name}, gpu={getattr(det, 'using_gpu', False)}"
            )
            if row >= 0:
                self.table.setItem(row, 3, QTableWidgetItem("PASS"))
                self.table.setItem(row, 4, QTableWidgetItem(msg))
        except Exception as exc:
            if row >= 0:
                self.table.setItem(row, 3, QTableWidgetItem("FAIL"))
                self.table.setItem(row, 4, QTableWidgetItem(str(exc)))
            QMessageBox.warning(self, "Test failed", str(exc))

    def _download_ngc(self, model_id: str) -> None:
        try:
            path = ensure_ngc_model_onnx(model_id, force_refresh=True)
            QMessageBox.information(self, "NGC", f"Downloaded to {path}")
            self.refresh_table()
        except Exception as exc:
            QMessageBox.warning(self, "NGC download", str(exc))

    def _export_onnx(self, model_id: str) -> None:
        try:
            path = ensure_onnx_model(model_id, 640)
            QMessageBox.information(self, "Export", f"ONNX ready: {path}")
            self.refresh_table()
        except Exception as exc:
            QMessageBox.warning(self, "Export", str(exc))

    def _row_for_id(self, model_id: str) -> int:
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item and ModelRegistry().get(model_id):
                mid = model_id
                m = ModelRegistry().get(mid)
                if m and item.text() == m.display_name:
                    return r
        return -1

    def _run_script(self) -> None:
        py = sys.executable
        script = project_root() / "scripts" / "verify_models.py"
        proc = subprocess.run(
            [py, str(script)],
            capture_output=True,
            text=True,
            cwd=str(project_root()),
        )
        QMessageBox.information(
            self,
            "verify_models.py",
            (proc.stdout or proc.stderr or "done")[-4000:],
        )
        self.refresh_table()
