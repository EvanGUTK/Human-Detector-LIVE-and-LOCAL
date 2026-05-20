"""Lab home — test clips and quick actions."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.utils.config import app_data_dir, load_config


def clips_dir() -> Path:
    p = app_data_dir() / "clips"
    p.mkdir(parents=True, exist_ok=True)
    return p


def clips_index_path() -> Path:
    return clips_dir() / "index.json"


class LabDashboard(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._parent_window = parent
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>AI Testing Lab</b> — quick start"))
        self.model_card = QLabel("Active model: —")
        self.model_card.setWordWrap(True)
        layout.addWidget(self.model_card)

        layout.addWidget(QLabel("Test clips"))
        self.clip_list = QListWidget()
        layout.addWidget(self.clip_list)
        cr = QHBoxLayout()
        self.btn_add_clip = QPushButton("Add clip…")
        self.btn_add_clip.clicked.connect(self._add_clip)
        self.btn_open_compare = QPushButton("Open in Compare")
        self.btn_open_compare.clicked.connect(self._open_compare)
        self.btn_health = QPushButton("Model health")
        self.btn_health.clicked.connect(self._goto_health)
        cr.addWidget(self.btn_add_clip)
        cr.addWidget(self.btn_open_compare)
        cr.addWidget(self.btn_health)
        layout.addLayout(cr)
        layout.addStretch()
        self.refresh()

    def refresh(self) -> None:
        cfg = load_config()
        mid = str(cfg.get("active_model_id", cfg.get("model_name", "yolo11s")))
        imgsz = int(cfg.get("model_imgsz", 640))
        self.model_card.setText(
            f"Active model: <b>{mid}</b> @ {imgsz}px<br>"
            f"Source: {cfg.get('input_source', 'webcam')}<br>"
            f"Preview: {cfg.get('preview_mode', 'performance')}"
        )
        self.clip_list.clear()
        for entry in self._load_index():
            self.clip_list.addItem(f"{entry.get('name', '?')}  ({entry.get('path', '')})")

    def _load_index(self) -> list[dict]:
        path = clips_index_path()
        if not path.is_file():
            return []
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def _save_index(self, entries: list[dict]) -> None:
        with clips_index_path().open("w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

    def _add_clip(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Add test clip", "", "Video (*.mp4 *.avi *.mkv *.mov)"
        )
        if not path:
            return
        src = Path(path)
        dest = clips_dir() / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{src.name}"
        shutil.copy2(src, dest)
        entries = self._load_index()
        entries.insert(0, {"name": src.name, "path": str(dest)})
        self._save_index(entries[:20])
        self.refresh()

    def _open_compare(self) -> None:
        item = self.clip_list.currentItem()
        if not item:
            return
        entries = self._load_index()
        if not entries:
            return
        idx = self.clip_list.currentRow()
        if idx < 0 or idx >= len(entries):
            return
        p = entries[idx].get("path", "")
        mw = self._parent_window
        if mw is None:
            return
        if hasattr(mw, "compare_panel"):
            mw.compare_panel.set_video_path(p)
            if hasattr(mw, "lab_tabs"):
                mw.lab_tabs.setCurrentWidget(mw.compare_panel)

    def _goto_health(self) -> None:
        mw = self._parent_window
        if mw and hasattr(mw, "health_panel") and hasattr(mw, "lab_tabs"):
            mw.lab_tabs.setCurrentWidget(mw.health_panel)
