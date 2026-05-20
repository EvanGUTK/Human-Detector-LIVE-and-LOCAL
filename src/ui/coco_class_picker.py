"""Dialog to pick COCO-80 detection classes (search + checkboxes)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from src.utils.coco_classes import COCO80_NAMES, CCTV_TRAFFIC_CLASS_IDS


class CocoClassPickerDialog(QDialog):
    def __init__(self, selected_ids: list[int], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("COCO classes (80)")
        self.resize(420, 520)
        self._selected: set[int] = {i for i in selected_ids if 0 <= i < 80}

        root = QVBoxLayout(self)
        self.summary = QLabel()
        root.addWidget(self.summary)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search:"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by class name…")
        self.search.textChanged.connect(self._apply_filter)
        search_row.addWidget(self.search)
        root.addLayout(search_row)

        preset_row = QHBoxLayout()
        btn_traffic = QPushButton("CCTV traffic")
        btn_traffic.setToolTip("Person, bicycle, car, motorcycle, bus, truck")
        btn_traffic.clicked.connect(self._preset_traffic)
        btn_person = QPushButton("Person only")
        btn_person.clicked.connect(lambda: self._set_preset({0}))
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(lambda: self._set_preset(set()))
        btn_all = QPushButton("Select all (test)")
        btn_all.clicked.connect(lambda: self._set_preset(set(range(80))))
        preset_row.addWidget(btn_traffic)
        preset_row.addWidget(btn_person)
        preset_row.addWidget(btn_clear)
        preset_row.addWidget(btn_all)
        root.addLayout(preset_row)

        self.list_w = QListWidget()
        self.list_w.setAlternatingRowColors(True)
        for i, name in enumerate(COCO80_NAMES):
            item = QListWidgetItem(f"{i:2d}  {name}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if i in self._selected else Qt.CheckState.Unchecked
            )
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.list_w.addItem(item)
        self.list_w.itemChanged.connect(self._on_item_changed)
        root.addWidget(self.list_w)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._update_summary()
        self._apply_filter()

    def _preset_traffic(self) -> None:
        self._set_preset(set(CCTV_TRAFFIC_CLASS_IDS))

    def _set_preset(self, ids: set[int]) -> None:
        self._selected = {i for i in ids if 0 <= i < 80}
        if not self._selected:
            self._selected = {0}
        self.list_w.blockSignals(True)
        for row in range(self.list_w.count()):
            item = self.list_w.item(row)
            if item is None:
                continue
            idx = int(item.data(Qt.ItemDataRole.UserRole))
            item.setCheckState(
                Qt.CheckState.Checked if idx in self._selected else Qt.CheckState.Unchecked
            )
        self.list_w.blockSignals(False)
        self._update_summary()

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        idx = int(item.data(Qt.ItemDataRole.UserRole))
        if item.checkState() == Qt.CheckState.Checked:
            self._selected.add(idx)
        else:
            self._selected.discard(idx)
        if not self._selected:
            self._selected.add(0)
            self.list_w.blockSignals(True)
            for row in range(self.list_w.count()):
                it = self.list_w.item(row)
                if it and int(it.data(Qt.ItemDataRole.UserRole)) == 0:
                    it.setCheckState(Qt.CheckState.Checked)
                    break
            self.list_w.blockSignals(False)
        self._update_summary()

    def _apply_filter(self) -> None:
        q = self.search.text().strip().lower()
        for row in range(self.list_w.count()):
            item = self.list_w.item(row)
            if item is None:
                continue
            text = item.text().lower()
            item.setHidden(bool(q) and q not in text)

    def _update_summary(self) -> None:
        n = len(self._selected)
        self.summary.setText(f"{n} class(es) selected.")

    def selected_ids(self) -> list[int]:
        return sorted(self._selected)
