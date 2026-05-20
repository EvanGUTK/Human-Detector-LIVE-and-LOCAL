"""Zones management panel."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.zones import ZoneManager

ZONE_EMPTY_HINT = (
    "No zones yet.\n"
    "1. Go to Monitor tab → Start\n"
    "2. Draw Zone → click corners → double-click to finish"
)


class ZonesPanel(QWidget):
    zone_help_requested = pyqtSignal()
    test_alert_requested = pyqtSignal()
    export_log_requested = pyqtSignal()
    zone_delete_requested = pyqtSignal()

    def __init__(self, zone_manager: ZoneManager, parent=None) -> None:
        super().__init__(parent)
        self.zone_manager = zone_manager
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("Zones"))
        btn_help = QPushButton("?")
        btn_help.setMaximumWidth(28)
        btn_help.setToolTip("How zones work")
        btn_help.clicked.connect(self.zone_help_requested.emit)
        header.addWidget(btn_help)
        layout.addLayout(header)

        self.hint = QLabel(
            "<b>Select a zone in the list below, then click Delete "
            "(or press the Delete key).</b>"
        )
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("padding: 6px;")
        layout.addWidget(self.hint)

        self.empty_label = QLabel(ZONE_EMPTY_HINT)
        self.empty_label.setWordWrap(True)
        self.empty_label.setStyleSheet("color: #aaa; padding: 4px;")
        layout.addWidget(self.empty_label)

        self.zone_list = QListWidget()
        self.zone_list.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.zone_list.setAlternatingRowColors(True)
        self.zone_list.setStyleSheet(
            """
            QListWidget::item:selected {
                background-color: #2d5a87;
                color: #ffffff;
                border: 1px solid #4a9eff;
            }
            QListWidget::item:hover {
                background-color: #3a3a4a;
            }
            """
        )
        layout.addWidget(self.zone_list)

        row = QHBoxLayout()
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setToolTip("Remove the selected zone (or press Delete)")
        self.btn_toggle = QPushButton("Toggle")
        self.btn_toggle.setToolTip("Enable or disable alerts for this zone")
        self.btn_rename = QPushButton("Rename")
        self.btn_rename.setToolTip("Change the zone display name")
        row.addWidget(self.btn_delete)
        row.addWidget(self.btn_toggle)
        row.addWidget(self.btn_rename)
        layout.addLayout(row)

        self.btn_test = QPushButton("Test alert")
        self.btn_test.clicked.connect(self.test_alert_requested.emit)
        layout.addWidget(self.btn_test)

        layout.addWidget(QLabel("Event log"))
        self.event_list = QListWidget()
        self.event_list.setMaximumHeight(140)
        layout.addWidget(self.event_list)
        self.btn_export = QPushButton("Export log CSV…")
        self.btn_export.clicked.connect(self.export_log_requested.emit)
        layout.addWidget(self.btn_export)
        layout.addStretch()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Delete and self.selected_zone_id():
            self.zone_delete_requested.emit()
        else:
            super().keyPressEvent(event)

    def refresh_zones(self) -> None:
        self.zone_list.clear()
        has = bool(self.zone_manager.zones)
        self.empty_label.setVisible(not has)
        self.zone_list.setVisible(has)
        for z in self.zone_manager.zones:
            state = "on" if z.enabled else "off"
            item = QListWidgetItem(f"{z.name} [{state}] ({len(z.points)} pts)")
            item.setData(Qt.ItemDataRole.UserRole, z.id)
            self.zone_list.addItem(item)

    def refresh_events(self, messages: list[str]) -> None:
        self.event_list.clear()
        for msg in messages:
            self.event_list.addItem(msg)

    def selected_zone_id(self) -> str | None:
        item = self.zone_list.currentItem()
        if item:
            return item.data(Qt.ItemDataRole.UserRole)
        return None
