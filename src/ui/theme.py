"""Application-wide Qt stylesheet tokens."""

from __future__ import annotations

STYLESHEET = """
QMainWindow, QWidget {
    background-color: #121418;
    color: #e8eaed;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #2a2f38;
    border-radius: 4px;
    top: -1px;
}
QTabBar::tab {
    background: #1c2028;
    color: #aab0b8;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: #252b36;
    color: #ffffff;
    border-bottom: 2px solid #3d8bfd;
}
QToolBar {
    background: #1a1e26;
    border-bottom: 1px solid #2a2f38;
    spacing: 6px;
    padding: 4px;
}
QToolButton, QPushButton {
    background-color: #2a3140;
    color: #e8eaed;
    border: 1px solid #3a4254;
    border-radius: 4px;
    padding: 6px 12px;
}
QPushButton:hover, QToolButton:hover {
    background-color: #354055;
}
QPushButton:pressed {
    background-color: #3d8bfd;
}
QPushButton:disabled {
    background-color: #1c2028;
    color: #5a6270;
    border-color: #2a2f38;
}
QPushButton:focus {
    border: 1px solid #3d8bfd;
}
QSlider::handle:horizontal {
    background: #3d8bfd;
    width: 12px;
    margin: -4px 0;
    border-radius: 6px;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #2a2f38;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #9aa3b2;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #1c2028;
    border: 1px solid #3a4254;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 22px;
}
QStatusBar {
    background: #1a1e26;
    color: #9aa3b2;
}
QScrollArea {
    border: none;
}
QLabel#PerfHudTitle {
    color: #3d8bfd;
    font-weight: bold;
}
QLabel#PerfWarn {
    color: #ff8c66;
}
"""


def apply_theme(app) -> None:
    app.setStyleSheet(STYLESHEET)
