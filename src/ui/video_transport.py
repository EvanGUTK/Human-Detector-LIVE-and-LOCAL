"""Shared video transport bar (play/pause video, seek, speed)."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)


class VideoTransportBar(QWidget):
    """Controls file playback — not detection pause."""

    seek_requested = pyqtSignal(int)
    pause_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(float)
    loop_changed = pyqtSignal(bool)
    step_requested = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)

        self.btn_play = QPushButton("Pause video")
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self._on_play_toggle)
        row.addWidget(self.btn_play)

        self.btn_back10 = QPushButton("-10")
        self.btn_back10.clicked.connect(lambda: self.step_requested.emit(-10))
        self.btn_back1 = QPushButton("-1")
        self.btn_back1.clicked.connect(lambda: self.step_requested.emit(-1))
        self.btn_fwd1 = QPushButton("+1")
        self.btn_fwd1.clicked.connect(lambda: self.step_requested.emit(1))
        self.btn_fwd10 = QPushButton("+10")
        self.btn_fwd10.clicked.connect(lambda: self.step_requested.emit(10))
        for b in (self.btn_back10, self.btn_back1, self.btn_fwd1, self.btn_fwd10):
            row.addWidget(b)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.sliderReleased.connect(self._on_slider_released)
        row.addWidget(self.slider, stretch=1)

        self.lbl_time = QLabel("00:00 / 00:00")
        row.addWidget(self.lbl_time)

        self.speed_combo = QComboBox()
        for v in (0.25, 0.5, 1.0, 1.5, 2.0, 4.0):
            self.speed_combo.addItem(f"{v}x", v)
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(self._on_speed)
        row.addWidget(self.speed_combo)

        self.chk_loop = QCheckBox("Loop")
        self.chk_loop.setChecked(True)
        self.chk_loop.toggled.connect(lambda c: self.loop_changed.emit(c))
        row.addWidget(self.chk_loop)

        self._frame_count = 0
        self._fps = 30.0
        self._current = 0
        self._video_paused = False

    def set_metadata(self, frame_count: int, fps: float, current: int = 0) -> None:
        self._frame_count = max(0, frame_count)
        self._fps = fps if fps > 1e-3 else 30.0
        self._current = max(0, min(current, max(0, self._frame_count - 1)))
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, self._frame_count - 1))
        self.slider.setValue(self._current)
        self.slider.blockSignals(False)
        self._update_label()

    def update_position(self, current: int) -> None:
        self._current = max(0, current)
        if self._frame_count > 0:
            self._current = min(self._current, self._frame_count - 1)
        self.slider.blockSignals(True)
        self.slider.setValue(self._current)
        self.slider.blockSignals(False)
        self._update_label()

    def set_video_paused(self, paused: bool) -> None:
        self._video_paused = paused
        self.btn_play.blockSignals(True)
        self.btn_play.setChecked(paused)
        self.btn_play.setText("Play video" if paused else "Pause video")
        self.btn_play.blockSignals(False)

    def _update_label(self) -> None:
        def fmt(frame: int) -> str:
            sec = frame / self._fps if self._fps > 0 else 0.0
            m = int(sec // 60)
            s = sec % 60
            return f"{m:02d}:{s:05.2f}"

        total = max(0, self._frame_count - 1)
        self.lbl_time.setText(f"{fmt(self._current)} / {fmt(total)}  (f{self._current}/{total})")

    def _on_play_toggle(self) -> None:
        paused = self.btn_play.isChecked()
        self.set_video_paused(paused)
        self.pause_toggled.emit(paused)

    def _on_slider_released(self) -> None:
        self.seek_requested.emit(self.slider.value())

    def _on_speed(self) -> None:
        spd = float(self.speed_combo.currentData() or 1.0)
        self.speed_changed.emit(spd)
