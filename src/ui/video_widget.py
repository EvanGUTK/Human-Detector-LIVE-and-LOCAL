"""Video display and zone drawing."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy


class VideoWidget(QLabel):
    zone_finished = pyqtSignal(list)  # list of (x, y) in frame coordinates
    bbox_drawn = pyqtSignal(float, float, float, float)  # x1,y1,x2,y2 frame coords

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #1a1a1a;")

        self._frame: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self._draw_mode = False
        self._draft_points: list[tuple[float, float]] = []
        self._scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._display_w = 0
        self._display_h = 0
        self._banner_text: str | None = None
        self._bbox_mode = False
        self._bbox_start: tuple[float, float] | None = None
        self._bbox_current: tuple[float, float] | None = None
        self._overlay_boxes: list[tuple[float, float, float, float, str]] = []

    @property
    def draw_mode(self) -> bool:
        return self._draw_mode

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = enabled
        if enabled:
            self._banner_text = (
                "DRAW ZONE: click corners on video — double-click or Enter to finish"
            )
        else:
            self._banner_text = None
            self._draft_points.clear()
        self.update()

    def set_banner(self, text: str | None) -> None:
        self._banner_text = text
        self.update()

    def set_bbox_mode(self, enabled: bool) -> None:
        self._bbox_mode = enabled
        if enabled:
            self._draw_mode = False
            self._draft_points.clear()
            self._banner_text = "DRAW BOX: drag rectangle on video"
        elif not self._draw_mode:
            self._banner_text = None
        self._bbox_start = None
        self._bbox_current = None
        self.update()

    def set_overlay_boxes(
        self, boxes: list[tuple[float, float, float, float, str]]
    ) -> None:
        self._overlay_boxes = boxes
        self.update()

    def set_frame(self, frame_bgr: np.ndarray | None) -> None:
        if frame_bgr is None:
            return
        self._frame = frame_bgr
        h, w = frame_bgr.shape[:2]
        lw, lh = self.width(), self.height()
        if lw > 64 and lh > 64 and (w > lw or h > lh):
            scale = min(lw / w, lh / h)
            dw = max(1, int(w * scale))
            dh = max(1, int(h * scale))
            display_bgr = cv2.resize(
                frame_bgr, (dw, dh), interpolation=cv2.INTER_LINEAR
            )
        else:
            display_bgr = frame_bgr
        rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        dh, dw, ch = rgb.shape
        bytes_per_line = ch * dw
        qimg = QImage(rgb.data, dw, dh, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg.copy())
        self._update_layout_metrics()
        self.update()

    def _update_layout_metrics(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        lw, lh = self.width(), self.height()
        self._scale = min(lw / pw, lh / ph)
        self._display_w = int(pw * self._scale)
        self._display_h = int(ph * self._scale)
        self._offset_x = (lw - self._display_w) / 2
        self._offset_y = (lh - self._display_h) / 2

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_layout_metrics()

    def widget_to_frame(self, wx: float, wy: float) -> tuple[float, float] | None:
        if self._frame is None or self._scale <= 0:
            return None
        fx = (wx - self._offset_x) / self._scale
        fy = (wy - self._offset_y) / self._scale
        fh, fw = self._frame.shape[:2]
        if fx < 0 or fy < 0 or fx >= fw or fy >= fh:
            return None
        return fx, fy

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._pixmap and not self._pixmap.isNull():
            self._update_layout_metrics()
            painter.drawPixmap(
                int(self._offset_x),
                int(self._offset_y),
                self._display_w,
                self._display_h,
                self._pixmap,
            )
        for x1, y1, x2, y2, label in self._overlay_boxes:
            p1 = (
                int(self._offset_x + x1 * self._scale),
                int(self._offset_y + y1 * self._scale),
            )
            p2 = (
                int(self._offset_x + x2 * self._scale),
                int(self._offset_y + y2 * self._scale),
            )
            painter.setPen(QPen(Qt.GlobalColor.cyan, 2))
            painter.drawRect(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1])
            painter.drawText(p1[0], max(p1[1] - 4, 12), label)
        if self._bbox_mode and self._bbox_start and self._bbox_current:
            x1, y1 = self._bbox_start
            x2, y2 = self._bbox_current
            p1 = (
                int(self._offset_x + x1 * self._scale),
                int(self._offset_y + y1 * self._scale),
            )
            p2 = (
                int(self._offset_x + x2 * self._scale),
                int(self._offset_y + y2 * self._scale),
            )
            painter.setPen(QPen(Qt.GlobalColor.magenta, 2))
            painter.drawRect(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1])
        if self._draw_mode and self._draft_points:
            pen = QPen(Qt.GlobalColor.yellow, 2)
            painter.setPen(pen)
            pts = [
                (
                    int(self._offset_x + x * self._scale),
                    int(self._offset_y + y * self._scale),
                )
                for x, y in self._draft_points
            ]
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
            for px, py in pts:
                painter.drawEllipse(px - 4, py - 4, 8, 8)
        if self._banner_text:
            painter.setPen(Qt.GlobalColor.yellow)
            font = painter.font()
            pt = font.pointSize()
            if pt <= 0:
                pt = font.pointSizeF()
            if pt <= 0:
                pt = 11.0
            font.setPointSize(max(8, int(pt)))
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(12, 28, self._banner_text)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._bbox_mode:
            if event.button() != Qt.MouseButton.LeftButton or self._frame is None:
                return
            pos = self.widget_to_frame(event.position().x(), event.position().y())
            if pos is None:
                return
            self._bbox_start = pos
            self._bbox_current = pos
            self.update()
            return
        if not self._draw_mode:
            super().mousePressEvent(event)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._frame is None:
            return
        pos = self.widget_to_frame(event.position().x(), event.position().y())
        if pos is None:
            return
        self._draft_points.append(pos)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._bbox_mode and self._bbox_start is not None:
            pos = self.widget_to_frame(event.position().x(), event.position().y())
            if pos:
                self._bbox_current = pos
                self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._bbox_mode and self._bbox_start and self._bbox_current:
            x1, y1 = self._bbox_start
            x2, y2 = self._bbox_current
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            if (x2 - x1) > 4 and (y2 - y1) > 4:
                self.bbox_drawn.emit(x1, y1, x2, y2)
            self._bbox_start = None
            self._bbox_current = None
            self.update()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._draw_mode:
            self._finish_zone()

    def keyPressEvent(self, event) -> None:
        if self._draw_mode and event.key() in (
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
        ):
            self._finish_zone()
        else:
            super().keyPressEvent(event)

    def _finish_zone(self) -> None:
        if len(self._draft_points) >= 3:
            self.zone_finished.emit(list(self._draft_points))
        self._draft_points.clear()
        self.update()

    def cancel_draft(self) -> None:
        self._draft_points.clear()
        self.update()
