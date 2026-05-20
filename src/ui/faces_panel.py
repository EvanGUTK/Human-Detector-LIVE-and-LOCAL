"""Face enrollment and gallery panel."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.face_recognizer import FaceRecognizer


class FacesPanel(QWidget):
    capture_requested = pyqtSignal()
    gallery_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._recognizer: FaceRecognizer | None = None
        layout = QVBoxLayout(self)

        layout.addWidget(
            QLabel(
                "All face data stays on this PC — nothing is uploaded.\n"
                "Enroll 3+ clear front-facing photos per person for best results."
            )
        )

        row = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Person name")
        row.addWidget(self.name_input)
        self.btn_capture = QPushButton("Capture from video")
        self.btn_capture.clicked.connect(self.capture_requested.emit)
        row.addWidget(self.btn_capture)
        self.btn_import = QPushButton("Import image…")
        self.btn_import.clicked.connect(self._import_image)
        row.addWidget(self.btn_import)
        layout.addLayout(row)

        self.people_list = QListWidget()
        layout.addWidget(self.people_list)

        btns = QHBoxLayout()
        self.btn_rename = QPushButton("Rename")
        self.btn_delete = QPushButton("Delete")
        self.btn_rename.clicked.connect(self._rename)
        self.btn_delete.clicked.connect(self._delete)
        btns.addWidget(self.btn_rename)
        btns.addWidget(self.btn_delete)
        layout.addLayout(btns)

        self.status_label = QLabel("Face recognition: not loaded")
        layout.addWidget(self.status_label)
        layout.addStretch()

    def set_recognizer(self, recognizer: FaceRecognizer | None) -> None:
        self._recognizer = recognizer
        if recognizer is None:
            self.status_label.setText("Face recognition: disabled")
        elif recognizer.is_ready:
            self.status_label.setText(f"Face recognition: ready ({len(recognizer.list_people())} people)")
        else:
            self.status_label.setText(f"Face recognition: {recognizer.error or 'unavailable'}")
        self._refresh_list()

    def _refresh_list(self) -> None:
        self.people_list.clear()
        if not self._recognizer:
            return
        for pid, name in self._recognizer.list_people():
            self.people_list.addItem(f"{name} ({pid})")
            item = self.people_list.item(self.people_list.count() - 1)
            item.setData(Qt.ItemDataRole.UserRole, pid)

    def enroll_frame(self, frame_bgr) -> None:
        if not self._recognizer or not self._recognizer.is_ready:
            QMessageBox.warning(self, "Faces", "Face recognition is not available.")
            return
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.information(self, "Faces", "Enter a name first.")
            return
        pid = self._recognizer.enroll_image(name, frame_bgr)
        if pid:
            self.name_input.clear()
            self._refresh_list()
            self.gallery_changed.emit()
            QMessageBox.information(self, "Faces", f"Enrolled '{name}'.")
        else:
            QMessageBox.warning(self, "Faces", "No face detected in frame. Try again.")

    def _import_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import face photo", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not path or not self._recognizer:
            return
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.information(self, "Faces", "Enter a name first.")
            return
        pid = self._recognizer.enroll_file(name, Path(path))
        if pid:
            self.name_input.clear()
            self._refresh_list()
            self.gallery_changed.emit()

    def _selected_person_id(self) -> str | None:
        item = self.people_list.currentItem()
        if item:
            return item.data(Qt.ItemDataRole.UserRole)
        return None

    def _rename(self) -> None:
        pid = self._selected_person_id()
        if not pid or not self._recognizer:
            return
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "Rename", "New name:")
        if ok and name.strip():
            self._recognizer.rename_person(pid, name.strip())
            self._refresh_list()
            self.gallery_changed.emit()

    def _delete(self) -> None:
        pid = self._selected_person_id()
        if not pid or not self._recognizer:
            return
        self._recognizer.delete_person(pid)
        self._refresh_list()
        self.gallery_changed.emit()
