"""First-run setup wizard."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)


class FirstRunDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome to Person Detector")
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Quick setup for home monitoring.\n"
                "You can change these later in Settings or the toolbar."
            )
        )
        form = QFormLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["performance", "compatibility"])
        self.profile_combo.setItemText(0, "Performance (GPU — recommended)")
        self.profile_combo.setItemText(1, "Compatibility (CPU)")
        form.addRow("Performance:", self.profile_combo)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_profile(self) -> str:
        return "performance" if self.profile_combo.currentIndex() == 0 else "compatibility"
