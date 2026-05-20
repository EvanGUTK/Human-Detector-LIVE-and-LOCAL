"""Help dialogs for Person Detector."""

from PyQt6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout


class ZonesHelpDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("How zones work")
        self.setMinimumWidth(480)

        text = """
<h3>What is a zone?</h3>
<p>A <b>zone</b> is a colored area you draw on the video. When a <b>tracked person</b>
enters that area, the app alerts you (sound, toast, red flash).</p>

<h3>How to create a zone</h3>
<ol>
<li>Click <b>Start</b> so detection is running.</li>
<li>Click <b>Draw Zone</b> in the toolbar.</li>
<li>Click each corner of the area on the video.</li>
<li><b>Double-click</b> or press <b>Enter</b> to finish (at least 3 points).</li>
<li>Enter a name (e.g. Doorway).</li>
</ol>

<h3>What triggers an alert?</h3>
<p>The app uses the <b>feet point</b> (red dot at the bottom-center of each green box),
not the hands or center of the body. One alert per person per zone until they leave.</p>

<p><b>Pose skeleton</b> (if enabled) is visual only — zones still use person boxes.</p>

<h3>Zone list buttons</h3>
<ul>
<li><b>Delete</b> — remove selected zone</li>
<li><b>Toggle</b> — enable/disable alerts for that zone</li>
<li><b>Rename</b> — change the zone name</li>
<li><b>Test alert</b> — preview sound/flash without a person</li>
</ul>
"""
        label = QLabel(text)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(label)
        layout.addWidget(close_btn)
