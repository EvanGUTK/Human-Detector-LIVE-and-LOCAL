"""Analytics tab with live charts."""

from __future__ import annotations

from PyQt6.QtWidgets import QFormLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

try:
    import pyqtgraph as pg

    pg.setConfigOptions(antialias=True)
    HAS_PG = True
except ImportError:
    HAS_PG = False

from src.core.metrics import MetricsStore


class AnalyticsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        if HAS_PG:
            self.fps_plot = pg.PlotWidget(title="FPS")
            self.fps_plot.setLabel("left", "FPS")
            self.fps_plot.setMaximumHeight(180)
            self.fps_curve = self.fps_plot.plot(pen="g")
            layout.addWidget(self.fps_plot)

            self.infer_plot = pg.PlotWidget(title="Inference (ms)")
            self.infer_plot.setLabel("left", "ms")
            self.infer_plot.setMaximumHeight(180)
            self.infer_curve = self.infer_plot.plot(pen="c")
            layout.addWidget(self.infer_plot)

            self.pre_plot = pg.PlotWidget(title="Preprocess (ms)")
            self.pre_plot.setMaximumHeight(140)
            self.pre_curve = self.pre_plot.plot(pen="y")
            layout.addWidget(self.pre_plot)

            self.draw_plot = pg.PlotWidget(title="Draw (ms)")
            self.draw_plot.setMaximumHeight(140)
            self.draw_curve = self.draw_plot.plot(pen="m")
            layout.addWidget(self.draw_plot)
        else:
            layout.addWidget(QLabel("Install pyqtgraph for live charts (pip install pyqtgraph)"))

        summary = QGroupBox("Session summary")
        form = QFormLayout(summary)
        self.lbl_fps = QLabel("—")
        self.lbl_infer = QLabel("—")
        self.lbl_pre = QLabel("—")
        self.lbl_draw = QLabel("—")
        self.lbl_enters = QLabel("0")
        self.lbl_dwell = QLabel("—")
        self.lbl_identities = QLabel("0")
        self.lbl_top_zone = QLabel("—")
        form.addRow("FPS (min / avg / max):", self.lbl_fps)
        form.addRow("Infer ms (min / avg / max):", self.lbl_infer)
        form.addRow("Preprocess ms (avg):", self.lbl_pre)
        form.addRow("Draw ms (avg):", self.lbl_draw)
        form.addRow("Zone enters:", self.lbl_enters)
        form.addRow("Dwell by zone:", self.lbl_dwell)
        form.addRow("Known identities:", self.lbl_identities)
        form.addRow("Most active zone:", self.lbl_top_zone)
        layout.addWidget(summary)
        self.btn_export_session = QPushButton("Export session metrics CSV…")
        layout.addWidget(self.btn_export_session)
        layout.addStretch()

    def update_metrics(self, metrics: MetricsStore | None) -> None:
        if metrics is None:
            return
        if HAS_PG:
            self.fps_curve.setData(list(metrics.fps_samples))
            self.infer_curve.setData(list(metrics.infer_samples))
            if hasattr(self, "pre_curve"):
                self.pre_curve.setData(list(metrics.preprocess_samples))
                self.draw_curve.setData(list(metrics.draw_samples))
        fmin, favg, fmax = metrics.fps_summary()
        imin, iavg, imax = metrics.infer_summary()
        self.lbl_fps.setText(f"{fmin:.1f} / {favg:.1f} / {fmax:.1f}")
        self.lbl_infer.setText(f"{imin:.1f} / {iavg:.1f} / {imax:.1f}")
        if metrics.preprocess_samples:
            pre_avg = sum(metrics.preprocess_samples) / len(metrics.preprocess_samples)
            self.lbl_pre.setText(f"{pre_avg:.1f}")
        else:
            self.lbl_pre.setText("—")
        if metrics.draw_samples:
            dr_avg = sum(metrics.draw_samples) / len(metrics.draw_samples)
            self.lbl_draw.setText(f"{dr_avg:.1f}")
        else:
            self.lbl_draw.setText("—")
        self.lbl_enters.setText(str(metrics.total_enters))
        dwell_parts = [
            f"{z}: {metrics.zone_dwell_total.get(z, 0):.1f}s"
            for z in sorted(metrics.zone_dwell_total)
        ]
        self.lbl_dwell.setText(", ".join(dwell_parts) if dwell_parts else "—")
        self.lbl_identities.setText(str(len(metrics.unique_identities)))
        self.lbl_top_zone.setText(metrics.top_zone_by_activity())
