"""Performance HUD and slow-mode diagnostics."""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.core.pipeline import FrameResult


class PerfHudPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("PerfHudPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.lbl_capture = QLabel("Capture: —")
        self.lbl_infer = QLabel("Infer: —")
        self.lbl_timings = QLabel("Timings: —")
        self.lbl_backend = QLabel("Backend: —")
        self.lbl_warnings = QLabel("")
        self.lbl_warnings.setObjectName("PerfWarn")
        self.lbl_warnings.setWordWrap(True)
        for w in (
            self.lbl_capture,
            self.lbl_infer,
            self.lbl_timings,
            self.lbl_backend,
            self.lbl_warnings,
        ):
            w.setStyleSheet("background: rgba(0,0,0,140); padding: 4px; border-radius: 3px;")
            layout.addWidget(w)
        layout.addStretch()

    def update_from_result(
        self,
        result: FrameResult | None,
        *,
        backend: str,
        using_gpu: bool,
        model_id: str,
        imgsz: int,
        screen_fps_cap: int,
        force_cpu: bool,
    ) -> None:
        if result is None:
            return
        cap = f"{result.capture_w}x{result.capture_h}" if result.capture_w else "—"
        inf = f"{result.infer_w}x{result.infer_h}" if result.infer_w else "—"
        self.lbl_capture.setText(
            f"Capture {cap} | Preview {result.preview_mode} | Cap FPS "
            f"{'uncapped' if screen_fps_cap <= 0 else screen_fps_cap}"
        )
        self.lbl_infer.setText(f"Infer frame {inf} | Model {model_id} @ {imgsz}")
        gpu = "CUDA" if using_gpu else "CPU"
        self.lbl_backend.setText(f"Backend {backend} ({gpu}) | FPS {result.fps:.1f}")
        self.lbl_timings.setText(
            f"pre {result.preprocess_ms:.1f} | det {result.detect_ms:.1f} | "
            f"track {result.track_ms:.1f} | zones {result.zones_ms:.1f} | "
            f"draw {result.draw_ms:.1f} | total {result.total_ms:.1f} ms"
        )
        warns: list[str] = []
        if not using_gpu or force_cpu:
            warns.append("Running on CPU — install CUDA 12 + cuDNN 9 on PATH.")
        if result.fps < 30:
            if result.capture_w > 1920 or result.capture_h > 1080:
                warns.append("4K+ capture — use ROI or Performance preview.")
            if screen_fps_cap > 0 and screen_fps_cap <= 30:
                warns.append(f"Screen FPS capped at {screen_fps_cap} — set 0 for uncapped.")
            if result.infer_w == result.capture_w and result.capture_w > 1280:
                warns.append("Inferring at full resolution — lower infer max in Settings.")
            if result.detect_ms > 25:
                warns.append(f"Slow inference ({result.detect_ms:.0f} ms) — try Fast preset.")
        self.lbl_warnings.setText("\n".join(warns) if warns else "")
