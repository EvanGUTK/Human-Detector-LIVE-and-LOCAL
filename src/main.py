"""Person Detector — application entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path (script and PyInstaller)
if getattr(sys, "frozen", False):
    _base = Path(sys._MEIPASS)
else:
    _base = Path(__file__).resolve().parents[1]
if str(_base) not in sys.path:
    sys.path.insert(0, str(_base))


def _setup_logging() -> None:
    from src.utils.config import log_path

    log_path().parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path(), encoding="utf-8"),
        ],
    )


def main() -> int:
    _setup_logging()
    logger = logging.getLogger(__name__)

    from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog
    from PyQt6.QtCore import Qt

    from src.core.detector import YoloDetector, yolo_kwargs_from_config
    from src.lab.model_registry import ModelRegistry
    from src.core.performance_profiles import apply_profile
    from src.core.profile_manager import ProfileManager
    from src.ui.setup_wizard import SetupWizard
    from src.ui.theme import apply_theme
    from src.utils.config import project_root
    from src.ui.main_window import MainWindow
    from src.utils.config import load_config, save_config

    app = QApplication(sys.argv)
    app.setApplicationName("Person Detector")
    apply_theme(app)

    cfg = load_config()
    pm = ProfileManager()
    active = str(cfg.get("active_profile", "Default room"))
    if active:
        cfg = pm.apply_to_config(cfg, active)

    if not cfg.get("first_run_complete"):
        dlg = SetupWizard(cfg, project_root())
        if dlg.exec():
            cfg.update(dlg.result_config())
            cfg["active_profile"] = "Default room"
            pm.save("Default room", cfg)
            save_config(cfg)
        else:
            cfg["first_run_complete"] = True
            save_config(cfg)

    prof = str(cfg.get("model_preset", cfg.get("performance_profile", "balanced")))
    if prof == "performance":
        prof = "balanced"
    if prof != "custom":
        cfg = apply_profile(cfg, prof)

    progress = QProgressDialog("Preparing AI model (first run may take a minute)...", None, 0, 0)
    progress.setWindowTitle("Person Detector")
    progress.setWindowModality(Qt.WindowModality.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.show()
    app.processEvents()

    try:
        model_id = str(cfg.get("active_model_id", cfg.get("model_name", "yolo11s")))
        imgsz = int(cfg.get("model_imgsz", 640))
        onnx_path = ModelRegistry().resolve_onnx(model_id, imgsz)
        detector = YoloDetector(
            onnx_path,
            model_name=model_id,
            **yolo_kwargs_from_config(cfg),
        )
        logger.info("Inference backend: %s (gpu=%s)", detector.backend_name, detector.using_gpu)
    except Exception as exc:
        progress.close()
        logger.exception("Failed to load model")
        from src.utils.config import log_path

        QMessageBox.critical(
            None,
            "Model error",
            f"Could not load detection model:\n{exc}\n\nSee log: {log_path()}",
        )
        return 1

    progress.close()

    if not getattr(detector, "using_gpu", False) and not cfg.get("force_cpu"):
        QMessageBox.warning(
            None,
            "Running on CPU",
            "GPU inference is not active (expect ~30 FPS).\n\n"
            "For 60+ FPS: install CUDA 12.x and run scripts\\check_gpu.ps1\n"
            "Or switch toolbar to Compatibility profile for stable CPU mode.",
        )

    window = MainWindow(detector, cfg)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
