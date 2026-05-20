"""Download/export models to ONNX for GPU inference."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from src.utils.config import models_dir

logger = logging.getLogger(__name__)


def onnx_path_for_model(model_name: str, imgsz: int = 640) -> Path:
    """ONNX file path; imgsz 640 prefers legacy ``{name}.onnx`` if present."""
    md = models_dir()
    if imgsz == 640:
        legacy = md / f"{model_name}.onnx"
        if legacy.is_file():
            return legacy
    return md / f"{model_name}_{imgsz}.onnx"


def ensure_onnx_model(model_name: str = "yolo11s", imgsz: int = 640) -> Path:
    """Return path to ONNX model, exporting via ultralytics if missing."""
    if model_name.startswith("rtdetr-"):
        return ensure_onnx_rtdetr(model_name, imgsz)
    out = onnx_path_for_model(model_name, imgsz)
    if out.is_file():
        logger.info("Using existing ONNX model: %s", out)
        return out

    logger.info("Exporting %s to ONNX (imgsz=%d)...", model_name, imgsz)
    import os

    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO

    pt_path = models_dir() / f"{model_name}.pt"
    yolo = YOLO(str(pt_path) if pt_path.is_file() else f"{model_name}.pt")
    if not pt_path.is_file():
        import shutil

        src = Path(f"{model_name}.pt")
        if src.is_file():
            shutil.move(str(src), str(pt_path))
    models_dir().mkdir(parents=True, exist_ok=True)
    export_name = f"{model_name}_sz{imgsz}"
    export_path = yolo.export(
        format="onnx",
        imgsz=imgsz,
        opset=17,
        dynamic=False,
        simplify=True,
        project=str(models_dir()),
        name=export_name,
    )
    return _finalize_exported_onnx(model_name, imgsz, export_name, Path(str(export_path)))


def ensure_onnx_rtdetr(model_name: str = "rtdetr-l", imgsz: int = 640) -> Path:
    out = onnx_path_for_model(model_name, imgsz)
    if out.is_file():
        logger.info("Using existing ONNX model: %s", out)
        return out
    logger.info("Exporting %s to ONNX (imgsz=%d)...", model_name, imgsz)
    import os

    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import RTDETR

    pt_path = models_dir() / f"{model_name}.pt"
    model = RTDETR(str(pt_path) if pt_path.is_file() else f"{model_name}.pt")
    if not pt_path.is_file():
        src = Path(f"{model_name}.pt")
        if src.is_file():
            shutil.move(str(src), str(pt_path))
    models_dir().mkdir(parents=True, exist_ok=True)
    export_name = f"{model_name}_sz{imgsz}"
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=17,
        dynamic=False,
        simplify=True,
        project=str(models_dir()),
        name=export_name,
    )
    return _finalize_exported_onnx(model_name, imgsz, export_name, Path(str(export_path)))


def _finalize_exported_onnx(
    model_name: str,
    imgsz: int,
    export_name: str,
    exported: Path,
) -> Path:
    out = onnx_path_for_model(model_name, imgsz)
    exported = Path(str(exported))
    candidates: list[Path] = []
    if exported.is_file():
        candidates.append(exported)
    sub = models_dir() / export_name
    if sub.is_dir():
        candidates.extend(sorted(sub.rglob("*.onnx")))
    chosen = next((p for p in candidates if p.is_file()), None)
    if chosen is None:
        raise FileNotFoundError(
            f"ONNX export produced no file for {model_name} (imgsz={imgsz}). "
            f"Tried: {exported}, under {sub}"
        )
    if chosen.resolve() != out.resolve():
        if out.is_file():
            out.unlink()
        shutil.move(str(chosen), str(out))
        # remove empty export folder
        if sub.is_dir() and not any(sub.iterdir()):
            sub.rmdir()
    logger.info("ONNX model ready: %s", out)
    return out
