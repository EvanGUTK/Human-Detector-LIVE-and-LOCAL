"""Verify built-in models load and run a dummy inference."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.core.detector import YoloDetector, yolo_kwargs_from_config
from src.lab.model_registry import ModelRegistry
from src.lab.ngc_download import NGC_MODELS
from src.utils.config import load_config


TARGETS = ("yolo11n", "rtdetr-s", "peoplenet", "detectnet_v2")


def verify_one(model_id: str, cfg: dict, *, skip_gpu: bool) -> tuple[bool, str]:
    if skip_gpu:
        cfg = dict(cfg)
        cfg["force_cpu"] = True
        cfg["ort_io_binding"] = False
    reg = ModelRegistry()
    imgsz = int(cfg.get("model_imgsz", 640))
    try:
        onnx = reg.resolve_onnx(model_id, imgsz)
    except Exception as exc:
        return False, f"resolve: {exc}"
    if not Path(onnx).is_file():
        return False, f"missing ONNX: {onnx}"
    try:
        det = YoloDetector(onnx, model_name=model_id, **yolo_kwargs_from_config(cfg))
    except Exception as exc:
        return False, f"init: {exc}"
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    try:
        dets = det.detect(frame)
    except Exception as exc:
        return False, f"infer: {exc}"
    ms = (time.perf_counter() - t0) * 1000.0
    gpu = getattr(det, "using_gpu", False)
    return True, f"ok infer={ms:.1f}ms boxes={len(dets)} gpu={gpu} backend={det.backend_name}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--models", nargs="*", default=list(TARGETS))
    args = parser.parse_args()
    cfg = load_config()
    failed = 0
    for mid in args.models:
        ok, msg = verify_one(mid, cfg, skip_gpu=args.skip_gpu)
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {mid}: {msg}")
        if not ok and mid in NGC_MODELS:
            print(f"       NGC resource: {NGC_MODELS[mid].resource}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
