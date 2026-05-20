"""Headless FPS benchmark (no GUI)."""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.capture import WebcamCapture
from src.core.detector import PersonDetector, yolo_kwargs_from_config
from src.core.screen_capture import ScreenCapture
from src.lab.model_registry import ModelRegistry
from src.utils.config import load_config
from src.utils.frame_resize import resize_for_infer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument(
        "--source",
        choices=("webcam", "screen"),
        default="webcam",
        help="webcam or full-screen desktop capture",
    )
    parser.add_argument(
        "--infer-max-width",
        type=int,
        default=None,
        help="Cap width before ONNX (0 = full frame); defaults from config",
    )
    parser.add_argument(
        "--infer-max-height",
        type=int,
        default=None,
        help="Cap height before ONNX (0 = full frame); defaults from config",
    )
    args = parser.parse_args()

    cfg = load_config()
    model = str(cfg.get("active_model_id", cfg.get("model_name", "yolo11s")))
    imgsz = int(cfg.get("model_imgsz", 640))
    max_w = (
        args.infer_max_width
        if args.infer_max_width is not None
        else int(cfg.get("infer_max_width", 1280))
    )
    max_h = (
        args.infer_max_height
        if args.infer_max_height is not None
        else int(cfg.get("infer_max_height", 720))
    )
    onnx = ModelRegistry().resolve_onnx(model, imgsz)
    detector = PersonDetector(
        onnx,
        model_name=model,
        **yolo_kwargs_from_config(cfg),
    )
    print(f"Backend: {detector.backend_name} GPU={detector.using_gpu}")

    if args.source == "screen":
        cap = ScreenCapture(
            monitor_index=int(cfg.get("screen_monitor", 1)),
            region=cfg.get("screen_region"),
            fps_cap=float(cfg.get("screen_fps_cap", 0)),
        )
        src_label = "screen"
    else:
        cam_idx = args.camera if args.camera is not None else int(cfg.get("camera_index", 0))
        cap = WebcamCapture(
            cam_idx,
            int(cfg.get("capture_width", 1280)),
            int(cfg.get("capture_height", 720)),
        )
        src_label = f"webcam:{cam_idx}"

    if not cap.start():
        print("Failed to open source:", getattr(cap, "error", None))
        sys.exit(1)

    print(
        f"Benchmarking {args.seconds}s — source={src_label} model={model} imgsz={imgsz} "
        f"infer_max={max_w}x{max_h}"
    )
    time.sleep(1.0)
    frames = 0
    infer_times: list[float] = []
    pre_times: list[float] = []
    t_end = time.perf_counter() + args.seconds
    last_id = -1

    while time.perf_counter() < t_end:
        frame, fid = cap.get_latest(copy=False)
        if frame is None or fid == last_id:
            time.sleep(0.001)
            continue
        last_id = fid
        t0 = time.perf_counter()
        infer_frame, _ = resize_for_infer(frame, max_w, max_h)
        pre_times.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        detector.detect(infer_frame)
        infer_times.append(time.perf_counter() - t1)
        frames += 1

    cap.stop()
    if not infer_times:
        print("No frames processed.")
        sys.exit(1)

    avg_infer = sum(infer_times) / len(infer_times) * 1000
    avg_pre = sum(pre_times) / len(pre_times) * 1000
    fps = frames / args.seconds
    print(f"Frames: {frames}")
    print(f"Pipeline FPS: {fps:.1f}")
    print(f"Avg preprocess: {avg_pre:.2f} ms")
    print(f"Avg inference: {avg_infer:.2f} ms")
    print(f"Peak inference FPS: {1.0 / max(infer_times):.1f}")


if __name__ == "__main__":
    main()
