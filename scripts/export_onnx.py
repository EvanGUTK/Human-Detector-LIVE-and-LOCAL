"""CLI: export YOLO model to ONNX."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.model_setup import ensure_onnx_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO to ONNX")
    parser.add_argument("--model", default="yolo11s", help="Model name (yolo11n, yolo11s)")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    path = ensure_onnx_model(args.model, args.imgsz)
    print(f"Exported: {path}")


if __name__ == "__main__":
    main()
