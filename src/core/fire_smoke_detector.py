"""Optional second-stage fire/smoke ONNX detector (user-supplied weights)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.core.detector import Detection, _letterbox, _nms

logger = logging.getLogger(__name__)

# Default display names when model has 1 or 2 classes (typical fire/smoke heads)
_FIRE_LABELS = ("fire", "smoke")


class FireSmokeDetector:
    """YOLO-style ONNX; merges as overlay with class_name prefix fire_ / smoke_."""

    def __init__(
        self,
        onnx_path: Path | None,
        enabled: bool,
        force_cpu: bool = False,
        full_frame: bool = True,
        confidence: float = 0.35,
        iou: float = 0.5,
        max_detections: int = 20,
        imgsz: int = 640,
    ) -> None:
        self.enabled = bool(enabled and onnx_path and Path(onnx_path).is_file())
        self.full_frame = full_frame
        self.confidence = confidence
        self.iou = iou
        self.max_detections = max_detections
        self.imgsz = imgsz
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self._num_classes = 2

        if not self.enabled:
            return
        path = Path(onnx_path)
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            prov = (
                ["CPUExecutionProvider"]
                if force_cpu
                else [
                    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                    "CPUExecutionProvider",
                ]
            )
            self.session = ort.InferenceSession(str(path), sess_options=so, providers=prov)
            self.input_name = self.session.get_inputs()[0].name
        except Exception as exc:
            logger.warning("Fire/smoke ONNX failed to load: %s", exc)
            self.session = None
            self.enabled = False

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if not self.enabled or self.session is None or self.input_name is None:
            return []
        h0, w0 = frame_bgr.shape[:2]
        if self.full_frame:
            roi = frame_bgr
            ox, oy = 0, 0
        else:
            cx, cy = w0 // 2, h0 // 2
            rw, rh = w0 // 2, h0 // 2
            x1, y1 = max(0, cx - rw // 2), max(0, cy - rh // 2)
            x2, y2 = min(w0, x1 + rw), min(h0, y1 + rh)
            roi = frame_bgr[y1:y2, x1:x2]
            ox, oy = x1, y1

        img, ratio, (pad_x, pad_y) = _letterbox(roi, (self.imgsz, self.imgsz))
        blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[np.newaxis, ...])

        outputs = self.session.run(None, {self.input_name: blob})
        preds = outputs[0]
        if preds.ndim == 3:
            preds = preds[0]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        boxes_xywh = preds[:, :4]
        scores = preds[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(scores)), class_ids]
        self._num_classes = int(scores.shape[1]) if scores.ndim == 2 else 2

        mask = confs >= self.confidence
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        boxes = np.zeros_like(boxes_xywh)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= ratio
        rh, rw = roi.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, rw)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, rh)
        boxes[:, [0, 2]] += ox
        boxes[:, [1, 3]] += oy
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

        keep = _nms(boxes, confs, self.iou)[: self.max_detections]
        out: list[Detection] = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            cid = int(class_ids[i])
            label = _FIRE_LABELS[cid] if 0 <= cid < len(_FIRE_LABELS) else f"class{cid}"
            if label == "fire":
                display_name = "fire_alert"
            elif label == "smoke":
                display_name = "smoke_alert"
            else:
                display_name = f"fire_{label}"
            d = Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(confs[i]),
                class_id=80 + min(cid, 7),
                display_name=display_name,
            )
            out.append(d)
        return out
