"""Detector backends for YOLO, RT-DETR, and TAO-style ONNX models."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

from src.core.tao_decoders import (
    decode_detectnet_v2_outputs,
    decode_peoplenet_outputs,
    decode_rtdetr_outputs,
)
from src.utils.config import CAR_CLASS_ID, CLASS_NAMES, PERSON_CLASS_ID, models_dir

logger = logging.getLogger(__name__)


def yolo_kwargs_from_config(cfg: dict) -> dict:
    """Common keyword args for YoloDetector from application config dict."""
    ids = cfg.get("detect_class_ids", [0])
    if not isinstance(ids, list):
        ids = [0]
    pcc = cfg.get("confidence_per_class")
    if not isinstance(pcc, dict):
        pcc = {}
    return {
        "imgsz": int(cfg.get("model_imgsz", 640)),
        "confidence": float(cfg.get("confidence", 0.45)),
        "iou": float(cfg.get("iou", 0.5)),
        "max_detections": int(cfg.get("max_detections", 50)),
        "force_cpu": bool(cfg.get("force_cpu", False)),
        "detect_class_ids": ids,
        "confidence_per_class": pcc,
        "enable_io_binding": bool(cfg.get("ort_io_binding", True)),
    }


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = PERSON_CLASS_ID
    display_name: str | None = None

    @property
    def class_name(self) -> str:
        if self.display_name:
            return self.display_name
        return CLASS_NAMES.get(self.class_id, f"class_{self.class_id}")


def _letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    h, w = image.shape[:2]
    nh, nw = new_shape
    r = min(nw / w, nh / h)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = (nw - new_unpad[0]) / 2, (nh - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return image, r, (left, top)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_threshold]
    return keep


class YoloDetector:
    """Multi-class YOLO detector with configurable COCO class filter."""

    def __init__(
        self,
        onnx_path: Path,
        model_name: str = "yolo11s",
        imgsz: int = 640,
        confidence: float = 0.45,
        iou: float = 0.5,
        max_detections: int = 50,
        force_cpu: bool = False,
        detect_person: bool = True,
        detect_car: bool = False,
        detect_class_ids: Sequence[int] | None = None,
        confidence_per_class: dict[str, float] | None = None,
        faded_conf_floor: float = 0.12,
        enable_io_binding: bool | None = None,
    ) -> None:
        self.imgsz = imgsz
        self.confidence = confidence
        self.iou = iou
        self.max_detections = max_detections
        self.model_name = model_name
        self.force_cpu = force_cpu
        self.confidence_per_class = dict(confidence_per_class or {})
        self.faded_conf_floor = faded_conf_floor
        self.debug_collect_faded = False
        self.last_faded_detections: list[Detection] = []

        if detect_class_ids is not None:
            self.detect_class_ids = self._normalize_class_ids(detect_class_ids)
        else:
            ids: list[int] = []
            if detect_person:
                ids.append(PERSON_CLASS_ID)
            if detect_car:
                ids.append(CAR_CLASS_ID)
            self.detect_class_ids = ids if ids else [PERSON_CLASS_ID]

        self._backend = "onnx"
        self.using_gpu = False
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.output_names: list[str] = []
        self._session_uses_cuda = False
        self._input_shape: tuple[int, int, int, int] | None = None
        if enable_io_binding is not None:
            self._enable_iobinding = enable_io_binding
        else:
            self._enable_iobinding = os.environ.get("PD_USE_ORT_IOBINDING", "1") == "1"
        self._model_family = self._detect_model_family(str(model_name), onnx_path.name)
        self._rtdetr_scale_fill = self._model_family == "rtdetr"
        self._yolo = None
        self._onnx_path = onnx_path

        if force_cpu:
            if self._init_onnx_cpu(onnx_path):
                logger.info("Detector backend: ONNX CPU (compatibility profile)")
            else:
                raise RuntimeError("No usable CPU inference backend found")
        elif self._init_onnx_cuda(onnx_path):
            logger.info("Detector backend: ONNX Runtime CUDA")
        elif self._init_torch_cuda(model_name, onnx_path):
            logger.info("Detector backend: Ultralytics PyTorch CUDA")
        elif self._init_onnx_cpu(onnx_path):
            logger.warning("Detector backend: ONNX CPU (install CUDA 12 for 60+ FPS)")
        else:
            raise RuntimeError("No usable inference backend found")

        self._warmup()

    @staticmethod
    def _normalize_class_ids(ids: Sequence[int]) -> list[int]:
        out: set[int] = set()
        for x in ids:
            xi = int(x)
            if 0 <= xi < 80:
                out.add(xi)
        return sorted(out) if out else [PERSON_CLASS_ID]

    def enabled_class_ids(self) -> set[int]:
        return set(self.detect_class_ids)

    def _threshold_for_class_id(self, class_id: int) -> float:
        name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        if name in self.confidence_per_class:
            return float(self.confidence_per_class[name])
        return float(self.confidence)

    def _min_torch_conf(self, enabled: set[int]) -> float:
        """Low floor for Ultralytics predict(); we filter per-class after."""
        vals = [self._threshold_for_class_id(c) for c in enabled]
        m = min(vals) if vals else float(self.confidence)
        return max(0.001, min(m * 0.25, m))

    def set_class_filter(self, detect_person: bool, detect_car: bool) -> None:
        """Legacy: map quick person/car toggles to COCO ids."""
        ids: list[int] = []
        if detect_person:
            ids.append(PERSON_CLASS_ID)
        if detect_car:
            ids.append(CAR_CLASS_ID)
        self.detect_class_ids = ids if ids else [PERSON_CLASS_ID]

    def set_detect_class_ids(self, class_ids: Sequence[int]) -> None:
        self.detect_class_ids = self._normalize_class_ids(class_ids)

    def set_confidence_per_class(self, m: dict[str, float] | None) -> None:
        self.confidence_per_class = dict(m or {})

    def _init_onnx_cuda(self, onnx_path: Path) -> bool:
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.enable_mem_pattern = True
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(
                str(onnx_path),
                sess_options=so,
                providers=[
                    (
                        "CUDAExecutionProvider",
                        {
                            "cudnn_conv_algo_search": "HEURISTIC",
                            "arena_extend_strategy": "kNextPowerOfTwo",
                        },
                    ),
                    "CPUExecutionProvider",
                ],
            )
            if "CUDAExecutionProvider" not in session.get_providers():
                return False
            self.session = session
            self.input_name = session.get_inputs()[0].name
            self.output_names = [o.name for o in session.get_outputs()]
            self._input_shape = tuple(int(x) for x in session.get_inputs()[0].shape if isinstance(x, int)) or None
            self._backend = "onnx"
            self.using_gpu = True
            self._session_uses_cuda = True
            return True
        except Exception as exc:
            logger.warning("ONNX CUDA init failed: %s", exc)
            return False

    def _init_onnx_cpu(self, onnx_path: Path) -> bool:
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(onnx_path),
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            self._input_shape = tuple(int(x) for x in self.session.get_inputs()[0].shape if isinstance(x, int)) or None
            self._backend = "onnx"
            self._session_uses_cuda = False
            return True
        except Exception as exc:
            logger.error("ONNX CPU init failed: %s", exc)
            return False

    def _init_torch_cuda(self, model_name: str, onnx_path: Path) -> bool:
        try:
            import torch
            from ultralytics import RTDETR, YOLO

            if not torch.cuda.is_available():
                return False
            pt = models_dir() / f"{model_name}.pt"
            if not pt.is_file():
                pt = Path(f"{model_name}.pt")
            if not pt.is_file():
                return False
            if self._model_family == "rtdetr":
                self._yolo = RTDETR(str(pt))
            else:
                self._yolo = YOLO(str(pt))
            self._yolo.to("cuda")
            self._backend = "torch"
            self.using_gpu = True
            return True
        except Exception as exc:
            logger.warning("PyTorch CUDA init failed: %s", exc)
            return False

    @property
    def backend_name(self) -> str:
        return self._backend

    def _warmup(self) -> None:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.detect(dummy)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        enabled = self.enabled_class_ids()
        if not enabled:
            self.last_faded_detections = []
            return []
        if self._backend == "torch":
            dets = self._detect_torch(frame_bgr, enabled)
        else:
            dets = self._detect_onnx(frame_bgr, enabled)
        return self._remap_custom_classes(dets)

    @staticmethod
    def _detect_model_family(model_name: str, onnx_name: str) -> str:
        name = f"{model_name} {onnx_name}".lower()
        if "peoplenet" in name:
            return "peoplenet"
        if "detectnet" in name:
            return "detectnet_v2"
        if "rtdetr" in name:
            return "rtdetr"
        return "yolo"

    def _remap_custom_classes(self, detections: list[Detection]) -> list[Detection]:
        """Custom fine-tuned models use class 1=car; COCO uses class 2=car."""
        if not str(self.model_name).startswith("custom_"):
            return detections
        out: list[Detection] = []
        for d in detections:
            cid = CAR_CLASS_ID if d.class_id == 1 else d.class_id
            out.append(
                Detection(
                    d.x1,
                    d.y1,
                    d.x2,
                    d.y2,
                    d.confidence,
                    class_id=cid,
                    display_name=d.display_name,
                )
            )
        return out

    def _detect_torch(self, frame_bgr: np.ndarray, enabled: set[int]) -> list[Detection]:
        assert self._yolo is not None
        floor = self._min_torch_conf(enabled)
        if self._model_family == "rtdetr":
            results = self._yolo.predict(
                frame_bgr,
                imgsz=self.imgsz,
                conf=floor,
                classes=sorted(enabled),
                max_det=self.max_detections * 4,
                verbose=False,
                device=0,
            )[0]
        else:
            results = self._yolo.predict(
                frame_bgr,
                imgsz=self.imgsz,
                conf=floor,
                iou=self.iou,
                classes=sorted(enabled),
                max_det=self.max_detections * 4,
                verbose=False,
                device=0,
            )[0]
        detections: list[Detection] = []
        faded: list[Detection] = []
        if results.boxes is None:
            self.last_faded_detections = []
            return detections
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cid = int(box.cls[0]) if box.cls is not None else PERSON_CLASS_ID
            thr = self._threshold_for_class_id(cid)
            det = Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=conf,
                class_id=cid,
            )
            if conf >= thr:
                detections.append(det)
            elif self.debug_collect_faded and conf >= self.faded_conf_floor:
                faded.append(det)
        if self.debug_collect_faded:
            faded.sort(key=lambda d: d.confidence, reverse=True)
            self.last_faded_detections = faded[: self.max_detections]
        else:
            self.last_faded_detections = []
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: self.max_detections]

    def _detect_onnx(self, frame_bgr: np.ndarray, enabled: set[int]) -> list[Detection]:
        assert self.session is not None and self.input_name is not None
        if self._model_family in ("peoplenet", "detectnet_v2"):
            return self._detect_onnx_tao(frame_bgr, enabled)
        if self._model_family == "rtdetr":
            return self._detect_onnx_rtdetr(frame_bgr, enabled)
        return self._detect_onnx_yolo(frame_bgr, enabled)

    def _run_onnx(self, blob: np.ndarray) -> list[np.ndarray]:
        assert self.session is not None and self.input_name is not None
        if self._enable_iobinding and self._session_uses_cuda:
            try:
                io = self.session.io_binding()
                io.bind_cpu_input(self.input_name, blob)
                for oname in self.output_names:
                    io.bind_output(oname)
                self.session.run_with_iobinding(io)
                return [np.asarray(x) for x in io.copy_outputs_to_cpu()]
            except Exception as exc:
                logger.debug("IOBinding fallback to run(): %s", exc)
        return self.session.run(None, {self.input_name: blob})

    def _detect_onnx_yolo(self, frame_bgr: np.ndarray, enabled: set[int]) -> list[Detection]:
        assert self.session is not None
        h0, w0 = frame_bgr.shape[:2]
        img, ratio, (pad_x, pad_y) = _letterbox(frame_bgr, (self.imgsz, self.imgsz))
        blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[np.newaxis, ...])

        outputs = self._run_onnx(blob)
        preds = outputs[0]
        if preds.ndim == 3:
            preds = preds[0]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        boxes_xywh = preds[:, :4]
        scores = preds[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(scores)), class_ids]

        enabled_list = list(enabled)
        mask_in_class = np.isin(class_ids, enabled_list)
        thr_arr = np.array(
            [self._threshold_for_class_id(int(c)) for c in class_ids], dtype=np.float32
        )
        mask_pass = mask_in_class & (confs >= thr_arr)
        if self.debug_collect_faded:
            mask_faded = (
                mask_in_class
                & (confs >= self.faded_conf_floor)
                & (confs < thr_arr)
            )
        else:
            mask_faded = np.zeros_like(mask_pass, dtype=bool)

        if not np.any(mask_pass) and not (self.debug_collect_faded and np.any(mask_faded)):
            self.last_faded_detections = []
            return []

        boxes_xywh_p = boxes_xywh[mask_pass]
        confs_p = confs[mask_pass]
        class_ids_p = class_ids[mask_pass]

        boxes = np.zeros_like(boxes_xywh_p)
        boxes[:, 0] = boxes_xywh_p[:, 0] - boxes_xywh_p[:, 2] / 2
        boxes[:, 1] = boxes_xywh_p[:, 1] - boxes_xywh_p[:, 3] / 2
        boxes[:, 2] = boxes_xywh_p[:, 0] + boxes_xywh_p[:, 2] / 2
        boxes[:, 3] = boxes_xywh_p[:, 1] + boxes_xywh_p[:, 3] / 2

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

        keep = _nms(boxes, confs_p, self.iou)[: self.max_detections]
        detections: list[Detection] = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(confs_p[i]),
                    class_id=int(class_ids_p[i]),
                )
            )

        if self.debug_collect_faded and np.any(mask_faded):
            bx_f = boxes_xywh[mask_faded]
            cf_f = confs[mask_faded]
            cid_f = class_ids[mask_faded]
            b2 = np.zeros_like(bx_f)
            b2[:, 0] = bx_f[:, 0] - bx_f[:, 2] / 2
            b2[:, 1] = bx_f[:, 1] - bx_f[:, 3] / 2
            b2[:, 2] = bx_f[:, 0] + bx_f[:, 2] / 2
            b2[:, 3] = bx_f[:, 1] + bx_f[:, 3] / 2
            b2[:, [0, 2]] -= pad_x
            b2[:, [1, 3]] -= pad_y
            b2 /= ratio
            b2[:, [0, 2]] = b2[:, [0, 2]].clip(0, w0)
            b2[:, [1, 3]] = b2[:, [1, 3]].clip(0, h0)
            kf = _nms(b2, cf_f, self.iou)[: self.max_detections]
            faded: list[Detection] = []
            for j in kf:
                x1, y1, x2, y2 = b2[j]
                faded.append(
                    Detection(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=float(cf_f[j]),
                        class_id=int(cid_f[j]),
                    )
                )
            self.last_faded_detections = faded
        else:
            self.last_faded_detections = []

        return detections

    def _detect_onnx_rtdetr(self, frame_bgr: np.ndarray, enabled: set[int]) -> list[Detection]:
        h0, w0 = frame_bgr.shape[:2]
        img = cv2.resize(frame_bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[np.newaxis, ...])
        outputs = self._run_onnx(blob)
        preds = outputs[0]
        for alt in outputs[1:]:
            if alt.size > preds.size:
                preds = alt
        boxes, confs, class_ids = decode_rtdetr_outputs(
            preds,
            image_w=self.imgsz,
            image_h=self.imgsz,
            score_threshold=self.faded_conf_floor,
        )
        if boxes.shape[0] > 0:
            sx = float(w0) / float(self.imgsz)
            sy = float(h0) / float(self.imgsz)
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)
        return self._filter_finalize(boxes, confs, class_ids, enabled)

    def _detect_onnx_tao(self, frame_bgr: np.ndarray, enabled: set[int]) -> list[Detection]:
        h0, w0 = frame_bgr.shape[:2]
        target_w = 960
        target_h = 544
        if self._input_shape and len(self._input_shape) == 4:
            _, _, ih, iw = self._input_shape
            if ih > 0 and iw > 0:
                target_h, target_w = ih, iw
        img = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[np.newaxis, ...])
        outputs = self._run_onnx(blob)
        if len(outputs) < 2:
            self.last_faded_detections = []
            return []
        out_a, out_b = outputs[0], outputs[1]
        # Heuristic: boxes tensor has channels multiple of 4 and usually > scores channels.
        if out_a.ndim == 4 and out_b.ndim == 4:
            if out_a.shape[1] >= out_b.shape[1]:
                boxes_t, scores_t = out_a, out_b
            else:
                boxes_t, scores_t = out_b, out_a
        else:
            self.last_faded_detections = []
            return []
        decode_fn = (
            decode_detectnet_v2_outputs
            if self._model_family == "detectnet_v2"
            else decode_peoplenet_outputs
        )
        boxes, confs, class_ids = decode_fn(
            boxes_t,
            scores_t,
            image_w=target_w,
            image_h=target_h,
            score_threshold=self.faded_conf_floor,
        )
        if boxes.shape[0] == 0:
            self.last_faded_detections = []
            return []
        sx = float(w0) / float(target_w)
        sy = float(h0) / float(target_h)
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)
        return self._filter_finalize(boxes, confs, class_ids, enabled)

    def _filter_finalize(
        self,
        boxes_xyxy: np.ndarray,
        confs: np.ndarray,
        class_ids: np.ndarray,
        enabled: set[int],
    ) -> list[Detection]:
        if boxes_xyxy.size == 0:
            self.last_faded_detections = []
            return []
        enabled_list = list(enabled)
        mask_in_class = np.isin(class_ids, enabled_list)
        thr_arr = np.array([self._threshold_for_class_id(int(c)) for c in class_ids], dtype=np.float32)
        mask_pass = mask_in_class & (confs >= thr_arr)
        if self.debug_collect_faded:
            mask_faded = mask_in_class & (confs >= self.faded_conf_floor) & (confs < thr_arr)
        else:
            mask_faded = np.zeros_like(mask_pass, dtype=bool)
        if not np.any(mask_pass) and not (self.debug_collect_faded and np.any(mask_faded)):
            self.last_faded_detections = []
            return []
        boxes_p = boxes_xyxy[mask_pass]
        confs_p = confs[mask_pass]
        class_ids_p = class_ids[mask_pass]
        keep = _nms(boxes_p, confs_p, self.iou)[: self.max_detections]
        detections: list[Detection] = []
        for i in keep:
            x1, y1, x2, y2 = boxes_p[i]
            cid = int(class_ids_p[i])
            dname = None
            if self._model_family in ("peoplenet", "detectnet_v2"):
                names = {0: "person", 1: "bag", 2: "face"}
                dname = names.get(cid, f"class_{cid}")
            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(confs_p[i]),
                    class_id=cid,
                    display_name=dname,
                )
            )
        if self.debug_collect_faded and np.any(mask_faded):
            b2 = boxes_xyxy[mask_faded]
            cf_f = confs[mask_faded]
            cid_f = class_ids[mask_faded]
            kf = _nms(b2, cf_f, self.iou)[: self.max_detections]
            self.last_faded_detections = [
                Detection(
                    x1=float(b2[j][0]),
                    y1=float(b2[j][1]),
                    x2=float(b2[j][2]),
                    y2=float(b2[j][3]),
                    confidence=float(cf_f[j]),
                    class_id=int(cid_f[j]),
                )
                for j in kf
            ]
        else:
            self.last_faded_detections = []
        return detections

    def update_thresholds(
        self,
        confidence: float | None = None,
        iou: float | None = None,
        max_detections: int | None = None,
    ) -> None:
        if confidence is not None:
            self.confidence = confidence
        if iou is not None:
            self.iou = iou
        if max_detections is not None:
            self.max_detections = max_detections


# Backward compatibility
PersonDetector = YoloDetector
