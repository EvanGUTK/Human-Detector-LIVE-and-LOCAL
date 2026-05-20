"""TAO/PeopleNet style output decoding helpers."""

from __future__ import annotations

import numpy as np


def decode_peoplenet_outputs(
    boxes: np.ndarray,
    scores: np.ndarray,
    image_w: int,
    image_h: int,
    score_threshold: float = 0.35,
    box_scale: float = 35.0,
    box_offset: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode TAO DetectNet_v2/PeopleNet outputs to xyxy boxes.

    Returns:
        boxes_xyxy: (N,4) float32 in image coordinates
        confs: (N,) float32
        class_ids: (N,) int32
    """
    # Expect (1, C, H, W) for both tensors.
    if boxes.ndim != 4 or scores.ndim != 4:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    b = boxes[0]
    s = scores[0]
    grid_h, grid_w = b.shape[1], b.shape[2]
    if grid_h <= 0 or grid_w <= 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    cell_h = float(image_h) / float(grid_h)
    cell_w = float(image_w) / float(grid_w)
    mx, my = np.meshgrid(np.arange(grid_w, dtype=np.float32), np.arange(grid_h, dtype=np.float32))

    det_boxes: list[np.ndarray] = []
    det_scores: list[np.ndarray] = []
    det_cls: list[np.ndarray] = []

    # boxes layout: 4 values per class, scores layout: 1 map per class.
    ncls = min(int(s.shape[0]), int(b.shape[0] // 4))
    for ci in range(ncls):
        sc = s[ci]
        keep = sc > float(score_threshold)
        if not np.any(keep):
            continue
        x1m = b[ci * 4 + 0]
        y1m = b[ci * 4 + 1]
        x2m = b[ci * 4 + 2]
        y2m = b[ci * 4 + 3]
        x1 = (-(x1m + box_offset) * box_scale) + (mx * cell_w)
        y1 = (-(y1m + box_offset) * box_scale) + (my * cell_h)
        x2 = ((x2m + box_offset) * box_scale) + (mx * cell_w)
        y2 = ((y2m + box_offset) * box_scale) + (my * cell_h)
        bx = np.stack([x1, y1, x2, y2], axis=-1)[keep]
        cf = sc[keep]
        cls = np.full((cf.shape[0],), ci, dtype=np.int32)
        det_boxes.append(bx.astype(np.float32, copy=False))
        det_scores.append(cf.astype(np.float32, copy=False))
        det_cls.append(cls)

    if not det_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    out_boxes = np.concatenate(det_boxes, axis=0)
    out_scores = np.concatenate(det_scores, axis=0)
    out_cls = np.concatenate(det_cls, axis=0)
    out_boxes[:, [0, 2]] = out_boxes[:, [0, 2]].clip(0, float(image_w))
    out_boxes[:, [1, 3]] = out_boxes[:, [1, 3]].clip(0, float(image_h))
    return out_boxes, out_scores, out_cls


# Alias — DetectNet_v2 deployable ONNX uses the same grid decode as PeopleNet
decode_detectnet_v2_outputs = decode_peoplenet_outputs


def decode_rtdetr_outputs(
    preds: np.ndarray,
    image_w: int,
    image_h: int,
    *,
    score_threshold: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode Ultralytics RT-DETR ONNX output to xyxy in original image coords.
    Supports (1,N,6), (N,6), (1,6,N) with xyxy or xywh normalized to letterbox size.
    """
    arr = np.asarray(preds)
    if arr.ndim == 3:
        if arr.shape[-1] >= 6:
            rows = arr[0]
        elif arr.shape[1] >= 6:
            rows = arr[0].T
        else:
            return _empty_tao()
    elif arr.ndim == 2 and arr.shape[-1] >= 6:
        rows = arr
    else:
        return _empty_tao()

    if rows.shape[-1] < 6:
        return _empty_tao()

    data = rows[:, :6].astype(np.float32)
    confs = data[:, 4]
    class_ids = data[:, 5].astype(np.int32)
    keep = confs >= float(score_threshold)
    if not np.any(keep):
        return _empty_tao()
    data = data[keep]
    confs = confs[keep]
    class_ids = class_ids[keep]

    sx = float(image_w) / float(max(image_w, 1))
    sy = float(image_h) / float(max(image_h, 1))
    # Heuristic: if coords look normalized 0..1
    if float(np.max(data[:, :4])) <= 1.5:
        scale_x, scale_y = float(image_w), float(image_h)
    else:
        scale_x, scale_y = sx, sy

    boxes = np.zeros((data.shape[0], 4), dtype=np.float32)
    x1, y1, x2, y2 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    # xywh center format if x2 < x1 or areas look like wh
    wh_like = np.mean((x2 > 0) & (y2 > 0) & (x2 < 1.0) & (y2 < 1.0))
    if wh_like > 0.5:
        boxes[:, 0] = (x1 - x2 / 2.0) * scale_x
        boxes[:, 1] = (y1 - y2 / 2.0) * scale_y
        boxes[:, 2] = (x1 + x2 / 2.0) * scale_x
        boxes[:, 3] = (y1 + y2 / 2.0) * scale_y
    else:
        boxes[:, 0] = x1 * scale_x
        boxes[:, 1] = y1 * scale_y
        boxes[:, 2] = x2 * scale_x
        boxes[:, 3] = y2 * scale_y

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, float(image_w))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, float(image_h))
    return boxes, confs.astype(np.float32), class_ids


def _empty_tao() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int32),
    )

