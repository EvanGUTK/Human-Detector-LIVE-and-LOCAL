"""Inspect ONNX sessions for model health / debugging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TensorInfo:
    name: str
    shape: list[int | str]
    dtype: str


@dataclass
class ModelProbeResult:
    model_id: str
    onnx_path: str
    ok: bool
    message: str = ""
    inputs: list[TensorInfo] = field(default_factory=list)
    outputs: list[TensorInfo] = field(default_factory=list)
    infer_ms: float = 0.0
    box_count: int = 0
    using_gpu: bool = False
    backend: str = ""


def _shape_list(shape) -> list[int | str]:
    out: list[int | str] = []
    for d in shape:
        if isinstance(d, (int, str)):
            out.append(d)
        else:
            out.append(str(d))
    return out


def probe_onnx_session(session, model_id: str, onnx_path: Path) -> ModelProbeResult:
    import onnxruntime as ort

    inputs: list[TensorInfo] = []
    for inp in session.get_inputs():
        inputs.append(
            TensorInfo(
                name=inp.name,
                shape=_shape_list(inp.shape),
                dtype=str(inp.type),
            )
        )
    outputs: list[TensorInfo] = []
    for out in session.get_outputs():
        outputs.append(
            TensorInfo(
                name=out.name,
                shape=_shape_list(out.shape),
                dtype=str(out.type),
            )
        )
    providers = session.get_providers()
    using_gpu = "CUDAExecutionProvider" in providers
    return ModelProbeResult(
        model_id=model_id,
        onnx_path=str(onnx_path),
        ok=True,
        message="Session loaded",
        inputs=inputs,
        outputs=outputs,
        using_gpu=using_gpu,
        backend="onnx",
    )


def probe_infer_dummy(
    result: ModelProbeResult,
    session,
    *,
    width: int = 640,
    height: int = 640,
) -> ModelProbeResult:
    import time

    import onnxruntime as ort

    inp = session.get_inputs()[0]
    shape = list(inp.shape)
    # Replace dynamic dims
    for i, d in enumerate(shape):
        if not isinstance(d, int) or d <= 0:
            if i == 0:
                shape[i] = 1
            elif i == 2:
                shape[i] = height
            elif i == 3:
                shape[i] = width
            else:
                shape[i] = 3
    if len(shape) == 4:
        _, c, h, w = shape
        blob = np.zeros((1, int(c), int(h), int(w)), dtype=np.float32)
    else:
        blob = np.zeros(tuple(int(x) for x in shape), dtype=np.float32)
    t0 = time.perf_counter()
    try:
        session.run(None, {inp.name: blob})
        result.infer_ms = (time.perf_counter() - t0) * 1000.0
        result.ok = True
    except Exception as exc:
        result.ok = False
        result.message = f"Inference failed: {exc}"
    return result
