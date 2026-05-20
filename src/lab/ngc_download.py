"""Minimal NGC download helpers for TAO ONNX models."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.utils.config import models_dir


@dataclass(frozen=True)
class NgcModelSpec:
    model_id: str
    resource: str
    onnx_name_hints: tuple[str, ...]
    family: str
    class_names: list[str]


NGC_MODELS: dict[str, NgcModelSpec] = {
    "peoplenet": NgcModelSpec(
        model_id="peoplenet",
        resource="nvidia/tao/peoplenet:deployable_v2.6",
        onnx_name_hints=("peoplenet", "people", "resnet34"),
        family="peoplenet",
        class_names=["person", "bag", "face"],
    ),
    "detectnet_v2": NgcModelSpec(
        model_id="detectnet_v2",
        resource="nvidia/tao/detectnet_v2:deployable_resnet18_960x544",
        onnx_name_hints=("detectnet", "resnet18", "efficientnet"),
        family="detectnet_v2",
        class_names=["person", "bag", "face"],
    ),
}


def tao_onnx_path(model_id: str) -> Path:
    return models_dir() / f"{model_id}_tao.onnx"


def _pick_onnx(candidates: list[Path], spec: NgcModelSpec) -> Path | None:
    if not candidates:
        return None
    scored: list[tuple[int, Path]] = []
    for p in candidates:
        name = p.name.lower()
        score = 0
        if spec.model_id.replace("_", "") in name:
            score += 10
        for hint in spec.onnx_name_hints:
            if hint in name:
                score += 5
        if "peoplenet" in spec.model_id and "peoplenet" in name:
            score += 8
        if "detectnet" in spec.model_id and "detectnet" in name:
            score += 8
        if "peoplenet" in name and "detectnet" in spec.model_id:
            score -= 20
        if "detectnet" in name and "peoplenet" in spec.model_id:
            score -= 20
        scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], -len(str(x[1]))))
    best = scored[0]
    return best[1] if best[0] > 0 else candidates[0]


def ensure_ngc_model_onnx(model_id: str, *, force_refresh: bool = False) -> Path:
    spec = NGC_MODELS.get(model_id)
    if spec is None:
        raise ValueError(f"Unsupported NGC model: {model_id}")
    out = tao_onnx_path(model_id)
    if out.is_file() and not force_refresh:
        return out

    cache_dir = models_dir() / "ngc" / model_id
    candidates = sorted(cache_dir.rglob("*.onnx")) if cache_dir.is_dir() else []

    if not candidates:
        api_key = os.environ.get("NGC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                f"NGC model {model_id} not cached at {cache_dir}. "
                "Set NGC_API_KEY and install NGC CLI to auto-download, or place ONNX manually."
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ngc",
            "registry",
            "model",
            "download-version",
            spec.resource,
            "--dest",
            str(cache_dir),
            "--yes",
        ]
        env = dict(os.environ)
        env["NGC_API_KEY"] = api_key
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"NGC download failed for {model_id} ({spec.resource}): "
                f"{proc.stderr.strip() or proc.stdout.strip() or 'unknown error'}"
            )
        candidates = sorted(cache_dir.rglob("*.onnx"))

    chosen = _pick_onnx([p for p in candidates if p.is_file()], spec)
    if chosen is None:
        raise FileNotFoundError(
            f"No ONNX file for {model_id} under {cache_dir}. "
            f"Downloaded resource: {spec.resource}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        out.unlink()
    shutil.copy2(chosen, out)
    return out
