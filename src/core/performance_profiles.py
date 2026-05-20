"""Model / performance presets (Fast, Balanced, Quality, Custom, CPU compatibility)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

FAST = "fast"
BALANCED = "balanced"
QUALITY = "quality"
CUSTOM = "custom"
COMPATIBILITY = "compatibility"

# Legacy alias
PERFORMANCE = BALANCED

_PRESETS: dict[str, dict[str, Any]] = {
    FAST: {
        "active_model_id": "yolo11n",
        "model_name": "yolo11n",
        "model_imgsz": 640,
        "force_cpu": False,
        "screen_fps_cap": 0,
        "infer_max_width": 1280,
        "infer_max_height": 720,
        "preview_mode": "performance",
        "show_faded_low_conf": False,
    },
    BALANCED: {
        "active_model_id": "yolo11s",
        "model_name": "yolo11s",
        "model_imgsz": 640,
        "force_cpu": False,
        "screen_fps_cap": 0,
        "infer_max_width": 1280,
        "infer_max_height": 720,
        "preview_mode": "performance",
    },
    QUALITY: {
        "active_model_id": "yolo11m",
        "model_name": "yolo11m",
        "model_imgsz": 960,
        "force_cpu": False,
        "screen_fps_cap": 0,
        "infer_max_width": 1920,
        "infer_max_height": 1080,
        "preview_mode": "full",
    },
    CUSTOM: {},
    COMPATIBILITY: {
        "active_model_id": "yolo11n",
        "model_name": "yolo11n",
        "model_imgsz": 416,
        "force_cpu": True,
        "screen_fps_cap": 30,
        "show_pose_skeleton": False,
        "show_hand_skeleton": False,
        "preview_mode": "full",
    },
}


def apply_profile(cfg: dict[str, Any], profile_name: str) -> dict[str, Any]:
    """Merge preset values into config; preserve user zones and paths."""
    out = deepcopy(cfg)
    key = profile_name
    if key == "performance":
        key = BALANCED
    preset = _PRESETS.get(key, {})
    out["performance_profile"] = key
    out["model_preset"] = key
    for k, value in preset.items():
        out[k] = value
    return out


def profile_label(name: str) -> str:
    labels = {
        FAST: "Fast (yolo11n)",
        BALANCED: "Balanced (yolo11s)",
        QUALITY: "Quality (yolo11m)",
        CUSTOM: "Custom (Settings)",
        COMPATIBILITY: "Compatibility (CPU)",
        "performance": "Balanced (yolo11s)",
    }
    return labels.get(name, name)


def preset_ids() -> list[str]:
    return [FAST, BALANCED, QUALITY, CUSTOM, COMPATIBILITY]
