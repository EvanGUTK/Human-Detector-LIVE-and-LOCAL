"""Application configuration load/save."""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PERSON_CLASS_ID = 0
CAR_CLASS_ID = 2

# Legacy 2-class map; full names in coco_classes.COCO80_NAMES
CLASS_NAMES = {i: n for i, n in enumerate(
    (
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    )
)}

_DEFAULTS: dict[str, Any] = {
    "camera_index": 0,
    "capture_width": 1280,
    "capture_height": 720,
    "mirror": False,
    "model_name": "yolo11s",
    "model_imgsz": 640,
    "confidence": 0.45,
    "iou": 0.5,
    "max_detections": 50,
    "alert_sound": True,
    "alert_toast": True,
    "alert_flash_ms": 800,
    "zones": [],
    "show_pose_skeleton": False,
    "show_hand_skeleton": True,
    "skeleton_stride": 2,
    "performance_profile": "performance",
    "force_cpu": False,
    "ui_mode": "dashboard",
    "input_source": "webcam",
    "file_path": "",
    "file_loop": True,
    "file_playback_speed": 1.0,
    "show_debug_overlay": False,
    "show_zones_on_frame": True,
    "show_boxes_on_frame": True,
    "show_hud_on_frame": True,
    "face_enabled": True,
    "face_match_threshold": 0.45,
    "face_stride": 3,
    "face_gallery_id": "Default room",
    "active_profile": "Default room",
    "minimize_to_tray": True,
    "first_run_complete": False,
    "detect_person": True,
    "detect_car": False,
    "active_model_id": "yolo11s",
    "screen_monitor": 1,
    "screen_region": None,
    "screen_fps_cap": 0,
    "infer_max_width": 1280,
    "infer_max_height": 720,
    "preview_mode": "performance",
    "ort_io_binding": True,
    "model_preset": "balanced",
    "compare_source": "video",
    "compare_playback_speed": 1.0,
    "detect_class_ids": [0],
    "zone_alert_class_ids": [0],
    "face_rec_class_ids": [0],
    "confidence_per_class": {},
    "show_faded_low_conf": False,
    "fire_smoke_enabled": False,
    "fire_smoke_onnx_path": "",
    "fire_smoke_full_frame": True,
}


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def app_data_dir() -> Path:
    base = os.environ.get("APPDATA", os.path.expanduser("~"))
    path = Path(base) / "PersonDetector"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    root = project_root()
    path = root / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_path() -> Path:
    return app_data_dir() / "person_detector.log"


def user_config_path() -> Path:
    return app_data_dir() / "config.json"


def default_yaml_path() -> Path:
    return project_root() / "config" / "default.yaml"


def load_defaults_from_yaml() -> dict[str, Any]:
    path = default_yaml_path()
    if not path.is_file():
        return deepcopy(_DEFAULTS)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = deepcopy(_DEFAULTS)
    merged.update(data)
    return merged


def migrate_config_inplace(cfg: dict[str, Any]) -> None:
    """Ensure new keys exist; derive detect_class_ids from legacy booleans."""
    if cfg.get("detect_class_ids") is None:
        ids: list[int] = []
        if cfg.get("detect_person", True):
            ids.append(PERSON_CLASS_ID)
        if cfg.get("detect_car", False):
            ids.append(CAR_CLASS_ID)
        cfg["detect_class_ids"] = ids if ids else [PERSON_CLASS_ID]
    if "confidence_per_class" not in cfg or not isinstance(cfg.get("confidence_per_class"), dict):
        cfg["confidence_per_class"] = {}
    if cfg.get("zone_alert_class_ids") is None:
        cfg["zone_alert_class_ids"] = list(cfg.get("detect_class_ids", [PERSON_CLASS_ID]))
    if cfg.get("face_rec_class_ids") is None:
        cfg["face_rec_class_ids"] = [PERSON_CLASS_ID]
    if "show_faded_low_conf" not in cfg:
        cfg["show_faded_low_conf"] = False
    if "fire_smoke_enabled" not in cfg:
        cfg["fire_smoke_enabled"] = False
    if "fire_smoke_onnx_path" not in cfg:
        cfg["fire_smoke_onnx_path"] = ""
    if "fire_smoke_full_frame" not in cfg:
        cfg["fire_smoke_full_frame"] = True
    if "infer_max_width" not in cfg:
        cfg["infer_max_width"] = 1280
    if "infer_max_height" not in cfg:
        cfg["infer_max_height"] = 720
    if cfg.get("preview_mode") not in ("full", "performance"):
        cfg["preview_mode"] = "performance"
    if "ort_io_binding" not in cfg:
        cfg["ort_io_binding"] = True
    if cfg.get("performance_profile") == "performance":
        cfg["performance_profile"] = "balanced"
        cfg["model_preset"] = cfg.get("model_preset") or "balanced"
    if "model_preset" not in cfg:
        cfg["model_preset"] = str(cfg.get("performance_profile", "balanced"))
    sync_legacy_detect_flags(cfg)


def sync_legacy_detect_flags(cfg: dict[str, Any]) -> None:
    """Normalize detect_class_ids and mirror legacy detect_person / detect_car booleans."""
    raw = cfg.get("detect_class_ids")
    if not isinstance(raw, list) or not raw:
        raw = [PERSON_CLASS_ID]
    s: set[int] = set()
    for x in raw:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= xi < 80:
            s.add(xi)
    if not s:
        s.add(PERSON_CLASS_ID)
    cfg["detect_class_ids"] = sorted(s)
    cfg["detect_person"] = PERSON_CLASS_ID in s
    cfg["detect_car"] = CAR_CLASS_ID in s
    for key in ("zone_alert_class_ids", "face_rec_class_ids"):
        raw_ids = cfg.get(key)
        if not isinstance(raw_ids, list) or not raw_ids:
            cfg[key] = [PERSON_CLASS_ID]
            continue
        clean: set[int] = set()
        for x in raw_ids:
            try:
                xi = int(x)
            except (TypeError, ValueError):
                continue
            if 0 <= xi < 80:
                clean.add(xi)
        cfg[key] = sorted(clean) if clean else [PERSON_CLASS_ID]


def load_config() -> dict[str, Any]:
    cfg = load_defaults_from_yaml()
    user_path = user_config_path()
    if user_path.is_file():
        with user_path.open(encoding="utf-8") as f:
            user = json.load(f)
        if isinstance(user, dict):
            cfg.update(user)
    migrate_config_inplace(cfg)
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    user_path = user_config_path()
    with user_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
