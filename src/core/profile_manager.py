"""Room profile save/load (zones, settings per room)."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.utils.config import app_data_dir

logger = logging.getLogger(__name__)

PROFILE_KEYS = (
    "camera_index",
    "capture_width",
    "capture_height",
    "mirror",
    "model_name",
    "model_imgsz",
    "confidence",
    "confidence_per_class",
    "iou",
    "max_detections",
    "alert_sound",
    "alert_toast",
    "alert_flash_ms",
    "show_pose_skeleton",
    "show_hand_skeleton",
    "skeleton_stride",
    "performance_profile",
    "force_cpu",
    "input_source",
    "file_path",
    "file_loop",
    "file_playback_speed",
    "ui_mode",
    "show_debug_overlay",
    "show_faded_low_conf",
    "show_zones_on_frame",
    "show_boxes_on_frame",
    "show_hud_on_frame",
    "face_match_threshold",
    "face_stride",
    "face_enabled",
    "zones",
    "face_gallery_id",
    "detect_person",
    "detect_car",
    "detect_class_ids",
    "zone_alert_class_ids",
    "face_rec_class_ids",
    "active_model_id",
    "fire_smoke_enabled",
    "fire_smoke_onnx_path",
    "fire_smoke_full_frame",
    "screen_monitor",
    "screen_region",
    "screen_fps_cap",
    "infer_max_width",
    "infer_max_height",
    "preview_mode",
    "ort_io_binding",
    "model_preset",
)


def profiles_dir() -> Path:
    path = app_data_dir() / "profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _profile_path(name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip()
    if not safe:
        safe = "profile"
    return profiles_dir() / f"{safe}.json"


class ProfileManager:
    def __init__(self) -> None:
        self._ensure_default()

    def _ensure_default(self) -> None:
        path = _profile_path("Default room")
        if not path.is_file():
            self.save("Default room", {"zones": []})

    def list_profiles(self) -> list[str]:
        names = []
        for p in sorted(profiles_dir().glob("*.json")):
            names.append(p.stem)
        return names or ["Default room"]

    def save(self, name: str, cfg: dict[str, Any]) -> None:
        data = {k: cfg.get(k) for k in PROFILE_KEYS if k in cfg}
        if "zones" not in data:
            data["zones"] = cfg.get("zones", [])
        data["face_gallery_id"] = cfg.get("face_gallery_id", name)
        path = _profile_path(name)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved profile %s", name)

    def load(self, name: str) -> dict[str, Any]:
        path = _profile_path(name)
        if not path.is_file():
            return {}
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    def delete(self, name: str) -> bool:
        path = _profile_path(name)
        if path.is_file():
            path.unlink()
            return True
        return False

    def apply_to_config(self, cfg: dict[str, Any], profile_name: str) -> dict[str, Any]:
        from src.utils.config import CAR_CLASS_ID, PERSON_CLASS_ID, migrate_config_inplace

        out = deepcopy(cfg)
        loaded = self.load(profile_name)
        out.update(loaded)
        if (
            "detect_class_ids" not in loaded
            and ("detect_person" in loaded or "detect_car" in loaded)
        ):
            ids: list[int] = []
            if loaded.get("detect_person", True):
                ids.append(PERSON_CLASS_ID)
            if loaded.get("detect_car", False):
                ids.append(CAR_CLASS_ID)
            out["detect_class_ids"] = ids if ids else [PERSON_CLASS_ID]
        migrate_config_inplace(out)
        out["active_profile"] = profile_name
        if "face_gallery_id" not in loaded:
            out["face_gallery_id"] = profile_name
        return out

    def extract_from_config(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return {k: deepcopy(cfg[k]) for k in PROFILE_KEYS if k in cfg}
