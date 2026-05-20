"""Zip YOLO export for sharing."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.lab.paths import project_dir


def export_yolo_zip(project_id: str, dest_zip: Path) -> Path:
    root = project_dir(project_id) / "exports" / "yolo"
    if not root.is_dir():
        raise FileNotFoundError("Export dataset first from Train tab")
    dest_zip = dest_zip.with_suffix(".zip")
    if dest_zip.is_file():
        dest_zip.unlink()
    shutil.make_archive(str(dest_zip.with_suffix("")), "zip", root)
    return dest_zip
