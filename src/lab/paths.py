"""Lab storage paths."""

from __future__ import annotations

from pathlib import Path

from src.utils.config import app_data_dir


def lab_dir() -> Path:
    path = app_data_dir() / "lab"
    path.mkdir(parents=True, exist_ok=True)
    return path


def projects_dir() -> Path:
    path = lab_dir() / "projects"
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_dir(project_id: str) -> Path:
    path = projects_dir() / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def registry_path() -> Path:
    return app_data_dir() / "models_registry.json"
