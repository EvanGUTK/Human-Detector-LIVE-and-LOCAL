"""Training project CRUD."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.lab.paths import project_dir, projects_dir


@dataclass
class LabProject:
    id: str
    name: str
    video_path: str = ""
    val_ratio: float = 0.2
    notes: str = ""
    created: str = ""
    classes: list[str] = field(default_factory=lambda: ["person", "car"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "video_path": self.video_path,
            "val_ratio": self.val_ratio,
            "notes": self.notes,
            "created": self.created,
            "classes": self.classes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabProject:
        return cls(
            id=str(data.get("id", uuid.uuid4().hex[:8])),
            name=str(data.get("name", "Project")),
            video_path=str(data.get("video_path", "")),
            val_ratio=float(data.get("val_ratio", 0.2)),
            notes=str(data.get("notes", "")),
            created=str(data.get("created", "")),
            classes=list(data.get("classes", ["person", "car"])),
        )


class ProjectManager:
    def list_projects(self) -> list[LabProject]:
        projects: list[LabProject] = []
        for path in sorted(projects_dir().glob("*/project.json")):
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            projects.append(LabProject.from_dict(data))
        return projects

    def get(self, project_id: str) -> LabProject | None:
        path = project_dir(project_id) / "project.json"
        if not path.is_file():
            return None
        with path.open(encoding="utf-8") as f:
            return LabProject.from_dict(json.load(f))

    def create(self, name: str, video_path: str = "") -> LabProject:
        pid = uuid.uuid4().hex[:10]
        proj = LabProject(
            id=pid,
            name=name,
            video_path=video_path,
            created=datetime.now().isoformat(timespec="seconds"),
        )
        self.save(proj)
        return proj

    def save(self, project: LabProject) -> None:
        root = project_dir(project.id)
        with (root / "project.json").open("w", encoding="utf-8") as f:
            json.dump(project.to_dict(), f, indent=2)

    def delete(self, project_id: str) -> None:
        root = project_dir(project_id)
        if root.is_dir():
            for f in root.rglob("*"):
                if f.is_file():
                    f.unlink()
            for d in sorted(root.rglob("*"), reverse=True):
                if d.is_dir():
                    d.rmdir()
            root.rmdir()
