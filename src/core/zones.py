"""Polygon zones and enter/exit event detection."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np

from src.utils.config import PERSON_CLASS_ID

from src.core.tracker import TrackedObject

TrackedPerson = TrackedObject


class ZoneEventType(str, Enum):
    ENTER = "enter"
    EXIT = "exit"


@dataclass
class Zone:
    id: str
    name: str
    points: list[tuple[float, float]]
    enabled: bool = True
    color: tuple[int, int, int] = (0, 200, 255)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "points": self.points,
            "enabled": self.enabled,
            "color": list(self.color),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Zone:
        color = data.get("color", [0, 200, 255])
        return cls(
            id=str(data.get("id", uuid.uuid4().hex[:8])),
            name=str(data.get("name", "Zone")),
            points=[(float(p[0]), float(p[1])) for p in data.get("points", [])],
            enabled=bool(data.get("enabled", True)),
            color=(int(color[0]), int(color[1]), int(color[2])),
        )


@dataclass
class ZoneEvent:
    event_type: ZoneEventType
    zone_id: str
    zone_name: str
    track_id: int
    identity: str = "Unknown"
    match_score: float = 0.0


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    contour = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


class ZoneManager:
    def __init__(self, zones: list[Zone] | None = None) -> None:
        self.zones: list[Zone] = zones or []
        self._inside: dict[tuple[int, str], bool] = {}

    def set_zones(self, zones: list[Zone]) -> None:
        zone_ids = {z.id for z in zones}
        self.zones = zones
        self._inside = {k: v for k, v in self._inside.items() if k[1] in zone_ids}

    def add_zone(self, name: str, points: list[tuple[float, float]]) -> Zone:
        zone = Zone(id=uuid.uuid4().hex[:8], name=name, points=points)
        self.zones.append(zone)
        return zone

    def remove_zone(self, zone_id: str) -> None:
        self.zones = [z for z in self.zones if z.id != zone_id]
        self._inside = {k: v for k, v in self._inside.items() if k[1] != zone_id}

    def check_tracks(
        self,
        tracks: list[TrackedPerson],
        identity_scores: dict[int, float] | None = None,
        allowed_class_ids: set[int] | None = None,
    ) -> list[ZoneEvent]:
        events: list[ZoneEvent] = []
        active_keys: set[tuple[int, str]] = set()
        scores = identity_scores or {}
        allowed = allowed_class_ids or {PERSON_CLASS_ID}

        for track in tracks:
            if track.class_id not in allowed:
                continue
            ident = track.identity or "Unknown"
            score = scores.get(track.track_id, 0.0)
            for zone in self.zones:
                if not zone.enabled or len(zone.points) < 3:
                    continue
                key = (track.track_id, zone.id)
                active_keys.add(key)
                inside = point_in_polygon(track.feet_x, track.feet_y, zone.points)
                was_inside = self._inside.get(key, False)

                if inside and not was_inside:
                    events.append(
                        ZoneEvent(
                            ZoneEventType.ENTER,
                            zone.id,
                            zone.name,
                            track.track_id,
                            ident,
                            score,
                        )
                    )
                elif not inside and was_inside:
                    events.append(
                        ZoneEvent(
                            ZoneEventType.EXIT,
                            zone.id,
                            zone.name,
                            track.track_id,
                            ident,
                            score,
                        )
                    )
                self._inside[key] = inside

        stale = [k for k in self._inside if k not in active_keys]
        for k in stale:
            del self._inside[k]

        return events

    def zones_from_config(self, raw: list[dict[str, Any]]) -> None:
        self.zones = [Zone.from_dict(z) for z in raw]

    def zones_to_config(self) -> list[dict[str, Any]]:
        return [z.to_dict() for z in self.zones]
