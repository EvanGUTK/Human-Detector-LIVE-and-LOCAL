"""Session event log for testing and export."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LogEntry:
    timestamp: float
    event_type: str
    track_id: int
    zone_name: str
    message: str
    fps: float = 0.0
    infer_ms: float = 0.0
    identity: str = ""
    match_score: float = 0.0


class EventLog:
    def __init__(self, max_entries: int = 500) -> None:
        self._entries: list[LogEntry] = []
        self._max = max_entries

    def append(
        self,
        event_type: str,
        message: str,
        track_id: int = -1,
        zone_name: str = "",
        fps: float = 0.0,
        infer_ms: float = 0.0,
        identity: str = "",
        match_score: float = 0.0,
    ) -> None:
        self._entries.append(
            LogEntry(
                time.time(),
                event_type,
                track_id,
                zone_name,
                message,
                fps,
                infer_ms,
                identity,
                match_score,
            )
        )
        if len(self._entries) > self._max:
            self._entries.pop(0)

    @property
    def entries(self) -> list[LogEntry]:
        return list(self._entries)

    def recent_messages(self, n: int = 15) -> list[str]:
        return [e.message for e in self._entries[-n:]]

    def export_csv(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "event_type",
                    "track_id",
                    "zone_name",
                    "message",
                    "fps",
                    "infer_ms",
                    "identity",
                    "match_score",
                ]
            )
            for e in self._entries:
                writer.writerow(
                    [
                        e.timestamp,
                        e.event_type,
                        e.track_id,
                        e.zone_name,
                        e.message,
                        f"{e.fps:.2f}",
                        f"{e.infer_ms:.2f}",
                        e.identity,
                        f"{e.match_score:.3f}",
                    ]
                )
