"""Session metrics and ring buffers for analytics."""

from __future__ import annotations

import csv
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ZoneDwellRecord:
    zone_id: str
    zone_name: str
    track_id: int
    identity: str
    duration_sec: float
    timestamp: float


class MetricsStore:
    def __init__(self, buffer_size: int = 600) -> None:
        self._buffer_size = buffer_size
        self.fps_samples: deque[float] = deque(maxlen=buffer_size)
        self.infer_samples: deque[float] = deque(maxlen=buffer_size)
        self.total_samples: deque[float] = deque(maxlen=buffer_size)
        self.detect_samples: deque[float] = deque(maxlen=buffer_size)
        self.fire_samples: deque[float] = deque(maxlen=buffer_size)
        self.track_samples: deque[float] = deque(maxlen=buffer_size)
        self.face_samples: deque[float] = deque(maxlen=buffer_size)
        self.zones_samples: deque[float] = deque(maxlen=buffer_size)
        self.preprocess_samples: deque[float] = deque(maxlen=buffer_size)
        self.draw_samples: deque[float] = deque(maxlen=buffer_size)
        self.zone_enter_count: dict[str, int] = {}
        self.zone_dwell_total: dict[str, float] = {}
        self.dwell_records: list[ZoneDwellRecord] = []
        self.unique_identities: set[str] = set()
        self.total_enters = 0
        self.total_exits = 0
        self._dwell_start: dict[tuple[int, str], tuple[float, str, str]] = {}
        self.session_start = time.time()

    def reset(self) -> None:
        self.fps_samples.clear()
        self.infer_samples.clear()
        self.total_samples.clear()
        self.detect_samples.clear()
        self.fire_samples.clear()
        self.track_samples.clear()
        self.face_samples.clear()
        self.zones_samples.clear()
        self.preprocess_samples.clear()
        self.draw_samples.clear()
        self.zone_enter_count.clear()
        self.zone_dwell_total.clear()
        self.dwell_records.clear()
        self.unique_identities.clear()
        self.total_enters = 0
        self.total_exits = 0
        self._dwell_start.clear()
        self.session_start = time.time()

    def record_frame(
        self,
        fps: float,
        infer_ms: float,
        total_ms: float | None = None,
        detect_ms: float | None = None,
        fire_ms: float | None = None,
        track_ms: float | None = None,
        face_ms: float | None = None,
        zones_ms: float | None = None,
        preprocess_ms: float | None = None,
        draw_ms: float | None = None,
    ) -> None:
        self.fps_samples.append(fps)
        self.infer_samples.append(infer_ms)
        if total_ms is not None:
            self.total_samples.append(total_ms)
        if detect_ms is not None:
            self.detect_samples.append(detect_ms)
        if fire_ms is not None:
            self.fire_samples.append(fire_ms)
        if track_ms is not None:
            self.track_samples.append(track_ms)
        if face_ms is not None:
            self.face_samples.append(face_ms)
        if zones_ms is not None:
            self.zones_samples.append(zones_ms)
        if preprocess_ms is not None:
            self.preprocess_samples.append(preprocess_ms)
        if draw_ms is not None:
            self.draw_samples.append(draw_ms)

    def record_identity(self, name: str) -> None:
        if name:
            self.unique_identities.add(name)

    def record_zone_enter(
        self, zone_id: str, zone_name: str, track_id: int, identity: str = "Unknown"
    ) -> None:
        self.total_enters += 1
        self.zone_enter_count[zone_name] = self.zone_enter_count.get(zone_name, 0) + 1
        key = (track_id, zone_id)
        self._dwell_start[key] = (time.time(), zone_name, identity)

    def record_zone_exit(
        self, zone_id: str, zone_name: str, track_id: int, identity: str = "Unknown"
    ) -> None:
        self.total_exits += 1
        key = (track_id, zone_id)
        start = self._dwell_start.pop(key, None)
        if start is None:
            return
        t0, zname, ident = start
        duration = time.time() - t0
        self.zone_dwell_total[zname] = self.zone_dwell_total.get(zname, 0.0) + duration
        self.dwell_records.append(
            ZoneDwellRecord(zone_id, zname, track_id, ident, duration, time.time())
        )

    def fps_summary(self) -> tuple[float, float, float]:
        if not self.fps_samples:
            return 0.0, 0.0, 0.0
        vals = list(self.fps_samples)
        return min(vals), sum(vals) / len(vals), max(vals)

    def infer_summary(self) -> tuple[float, float, float]:
        if not self.infer_samples:
            return 0.0, 0.0, 0.0
        vals = list(self.infer_samples)
        return min(vals), sum(vals) / len(vals), max(vals)

    def top_zone_by_activity(self) -> str:
        if not self.zone_enter_count:
            return "—"
        return max(self.zone_enter_count, key=self.zone_enter_count.get)  # type: ignore[arg-type]

    def export_session_csv(self, path: Path) -> None:
        fps_min, fps_avg, fps_max = self.fps_summary()
        inf_min, inf_avg, inf_max = self.infer_summary()
        tot_vals = list(self.total_samples)
        tot_avg = sum(tot_vals) / len(tot_vals) if tot_vals else 0.0
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["session_duration_sec", f"{time.time() - self.session_start:.1f}"])
            w.writerow(["fps_min", f"{fps_min:.2f}"])
            w.writerow(["fps_avg", f"{fps_avg:.2f}"])
            w.writerow(["fps_max", f"{fps_max:.2f}"])
            w.writerow(["infer_ms_min", f"{inf_min:.2f}"])
            w.writerow(["infer_ms_avg", f"{inf_avg:.2f}"])
            w.writerow(["infer_ms_max", f"{inf_max:.2f}"])
            w.writerow(["total_ms_avg", f"{tot_avg:.2f}"])
            w.writerow(["total_enters", self.total_enters])
            w.writerow(["total_exits", self.total_exits])
            w.writerow(["unique_identities", len(self.unique_identities)])
            w.writerow(["top_zone", self.top_zone_by_activity()])
            w.writerow([])
            w.writerow(["zone_name", "enter_count", "dwell_total_sec"])
            for zname, count in sorted(self.zone_enter_count.items()):
                dwell = self.zone_dwell_total.get(zname, 0.0)
                w.writerow([zname, count, f"{dwell:.2f}"])
