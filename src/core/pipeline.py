"""Detection pipeline: capture + inference + tracking + zones + faces."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.core.alerts import AlertManager
from src.core.detector import Detection, YoloDetector
from src.core.face_recognizer import FaceRecognizer, UNKNOWN
from src.core.fire_smoke_detector import FireSmokeDetector
from src.core.frame_source import FrameSource
from src.core.metrics import MetricsStore
from src.core.skeleton_overlay import SkeletonOverlay
from src.core.tracker import ObjectTracker, TrackedObject

TrackedPerson = TrackedObject
from src.core.zones import ZoneEvent, ZoneEventType, ZoneManager, point_in_polygon
from src.utils.config import CLASS_NAMES, PERSON_CLASS_ID
from src.utils.frame_resize import (
    InferScale,
    resize_for_infer,
    scale_detections,
    upscale_frame_for_display,
)

logger = logging.getLogger(__name__)

_VEHICLE_IDS = frozenset({1, 2, 3, 5, 7})
_ANIMAL_IDS = frozenset({14, 15, 16, 17, 18, 19, 20, 21, 22, 23})


def _track_color(class_id: int) -> tuple[int, int, int]:
    if class_id == PERSON_CLASS_ID:
        return (0, 255, 0)
    if class_id in _VEHICLE_IDS:
        return (0, 165, 255)
    if class_id in _ANIMAL_IDS:
        return (255, 200, 100)
    return (200, 180, 255)


@dataclass
class DebugTrackInfo:
    track_id: int
    feet_x: float
    feet_y: float
    zone_states: dict[str, bool]  # zone_id -> inside
    identity: str
    match_score: float


@dataclass
class DebugInfo:
    tracks: list[DebugTrackInfo] = field(default_factory=list)


@dataclass
class FrameResult:
    frame: np.ndarray
    tracks: list[TrackedPerson] = field(default_factory=list)
    zone_events: list[ZoneEvent] = field(default_factory=list)
    fps: float = 0.0
    infer_ms: float = 0.0
    detect_ms: float = 0.0
    fire_ms: float = 0.0
    track_ms: float = 0.0
    face_ms: float = 0.0
    zones_ms: float = 0.0
    total_ms: float = 0.0
    preprocess_ms: float = 0.0
    draw_ms: float = 0.0
    capture_w: int = 0
    capture_h: int = 0
    infer_w: int = 0
    infer_h: int = 0
    preview_mode: str = "performance"
    in_frame_count: int = 0
    unique_session_count: int = 0
    source_type: str = "webcam"
    source_label: str = ""
    identities: dict[int, str] = field(default_factory=dict)
    debug: DebugInfo | None = None
    faded_detections: list[Detection] = field(default_factory=list)
    fire_smoke_detections: list[Detection] = field(default_factory=list)


class DetectionPipeline:
    def __init__(
        self,
        detector: YoloDetector,
        capture: FrameSource,
        zone_manager: ZoneManager,
        alert_manager: AlertManager,
        metrics: MetricsStore | None = None,
        face_recognizer: FaceRecognizer | None = None,
        show_pose_skeleton: bool = False,
        show_hand_skeleton: bool = True,
        skeleton_stride: int = 2,
        show_zones_on_frame: bool = True,
        show_boxes_on_frame: bool = True,
        show_hud_on_frame: bool = True,
        show_debug_overlay: bool = False,
        ui_mode: str = "dashboard",
        show_faded_low_conf: bool = False,
        fire_smoke_detector: FireSmokeDetector | None = None,
        zone_alert_class_ids: set[int] | None = None,
        face_rec_class_ids: set[int] | None = None,
        infer_max_width: int = 1280,
        infer_max_height: int = 720,
        preview_mode: str = "performance",
    ) -> None:
        self.detector = detector
        self.capture = capture
        self.zone_manager = zone_manager
        self.alert_manager = alert_manager
        self.metrics = metrics
        self.face_recognizer = face_recognizer
        self.show_zones_on_frame = show_zones_on_frame
        self.show_boxes_on_frame = show_boxes_on_frame
        self.show_hud_on_frame = show_hud_on_frame
        self.show_debug_overlay = show_debug_overlay
        self.ui_mode = ui_mode
        self.show_faded_low_conf = show_faded_low_conf
        self._fire_smoke = fire_smoke_detector
        self.zone_alert_class_ids = set(zone_alert_class_ids or {PERSON_CLASS_ID})
        self.face_rec_class_ids = set(face_rec_class_ids or {PERSON_CLASS_ID})
        self.infer_max_width = int(infer_max_width)
        self.infer_max_height = int(infer_max_height)
        self.preview_mode = (
            preview_mode if preview_mode in ("full", "performance") else "performance"
        )

        self.tracker = ObjectTracker()
        self._skeleton: SkeletonOverlay | None = None
        if show_pose_skeleton or show_hand_skeleton:
            self._skeleton = SkeletonOverlay(
                show_pose=show_pose_skeleton,
                show_hands=show_hand_skeleton,
                stride=skeleton_stride,
            )

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._lock = threading.Lock()
        self._latest_result: FrameResult | None = None
        self._fps_samples: list[float] = []
        self._last_frame_id = -1
        self._pulse_zone_id: str | None = None
        self._pulse_until: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if not self.capture.start():
            return False
        self._stop.clear()
        self._pause.clear()
        self.tracker.reset()
        if self.face_recognizer:
            self.face_recognizer.reset_session()
        if self._skeleton is not None:
            self._skeleton.reset()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self.capture.stop()
        if self._skeleton is not None:
            self._skeleton.close()
        with self._lock:
            self._latest_result = None

    def pulse_zone(self, zone_id: str, duration_ms: int = 2500) -> None:
        self._pulse_zone_id = zone_id
        self._pulse_until = time.time() + duration_ms / 1000.0

    def pause(self, paused: bool) -> None:
        if paused:
            self._pause.set()
        else:
            self._pause.clear()

    def get_result(self) -> FrameResult | None:
        with self._lock:
            return self._latest_result

    def _loop(self) -> None:
        last_t = time.perf_counter()
        while not self._stop.is_set():
            if self._pause.is_set():
                time.sleep(0.02)
                continue

            need_capture_copy = self.preview_mode == "full"
            frame, fid = self.capture.get_latest(copy=need_capture_copy)
            if frame is None or fid == self._last_frame_id:
                time.sleep(0.001)
                continue
            self._last_frame_id = fid
            cap_h, cap_w = frame.shape[:2]

            t0 = time.perf_counter()
            t_pre0 = time.perf_counter()
            infer_frame, infer_scale = resize_for_infer(
                frame,
                self.infer_max_width,
                self.infer_max_height,
            )
            preprocess_ms = (time.perf_counter() - t_pre0) * 1000.0

            self.detector.debug_collect_faded = self.show_faded_low_conf
            t_det0 = time.perf_counter()
            detections = scale_detections(
                self.detector.detect(infer_frame), infer_scale
            )
            detect_ms = (time.perf_counter() - t_det0) * 1000.0
            faded = (
                scale_detections(
                    list(self.detector.last_faded_detections), infer_scale
                )
                if self.show_faded_low_conf
                else []
            )
            fire_dets: list[Detection] = []
            t_fire0 = time.perf_counter()
            if self._fire_smoke is not None:
                fire_raw = self._fire_smoke.detect(infer_frame)
                fire_dets = scale_detections(fire_raw, infer_scale)
            fire_ms = (time.perf_counter() - t_fire0) * 1000.0
            t_track0 = time.perf_counter()
            tracks = self.tracker.update(detections)
            track_ms = (time.perf_counter() - t_track0) * 1000.0

            identity_scores: dict[int, float] = {}
            identities: dict[int, str] = {}
            face_tracks = [t for t in tracks if t.class_id in self.face_rec_class_ids]
            zone_tracks = [t for t in tracks if t.class_id in self.zone_alert_class_ids]
            t_face0 = time.perf_counter()
            if self.face_recognizer and self.face_recognizer.enabled and face_tracks:
                identities = self.face_recognizer.identify_tracks(
                    frame, face_tracks, stride=self.face_recognizer.face_stride
                )
                for t in tracks:
                    identity_scores[t.track_id] = self.face_recognizer.get_match_score(
                        t.track_id
                    )
                    if t.identity and t.identity != UNKNOWN:
                        if self.metrics:
                            self.metrics.record_identity(t.identity)
            face_ms = (time.perf_counter() - t_face0) * 1000.0

            t_zone0 = time.perf_counter()
            events = self.zone_manager.check_tracks(
                zone_tracks,
                identity_scores,
                allowed_class_ids=self.zone_alert_class_ids,
            )
            self.alert_manager.handle_events(events)

            if self.metrics:
                for ev in events:
                    if ev.event_type == ZoneEventType.ENTER:
                        self.metrics.record_zone_enter(
                            ev.zone_id, ev.zone_name, ev.track_id, ev.identity
                        )
                    elif ev.event_type == ZoneEventType.EXIT:
                        self.metrics.record_zone_exit(
                            ev.zone_id, ev.zone_name, ev.track_id, ev.identity
                        )
            zones_ms = (time.perf_counter() - t_zone0) * 1000.0

            infer_ms = detect_ms

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 0:
                self._fps_samples.append(1.0 / dt)
                if len(self._fps_samples) > 30:
                    self._fps_samples.pop(0)
            fps = sum(self._fps_samples) / len(self._fps_samples) if self._fps_samples else 0.0
            total_ms_so_far = (time.perf_counter() - t0) * 1000.0

            if self.preview_mode == "performance":
                draw_base = infer_frame
            else:
                draw_base = frame if need_capture_copy else frame.copy()

            if self._skeleton is not None and face_tracks:
                if self.preview_mode == "performance" and infer_scale.active:
                    sk_boxes = [
                        (
                            t.x1 / infer_scale.scale_x,
                            t.y1 / infer_scale.scale_y,
                            t.x2 / infer_scale.scale_x,
                            t.y2 / infer_scale.scale_y,
                        )
                        for t in face_tracks
                    ]
                    draw_base = self._skeleton.apply(infer_frame, person_boxes=sk_boxes)
                else:
                    boxes = [(t.x1, t.y1, t.x2, t.y2) for t in face_tracks]
                    draw_base = self._skeleton.apply(draw_base, person_boxes=boxes)

            debug_info = self._build_debug(tracks) if self.show_debug_overlay else None
            t_draw0 = time.perf_counter()
            perf_scale = infer_scale if self.preview_mode == "performance" else None
            annotated = self._draw(
                draw_base,
                tracks,
                fps,
                infer_ms,
                total_ms_so_far,
                detect_ms,
                fire_ms,
                track_ms,
                face_ms,
                zones_ms,
                debug_info,
                faded,
                fire_dets,
                coord_scale=perf_scale,
            )
            draw_ms = (time.perf_counter() - t_draw0) * 1000.0

            if self.preview_mode == "performance" and (cap_w, cap_h) != (
                annotated.shape[1],
                annotated.shape[0],
            ):
                annotated = upscale_frame_for_display(annotated, cap_w, cap_h)
            total_ms = (time.perf_counter() - t0) * 1000.0

            if self.metrics:
                self.metrics.record_frame(
                    fps,
                    infer_ms=infer_ms,
                    total_ms=total_ms,
                    detect_ms=detect_ms,
                    fire_ms=fire_ms,
                    track_ms=track_ms,
                    face_ms=face_ms,
                    zones_ms=zones_ms,
                    preprocess_ms=preprocess_ms,
                    draw_ms=draw_ms,
                )

            result = FrameResult(
                frame=annotated,
                tracks=tracks,
                zone_events=events,
                fps=fps,
                infer_ms=infer_ms,
                detect_ms=detect_ms,
                fire_ms=fire_ms,
                track_ms=track_ms,
                face_ms=face_ms,
                zones_ms=zones_ms,
                total_ms=total_ms,
                preprocess_ms=preprocess_ms,
                draw_ms=draw_ms,
                capture_w=cap_w,
                capture_h=cap_h,
                infer_w=infer_scale.infer_w,
                infer_h=infer_scale.infer_h,
                preview_mode=self.preview_mode,
                in_frame_count=len(tracks),
                unique_session_count=len(self.tracker.seen_ids),
                source_type=getattr(self.capture, "source_type", "webcam"),
                source_label=getattr(self.capture, "source_label", ""),
                identities=identities,
                debug=debug_info,
                faded_detections=faded,
                fire_smoke_detections=fire_dets,
            )
            with self._lock:
                self._latest_result = result

    def _build_debug(self, tracks: list[TrackedPerson]) -> DebugInfo:
        infos: list[DebugTrackInfo] = []
        for track in tracks:
            states: dict[str, bool] = {}
            for zone in self.zone_manager.zones:
                if len(zone.points) >= 3:
                    states[zone.id] = point_in_polygon(
                        track.feet_x, track.feet_y, zone.points
                    )
            score = 0.0
            if self.face_recognizer:
                score = self.face_recognizer.get_match_score(track.track_id)
            infos.append(
                DebugTrackInfo(
                    track.track_id,
                    track.feet_x,
                    track.feet_y,
                    states,
                    track.identity or UNKNOWN,
                    score,
                )
            )
        return DebugInfo(tracks=infos)

    def _to_draw_xy(
        self, x: float, y: float, coord_scale: InferScale | None
    ) -> tuple[int, int]:
        if coord_scale and coord_scale.active:
            return int(x / coord_scale.scale_x), int(y / coord_scale.scale_y)
        return int(x), int(y)

    def _zone_pts_draw(
        self, points: list[tuple[float, float]], coord_scale: InferScale | None
    ) -> np.ndarray:
        if coord_scale and coord_scale.active:
            pts = [
                (int(p[0] / coord_scale.scale_x), int(p[1] / coord_scale.scale_y))
                for p in points
            ]
        else:
            pts = [(int(p[0]), int(p[1])) for p in points]
        return np.array(pts, dtype=np.int32)

    def _draw(
        self,
        frame: np.ndarray,
        tracks: list[TrackedPerson],
        fps: float,
        infer_ms: float,
        total_ms: float,
        detect_ms: float,
        fire_ms: float,
        track_ms: float,
        face_ms: float,
        zones_ms: float,
        debug_info: DebugInfo | None,
        faded_detections: list[Detection],
        fire_smoke_detections: list[Detection],
        coord_scale: InferScale | None = None,
    ) -> np.ndarray:
        overlay_mode = self.ui_mode == "overlay"
        show_zones = self.show_zones_on_frame
        show_boxes = self.show_boxes_on_frame and not overlay_mode
        show_hud = self.show_hud_on_frame and not overlay_mode
        needs_zone_blend = show_zones and any(
            len(z.points) >= 3 for z in self.zone_manager.zones
        )
        out = frame.copy() if needs_zone_blend else frame

        now = time.time()
        pulse_active = now < self._pulse_until
        if show_zones:
            for zone in self.zone_manager.zones:
                if len(zone.points) < 3:
                    continue
                pts = self._zone_pts_draw(zone.points, coord_scale)
                if needs_zone_blend:
                    overlay = out.copy()
                else:
                    overlay = out
                color = zone.color
                cv2.fillPoly(overlay, [pts], color)
                alpha = 0.25 if zone.enabled else 0.1
                cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
                thickness = 4 if pulse_active and zone.id == self._pulse_zone_id else 2
                pulse_color = (
                    (0, 255, 255) if pulse_active and zone.id == self._pulse_zone_id else color
                )
                cv2.polylines(out, [pts], True, pulse_color, thickness)
                if zone.points:
                    cx = int(sum(p[0] for p in zone.points) / len(zone.points))
                    cy = int(sum(p[1] for p in zone.points) / len(zone.points))
                    cv2.putText(
                        out,
                        zone.name,
                        (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

        flash = self.alert_manager.state.is_flashing()
        flash_zone = self.alert_manager.state.flash_zone_id
        if show_boxes:
            for track in tracks:
                x1, y1 = self._to_draw_xy(track.x1, track.y1, coord_scale)
                x2, y2 = self._to_draw_xy(track.x2, track.y2, coord_scale)
                color = _track_color(track.class_id)
                if track.class_id == PERSON_CLASS_ID:
                    ident = track.identity or UNKNOWN
                    label = f"#{track.track_id} {ident} {track.confidence:.2f}"
                else:
                    label = f"#{track.track_id} {track.class_name} {track.confidence:.2f}"
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    out,
                    label,
                    (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                if track.class_id == PERSON_CLASS_ID:
                    fx, fy = self._to_draw_xy(track.feet_x, track.feet_y, coord_scale)
                    cv2.circle(out, (fx, fy), 4, (0, 0, 255), -1)

            for fd in faded_detections:
                x1, y1 = self._to_draw_xy(fd.x1, fd.y1, coord_scale)
                x2, y2 = self._to_draw_xy(fd.x2, fd.y2, coord_scale)
                cname = fd.class_name
                overlay = out.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (128, 128, 128), 1)
                cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
                cv2.putText(
                    out,
                    f"{cname} {fd.confidence:.2f}",
                    (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (160, 160, 160),
                    1,
                )

            for fd in fire_smoke_detections:
                x1, y1 = self._to_draw_xy(fd.x1, fd.y1, coord_scale)
                x2, y2 = self._to_draw_xy(fd.x2, fd.y2, coord_scale)
                color = (0, 0, 255) if "fire" in (fd.display_name or "") else (255, 0, 255)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    out,
                    f"{fd.display_name or 'event'} {fd.confidence:.2f}",
                    (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        if self.show_debug_overlay and debug_info:
            for info in debug_info.tracks:
                fx, fy = self._to_draw_xy(info.feet_x, info.feet_y, coord_scale)
                cv2.circle(out, (fx, fy), 8, (255, 0, 255), 2)
                y_off = 0
                for zone in self.zone_manager.zones:
                    inside = info.zone_states.get(zone.id, False)
                    state = "in" if inside else "out"
                    cv2.putText(
                        out,
                        f"{zone.name}:{state}",
                        (fx + 10, fy + y_off),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 0) if inside else (128, 128, 128),
                        1,
                    )
                    y_off += 14
                if info.match_score > 0:
                    cv2.putText(
                        out,
                        f"{info.identity} ({info.match_score:.2f})",
                        (fx + 10, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                    )

        for zone in self.zone_manager.zones:
            if flash and zone.id == flash_zone and len(zone.points) >= 3:
                pts = self._zone_pts_draw(zone.points, coord_scale)
                cv2.polylines(out, [pts], True, (0, 0, 255), 4)

        if show_hud:
            cv2.putText(
                out,
                f"FPS: {fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                out,
                f"Infer: {infer_ms:.1f}ms | Total: {total_ms:.1f}ms | In frame: {len(tracks)} | "
                f"Unique: {len(self.tracker.seen_ids)}",
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            if self.show_debug_overlay:
                cv2.putText(
                    out,
                    (
                        f"detect {detect_ms:.1f} | fire {fire_ms:.1f} | track {track_ms:.1f} | "
                        f"face {face_ms:.1f} | zones {zones_ms:.1f} ms"
                    ),
                    (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 220, 255),
                    1,
                )
        return out
