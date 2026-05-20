"""Local face enrollment and recognition (InsightFace when available)."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import app_data_dir

logger = logging.getLogger(__name__)

UNKNOWN = "Unknown"


@dataclass
class FaceMatch:
    person_id: str
    name: str
    score: float


@dataclass
class GalleryPerson:
    person_id: str
    name: str
    embeddings: list[np.ndarray]


def faces_dir(profile_id: str) -> Path:
    path = app_data_dir() / "faces" / profile_id
    path.mkdir(parents=True, exist_ok=True)
    return path


class FaceRecognizer:
    def __init__(
        self,
        profile_id: str = "default",
        match_threshold: float = 0.45,
        enabled: bool = True,
        face_stride: int = 3,
    ) -> None:
        self.profile_id = profile_id
        self.match_threshold = match_threshold
        self.enabled = enabled
        self.face_stride = max(1, face_stride)
        self._gallery: list[GalleryPerson] = []
        self._app = None
        self._error: str | None = None
        self._frame_counter = 0
        self._identity_votes: dict[int, list[str]] = {}
        self._last_scores: dict[int, float] = {}
        self._last_identity_by_track: dict[int, str] = {}

        if enabled:
            self._init_model()
        self.reload_gallery()

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def is_ready(self) -> bool:
        return self.enabled and self._app is not None and self._error is None

    def _init_model(self) -> None:
        try:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(
                name="buffalo_s",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=(320, 320))
            logger.info("InsightFace loaded for profile %s", self.profile_id)
        except Exception as exc:
            self._error = str(exc)
            logger.warning("InsightFace unavailable: %s", exc)
            try:
                from insightface.app import FaceAnalysis

                self._app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
                self._app.prepare(ctx_id=-1, det_size=(320, 320))
                self._error = None
                logger.info("InsightFace loaded on CPU")
            except Exception as exc2:
                self._error = str(exc2)
                self._app = None

    def reload_gallery(self) -> None:
        self._gallery.clear()
        root = faces_dir(self.profile_id)
        if not root.is_dir():
            return
        for person_dir in root.iterdir():
            if not person_dir.is_dir():
                continue
            meta_path = person_dir / "meta.json"
            if not meta_path.is_file():
                continue
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            name = str(meta.get("name", person_dir.name))
            pid = str(meta.get("person_id", person_dir.name))
            embeddings: list[np.ndarray] = []
            emb_path = person_dir / "embeddings.npy"
            if emb_path.is_file():
                arr = np.load(emb_path)
                if arr.ndim == 2:
                    for row in arr:
                        embeddings.append(row.astype(np.float32))
            else:
                for img_path in person_dir.glob("*.jpg"):
                    emb = self._embed_file(img_path)
                    if emb is not None:
                        embeddings.append(emb)
                for img_path in person_dir.glob("*.png"):
                    emb = self._embed_file(img_path)
                    if emb is not None:
                        embeddings.append(emb)
                if embeddings:
                    np.save(emb_path, np.stack(embeddings))
            if embeddings:
                self._gallery.append(GalleryPerson(pid, name, embeddings))

    def list_people(self) -> list[tuple[str, str]]:
        return [(p.person_id, p.name) for p in self._gallery]

    def enroll_image(self, name: str, image_bgr: np.ndarray) -> str | None:
        if not self.is_ready:
            return None
        emb = self._embed_image(image_bgr)
        if emb is None:
            return None
        pid = uuid.uuid4().hex[:8]
        person_dir = faces_dir(self.profile_id) / pid
        person_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(person_dir / "enroll_0.jpg"), image_bgr)
        np.save(person_dir / "embeddings.npy", emb.reshape(1, -1))
        with (person_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump({"person_id": pid, "name": name}, f, indent=2)
        self.reload_gallery()
        return pid

    def enroll_file(self, name: str, path: Path) -> str | None:
        img = cv2.imread(str(path))
        if img is None:
            return None
        return self.enroll_image(name, img)

    def delete_person(self, person_id: str) -> None:
        person_dir = faces_dir(self.profile_id) / person_id
        if person_dir.is_dir():
            for f in person_dir.iterdir():
                f.unlink()
            person_dir.rmdir()
        self.reload_gallery()

    def rename_person(self, person_id: str, new_name: str) -> None:
        meta_path = faces_dir(self.profile_id) / person_id / "meta.json"
        if not meta_path.is_file():
            return
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        meta["name"] = new_name
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        self.reload_gallery()

    def _embed_file(self, path: Path) -> np.ndarray | None:
        img = cv2.imread(str(path))
        if img is None:
            return None
        return self._embed_image(img)

    def _embed_image(self, image_bgr: np.ndarray) -> np.ndarray | None:
        if self._app is None:
            return None
        faces = self._app.get(image_bgr)
        if not faces:
            return None
        return faces[0].embedding.astype(np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def match_embedding(self, embedding: np.ndarray) -> FaceMatch | None:
        best: FaceMatch | None = None
        for person in self._gallery:
            for ref in person.embeddings:
                score = self._cosine(embedding, ref)
                if score >= self.match_threshold and (best is None or score > best.score):
                    best = FaceMatch(person.person_id, person.name, score)
        return best

    def _crop_face_region(self, frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        h, w = frame.shape[:2]
        bh = y2 - y1
        fy1 = max(0, int(y1))
        fy2 = min(h, int(y1 + bh * 0.55))
        fx1 = max(0, int(x1))
        fx2 = min(w, int(x2))
        if fy2 <= fy1 or fx2 <= fx1:
            return frame[int(y1) : int(y2), int(x1) : int(x2)]
        return frame[fy1:fy2, fx1:fx2]

    def identify_tracks(
        self,
        frame: np.ndarray,
        tracks: list,
        stride: int = 3,
    ) -> dict[int, str]:
        """Return track_id -> display name."""
        identities: dict[int, str] = {}
        if not self.enabled or not self.is_ready or not tracks:
            for t in tracks:
                identities[t.track_id] = UNKNOWN
            return identities

        self._frame_counter += 1
        run_detect = self._frame_counter % max(1, stride) == 0

        for track in tracks:
            tid = track.track_id
            if not run_detect:
                prev = track.identity or self._last_identity_by_track.get(tid, UNKNOWN)
                identities[tid] = prev
                track.identity = prev
                continue

            crop = self._crop_face_region(frame, track.x1, track.y1, track.x2, track.y2)
            if crop.size == 0:
                identities[tid] = UNKNOWN
                track.identity = UNKNOWN
                continue

            emb = self._embed_image(crop)
            if emb is None:
                prev = track.identity if track.identity else self._last_identity_by_track.get(tid, UNKNOWN)
                identities[tid] = prev
                track.identity = prev
                continue

            match = self.match_embedding(emb)
            if match:
                votes = self._identity_votes.setdefault(tid, [])
                votes.append(match.name)
                if len(votes) > 5:
                    votes.pop(0)
                name = max(set(votes), key=votes.count)
                identities[tid] = name
                track.identity = name
                self._last_identity_by_track[tid] = name
                self._last_scores[tid] = match.score
            else:
                identities[tid] = UNKNOWN
                track.identity = UNKNOWN
                self._last_identity_by_track[tid] = UNKNOWN
                self._last_scores[tid] = 0.0

        for track in tracks:
            identities.setdefault(track.track_id, track.identity or UNKNOWN)
        return identities

    def get_match_score(self, track_id: int) -> float:
        return self._last_scores.get(track_id, 0.0)

    def reset_session(self) -> None:
        self._identity_votes.clear()
        self._last_scores.clear()
        self._last_identity_by_track.clear()
        self._frame_counter = 0

    def close(self) -> None:
        self._app = None
