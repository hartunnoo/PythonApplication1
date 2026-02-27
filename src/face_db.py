"""
Dual-list face database (whitelist + blacklist) — deepface / ArcFace backend.

Folder layout:
  known_faces/
    whitelist/  → green box  (employees, VIPs, allowed visitors)
    blacklist/  → red box    (banned, flagged persons)

Each list extracts ArcFace embeddings (512-dim float32 vectors) for every
training image and stores them in a pickle cache.  At match time, cosine
distance between the live-face embedding and every stored embedding is
computed; the closest match wins if it falls within the threshold.

Cache invalidation uses the same mtime + exact-filename check as before,
so adding, removing, or replacing a photo automatically triggers a rebuild.
"""

from __future__ import annotations

import os
import pickle
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.logger_setup import get_logger

log = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LIST_WHITELIST = "whitelist"
LIST_BLACKLIST = "blacklist"

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PersonEmbedding:
    name: str
    embedding: np.ndarray   # 512-dim float32 for ArcFace


@dataclass
class ListModel:
    list_type:    str
    embeddings:   List[PersonEmbedding] = field(default_factory=list)
    trained:      bool       = False
    person_count: int        = 0
    sample_count: int        = 0
    names:        List[str]  = field(default_factory=list)   # unique person names


@dataclass
class FaceDatabase:
    whitelist: ListModel
    blacklist: ListModel
    detector:  cv2.CascadeClassifier   # Haar cascade — still used for live detection

    @property
    def any_trained(self) -> bool:
        return self.whitelist.trained or self.blacklist.trained

    @property
    def total_persons(self) -> int:
        return self.whitelist.person_count + self.blacklist.person_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )


def _name_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.replace("_", " ").replace("-", " ").strip()


def _base_name(display_name: str) -> str:
    parts = display_name.rsplit(" ", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else display_name


def _dir_mtime(directory: str) -> float:
    """
    Return the most recent modification time across the directory entry itself
    AND every image file inside it.  Any change (add/remove/edit) invalidates
    the cache.
    """
    try:
        dir_own = os.path.getmtime(directory)
        file_mtimes = [
            os.path.getmtime(os.path.join(directory, f))
            for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ]
        return max([dir_own] + file_mtimes) if file_mtimes else dir_own
    except OSError:
        return 0.0


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _embedding_cache_paths(cache_dir: str, lt: str) -> Tuple[str, str]:
    return (
        os.path.join(cache_dir, f"arcface_onnx_{lt}_embeddings.pkl"),
        os.path.join(cache_dir, f"arcface_onnx_{lt}_meta.txt"),
    )


def _save_embedding_cache(
    cache_dir: str,
    lt: str,
    embeddings: List[PersonEmbedding],
    mtime: float,
    image_paths: List[str],
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    pkl_path, meta_path = _embedding_cache_paths(cache_dir, lt)
    try:
        with open(pkl_path, "wb") as fh:
            pickle.dump(embeddings, fh)
        filenames = ",".join(sorted(os.path.basename(p) for p in image_paths))
        with open(meta_path, "w", encoding="utf-8") as fh:
            fh.write(f"mtime={mtime}\n")
            fh.write(f"files={filenames}\n")
        log.info("[%s] Embedding cache saved (%d samples).", lt, len(embeddings))
    except Exception as exc:
        log.warning("[%s] Embedding cache save failed: %s", lt, exc)


def _load_embedding_cache(
    cache_dir: str,
    lt: str,
    mtime: float,
    image_paths: List[str],
) -> Optional[List[PersonEmbedding]]:
    pkl_path, meta_path = _embedding_cache_paths(cache_dir, lt)
    if not os.path.exists(pkl_path) or not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()

        if float(lines[0].split("=", 1)[1]) < mtime:
            log.info("[%s] Embedding cache stale (mtime) — rebuilding.", lt)
            return None

        if len(lines) > 1 and lines[1].startswith("files="):
            cached_files  = set(lines[1].split("=", 1)[1].split(","))
            current_files = set(os.path.basename(p) for p in image_paths)
            if cached_files != current_files:
                log.info("[%s] Embedding cache stale (files changed) — rebuilding.", lt)
                return None

        with open(pkl_path, "rb") as fh:
            embeddings: List[PersonEmbedding] = pickle.load(fh)
        log.info("[%s] Embeddings loaded from cache (%d samples).", lt, len(embeddings))
        return embeddings
    except Exception as exc:
        log.warning("[%s] Embedding cache load failed (%s). Rebuilding.", lt, exc)
        return None


# ---------------------------------------------------------------------------
# List model loader
# ---------------------------------------------------------------------------

def _load_list_model(
    directory: str,
    list_type: str,
    cache_dir: str,
    detector: cv2.CascadeClassifier,
    deepface_model: str = "ArcFace",
) -> ListModel:
    os.makedirs(directory, exist_ok=True)
    image_paths = _image_files(directory)

    if not image_paths:
        log.info("[%s] No images in '%s'.", list_type, directory)
        return ListModel(list_type=list_type)

    source_mtime = _dir_mtime(directory)

    # Try cache first
    cached = _load_embedding_cache(cache_dir, list_type, source_mtime, image_paths)
    if cached is not None:
        unique_names = sorted({pe.name for pe in cached})
        return ListModel(
            list_type=list_type,
            embeddings=cached,
            trained=True,
            person_count=len(unique_names),
            sample_count=len(cached),
            names=unique_names,
        )

    # Build embeddings from training images using ONNX ArcFace
    from src.arcface_onnx import get_arcface
    arcface = get_arcface()

    log.info(
        "[%s] Extracting ArcFace embeddings from %d image(s)...",
        list_type, len(image_paths),
    )
    embeddings: List[PersonEmbedding] = []
    skipped = 0

    for path in image_paths:
        display = _name_from_path(path)
        person  = _base_name(display)

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            log.warning("[%s] Could not read '%s' — skipped.", list_type, os.path.basename(path))
            skipped += 1
            continue

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Stage 1: Haar cascade (strict) — most portrait photos
        haar_faces = detector.detectMultiScale(
            cv2.equalizeHist(gray), scaleFactor=1.1, minNeighbors=5, minSize=(40, 40),
        )

        # Stage 2: Haar relaxed — for harder shots (angle, distance, lighting)
        if not len(haar_faces):
            haar_faces = detector.detectMultiScale(
                cv2.equalizeHist(gray), scaleFactor=1.05, minNeighbors=3, minSize=(30, 30),
            )
            if len(haar_faces):
                log.info("[%s] '%s' needed relaxed Haar detection.", list_type, os.path.basename(path))

        if len(haar_faces):
            x, y, w, h = max(haar_faces, key=lambda f: f[2] * f[3])
            face_bgr = img_bgr[y:y + h, x:x + w]
        else:
            # Stage 3: No face detected — use the full image (portrait assumption)
            log.warning(
                "[%s] Haar missed '%s' — using full image as face crop.",
                list_type, os.path.basename(path),
            )
            face_bgr = img_bgr

        vec = arcface.get_embedding(face_bgr)
        if vec is not None:
            embeddings.append(PersonEmbedding(name=person, embedding=vec))
        else:
            log.warning(
                "[%s] Embedding failed for '%s' — skipped.",
                list_type, os.path.basename(path),
            )
            skipped += 1

    if not embeddings:
        log.error("[%s] No usable images — list disabled.", list_type)
        return ListModel(list_type=list_type)

    unique_names = sorted({pe.name for pe in embeddings})
    _save_embedding_cache(cache_dir, list_type, embeddings, source_mtime, image_paths)

    log.info(
        "[%s] Done: %d person(s), %d embeddings, %d skipped.",
        list_type, len(unique_names), len(embeddings), skipped,
    )
    return ListModel(
        list_type=list_type,
        embeddings=embeddings,
        trained=True,
        person_count=len(unique_names),
        sample_count=len(embeddings),
        names=unique_names,
    )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_face_database(
    whitelist_dir: str,
    blacklist_dir: str,
    cache_dir: str,
    deepface_model: str = "ArcFace",
) -> FaceDatabase:
    detector = cv2.CascadeClassifier(_CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"Haar cascade not found at '{_CASCADE_PATH}'.")

    whitelist = _load_list_model(whitelist_dir, LIST_WHITELIST, cache_dir, detector, deepface_model)
    blacklist = _load_list_model(blacklist_dir, LIST_BLACKLIST, cache_dir, detector, deepface_model)

    log.info(
        "Database ready — whitelist: %d person(s) | blacklist: %d person(s)",
        whitelist.person_count, blacklist.person_count,
    )
    return FaceDatabase(whitelist=whitelist, blacklist=blacklist, detector=detector)


# ---------------------------------------------------------------------------
# Thread-safe manager
# ---------------------------------------------------------------------------

class FaceDatabaseManager:
    def __init__(self, db: FaceDatabase) -> None:
        self._db   = db
        self._lock = threading.RLock()

    def get(self) -> FaceDatabase:
        with self._lock:
            return self._db

    def reload(self, whitelist_dir: str, blacklist_dir: str, cache_dir: str) -> None:
        log.info("Reloading face database...")
        new_db = load_face_database(whitelist_dir, blacklist_dir, cache_dir)
        with self._lock:
            self._db = new_db
        log.info("Reload done — %d total persons.", new_db.total_persons)
