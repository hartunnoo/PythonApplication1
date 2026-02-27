"""
Dual-list face matcher — deepface ArcFace embeddings + cosine distance.

Checks a detected face crop against BOTH whitelist and blacklist.
Blacklist takes priority — if a face matches both, it is flagged as blacklist.

Matching strategy:
  1. Extract a 512-dim ArcFace embedding from the BGR face crop.
  2. Compute cosine distance to every stored embedding in each list.
  3. Take the minimum distance per list.
  4. If min_distance <= threshold → match (unknown otherwise).
  5. Tiebreak: whitelist wins only if its distance is clearly lower
     (by _BLACKLIST_MARGIN_COSINE) than the blacklist distance.

Cosine distance interpretation:
  0.0       — identical vectors (same person, perfect photo)
  0.10–0.35 — same person, typical live variation
  0.40      — default threshold (configurable in settings.yaml)
  0.45+     — different people

MatchResult.list_type values:
  "whitelist" — known allowed person   → green box
  "blacklist" — known flagged person   → red box
  "unknown"   — not in either list     → grey box
  "idle"      — no face in frame       → blue state (set externally)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from src.face_db import FaceDatabase, ListModel, LIST_WHITELIST, LIST_BLACKLIST
from src.logger_setup import get_logger

log = get_logger(__name__)

LIST_UNKNOWN = "unknown"
LIST_IDLE    = "idle"

# How much lower (cosine distance) the whitelist match must be to override
# blacklist priority.  0.05 cosine units is a meaningful margin at this scale.
_BLACKLIST_MARGIN_COSINE = 0.05


@dataclass
class MatchResult:
    name: str
    confidence: float           # 0.0–1.0
    is_match: bool
    list_type: str              # "whitelist" | "blacklist" | "unknown" | "idle"
    raw_distance: float         # cosine distance (0.0 = identical)
    # Facial attributes — populated by camera_worker after recognition
    age:      Optional[int]  = None
    emotion:  Optional[str]  = None
    is_child: Optional[bool] = None

    def __str__(self) -> str:
        return (
            f"[{self.list_type.upper()}] {self.name} "
            f"(conf={self.confidence:.1%}, dist={self.raw_distance:.4f})"
        )


UNKNOWN_RESULT = MatchResult(
    name="UNKNOWN VISITOR",
    confidence=0.0,
    is_match=False,
    list_type=LIST_UNKNOWN,
    raw_distance=2.0,   # maximum cosine distance
)


class FaceMatcher:
    """
    Matches one BGR face crop against whitelist and blacklist using ArcFace
    embeddings and cosine distance.
    """

    def __init__(self, threshold: float = 0.40, model_name: str = "ArcFace") -> None:
        self.threshold  = threshold
        self.model_name = model_name
        # Pre-load ArcFace model at startup to avoid lag on first detection
        try:
            from src.arcface_onnx import get_arcface
            get_arcface()
            log.info(
                "FaceMatcher ready (threshold=%.2f, cosine distance, ONNX ArcFace).",
                threshold,
            )
        except Exception as exc:
            log.error("ArcFace model init failed: %s", exc)

    # ------------------------------------------------------------------

    def _extract_embedding(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 512-dim ArcFace embedding from a pre-cropped BGR face.
        Returns None on failure (crop too small, corrupt image, etc.).
        Uses ONNX Runtime — no tensorflow required.
        """
        from src.arcface_onnx import get_arcface
        return get_arcface().get_embedding(crop_bgr)

    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance: 0.0 = identical vectors, 2.0 = opposite."""
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 2.0
        return float(1.0 - float(np.dot(a, b)) / (norm_a * norm_b))

    # ------------------------------------------------------------------

    def _find_best_match(
        self,
        query_embedding: np.ndarray,
        model: ListModel,
    ) -> Optional[MatchResult]:
        """
        Find the closest PersonEmbedding in the list.
        Returns MatchResult if distance <= threshold, else None.
        confidence = 1.0 at distance=0, 0.0 at distance=threshold.
        """
        if not model.trained or not model.embeddings:
            return None

        best_dist = float("inf")
        best_name = ""

        for pe in model.embeddings:
            dist = self._cosine_distance(query_embedding, pe.embedding)
            if dist < best_dist:
                best_dist = dist
                best_name = pe.name

        if best_dist > self.threshold:
            return None   # Not in this list

        confidence = max(0.0, 1.0 - (best_dist / self.threshold))

        return MatchResult(
            name=best_name,
            confidence=confidence,
            is_match=True,
            list_type=model.list_type,
            raw_distance=best_dist,
        )

    # ------------------------------------------------------------------

    def match(self, crop_bgr: np.ndarray, db: FaceDatabase) -> MatchResult:
        """
        Match one BGR face crop against both lists.

        Blacklist wins UNLESS whitelist has a clearly lower cosine distance
        (by more than _BLACKLIST_MARGIN_COSINE).
        """
        embedding = self._extract_embedding(crop_bgr)
        if embedding is None:
            return UNKNOWN_RESULT

        bl_result = self._find_best_match(embedding, db.blacklist)
        wl_result = self._find_best_match(embedding, db.whitelist)

        if bl_result is None and wl_result is None:
            return UNKNOWN_RESULT
        if bl_result is None:
            return wl_result
        if wl_result is None:
            return bl_result

        # Both matched — whitelist wins only if clearly the better distance
        if wl_result.raw_distance < bl_result.raw_distance - _BLACKLIST_MARGIN_COSINE:
            return wl_result
        return bl_result

    def match_batch(
        self, crops: List[np.ndarray], db: FaceDatabase
    ) -> List[MatchResult]:
        return [self.match(c, db) for c in crops]
