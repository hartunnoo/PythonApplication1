"""
SVM face classifier — Accord.NET / scikit-learn equivalent.

Uses HOG (Histogram of Oriented Gradients) features extracted via scikit-image
and an RBF-kernel SVM from scikit-learn to classify face identities.

This provides much more reliable per-person probability scores than
raw LBPH distance, especially when 2+ persons are enrolled.

Behaviour:
  - < 2 persons in list: SVM disabled, LBPH used alone
  - >= 2 persons       : SVM primary, LBPH as fallback/ensemble vote
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.feature import hog as ski_hog

from src.logger_setup import get_logger

log = get_logger(__name__)

HOG_IMAGE_SIZE = (64, 64)
HOG_ORIENTATIONS   = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_hog(gray_img: np.ndarray) -> np.ndarray:
    """
    Extract HOG feature vector from a grayscale face image.
    Uses scikit-image (AForge.NET / Accord.NET equivalent).

    Returns a 1D float64 numpy array suitable for SVM input.
    """
    resized = cv2.resize(gray_img, HOG_IMAGE_SIZE)
    features = ski_hog(
        resized,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features.astype(np.float64)


# ---------------------------------------------------------------------------
# SVM Classifier
# ---------------------------------------------------------------------------

class SVMFaceClassifier:
    """
    scikit-learn SVM classifier over HOG features.

    Pipeline:
      HOG features → StandardScaler → RBF-SVM → predict_proba()

    Requires at least 2 distinct persons to train.
    Falls back gracefully when fewer persons are enrolled.
    """

    def __init__(self) -> None:
        self._pipeline: Optional[Pipeline] = None
        self._label_to_name: Dict[int, str] = {}
        self.trained: bool = False
        self.person_count: int = 0

    # ------------------------------------------------------------------

    def train(
        self,
        face_crops: List[np.ndarray],
        labels: List[int],
        label_to_name: Dict[int, str],
    ) -> None:
        """
        Train on a list of 100×100 grayscale face crops.
        Skips training if fewer than 2 distinct persons.
        """
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            log.info(
                "SVM training skipped: only %d person(s) enrolled "
                "(need >= 2). LBPH will be used alone.",
                len(unique_labels),
            )
            self.trained = False
            return

        log.info(
            "Training SVM: %d samples, %d person(s)...",
            len(face_crops), len(unique_labels),
        )
        features = [extract_hog(crop) for crop in face_crops]
        X = np.array(features)
        y = np.array(labels)

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
            )),
        ])
        self._pipeline.fit(X, y)
        self._label_to_name = label_to_name
        self.trained = True
        self.person_count = len(unique_labels)
        log.info("SVM trained: %d persons.", self.person_count)

    # ------------------------------------------------------------------

    def predict(
        self,
        gray_crop: np.ndarray,
    ) -> Tuple[Optional[str], float]:
        """
        Predict identity.
        Returns (name, confidence) or (None, 0.0) if not trained.
        confidence is the SVM's probability for the winning class (0.0–1.0).
        """
        if not self.trained or self._pipeline is None:
            return None, 0.0

        features = extract_hog(gray_crop).reshape(1, -1)
        label      = int(self._pipeline.predict(features)[0])
        proba      = self._pipeline.predict_proba(features)[0]
        confidence = float(np.max(proba))
        name       = self._label_to_name.get(label, "Unknown")
        return name, confidence

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            with open(path, "wb") as fh:
                pickle.dump({
                    "pipeline":      self._pipeline,
                    "label_to_name": self._label_to_name,
                    "trained":       self.trained,
                    "person_count":  self.person_count,
                }, fh)
        except Exception as exc:
            log.warning("SVM save failed: %s", exc)

    def load(self, path: str) -> bool:
        """Load from disk. Returns True on success."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            self._pipeline      = data["pipeline"]
            self._label_to_name = data["label_to_name"]
            self.trained        = data["trained"]
            self.person_count   = data["person_count"]
            if self.trained:
                log.info("SVM loaded from cache (%d persons).", self.person_count)
            return True
        except Exception as exc:
            log.warning("SVM load failed: %s", exc)
            return False
