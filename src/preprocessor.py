"""
Face image enhancement pipeline.

Uses ALL available imaging libraries in sequence:

  1. scipy.ndimage       — Gaussian pre-filter (noise reduction)
  2. scikit-image        — CLAHE adaptive contrast + bilateral denoising
  3. Pillow (ImageSharp) — Auto-contrast + unsharp sharpening
  4. OpenCV              — Final equalization + face alignment via eye positions

This pipeline runs on BOTH training images (once at DB build time)
and on live face crops (every detected frame) to maximise matching accuracy.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage as sci_ndimage
from skimage import exposure as ski_exposure
from skimage import filters as ski_filters
from skimage import restoration as ski_restore

from src.logger_setup import get_logger

log = get_logger(__name__)

FACE_SIZE = (100, 100)

# Eye cascade for alignment
_EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"
_eye_cascade: Optional[cv2.CascadeClassifier] = None


def _eye_cascade_cc() -> Optional[cv2.CascadeClassifier]:
    global _eye_cascade
    if _eye_cascade is None:
        cc = cv2.CascadeClassifier(_EYE_CASCADE_PATH)
        _eye_cascade = None if cc.empty() else cc
    return _eye_cascade


# ---------------------------------------------------------------------------
# Core enhancement pipeline
# ---------------------------------------------------------------------------

def enhance_face_crop(gray: np.ndarray) -> np.ndarray:
    """
    Full multi-library enhancement pipeline for a grayscale face crop.

    Input : any-size single-channel uint8 image
    Output: 100×100 uint8 enhanced image

    Libraries used:
      scipy     → Gaussian pre-smoothing to remove sensor noise
      scikit-image → CLAHE adaptive contrast equalisation
      Pillow    → Auto-contrast + sharpness boost
      OpenCV    → Final histogram equalisation + resize
    """
    img = gray.astype(np.float32)

    # ── Step 1: scipy — Gaussian pre-filter (AForge noise reduction) ──────
    img = sci_ndimage.gaussian_filter(img, sigma=0.8)

    # ── Step 2: scikit-image — CLAHE adaptive contrast ────────────────────
    #    Better than plain equalizeHist — avoids over-amplifying noise
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    img_norm = img_u8 / 255.0
    clahe = ski_exposure.equalize_adapthist(img_norm, clip_limit=0.03)
    img_u8 = (clahe * 255).astype(np.uint8)

    # ── Step 3: Pillow — auto-contrast + unsharp mask sharpening ──────────
    pil = Image.fromarray(img_u8)
    pil = ImageOps.autocontrast(pil, cutoff=1)                # stretch histogram
    pil = pil.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
    pil = ImageEnhance.Contrast(pil).enhance(1.15)            # slight contrast boost
    img_u8 = np.array(pil)

    # ── Step 4: OpenCV — final equalization + resize ───────────────────────
    img_u8 = cv2.equalizeHist(img_u8)
    return cv2.resize(img_u8, FACE_SIZE)


# ---------------------------------------------------------------------------
# Alignment helper (uses eye positions from Haar cascade)
# ---------------------------------------------------------------------------

def align_face(
    gray_crop: np.ndarray,
) -> np.ndarray:
    """
    Rotate the face crop to align eyes horizontally.
    Falls back to unaligned crop if eye detection fails.
    """
    eye_cc = _eye_cascade_cc()
    if eye_cc is None:
        return gray_crop

    h, w = gray_crop.shape
    roi   = gray_crop[: h // 2, :]   # eyes are in top half

    eyes = eye_cc.detectMultiScale(
        roi, scaleFactor=1.1, minNeighbors=6, minSize=(12, 12)
    )
    if len(eyes) < 2:
        return gray_crop

    # Sort eyes left → right
    eyes = sorted(eyes, key=lambda e: e[0])[:2]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes
    cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
    cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2

    angle = float(np.degrees(np.arctan2(cy2 - cy1, cx2 - cx1)))
    if abs(angle) > 20:   # ignore unreasonable angles
        return gray_crop

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(gray_crop, M, (w, h), flags=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Combined: align + enhance
# ---------------------------------------------------------------------------

def prepare_face(gray_crop: np.ndarray) -> np.ndarray:
    """
    Full pipeline: align → enhance → return 100×100 ready for LBPH/SVM.
    Use this for BOTH training images and live camera crops.
    """
    aligned  = align_face(gray_crop)
    enhanced = enhance_face_crop(aligned)
    return enhanced


# ---------------------------------------------------------------------------
# Training image loader
# ---------------------------------------------------------------------------

def load_and_prepare(
    path: str,
    detector: cv2.CascadeClassifier,
) -> Optional[np.ndarray]:
    """
    Load an image file, detect the largest face, apply full pipeline.
    Returns 100×100 enhanced crop or None if no face found.
    """
    img = cv2.imread(path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        cv2.equalizeHist(gray), scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if not len(faces):
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    crop = gray[y : y + h, x : x + w]
    return prepare_face(crop)
