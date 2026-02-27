"""
ArcFace face embedding extraction using ONNX Runtime.

Uses InsightFace's buffalo_l recognition model (w600k_r50.onnx, ResNet-50).
Model is downloaded automatically (~356 MB zip, extracted ~166 MB ONNX) on
first use and cached to ~/.arcface_onnx/.

Model spec:
  Input  : (1, 3, 112, 112) float32 — (pixel - 127.5) / 128.0  (BGR order)
  Output : (1, 512) float32 — L2-normalised ArcFace embedding
"""

from __future__ import annotations

import threading
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request

import cv2
import numpy as np

from src.logger_setup import get_logger

log = get_logger(__name__)

_ARCFACE_DIR    = Path.home() / ".arcface_onnx"
_MODEL_FILENAME = "w600k_r50.onnx"
_MODEL_PATH     = _ARCFACE_DIR / _MODEL_FILENAME
_INPUT_SIZE     = (112, 112)

# InsightFace buffalo_l GitHub release (~356 MB zip, contains w600k_r50.onnx)
_BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)


class ArcFaceONNX:
    """
    Wraps InsightFace's w600k_r50.onnx ArcFace model via ONNX Runtime.
    Downloaded once on first use; subsequent runs load from cache.
    """

    def __init__(self) -> None:
        import onnxruntime as ort

        if not _MODEL_PATH.exists():
            self._download_model()

        opts = ort.SessionOptions()
        opts.log_severity_level = 3   # suppress verbose ORT logs
        available = ort.get_available_providers()
        providers  = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers  = [p for p in providers if p in available] or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(
            str(_MODEL_PATH),
            sess_options=opts,
            providers=providers,
        )
        self._input_name = self._session.get_inputs()[0].name
        log.info("ArcFaceONNX ready — model: %s", _MODEL_FILENAME)

    # ------------------------------------------------------------------

    def get_embedding(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 512-dim ArcFace embedding from a BGR face crop.
        Returns None if the crop is too small or inference fails.

        Preprocessing matches InsightFace's get_feat():
          resize to 112×112 → (pixel - 127.5) / 128 → NCHW (BGR)
        """
        if face_bgr.shape[0] < 40 or face_bgr.shape[1] < 40:
            return None
        try:
            img = cv2.resize(face_bgr, _INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32)
            img = (img - 127.5) / 128.0          # normalise to ~[-1, 1]
            img = np.transpose(img, (2, 0, 1))   # HWC → CHW
            img = np.expand_dims(img, axis=0)    # CHW → NCHW (1, 3, 112, 112)

            outputs = self._session.run(None, {self._input_name: img})
            emb     = outputs[0][0].astype(np.float32)

            # Ensure L2-normalised
            norm = float(np.linalg.norm(emb))
            return emb / norm if norm > 0.0 else emb

        except Exception as exc:
            log.debug("ArcFace embedding failed: %s", exc)
            return None

    # ------------------------------------------------------------------

    @staticmethod
    def _download_model() -> None:
        """Download buffalo_l.zip and extract w600k_r50.onnx."""
        _ARCFACE_DIR.mkdir(parents=True, exist_ok=True)
        log.info(
            "Downloading ArcFace ONNX model — first-run only (~356 MB). Please wait..."
        )
        try:
            req  = Request(_BUFFALO_L_URL, headers={"User-Agent": "Mozilla/5.0"})
            data = bytearray()
            with urlopen(req, timeout=600) as resp:
                total  = int(resp.headers.get("Content-Length", 0))
                chunk  = 1 << 20   # 1 MB chunks
                last_pct = -10
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    data.extend(buf)
                    if total:
                        pct = len(data) * 100 // total
                        if pct >= last_pct + 10:
                            log.info("  ArcFace model download: %d%%", pct)
                            last_pct = pct

            log.info("Extracting %s from zip...", _MODEL_FILENAME)
            with zipfile.ZipFile(BytesIO(bytes(data))) as zf:
                for name in zf.namelist():
                    if name.endswith(_MODEL_FILENAME):
                        with zf.open(name) as src, open(_MODEL_PATH, "wb") as dst:
                            dst.write(src.read())
                        log.info("ArcFace model saved → %s", _MODEL_PATH)
                        return
            raise FileNotFoundError(f"{_MODEL_FILENAME} not found in downloaded zip.")

        except Exception as exc:
            raise RuntimeError(
                f"ArcFace model download failed: {exc}\n"
                f"Manual fix: download {_BUFFALO_L_URL}\n"
                f"            extract '{_MODEL_FILENAME}' to {_ARCFACE_DIR}"
            ) from exc


# ── Module-level singleton (lazy, thread-safe) ────────────────────────────────

_instance: Optional[ArcFaceONNX] = None
_instance_lock = threading.Lock()


def get_arcface() -> ArcFaceONNX:
    """Return the module-level ArcFace singleton (created and cached on first call)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ArcFaceONNX()
    return _instance
