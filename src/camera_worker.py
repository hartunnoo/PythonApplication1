"""
Per-camera worker thread — tri-thread, high-performance.

Thread A: capture loop  — reads frames, stores latest rendered frame
Thread B: detect loop   — runs Haar + ArcFace matching, writes detection results
Thread C: attr loop     — runs DeepFace.analyze (age/emotion) in background;
                          results are cached per person, applied by Thread B

Decoupling keeps detection fast even when attribute analysis (age/emotion) is slow.

Auto-screenshot: when a person is confirmed (consecutive-frame gate),
a labelled JPEG is saved to screenshots/ for auditing and dashboard display.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.alerter import Alerter
from src.config import CameraConfig, DetectionConfig, DisplayConfig, MatchingConfig
from src.display import FrameRenderer
from src.face_db import FaceDatabaseManager, LIST_WHITELIST, LIST_BLACKLIST
from src.logger_setup import get_logger
from src.matcher import FaceMatcher, MatchResult, LIST_UNKNOWN, UNKNOWN_RESULT

log = get_logger(__name__)

_MAX_FAILS = 30

# ── Capture-mode constants ─────────────────────────────────────────────────────
_STEADY_FRAMES = 18   # consecutive still detections before auto-capture
_STEADY_PIXELS = 22   # max face-centre drift (px) counted as "still"


def _detect_faces(
    frame_bgr: np.ndarray,
    detector: cv2.CascadeClassifier,
) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
    """
    Detect faces in a BGR frame using Haar cascade.

    Detection uses equalized grayscale for robustness.
    Crops returned are BGR (colour) for deepface ArcFace embedding extraction.
    deepface handles all internal preprocessing.
    """
    gray_raw = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq  = cv2.equalizeHist(gray_raw)    # equalized — only for detection
    raw = detector.detectMultiScale(
        gray_eq, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    locs, crops = [], []
    if len(raw):
        for x, y, w, h in raw:
            x, y = max(0, x), max(0, y)
            x2 = min(frame_bgr.shape[1], x + w)
            y2 = min(frame_bgr.shape[0], y + h)
            locs.append((y, x2, y2, x))
            crop = frame_bgr[y:y2, x:x2]    # BGR crop — deepface handles preprocessing
            crops.append(crop)
    return locs, crops


# ── Attribute cache entry ─────────────────────────────────────────────────────
# Keyed by person name; stores the most recent age/emotion/is_child result.
_AttrEntry = Tuple[Optional[int], Optional[str], Optional[bool]]


class CameraWorker(threading.Thread):
    def __init__(
        self,
        cam_cfg: CameraConfig,
        det_cfg: DetectionConfig,
        mat_cfg: MatchingConfig,
        db_manager: FaceDatabaseManager,
        matcher: FaceMatcher,
        alerter: Alerter,
        display_cfg: DisplayConfig,
        screenshots_dir: str = "screenshots",
        unknown_faces_dir: str = "known_faces/unknown",
    ) -> None:
        super().__init__(daemon=True, name=f"cam-{cam_cfg.label}")
        self._cam_cfg           = cam_cfg
        self._det_cfg           = det_cfg
        self._mat_cfg           = mat_cfg
        self._db_manager        = db_manager
        self._matcher           = matcher
        self._alerter           = alerter
        self._renderer          = FrameRenderer(display_cfg)
        self._screenshots_dir   = screenshots_dir
        self._unknown_faces_dir = unknown_faces_dir

        self._running = False
        self.paused   = False

        self._out_frame: Optional[np.ndarray] = None
        self._out_lock  = threading.Lock()

        self._raw_frame: Optional[np.ndarray] = None
        self._raw_lock  = threading.Lock()
        self._raw_event = threading.Event()

        self._det_locs:    List = []
        self._det_results: List[MatchResult] = []
        self._det_lock = threading.Lock()

        # Consecutive-frame counter keyed by (name, list_type) — NOT name alone.
        # This ensures a blacklist hit and a whitelist hit for the same name
        # (or two people swapping positions) are tracked independently.
        self._consec: Dict[Tuple[str, str], int] = {}

        # Screenshot cooldown: tracks last screenshot time per (name, list_type).
        # Prevents duplicate screenshots when the _consec counter resets mid-session.
        # Synced to the alerter cooldown so every alert fires with a fresh screenshot.
        self._screenshot_cooldown: Dict[Tuple[str, str], float] = {}

        # Thread C: attribute analysis (age/emotion) — runs in background
        # Queue items: (crop_bgr, person_name)
        # maxsize=1 means at most one pending job; new jobs drop old ones if busy.
        self._attr_queue: queue.Queue = queue.Queue(maxsize=1)
        # Cache: name → (age, emotion, is_child)
        self._attr_cache: Dict[str, _AttrEntry] = {}
        self._attr_lock = threading.Lock()

        self._error: Optional[str] = None

        # ── Capture mode (C key) ───────────────────────────────────────────
        self._capture_mode     = False
        self._capture_lock     = threading.Lock()
        self._capture_steady   = 0           # consecutive still-frames counter
        self._capture_last_box: Optional[Tuple[int,int,int,int]] = None
        self._capture_status   = ""          # message shown on guide overlay

        # Eye cascade — used in capture mode to reject side-facing heads
        _eye_xml = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        self._eye_cascade = cv2.CascadeClassifier(_eye_xml)

        # Latest raw (un-rendered) camera frame — used by IC scanner so OCR
        # never sees the left-panel UI overlay or face detection boxes.
        self._ic_frame: Optional[np.ndarray] = None
        self._ic_lock  = threading.Lock()

    @property
    def label(self) -> str:
        return self._cam_cfg.label

    @property
    def latest_raw_frame(self) -> Optional[np.ndarray]:
        """Return the latest unrendered camera frame (no UI overlays)."""
        with self._ic_lock:
            return self._ic_frame.copy() if self._ic_frame is not None else None

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        with self._out_lock:
            return self._out_frame.copy() if self._out_frame is not None else None

    def stop(self) -> None:
        self._running = False
        self._raw_event.set()
        # Unblock attr thread if waiting
        try:
            self._attr_queue.put_nowait(None)
        except queue.Full:
            pass

    # ── Thread A: capture ──────────────────────────────────────────────

    def run(self) -> None:
        cfg = self._cam_cfg
        log.info("[%s] Starting (device=%d)", cfg.label, cfg.device_index)

        # IP cameras (RTSP/HTTP) use the URL directly; local cameras use CAP_DSHOW
        if cfg.rtsp_url:
            log.info("[%s] Connecting to IP camera: %s", cfg.label, cfg.rtsp_url)
            cam = cv2.VideoCapture(cfg.rtsp_url)
        else:
            cam = cv2.VideoCapture(cfg.device_index, cv2.CAP_DSHOW)
        if not cam.isOpened():
            self._error = f"Cannot open [{cfg.label}] device {cfg.device_index}"
            log.error(self._error)
            self._set_error_frame()
            return

        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cam.set(cv2.CAP_PROP_FPS,          cfg.fps)
        cam.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        log.info("[%s] Opened at %dx%d",
                 cfg.label,
                 int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self._running = True

        dt = threading.Thread(target=self._detect_loop, daemon=True,
                              name=f"detect-{cfg.label}")
        dt.start()

        at = threading.Thread(target=self._attr_loop, daemon=True,
                              name=f"attr-{cfg.label}")
        at.start()

        fails = 0
        while self._running:
            if self.paused:
                time.sleep(0.05)
                continue

            ret, frame = cam.read()
            if not ret or frame is None:
                fails += 1
                if fails >= _MAX_FAILS:
                    log.error("[%s] Camera stopped responding.", cfg.label)
                    self._set_error_frame()
                    break
                continue
            fails = 0

            # Always keep latest raw frame for IC scanner (no UI overlay)
            with self._ic_lock:
                self._ic_frame = frame.copy()

            # Hand to detect thread (non-blocking drop if busy)
            if self._raw_lock.acquire(blocking=False):
                self._raw_frame = frame.copy()
                self._raw_lock.release()
                self._raw_event.set()

            with self._det_lock:
                locs    = self._det_locs
                results = self._det_results

            # Mild unsharp mask — reduces webcam softness without artifacts
            blur = cv2.GaussianBlur(frame, (0, 0), 2.0)
            frame = cv2.addWeighted(frame, 1.3, blur, -0.3, 0)

            rendered = self._renderer.render(frame, locs, results)
            self._draw_label(rendered, cfg.label)
            if self._capture_mode:
                self._draw_capture_guide(rendered)

            with self._out_lock:
                self._out_frame = rendered

        cam.release()
        self._running = False
        self._raw_event.set()
        dt.join(timeout=2)
        at.join(timeout=2)
        log.info("[%s] Stopped.", cfg.label)

    # ── Thread B: detection ────────────────────────────────────────────

    def _detect_loop(self) -> None:
        cfg = self._cam_cfg
        while self._running:
            self._raw_event.wait(timeout=1.0)
            self._raw_event.clear()
            if not self._running:
                break

            with self._raw_lock:
                frame = self._raw_frame
                self._raw_frame = None
            if frame is None:
                continue

            db    = self._db_manager.get()
            locs, crops = _detect_faces(frame, db.detector)
            results = self._matcher.match_batch(crops, db)

            # Apply cached attributes and submit new analysis jobs (non-blocking)
            for result, crop in zip(results, crops):
                # Apply last known age/emotion from cache
                with self._attr_lock:
                    cached = self._attr_cache.get(result.name)
                if cached is not None:
                    result.age, result.emotion, result.is_child = cached

                # Submit this crop for (re)analysis — drop if Thread C is busy
                try:
                    self._attr_queue.put_nowait((crop.copy(), result.name))
                except queue.Full:
                    pass  # Thread C still working; cached value is good enough

            with self._det_lock:
                self._det_locs    = locs
                self._det_results = results

            # Reset counters for (name, list_type) pairs no longer in frame
            seen_keys = {
                (r.name, r.list_type)
                for r in results
                if r.is_match and r.list_type in (LIST_WHITELIST, LIST_BLACKLIST)
            }
            for key in list(self._consec):
                if key not in seen_keys:
                    self._consec.pop(key, None)

            # Capture-mode: track steadiness and auto-fire when ready
            if self._capture_mode:
                self._process_capture_steady(locs, crops, frame)

            # Log unknown visitors (confirmed, throttled)
            for result, crop in zip(results, crops):
                if result.list_type == LIST_UNKNOWN:
                    key = ("UNKNOWN", cfg.label)
                    count = self._consec.get(key, 0) + 1  # type: ignore[arg-type]
                    self._consec[key] = count  # type: ignore[index]
                    if count == self._mat_cfg.confirm_frames:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        self._save_unknown_face(crop, ts)
                        unk_path = self._save_screenshot(frame, result)
                        self._alerter.log_unknown_visitor(
                            camera_label=cfg.label,
                            screenshot_path=unk_path,
                        )
                else:
                    self._consec.pop(("UNKNOWN", cfg.label), None)  # type: ignore[arg-type]

            for result in results:
                # Only track whitelist and blacklist — unknown faces are display-only
                if not result.is_match or result.list_type not in (LIST_WHITELIST, LIST_BLACKLIST):
                    continue
                if result.confidence < self._mat_cfg.min_confidence:
                    self._consec.pop((result.name, result.list_type), None)
                    continue

                key   = (result.name, result.list_type)
                count = self._consec.get(key, 0) + 1
                self._consec[key] = count

                if count >= self._mat_cfg.confirm_frames:
                    # Save screenshot when the alert cooldown allows it so every
                    # alert fires with a fresh photo (no more orphan screenshots
                    # or alerts missing their image).
                    now = time.monotonic()
                    sc_key = (result.name, result.list_type)
                    sc_elapsed = now - self._screenshot_cooldown.get(sc_key, 0.0)
                    screenshot_path: Optional[str] = None
                    if sc_elapsed >= self._alerter._cfg.cooldown_seconds:
                        screenshot_path = self._save_screenshot(frame, result)
                        if screenshot_path:
                            self._screenshot_cooldown[sc_key] = now

                    self._alerter.trigger(
                        result,
                        camera_label=cfg.label,
                        screenshot_path=screenshot_path,
                    )

    # ── Thread C: attribute analysis (age / emotion) ───────────────────

    def _attr_loop(self) -> None:
        """
        Background thread: pops face crops from the queue and runs
        DeepFace.analyze() for age + emotion. Results are stored in
        _attr_cache and applied to the next detection result for that person.

        Heavy models run here so Thread B (detection) is never blocked.
        """
        try:
            from deepface import DeepFace
        except ImportError:
            log.debug("deepface not available — age/emotion analysis disabled.")
            return

        while self._running:
            try:
                item = self._attr_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break   # sentinel — stop signal

            crop, name = item
            try:
                analysis = DeepFace.analyze(
                    img_path=crop,
                    actions=["age", "emotion"],
                    detector_backend="skip",
                    enforce_detection=False,
                    silent=True,
                )
                age     = int(analysis[0]["age"])
                emotion = analysis[0]["dominant_emotion"]
                with self._attr_lock:
                    self._attr_cache[name] = (age, emotion, age < 18)
            except Exception:
                pass

    # ── Screenshot helper ─────────────────────────────────────────────

    def _save_screenshot(
        self,
        frame: np.ndarray,
        result: MatchResult,
    ) -> Optional[str]:
        """
        Save a labelled JPEG snapshot of the current frame.
        Returns the saved path, or None on failure.

        Filename: {list_type}_{name}_{YYYYMMDD_HHMMSS}_{camera}.jpg
        """
        try:
            os.makedirs(self._screenshots_dir, exist_ok=True)
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_safe = result.name.replace(" ", "_").replace("/", "_")
            cam_safe  = self._cam_cfg.label.replace(" ", "_")
            filename  = f"{result.list_type}_{name_safe}_{ts}_{cam_safe}.jpg"
            out_path  = os.path.join(self._screenshots_dir, filename)

            snap = frame.copy()

            # Top banner: name + list + confidence
            # Green for whitelist, red for blacklist — matches live box colours
            if result.list_type == LIST_WHITELIST:
                col = (0, 210, 50)    # green
            elif result.list_type == LIST_BLACKLIST:
                col = (0, 0, 220)     # red
            else:
                col = (180, 180, 180) # grey fallback (should never reach here)
            label = f"  {result.name}  |  {result.list_type.upper()}  |  {result.confidence:.0%}"
            cv2.rectangle(snap, (0, 0), (snap.shape[1], 30), (10, 10, 20), cv2.FILLED)
            cv2.putText(snap, label, (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)

            # Bottom banner: timestamp + camera
            ts_label = (f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        f"  [{self._cam_cfg.label}]")
            h = snap.shape[0]
            cv2.rectangle(snap, (0, h - 28), (snap.shape[1], h), (10, 10, 20), cv2.FILLED)
            cv2.putText(snap, ts_label, (6, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imwrite(out_path, snap)
            log.info("[%s] Screenshot saved: %s", self._cam_cfg.label, out_path)
            return out_path

        except Exception as exc:
            log.warning("[%s] Screenshot failed: %s", self._cam_cfg.label, exc)
            return None

    # ── Capture mode ──────────────────────────────────────────────────

    def toggle_capture_mode(self) -> None:
        """Toggle capture-mode on/off (C key). When active, shows a positioning
        guide and auto-captures once the face has been still for _STEADY_FRAMES."""
        with self._capture_lock:
            if self._capture_mode:
                self._capture_mode   = False
                self._capture_steady = 0
                self._capture_status = ""
                log.info("[%s] Capture mode cancelled.", self._cam_cfg.label)
            else:
                self._capture_mode     = True
                self._capture_steady   = 0
                self._capture_last_box = None
                self._capture_status   = "Position your face in the oval"
                log.info("[%s] Capture mode activated.", self._cam_cfg.label)

    def _process_capture_steady(
        self,
        locs: list,
        crops: list,
        frame: np.ndarray,
    ) -> None:
        """Called from Thread B. Track face steadiness; auto-capture when ready."""
        if not locs:
            with self._capture_lock:
                self._capture_steady   = 0
                self._capture_last_box = None
                self._capture_status   = "Position your face in the oval"
            return

        # Frontality check — require both eyes to be visible.
        # If the person is turned sideways, the eye cascade finds < 2 eyes,
        # so the steady counter resets and capture does not start.
        crop_gray = cv2.cvtColor(crops[0], cv2.COLOR_BGR2GRAY)
        eyes = self._eye_cascade.detectMultiScale(
            crop_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15),
        )
        if len(eyes) < 2:
            with self._capture_lock:
                self._capture_steady   = 0
                self._capture_last_box = None
                self._capture_status   = "Face the camera directly"
            return

        top, right, bottom, left = locs[0]
        cx = (left + right) // 2
        cy = (top + bottom) // 2

        with self._capture_lock:
            last = self._capture_last_box
            if last is not None:
                lcx = (last[3] + last[1]) // 2
                lcy = (last[0] + last[2]) // 2
                moved = (abs(cx - lcx) > _STEADY_PIXELS or
                         abs(cy - lcy) > _STEADY_PIXELS)
            else:
                moved = True

            self._capture_last_box = locs[0]

            if moved:
                self._capture_steady = max(0, self._capture_steady - 1)
            else:
                self._capture_steady = min(_STEADY_FRAMES, self._capture_steady + 1)

            count = self._capture_steady
            seg   = max(1, _STEADY_FRAMES // 3)

            if count == 0:
                self._capture_status = "Position your face in the oval"
            elif count < seg:
                self._capture_status = "Hold still..."
            elif count < seg * 2:
                self._capture_status = "Hold still...  3"
            elif count < _STEADY_FRAMES:
                self._capture_status = "Hold still...  2"
            else:
                self._capture_status = "Capturing!"

        if count >= _STEADY_FRAMES:
            # ── auto-capture ──────────────────────────────────────────────
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            cam_safe = self._cam_cfg.label.replace(" ", "_")
            crop     = crops[0]

            # Full annotated screenshot → screenshots/
            ss_path = self._save_screenshot(frame, UNKNOWN_RESULT)

            # Face crop → known_faces/unknown/ (for later enrolment)
            try:
                os.makedirs(self._unknown_faces_dir, exist_ok=True)
                crop_path = os.path.join(
                    self._unknown_faces_dir, f"Unknown_{ts}_{cam_safe}.jpg"
                )
                cv2.imwrite(crop_path, crop)
                log.info("[%s] Auto-capture saved: %s", self._cam_cfg.label, crop_path)
            except Exception as exc:
                log.warning("[%s] Auto-capture crop failed: %s", self._cam_cfg.label, exc)

            # Log to JSONL so the dashboard shows this capture
            # force=True bypasses the 60s cooldown — user explicitly pressed C
            self._alerter.log_unknown_visitor(
                camera_label=self._cam_cfg.label,
                screenshot_path=ss_path,
                force=True,
            )

            with self._capture_lock:
                self._capture_mode   = False
                self._capture_steady = 0
                self._capture_status = ""

    def _draw_capture_guide(self, frame: np.ndarray) -> None:
        """Draw face-oval + body guide + status overlay when capture mode is active."""
        h, w = frame.shape[:2]

        with self._capture_lock:
            count  = self._capture_steady
            status = self._capture_status

        # Colour: white → amber → green as steadiness grows
        prog = count / max(_STEADY_FRAMES, 1)
        if prog == 0:
            col = (210, 210, 210)           # white — no face
        elif prog < 0.5:
            col = (0, 200, 255)             # amber — moving
        else:
            col = (40, 230, 60)             # green — steady

        # ── face oval ─────────────────────────────────────────────────────
        ox, oy = w // 2, int(h * 0.37)
        rx, ry = int(w * 0.115), int(h * 0.265)
        cv2.ellipse(frame, (ox, oy), (rx, ry), 0, 0, 360, col, 2, cv2.LINE_AA)

        # dashed tick marks at cardinal points for alignment
        for angle, (dx, dy) in [(0, (1, 0)), (90, (0, 1)),
                                 (180, (-1, 0)), (270, (0, -1))]:
            px = ox + int(rx * dx)
            py = oy + int(ry * dy)
            cv2.line(frame,
                     (px - int(8 * dy), py - int(8 * dx)),
                     (px + int(8 * dy), py + int(8 * dx)),
                     col, 2, cv2.LINE_AA)

        # ── shoulder / body guide ──────────────────────────────────────────
        neck_y  = oy + ry + 4
        sh_y    = oy + ry + int(h * 0.09)
        sh_half = int(w * 0.20)
        # slopes from oval bottom-left / bottom-right to shoulder ends
        cv2.line(frame, (ox - rx + 12, neck_y),
                 (ox - sh_half, sh_y), col, 2, cv2.LINE_AA)
        cv2.line(frame, (ox + rx - 12, neck_y),
                 (ox + sh_half, sh_y), col, 2, cv2.LINE_AA)
        # horizontal shoulder bar
        cv2.line(frame, (ox - sh_half, sh_y),
                 (ox + sh_half, sh_y), col, 2, cv2.LINE_AA)

        # ── top banner ────────────────────────────────────────────────────
        banner = "  FACE CAPTURE — hold still  |  C = cancel  "
        (tw, th), _ = cv2.getTextSize(
            banner, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
        bx = (w - tw) // 2
        cv2.rectangle(frame, (bx - 6, 5),
                      (bx + tw + 6, 5 + th + 10), (18, 18, 28), cv2.FILLED)
        cv2.putText(frame, banner, (bx, 5 + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 200, 255), 1, cv2.LINE_AA)

        # ── countdown number inside oval ──────────────────────────────────
        seg = max(1, _STEADY_FRAMES // 3)
        if count >= seg:
            n = 3 - min(2, (count - seg) // seg)   # 3 → 2 → 1
            cnt_str = str(n)
            (cw, ch2), _ = cv2.getTextSize(
                cnt_str, cv2.FONT_HERSHEY_SIMPLEX, 3.5, 5)
            cv2.putText(frame, cnt_str,
                        ((w - cw) // 2, oy + ch2 // 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.5, col, 5, cv2.LINE_AA)

        # ── status text below guide ───────────────────────────────────────
        if status:
            (sw, _), _ = cv2.getTextSize(
                status, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
            cv2.putText(frame, status,
                        ((w - sw) // 2, sh_y + int(h * 0.06)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, col, 2, cv2.LINE_AA)

        # ── steadiness progress bar ───────────────────────────────────────
        bw   = w // 2
        bx2  = (w - bw) // 2
        by   = h - 26
        bh   = 10
        fill = int(bw * prog)
        cv2.rectangle(frame, (bx2, by),
                      (bx2 + bw, by + bh), (35, 35, 45), cv2.FILLED)
        if fill > 0:
            cv2.rectangle(frame, (bx2, by),
                          (bx2 + fill, by + bh), col, cv2.FILLED)
        cv2.rectangle(frame, (bx2, by),
                      (bx2 + bw, by + bh), (90, 90, 110), 1)
        # label
        lbl = f"Steady  {int(prog * 100)}%"
        cv2.putText(frame, lbl, (bx2, by - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 170), 1, cv2.LINE_AA)

    def capture_unknown_face(self) -> int:
        """
        Manual capture (C key): detect faces in the current frame and save:
          1. Full annotated frame  → screenshots/  (same as whitelist/blacklist screenshots)
          2. Face crop only        → known_faces/unknown/  (ready to rename and enrol)

        Returns the number of faces saved.
        """
        frame = self.latest_raw_frame
        if frame is None:
            log.warning("[%s] Manual capture: no frame available.", self._cam_cfg.label)
            return 0

        db = self._db_manager.get()
        _, crops = _detect_faces(frame, db.detector)

        if not crops:
            log.info("[%s] Manual capture: no face detected in frame.", self._cam_cfg.label)
            return 0

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        cam_safe = self._cam_cfg.label.replace(" ", "_")
        saved    = 0

        for i, crop in enumerate(crops):
            suffix = f"_{i + 1}" if len(crops) > 1 else ""

            # 1. Full annotated screenshot → screenshots/
            ss_path = self._save_screenshot(frame, UNKNOWN_RESULT)

            # 2. Face crop → known_faces/unknown/  (for later enrolment)
            try:
                os.makedirs(self._unknown_faces_dir, exist_ok=True)
                crop_name = f"Unknown_{ts}{suffix}_{cam_safe}.jpg"
                crop_path = os.path.join(self._unknown_faces_dir, crop_name)
                cv2.imwrite(crop_path, crop)
                log.info("[%s] Unknown face crop saved: %s", self._cam_cfg.label, crop_path)
            except Exception as exc:
                log.warning("[%s] Unknown face crop save failed: %s", self._cam_cfg.label, exc)

            # Log to JSONL so the dashboard shows this capture
            # force=True bypasses the 60s cooldown — user explicitly pressed C
            self._alerter.log_unknown_visitor(
                camera_label=self._cam_cfg.label,
                screenshot_path=ss_path,
                force=True,
            )

            saved += 1

        return saved

    def _save_unknown_face(self, crop: np.ndarray, ts: str) -> Optional[str]:
        """
        Save the face crop of an unknown visitor to unknown_faces_dir.

        Filename follows the same naming convention as whitelist / blacklist:
            Unknown_{YYYYMMDD_HHMMSS}_{Camera}.jpg
        When loaded by the face DB the name reads: "Unknown YYYYMMDD HHMMSS Camera".
        Operators can rename and move the file to whitelist/ or blacklist/ to enrol
        the person after identification.
        """
        try:
            os.makedirs(self._unknown_faces_dir, exist_ok=True)
            cam_safe = self._cam_cfg.label.replace(" ", "_")
            filename = f"Unknown_{ts}_{cam_safe}.jpg"
            out_path = os.path.join(self._unknown_faces_dir, filename)
            cv2.imwrite(out_path, crop)
            log.info("[%s] Unknown face saved: %s", self._cam_cfg.label, out_path)
            return out_path
        except Exception as exc:
            log.warning("[%s] Unknown face save failed: %s", self._cam_cfg.label, exc)
            return None

    # ── Helpers ───────────────────────────────────────────────────────

    def _draw_label(self, frame: np.ndarray, label: str) -> None:
        h = frame.shape[0]
        text = f" {label} "
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        y = h - 10
        cv2.rectangle(frame, (0, y - th - bl - 4), (tw + 6, h), (25, 25, 35), cv2.FILLED)
        cv2.putText(frame, text, (4, y - bl),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 1, cv2.LINE_AA)

    def _set_error_frame(self) -> None:
        cfg   = self._cam_cfg
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"[{cfg.label}] unavailable",
                    (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2, cv2.LINE_AA)
        with self._out_lock:
            self._out_frame = frame
