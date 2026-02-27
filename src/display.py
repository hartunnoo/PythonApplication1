"""
Display renderer — dual-list color scheme + left scan-info panel.

Box colors:
  IDLE (no face)   → blue scanning border on frame edges
  WHITELIST match  → green box + name
  BLACKLIST match  → red box + name (semi-transparent red fill)
  UNKNOWN visitor  → green box + "UNKNOWN VISITOR"

Left panel (220 px wide):
  Camera label, scan status, last result, TRUE/FALSE indicator
"""

from __future__ import annotations

import math
import time
from collections import deque
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import DisplayConfig
from src.matcher import MatchResult, LIST_WHITELIST, LIST_BLACKLIST, LIST_UNKNOWN, LIST_IDLE

# ── Colors (BGR) ────────────────────────────────────────────────────────────
COL_IDLE      = (200, 120,  30)   # Blue-ish orange for idle border
COL_WHITELIST = (  0, 210,  50)   # Green
COL_BLACKLIST = (  0,   0, 220)   # Red
COL_UNKNOWN   = ( 30, 200,  80)   # Slightly different green
COL_WHITE     = (255, 255, 255)
COL_PANEL_BG  = ( 18,  18,  28)   # Very dark blue-grey

_FONT       = cv2.FONT_HERSHEY_DUPLEX
_FONT_S     = cv2.FONT_HERSHEY_SIMPLEX
_EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"

_eye_cascade: Optional[cv2.CascadeClassifier] = None

def _get_eye_cascade() -> Optional[cv2.CascadeClassifier]:
    global _eye_cascade
    if _eye_cascade is None:
        cc = cv2.CascadeClassifier(_EYE_CASCADE_PATH)
        _eye_cascade = cc if not cc.empty() else None
    return _eye_cascade

def _box_color(list_type: str) -> Tuple[int, int, int]:
    return {
        LIST_WHITELIST: COL_WHITELIST,
        LIST_BLACKLIST: COL_BLACKLIST,
        LIST_UNKNOWN:   COL_UNKNOWN,
        LIST_IDLE:      COL_IDLE,
    }.get(list_type, COL_UNKNOWN)


# ---------------------------------------------------------------------------
# FPS meter
# ---------------------------------------------------------------------------

class _FPSMeter:
    def __init__(self, window: int = 30) -> None:
        self._times: deque = deque(maxlen=window)

    def tick(self) -> None:
        self._times.append(time.monotonic())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# Scan info state (per camera, kept between frames)
# ---------------------------------------------------------------------------

class ScanInfo:
    """Tracks the last significant detection event for the left panel."""

    def __init__(self) -> None:
        self.scanning:    bool          = True
        self.last_name:   str           = "—"
        self.last_list:   str           = LIST_IDLE
        self.last_conf:   float         = 0.0
        self.last_time:   str           = "—"
        self.is_match:    bool          = False
        self.face_count:  int           = 0
        self.last_age:    Optional[int] = None
        self.last_emotion: Optional[str]= None
        self.last_is_child: Optional[bool] = None

    def update(self, results: List[MatchResult]) -> None:
        self.face_count = len(results)
        self.scanning = True

        # Pick the most significant result (blacklist > whitelist > unknown)
        priority = {LIST_BLACKLIST: 3, LIST_WHITELIST: 2, LIST_UNKNOWN: 1}
        if not results:
            return

        best = max(results, key=lambda r: (priority.get(r.list_type, 0), r.confidence))
        self.last_name     = best.name
        self.last_list     = best.list_type
        self.last_conf     = best.confidence
        self.last_time     = datetime.now().strftime("%H:%M:%S")
        self.is_match      = best.is_match
        self.last_age      = best.age
        self.last_emotion  = best.emotion
        self.last_is_child = best.is_child


# ---------------------------------------------------------------------------
# Per-camera frame renderer
# ---------------------------------------------------------------------------

class FrameRenderer:
    def __init__(self, cfg: DisplayConfig) -> None:
        self._cfg   = cfg
        self._fps   = _FPSMeter()
        self._scan  = ScanInfo()
        self._pulse = 0   # for idle animation counter

    def render(
        self,
        frame: np.ndarray,
        face_locations: List[Tuple[int, int, int, int]],
        results: List[MatchResult],
        *,
        paused: bool = False,
    ) -> np.ndarray:
        self._fps.tick()
        self._scan.update(results)
        self._pulse = (self._pulse + 1) % 60

        # Draw left panel first (so face boxes render on top of panel edge)
        panel = self._cfg.panel_width
        self._draw_panel(frame, panel)

        if not face_locations:
            # Idle state — animated blue scanning border
            self._draw_idle_border(frame, panel)
        else:
            for loc, result in zip(face_locations, results):
                self._draw_face(frame, loc, result)

        if paused:
            h, w = frame.shape[:2]
            cv2.putText(frame, "PAUSED", (w // 2 - 60, h // 2),
                        _FONT, 1.4, (0, 220, 220), 2)
        return frame

    # ------------------------------------------------------------------
    # Left scan-info panel
    # ------------------------------------------------------------------

    def _draw_panel(self, frame: np.ndarray, panel_w: int) -> None:
        h = frame.shape[0]
        s = self._scan

        # Dark background
        cv2.rectangle(frame, (0, 0), (panel_w, h), COL_PANEL_BG, cv2.FILLED)
        # Right border line
        cv2.line(frame, (panel_w, 0), (panel_w, h), (60, 60, 80), 1)

        x = 10
        y = 20

        # ── Title ──────────────────────────────────────────────────────
        cv2.putText(frame, "SCAN STATUS", (x, y), _FONT_S, 0.55, (150, 150, 200), 1)
        y += 4
        cv2.line(frame, (x, y), (panel_w - x, y), (60, 60, 90), 1)
        y += 18

        # ── FPS ────────────────────────────────────────────────────────
        cv2.putText(frame, f"FPS: {self._fps.fps:.0f}", (x, y), _FONT_S, 0.48, (100, 100, 140), 1)
        y += 22

        # ── Faces in frame ─────────────────────────────────────────────
        cv2.putText(frame, f"Faces: {s.face_count}", (x, y), _FONT_S, 0.48, (100, 100, 140), 1)
        y += 26

        cv2.line(frame, (x, y), (panel_w - x, y), (50, 50, 70), 1)
        y += 16

        # ── Status ─────────────────────────────────────────────────────
        cv2.putText(frame, "STATUS", (x, y), _FONT_S, 0.45, (120, 120, 160), 1)
        y += 18

        pulse_bright = abs(self._pulse - 30) / 30.0   # 0→1→0
        if s.face_count == 0:
            dot_col = tuple(int(c * (0.4 + 0.6 * pulse_bright)) for c in COL_IDLE)
            label   = "IDLE / SCANNING"
        elif s.last_list == LIST_BLACKLIST:
            dot_col = COL_BLACKLIST
            label   = "BLACKLIST HIT"
        elif s.last_list == LIST_WHITELIST:
            dot_col = COL_WHITELIST
            label   = "WHITELIST"
        else:
            dot_col = COL_UNKNOWN
            label   = "UNKNOWN FACE"

        cv2.circle(frame, (x + 6, y - 4), 5, dot_col, cv2.FILLED)
        cv2.putText(frame, label, (x + 16, y), _FONT_S, 0.42, dot_col, 1)
        y += 26

        cv2.line(frame, (x, y), (panel_w - x, y), (50, 50, 70), 1)
        y += 16

        # ── Last result ────────────────────────────────────────────────
        cv2.putText(frame, "LAST RESULT", (x, y), _FONT_S, 0.45, (120, 120, 160), 1)
        y += 18

        name_col = _box_color(s.last_list)
        name_display = s.last_name if len(s.last_name) <= 16 else s.last_name[:14] + ".."
        cv2.putText(frame, name_display, (x, y), _FONT_S, 0.52, name_col, 1)
        y += 20

        list_label = s.last_list.upper() if s.last_list != LIST_IDLE else "—"
        cv2.putText(frame, f"List: {list_label}", (x, y), _FONT_S, 0.44, (130, 130, 170), 1)
        y += 18

        conf_pct = f"{s.last_conf:.0%}" if s.last_conf > 0 else "—"
        cv2.putText(frame, f"Conf: {conf_pct}", (x, y), _FONT_S, 0.44, (130, 130, 170), 1)
        y += 18

        cv2.putText(frame, f"Time: {s.last_time}", (x, y), _FONT_S, 0.42, (100, 100, 140), 1)
        y += 18

        if s.last_emotion:
            cv2.putText(frame, f"Mood: {s.last_emotion.capitalize()}", (x, y),
                        _FONT_S, 0.42, (130, 130, 170), 1)
            y += 18

        if s.last_age is not None:
            age_cat = "Child" if s.last_is_child else "Adult"
            cv2.putText(frame, f"Age:  ~{s.last_age} ({age_cat})", (x, y),
                        _FONT_S, 0.42, (130, 130, 170), 1)
            y += 18

        y += 8

        cv2.line(frame, (x, y), (panel_w - x, y), (50, 50, 70), 1)
        y += 16

        # ── TRUE / FALSE indicator ─────────────────────────────────────
        cv2.putText(frame, "SCAN RESULT", (x, y), _FONT_S, 0.45, (120, 120, 160), 1)
        y += 22

        if s.face_count == 0:
            ind_text  = "SCANNING"
            ind_color = COL_IDLE
        elif s.is_match:
            ind_text  = "TRUE"
            ind_color = COL_WHITELIST if s.last_list == LIST_WHITELIST else COL_BLACKLIST
        else:
            ind_text  = "FALSE"
            ind_color = (100, 100, 120)

        # Pill background
        (tw, th), _ = cv2.getTextSize(ind_text, _FONT, 0.75, 2)
        pill_x1, pill_y1 = x, y - th - 4
        pill_x2, pill_y2 = x + tw + 16, y + 6
        cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), ind_color, cv2.FILLED)
        cv2.putText(frame, ind_text, (x + 8, y), _FONT, 0.75, COL_WHITE, 2)
        y += 40

        # ── Legend key ─────────────────────────────────────────────────
        cv2.line(frame, (x, y), (panel_w - x, y), (50, 50, 70), 1)
        y += 14
        cv2.putText(frame, "LEGEND", (x, y), _FONT_S, 0.42, (90, 90, 120), 1)
        y += 16

        for col, txt in [
            (COL_IDLE,      "IDLE/SCANNING"),
            (COL_WHITELIST, "WHITELIST"),
            (COL_UNKNOWN,   "UNKNOWN"),
            (COL_BLACKLIST, "BLACKLIST"),
        ]:
            cv2.rectangle(frame, (x, y - 8), (x + 10, y + 2), col, cv2.FILLED)
            cv2.putText(frame, txt, (x + 15, y), _FONT_S, 0.40, (110, 110, 150), 1)
            y += 16

        # ── Confidence guide table ──────────────────────────────────────
        cv2.line(frame, (x, y), (panel_w - x, y), (50, 50, 70), 1)
        y += 14
        cv2.putText(frame, "CONFIDENCE GUIDE", (x, y), _FONT_S, 0.42, (90, 90, 120), 1)
        y += 4
        cv2.line(frame, (x, y), (panel_w - x, y), (40, 40, 60), 1)
        y += 14

        _CONF_ROWS = [
            ("<  25%",  (80,  80,  95), "No match"),
            ("25-50%",  (30, 120, 200), "Weak / risky"),
            ("50-70%",  (20, 190, 190), "Moderate"),
            ("70-85%",  (50, 200, 100), "Good"),
            (">  85%",  ( 0, 220,  55), "Excellent"),
        ]
        for rng, swatch, meaning in _CONF_ROWS:
            # Highlight current confidence band when a match is active
            highlight = False
            if s.face_count > 0 and s.last_conf > 0:
                c = s.last_conf
                if (rng == "<  25%" and c < 0.25) or \
                   (rng == "25-50%" and 0.25 <= c < 0.50) or \
                   (rng == "50-70%" and 0.50 <= c < 0.70) or \
                   (rng == "70-85%" and 0.70 <= c < 0.85) or \
                   (rng == ">  85%" and c >= 0.85):
                    highlight = True

            row_bg = (35, 35, 50) if highlight else COL_PANEL_BG
            cv2.rectangle(frame, (x - 2, y - 11), (panel_w - x + 2, y + 4),
                          row_bg, cv2.FILLED)

            # Colour swatch
            cv2.rectangle(frame, (x, y - 9), (x + 9, y + 2), swatch, cv2.FILLED)
            if highlight:
                cv2.rectangle(frame, (x, y - 9), (x + 9, y + 2), COL_WHITE, 1)

            # Range text
            range_col = COL_WHITE if highlight else (140, 140, 175)
            cv2.putText(frame, rng, (x + 13, y), _FONT_S, 0.38, range_col, 1)

            # Meaning text
            mean_col = swatch if highlight else (100, 100, 135)
            cv2.putText(frame, meaning, (x + 58, y), _FONT_S, 0.37, mean_col, 1)
            y += 15

        # ── Author credit (pinned to bottom of panel) ───────────────────
        cv2.line(frame, (x, h - 54), (panel_w - x, h - 54), (40, 40, 58), 1)
        cr_col = (48, 48, 72)
        cv2.putText(frame, "His Majesty Office",     (x, h - 40), _FONT_S, 0.31, cr_col, 1)
        cv2.putText(frame, "Istana Nurul Iman",       (x, h - 28), _FONT_S, 0.31, cr_col, 1)
        cv2.putText(frame, "A.H. Suharddy Bin Mohd Soud", (x, h - 16), _FONT_S, 0.29, cr_col, 1)
        cv2.putText(frame, "Ketua Server & Security", (x, h -  4), _FONT_S, 0.29, cr_col, 1)

    # ------------------------------------------------------------------
    # Idle border animation
    # ------------------------------------------------------------------

    def _draw_idle_border(self, frame: np.ndarray, panel_w: int) -> None:
        h, w = frame.shape[:2]
        bright = int(80 + 80 * abs(self._pulse - 30) / 30.0)
        col = (bright, int(bright * 0.5), 10)
        t = 3
        # Border on three sides (right, top, bottom) — left is the panel
        cv2.rectangle(frame, (panel_w + t, t), (w - t, h - t), col, t)
        cv2.putText(frame, "SCANNING...", (panel_w + 12, 28),
                    _FONT_S, 0.65, col, 1)

    # ------------------------------------------------------------------
    # Face box rendering
    # ------------------------------------------------------------------

    def _draw_face(
        self,
        frame: np.ndarray,
        location: Tuple[int, int, int, int],
        result: MatchResult,
    ) -> None:
        top, right, bottom, left = location
        cfg   = self._cfg
        colour = _box_color(result.list_type)
        thick  = cfg.box_thickness

        # Semi-transparent fill for blacklist
        if result.list_type == LIST_BLACKLIST:
            overlay = frame.copy()
            cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 160), cv2.FILLED)
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

        # Main box
        cv2.rectangle(frame, (left, top), (right, bottom), colour, thick)

        # Corner accents
        clen = max(10, (right - left) // 6)
        for px, py, dx, dy in [
            (left, top, 1, 1), (right, top, -1, 1),
            (left, bottom, 1, -1), (right, bottom, -1, -1),
        ]:
            cv2.line(frame, (px, py), (px + dx * clen, py), colour, thick + 1)
            cv2.line(frame, (px, py), (px, py + dy * clen), colour, thick + 1)

        # Label badge
        if result.list_type == LIST_BLACKLIST:
            label = f"[BLACKLIST] {result.name}"
        elif result.list_type == LIST_WHITELIST:
            label = f"[WHITELIST] {result.name}"
        else:
            label = f"[UNKNOWN] UNKNOWN VISITOR"

        conf_str = f" {result.confidence:.0%}" if result.is_match else ""
        full_label = f" {label}{conf_str} "

        (tw, th), bl = cv2.getTextSize(full_label, _FONT, cfg.font_scale, 1)
        label_top = max(bottom, th + 4)
        cv2.rectangle(
            frame,
            (left, label_top - th - bl - 4),
            (left + tw, label_top + 2),
            colour, cv2.FILLED,
        )
        cv2.putText(frame, full_label, (left, label_top - bl - 2),
                    _FONT, cfg.font_scale, COL_WHITE, 1)

        # Attribute line: emotion | age | adult/child
        attr_parts = []
        if result.emotion:
            attr_parts.append(result.emotion.capitalize())
        if result.age is not None:
            age_cat = "Child" if result.is_child else "Adult"
            attr_parts.append(f"{result.age}y / {age_cat}")
        if attr_parts:
            attr_label = f"  {' | '.join(attr_parts)}  "
            attr_scale = cfg.font_scale * 0.78
            (atw, ath), abl = cv2.getTextSize(attr_label, _FONT_S, attr_scale, 1)
            attr_y = label_top + ath + abl + 4
            cv2.rectangle(frame,
                          (left, label_top + 3),
                          (left + atw, attr_y + 2),
                          (30, 30, 45), cv2.FILLED)
            cv2.putText(frame, attr_label, (left, attr_y - abl),
                        _FONT_S, attr_scale, colour, 1)

        # Eye circles
        eye_cc = _get_eye_cascade()
        if eye_cc is not None:
            face_gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            half_h = face_gray.shape[0] // 2
            roi    = face_gray[:half_h, :]
            eyes   = eye_cc.detectMultiScale(
                roi, scaleFactor=1.1, minNeighbors=6, minSize=(15, 15)
            )
            eye_col = (255, 220, 50)
            for ex, ey, ew, eh in eyes:
                cx = left + ex + ew // 2
                cy = top  + ey + eh // 2
                r  = max(ew, eh) // 2
                cv2.circle(frame, (cx, cy), r, eye_col, 2)
                cv2.circle(frame, (cx, cy), max(2, r // 4), eye_col, cv2.FILLED)


# ---------------------------------------------------------------------------
# Grid compositor
# ---------------------------------------------------------------------------

def make_grid(
    frames: List[Optional[np.ndarray]],
    cell_w: int = 640,
    cell_h: int = 360,
    border: int = 3,
    border_color: Tuple[int, int, int] = (40, 40, 55),
) -> np.ndarray:
    n    = len(frames)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    total_w = cols * cell_w + (cols + 1) * border
    total_h = rows * cell_h + (rows + 1) * border
    canvas  = np.full((total_h, total_w, 3), border_color, dtype=np.uint8)

    placeholder = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    cv2.putText(placeholder, "No signal", (cell_w // 2 - 60, cell_h // 2),
                _FONT_S, 0.8, (80, 80, 80), 1)

    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x   = border + col * (cell_w + border)
        y   = border + row * (cell_h + border)
        cell = cv2.resize(frame if frame is not None else placeholder, (cell_w, cell_h))
        canvas[y : y + cell_h, x : x + cell_w] = cell

    return canvas


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def create_window(title: str) -> None:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280, 720)


def show_frame(title: str, frame: np.ndarray) -> None:
    cv2.imshow(title, frame)


def read_key(wait_ms: int = 1) -> int:
    return cv2.waitKey(wait_ms) & 0xFF


def destroy_windows() -> None:
    cv2.destroyAllWindows()
