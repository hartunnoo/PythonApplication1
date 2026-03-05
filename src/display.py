"""
Display renderer — dual-list color scheme + left scan-info panel.

Box colors:
  IDLE (no face)   → blue scanning border on frame edges
  WHITELIST match  → green box + name
  BLACKLIST match  → red box + name (semi-transparent red fill)
  UNKNOWN visitor  → green box + "UNKNOWN VISITOR"

Left panel (280 px wide):
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
COL_PANEL_SECTION = (60, 60, 90)  # Section divider lines
COL_PANEL_SUB     = (50, 50, 70)  # Sub-divider lines
COL_LABEL_DIM     = (130, 130, 170)  # Dimmed info text
COL_LABEL_MUTED   = (100, 100, 140)  # Very dim text
COL_HEADER_DIM    = (90, 90, 120)    # Section header text

_AA         = cv2.LINE_AA          # Anti-aliased line type for sharp text
_FONT       = cv2.FONT_HERSHEY_DUPLEX
_FONT_S     = cv2.FONT_HERSHEY_SIMPLEX
_FONT_T     = cv2.FONT_HERSHEY_TRIPLEX   # Thicker stroked font for headers
_EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"

# Panel supersampling factor — render text at Nx resolution then downscale
# for sub-pixel smooth text. 2 = double resolution, 1 = disabled.
_PANEL_SS = 2

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
            # Shadow + main text for sharp PAUSED overlay
            cv2.putText(frame, "PAUSED", (w // 2 - 59, h // 2 + 1),
                        _FONT, 1.4, (0, 0, 0), 3, _AA)
            cv2.putText(frame, "PAUSED", (w // 2 - 60, h // 2),
                        _FONT, 1.4, (0, 220, 220), 2, _AA)
        return frame

    # ------------------------------------------------------------------
    # Left scan-info panel
    # ------------------------------------------------------------------

    def _draw_panel(self, frame: np.ndarray, panel_w: int) -> None:
        """Draw the left scan-info panel.

        When _PANEL_SS > 1 we render to a 2× canvas then downscale with
        INTER_AREA (best for shrinking) → sub-pixel smooth text that looks
        dramatically sharper than native-resolution LINE_AA alone.
        """
        h = frame.shape[0]
        ss = _PANEL_SS  # supersampling factor

        # Create hi-res canvas for the panel region
        ph, pw = h * ss, panel_w * ss
        panel = np.full((ph, pw, 3), COL_PANEL_BG, dtype=np.uint8)

        self._render_panel_content(panel, pw, ph, ss)

        # Downscale with INTER_AREA (best for shrink, preserves fine text detail)
        if ss > 1:
            panel = cv2.resize(panel, (panel_w, h), interpolation=cv2.INTER_AREA)

        # Blit panel onto frame
        frame[0:h, 0:panel_w] = panel

        # Right border line — drawn on final frame for pixel-precise alignment
        cv2.line(frame, (panel_w, 0), (panel_w, h), COL_PANEL_SECTION, 1, _AA)

    def _render_panel_content(self, panel: np.ndarray, pw: int, ph: int, ss: int) -> None:
        """Render all panel text/shapes at ss× resolution."""
        s = self._scan

        # Scale all coordinates and font sizes by ss
        x = 14 * ss
        y = 28 * ss
        line_h = lambda n: int(n * ss)  # helper for vertical spacing

        # ── Title ──────────────────────────────────────────────────────
        cv2.putText(panel, "SCAN STATUS", (x, y),
                    _FONT_T, 0.60 * ss, (170, 170, 220), max(1, ss), _AA)
        y += line_h(8)
        cv2.line(panel, (x, y), (pw - x, y), COL_PANEL_SECTION, max(1, ss), _AA)
        y += line_h(24)

        # ── FPS + Face count (single row) ─────────────────────────────
        cv2.putText(panel, f"FPS: {self._fps.fps:.0f}", (x, y),
                    _FONT_S, 0.52 * ss, COL_LABEL_MUTED, max(1, ss), _AA)
        cv2.putText(panel, f"Faces: {s.face_count}", (x + 110 * ss, y),
                    _FONT_S, 0.52 * ss, COL_LABEL_MUTED, max(1, ss), _AA)
        y += line_h(28)

        cv2.line(panel, (x, y), (pw - x, y), COL_PANEL_SUB, max(1, ss), _AA)
        y += line_h(20)

        # ── Status ─────────────────────────────────────────────────────
        cv2.putText(panel, "STATUS", (x, y),
                    _FONT_S, 0.50 * ss, (120, 120, 160), max(1, ss), _AA)
        y += line_h(24)

        pulse_bright = abs(self._pulse - 30) / 30.0
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

        cv2.circle(panel, (x + 8 * ss, y - 5 * ss), 7 * ss, dot_col, cv2.FILLED, _AA)
        cv2.putText(panel, label, (x + 22 * ss, y),
                    _FONT_S, 0.50 * ss, dot_col, max(1, ss), _AA)
        y += line_h(30)

        cv2.line(panel, (x, y), (pw - x, y), COL_PANEL_SUB, max(1, ss), _AA)
        y += line_h(20)

        # ── Last result ────────────────────────────────────────────────
        cv2.putText(panel, "LAST RESULT", (x, y),
                    _FONT_S, 0.50 * ss, (120, 120, 160), max(1, ss), _AA)
        y += line_h(24)

        name_col = _box_color(s.last_list)
        max_chars = max(18, (pw // ss - 28) // 10)
        name_display = s.last_name if len(s.last_name) <= max_chars else s.last_name[:max_chars - 2] + ".."
        cv2.putText(panel, name_display, (x, y),
                    _FONT, 0.58 * ss, name_col, max(1, ss), _AA)
        y += line_h(24)

        list_label = s.last_list.upper() if s.last_list != LIST_IDLE else "—"
        cv2.putText(panel, f"List: {list_label}", (x, y),
                    _FONT_S, 0.50 * ss, COL_LABEL_DIM, max(1, ss), _AA)
        y += line_h(22)

        conf_pct = f"{s.last_conf:.0%}" if s.last_conf > 0 else "—"
        cv2.putText(panel, f"Conf: {conf_pct}", (x, y),
                    _FONT_S, 0.50 * ss, COL_LABEL_DIM, max(1, ss), _AA)
        y += line_h(22)

        cv2.putText(panel, f"Time: {s.last_time}", (x, y),
                    _FONT_S, 0.48 * ss, COL_LABEL_MUTED, max(1, ss), _AA)
        y += line_h(22)

        if s.last_emotion:
            cv2.putText(panel, f"Mood: {s.last_emotion.capitalize()}", (x, y),
                        _FONT_S, 0.48 * ss, COL_LABEL_DIM, max(1, ss), _AA)
            y += line_h(22)

        if s.last_age is not None:
            age_cat = "Child" if s.last_is_child else "Adult"
            cv2.putText(panel, f"Age:  ~{s.last_age} ({age_cat})", (x, y),
                        _FONT_S, 0.48 * ss, COL_LABEL_DIM, max(1, ss), _AA)
            y += line_h(22)

        y += line_h(10)

        cv2.line(panel, (x, y), (pw - x, y), COL_PANEL_SUB, max(1, ss), _AA)
        y += line_h(20)

        # ── TRUE / FALSE indicator ─────────────────────────────────────
        cv2.putText(panel, "SCAN RESULT", (x, y),
                    _FONT_S, 0.50 * ss, (120, 120, 160), max(1, ss), _AA)
        y += line_h(26)

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
        (tw, th), _ = cv2.getTextSize(ind_text, _FONT, 0.85 * ss, 2 * ss)
        pill_x1, pill_y1 = x, y - th - 6 * ss
        pill_x2, pill_y2 = x + tw + 22 * ss, y + 8 * ss
        cv2.rectangle(panel, (pill_x1, pill_y1), (pill_x2, pill_y2), ind_color, cv2.FILLED)
        cv2.putText(panel, ind_text, (x + 10 * ss, y),
                    _FONT, 0.85 * ss, COL_WHITE, 2 * ss, _AA)
        y += line_h(44)

        # ── Legend key ─────────────────────────────────────────────────
        cv2.line(panel, (x, y), (pw - x, y), COL_PANEL_SUB, max(1, ss), _AA)
        y += line_h(18)
        cv2.putText(panel, "LEGEND", (x, y),
                    _FONT_S, 0.48 * ss, COL_HEADER_DIM, max(1, ss), _AA)
        y += line_h(20)

        for col, txt in [
            (COL_IDLE,      "IDLE / SCANNING"),
            (COL_WHITELIST, "WHITELIST"),
            (COL_UNKNOWN,   "UNKNOWN"),
            (COL_BLACKLIST, "BLACKLIST"),
        ]:
            cv2.rectangle(panel, (x, y - 10 * ss), (x + 13 * ss, y + 4 * ss),
                          col, cv2.FILLED)
            cv2.putText(panel, txt, (x + 20 * ss, y),
                        _FONT_S, 0.46 * ss, (120, 120, 160), max(1, ss), _AA)
            y += line_h(20)

        # ── Confidence guide table ──────────────────────────────────────
        y += line_h(4)
        cv2.line(panel, (x, y), (pw - x, y), COL_PANEL_SUB, max(1, ss), _AA)
        y += line_h(18)
        cv2.putText(panel, "CONFIDENCE GUIDE", (x, y),
                    _FONT_S, 0.48 * ss, COL_HEADER_DIM, max(1, ss), _AA)
        y += line_h(6)
        cv2.line(panel, (x, y), (pw - x, y), (40, 40, 60), max(1, ss), _AA)
        y += line_h(18)

        _CONF_ROWS = [
            ("<  25%",  (80,  80,  95), "No match"),
            ("25-50%",  (30, 120, 200), "Weak / risky"),
            ("50-70%",  (20, 190, 190), "Moderate"),
            ("70-85%",  (50, 200, 100), "Good"),
            (">  85%",  ( 0, 220,  55), "Excellent"),
        ]
        for rng, swatch, meaning in _CONF_ROWS:
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
            cv2.rectangle(panel, (x - 2 * ss, y - 13 * ss), (pw - x + 2 * ss, y + 6 * ss),
                          row_bg, cv2.FILLED)

            # Colour swatch
            cv2.rectangle(panel, (x, y - 11 * ss), (x + 12 * ss, y + 4 * ss),
                          swatch, cv2.FILLED)
            if highlight:
                cv2.rectangle(panel, (x, y - 11 * ss), (x + 12 * ss, y + 4 * ss),
                              COL_WHITE, max(1, ss))

            range_col = COL_WHITE if highlight else (140, 140, 175)
            cv2.putText(panel, rng, (x + 18 * ss, y),
                        _FONT_S, 0.44 * ss, range_col, max(1, ss), _AA)

            mean_col = swatch if highlight else (100, 100, 135)
            cv2.putText(panel, meaning, (x + 80 * ss, y),
                        _FONT_S, 0.44 * ss, mean_col, max(1, ss), _AA)
            y += line_h(19)

        # ── Author credit (pinned to bottom of panel) ───────────────────
        cv2.line(panel, (x, ph - 64 * ss), (pw - x, ph - 64 * ss),
                 (40, 40, 58), max(1, ss), _AA)
        cr_col = (60, 60, 88)
        cv2.putText(panel, "His Majesty Office",     (x, ph - 48 * ss),
                    _FONT_S, 0.38 * ss, cr_col, max(1, ss), _AA)
        cv2.putText(panel, "Istana Nurul Iman",       (x, ph - 34 * ss),
                    _FONT_S, 0.38 * ss, cr_col, max(1, ss), _AA)
        cv2.putText(panel, "A.H. Suharddy Bin Mohd Soud", (x, ph - 20 * ss),
                    _FONT_S, 0.35 * ss, cr_col, max(1, ss), _AA)
        cv2.putText(panel, "Ketua Server & Security", (x, ph -  6 * ss),
                    _FONT_S, 0.35 * ss, cr_col, max(1, ss), _AA)

    # ------------------------------------------------------------------
    # Idle border animation
    # ------------------------------------------------------------------

    def _draw_idle_border(self, frame: np.ndarray, panel_w: int) -> None:
        h, w = frame.shape[:2]
        bright = int(80 + 80 * abs(self._pulse - 30) / 30.0)
        col = (bright, int(bright * 0.5), 10)
        t = 3
        # Border on three sides (right, top, bottom) — left is the panel
        cv2.rectangle(frame, (panel_w + t, t), (w - t, h - t), col, t, _AA)
        # Shadow then text for readability
        cv2.putText(frame, "SCANNING...", (panel_w + 13, 31),
                    _FONT_S, 0.70, (0, 0, 0), 2, _AA)
        cv2.putText(frame, "SCANNING...", (panel_w + 12, 30),
                    _FONT_S, 0.70, col, 1, _AA)

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

        # Main box — anti-aliased edges
        cv2.rectangle(frame, (left, top), (right, bottom), colour, thick, _AA)

        # Corner accents
        clen = max(12, (right - left) // 5)
        for px, py, dx, dy in [
            (left, top, 1, 1), (right, top, -1, 1),
            (left, bottom, 1, -1), (right, bottom, -1, -1),
        ]:
            cv2.line(frame, (px, py), (px + dx * clen, py), colour, thick + 1, _AA)
            cv2.line(frame, (px, py), (px, py + dy * clen), colour, thick + 1, _AA)

        # Label badge
        if result.list_type == LIST_BLACKLIST:
            label = f"[BLACKLIST] {result.name}"
        elif result.list_type == LIST_WHITELIST:
            label = f"[WHITELIST] {result.name}"
        else:
            label = f"[UNKNOWN] UNKNOWN VISITOR"

        conf_str = f" {result.confidence:.0%}" if result.is_match else ""
        full_label = f" {label}{conf_str} "

        fs = cfg.font_scale
        (tw, th), bl = cv2.getTextSize(full_label, _FONT, fs, 1)
        label_top = max(bottom + 2, th + 4)
        cv2.rectangle(
            frame,
            (left, label_top - th - bl - 4),
            (left + tw + 2, label_top + 3),
            colour, cv2.FILLED,
        )
        # Text shadow for crispness
        cv2.putText(frame, full_label, (left + 1, label_top - bl - 1),
                    _FONT, fs, (0, 0, 0), 2, _AA)
        cv2.putText(frame, full_label, (left, label_top - bl - 2),
                    _FONT, fs, COL_WHITE, 1, _AA)

        # Attribute line: emotion | age | adult/child
        attr_parts = []
        if result.emotion:
            attr_parts.append(result.emotion.capitalize())
        if result.age is not None:
            age_cat = "Child" if result.is_child else "Adult"
            attr_parts.append(f"{result.age}y / {age_cat}")
        if attr_parts:
            attr_label = f"  {' | '.join(attr_parts)}  "
            attr_scale = fs * 0.80
            (atw, ath), abl = cv2.getTextSize(attr_label, _FONT_S, attr_scale, 1)
            attr_y = label_top + ath + abl + 5
            cv2.rectangle(frame,
                          (left, label_top + 4),
                          (left + atw + 2, attr_y + 3),
                          (30, 30, 45), cv2.FILLED)
            cv2.putText(frame, attr_label, (left, attr_y - abl),
                        _FONT_S, attr_scale, colour, 1, _AA)

        # Eye circles — anti-aliased
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
                cv2.circle(frame, (cx, cy), r, eye_col, 2, _AA)
                cv2.circle(frame, (cx, cy), max(2, r // 4), eye_col, cv2.FILLED, _AA)


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
                _FONT_S, 0.8, (80, 80, 80), 1, _AA)

    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x   = border + col * (cell_w + border)
        y   = border + row * (cell_h + border)
        src = frame if frame is not None else placeholder
        # Use INTER_AREA for shrinking (preserves detail), INTER_LANCZOS4 for enlarging (sharpest)
        sh, sw = src.shape[:2]
        interp = cv2.INTER_AREA if (sw > cell_w or sh > cell_h) else cv2.INTER_LANCZOS4
        cell = cv2.resize(src, (cell_w, cell_h), interpolation=interp)
        canvas[y : y + cell_h, x : x + cell_w] = cell

    return canvas


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def create_window(title: str) -> None:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1920, 1080)


def show_frame(title: str, frame: np.ndarray) -> None:
    cv2.imshow(title, frame)


def read_key(wait_ms: int = 1) -> int:
    return cv2.waitKey(wait_ms) & 0xFF


def destroy_windows() -> None:
    cv2.destroyAllWindows()
