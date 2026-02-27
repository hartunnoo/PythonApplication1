"""
Alert system — whitelist vs blacklist aware.

Blacklist → urgent red popup + JSON event log
Whitelist → informational green popup + JSON event log
Unknown   → no popup (just logged)

JSON events are written to logs/events.jsonl for ASP.NET Core dashboard.

Popup architecture:
  A single hidden Tk root lives on its own dedicated thread (_PopupManager).
  Each alert creates a Toplevel window on that root via a thread-safe Queue.
  This avoids the 'Tcl_AsyncDelete' crash that occurs when multiple Tk()
  instances are created across threads.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

from src.config import AlertConfig, SoundConfig, EmailConfig
from src.json_logger import JsonEventLogger
from src.logger_setup import get_logger, get_match_logger
from src.matcher import MatchResult, LIST_WHITELIST, LIST_BLACKLIST, LIST_UNKNOWN

log = get_logger(__name__)
match_log = get_match_logger()


@dataclass
class AlertEvent:
    name: str
    confidence: float
    list_type: str
    camera_label: str
    timestamp: datetime = field(default_factory=datetime.now)
    screenshot_path: str = ""

    def __str__(self) -> str:
        s = (
            f"{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  "
            f"{self.list_type.upper():<12}  [{self.camera_label}]  "
            f"{self.name}  confidence={self.confidence:.2%}"
        )
        if self.screenshot_path:
            s += f"  screenshot={self.screenshot_path}"
        return s


# ── Popup offset — stack multiple popups so they don't overlap ────────────────
_POPUP_STACK_OFFSET = 30   # pixels to shift each additional popup up/left


def _build_popup(
    parent: tk.Tk,
    name: str,
    confidence: float,
    list_type: str,
    camera_label: str,
    timestamp: datetime,
    duration_ms: int,
    stack_index: int = 0,
    age: Optional[int] = None,
    emotion: Optional[str] = None,
    is_child: Optional[bool] = None,
    screenshot_path: Optional[str] = None,
) -> None:
    """Build one alert popup as a Toplevel on the shared hidden root."""
    try:
        is_blacklist = list_type == LIST_BLACKLIST

        # ── Colour palette ────────────────────────────────────────────
        BG        = "#0f0f1a"
        BG_CARD   = "#16162a"
        BG_ROW    = "#1c1c30"
        HDR_COL   = "#b01c2e" if is_blacklist else "#1a7a3a"
        HDR_LIGHT = "#e02040" if is_blacklist else "#22aa50"
        ACCENT    = "#ff3355" if is_blacklist else "#22dd66"
        MUTED     = "#6e6e96"
        LABEL_COL = "#9090b8"
        VALUE_COL = "#d8d8f0"
        WHITE     = "#ffffff"

        win = tk.Toplevel(parent)
        win.title("SECURITY ALERT" if is_blacklist else "ACCESS GRANTED")
        win.resizable(False, False)
        win.attributes("-topmost", True)
        win.configure(bg=BG)

        W, H = 520, 420
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        off = stack_index * _POPUP_STACK_OFFSET
        win.geometry(f"{W}x{H}+{sw - W - 24 - off}+{sh - H - 60 - off}")

        # ── Header bar ───────────────────────────────────────────────
        hdr = tk.Frame(win, bg=HDR_COL, height=52)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        hdr_inner = tk.Frame(hdr, bg=HDR_COL)
        hdr_inner.place(relx=0.5, rely=0.5, anchor="center")

        status_txt = (
            "SECURITY ALERT  —  BLACKLIST DETECTED"
            if is_blacklist else
            "ACCESS GRANTED  —  WHITELIST MATCH"
        )
        tk.Label(hdr_inner, text=status_txt,
                 font=("Segoe UI", 11, "bold"), fg=WHITE, bg=HDR_COL).pack()

        # ── Thin accent line ─────────────────────────────────────────
        tk.Frame(win, bg=HDR_LIGHT, height=2).pack(fill=tk.X)

        # ── Body ─────────────────────────────────────────────────────
        body = tk.Frame(win, bg=BG, padx=20, pady=14)
        body.pack(fill=tk.BOTH, expand=True)

        # Name
        tk.Label(body, text=name,
                 font=("Segoe UI", 24, "bold"), fg=ACCENT, bg=BG,
                 anchor="w").pack(fill=tk.X)

        # Thin divider
        tk.Frame(body, bg=BG_ROW, height=1).pack(fill=tk.X, pady=(4, 10))

        # ── Info grid ────────────────────────────────────────────────
        grid = tk.Frame(body, bg=BG)
        grid.pack(fill=tk.X)

        def _row(parent_f, label, value, val_col=VALUE_COL, row=0):
            bg = BG_ROW if row % 2 == 0 else BG_CARD
            f = tk.Frame(parent_f, bg=bg)
            f.pack(fill=tk.X, pady=1)
            tk.Label(f, text=f"  {label}", font=("Segoe UI", 9), fg=LABEL_COL,
                     bg=bg, width=16, anchor="w").pack(side=tk.LEFT)
            tk.Label(f, text=value, font=("Segoe UI", 9, "bold"), fg=val_col,
                     bg=bg, anchor="w").pack(side=tk.LEFT, padx=(4, 0))

        r = 0
        _row(grid, "Status",      list_type.upper(),       val_col=ACCENT,    row=r); r += 1
        _row(grid, "Confidence",  f"{confidence:.1%}",     val_col=VALUE_COL, row=r); r += 1
        _row(grid, "Camera",      camera_label,            val_col=VALUE_COL, row=r); r += 1
        _row(grid, "Detected at", timestamp.strftime("%H:%M:%S  |  %d %b %Y"),
             val_col=VALUE_COL, row=r); r += 1

        if emotion:
            _row(grid, "Expression", emotion.capitalize(), val_col=VALUE_COL, row=r); r += 1
        if age is not None:
            age_cat = "Child" if is_child else "Adult"
            _row(grid, "Age (est.)", f"~{age} yrs  —  {age_cat}", val_col=VALUE_COL, row=r); r += 1

        # ── Confidence bar ───────────────────────────────────────────
        tk.Frame(body, bg=BG_ROW, height=1).pack(fill=tk.X, pady=(10, 4))
        bar_frame = tk.Frame(body, bg=BG)
        bar_frame.pack(fill=tk.X)
        tk.Label(bar_frame, text="Confidence", font=("Segoe UI", 8), fg=MUTED, bg=BG,
                 anchor="w").pack(side=tk.LEFT)
        tk.Label(bar_frame, text=f"{confidence:.1%}", font=("Segoe UI", 8, "bold"),
                 fg=ACCENT, bg=BG, anchor="e").pack(side=tk.RIGHT)

        bar_bg = tk.Frame(body, bg="#2a2a40", height=6)
        bar_bg.pack(fill=tk.X, pady=(2, 8))
        bar_fill_w = max(4, int(confidence * (W - 40)))
        tk.Frame(bar_bg, bg=ACCENT, width=bar_fill_w, height=6).place(x=0, y=0)

        # ── Auto-dismiss progress bar ─────────────────────────────────
        tk.Frame(body, bg=BG_ROW, height=1).pack(fill=tk.X, pady=(0, 6))
        prog_label = tk.Label(body, font=("Segoe UI", 8), fg=MUTED, bg=BG, anchor="w")
        prog_label.pack(fill=tk.X)

        prog_bg = tk.Frame(body, bg="#1e1e30", height=4)
        prog_bg.pack(fill=tk.X, pady=(2, 10))
        prog_fill = tk.Frame(prog_bg, bg=MUTED, height=4)
        prog_fill.place(x=0, y=0, relwidth=1.0)

        total_ms   = duration_ms
        start_time = time.monotonic()

        def _tick():
            if not win.winfo_exists():
                return
            elapsed   = (time.monotonic() - start_time) * 1000
            remaining = max(0, total_ms - elapsed)
            frac      = remaining / total_ms
            secs      = int(remaining / 1000) + 1
            prog_label.config(text=f"  Auto-closing in {secs}s")
            prog_fill.place(x=0, y=0, relwidth=frac)
            if remaining > 0:
                win.after(100, _tick)
            else:
                win.destroy()

        _tick()

        # ── Buttons ──────────────────────────────────────────────────
        btn_frame = tk.Frame(body, bg=BG)
        btn_frame.pack(fill=tk.X)

        tk.Button(
            btn_frame, text="Dismiss",
            font=("Segoe UI", 9, "bold"), bg="#2a2a40", fg=VALUE_COL,
            relief=tk.FLAT, padx=20, pady=6, cursor="hand2",
            activebackground="#35355a", activeforeground=WHITE,
            command=win.destroy,
        ).pack(side=tk.RIGHT, padx=(6, 0))

        tk.Button(
            btn_frame, text="Acknowledge",
            font=("Segoe UI", 9, "bold"), bg=HDR_COL, fg=WHITE,
            relief=tk.FLAT, padx=20, pady=6, cursor="hand2",
            activebackground=HDR_LIGHT, activeforeground=WHITE,
            command=win.destroy,
        ).pack(side=tk.RIGHT)

    except Exception as exc:
        log.error("Popup build failed: %s", exc)


# ── Popup Manager — single tkinter thread, queue-driven ──────────────────────

class _PopupManager:
    """
    Owns the single hidden Tk root and its mainloop.
    All popup requests arrive via a Queue and are dispatched with after().
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._root: Optional[tk.Tk] = None
        self._stack_count = 0
        t = threading.Thread(target=self._run, daemon=True, name="tk-popup-mgr")
        t.start()

    def show(self, **kwargs) -> None:
        self._queue.put(kwargs)

    def _run(self) -> None:
        try:
            root = tk.Tk()
            root.withdraw()          # invisible root — never shown to user
            self._root = root
            root.after(50, self._poll)
            root.mainloop()
        except Exception as exc:
            log.error("PopupManager mainloop failed: %s", exc)

    def _poll(self) -> None:
        try:
            while True:
                kwargs = self._queue.get_nowait()
                kwargs["stack_index"] = self._stack_count
                self._stack_count += 1
                _build_popup(self._root, **kwargs)
        except queue.Empty:
            pass
        if self._root and self._root.winfo_exists():
            self._root.after(50, self._poll)


# Module-level singleton
_manager: Optional[_PopupManager] = None
_manager_lock = threading.Lock()


def _get_manager() -> _PopupManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = _PopupManager()
    return _manager


# ── Cooldown tracker ──────────────────────────────────────────────────────────

class _CooldownTracker:
    def __init__(self, cooldown_seconds: int) -> None:
        self._cooldown = cooldown_seconds
        self._last: Dict[Tuple[str, str, str], float] = {}
        self._lock = threading.Lock()

    def is_allowed(self, name: str, list_type: str, camera_label: str) -> bool:
        key = (name, list_type, camera_label)
        with self._lock:
            return (time.monotonic() - self._last.get(key, 0.0)) >= self._cooldown

    def record(self, name: str, list_type: str, camera_label: str) -> None:
        key = (name, list_type, camera_label)
        with self._lock:
            self._last[key] = time.monotonic()


# ── Alerter ───────────────────────────────────────────────────────────────────

class Alerter:
    def __init__(
        self,
        cfg: AlertConfig,
        json_logger: Optional[JsonEventLogger] = None,
        sound_cfg: Optional[SoundConfig] = None,
        email_cfg: Optional[EmailConfig] = None,
    ) -> None:
        self._cfg      = cfg
        self._cooldown = _CooldownTracker(cfg.cooldown_seconds)
        self._json_log = json_logger

        # Sound alarm
        from src.sound_alarm import SoundAlarm
        from src.config import SoundConfig as _SC
        self._sound = SoundAlarm(sound_cfg or _SC())

        # Email notifier
        from src.email_notifier import EmailNotifier
        from src.config import EmailConfig as _EC
        self._email = EmailNotifier(email_cfg or _EC())

        # Unknown visitor cooldown (separate, longer default)
        self._unknown_cooldown = _CooldownTracker(60)

        # Warm up the popup manager (starts the tkinter thread early)
        _get_manager()
        log.info("Alerter ready (cooldown=%ds).", cfg.cooldown_seconds)

    def trigger(
        self,
        result: MatchResult,
        camera_label: str = "Camera",
        screenshot_path: Optional[str] = None,
    ) -> None:
        """
        Fire alert for whitelist and blacklist matches only.
        Unknown visitors are logged but get no popup.
        """
        if not result.is_match:
            return
        if result.list_type not in (LIST_WHITELIST, LIST_BLACKLIST):
            return

        name      = result.name
        list_type = result.list_type

        if not self._cooldown.is_allowed(name, list_type, camera_label):
            return

        self._cooldown.record(name, list_type, camera_label)
        event = AlertEvent(
            name=name,
            confidence=result.confidence,
            list_type=list_type,
            camera_label=camera_label,
            screenshot_path=screenshot_path or "",
        )
        match_log.info(str(event))
        log.info("ALERT: %s", event)

        # Write to JSON event log for dashboard
        if self._json_log is not None:
            self._json_log.log_event(
                name=name,
                list_type=list_type,
                confidence=result.confidence,
                camera_label=camera_label,
                screenshot_path=screenshot_path,
            )

        # Sound alarm (non-blocking)
        if list_type == LIST_BLACKLIST:
            self._sound.alert_blacklist()
        else:
            self._sound.alert_whitelist()

        # Email notification for blacklist only (non-blocking)
        if list_type == LIST_BLACKLIST:
            self._email.send_blacklist_alert(
                name=name,
                confidence=result.confidence,
                camera_label=camera_label,
                timestamp=event.timestamp,
                screenshot_path=screenshot_path,
                age=result.age,
                emotion=result.emotion,
            )

        # Queue the popup (non-blocking — returns immediately)
        _get_manager().show(
            name=name,
            confidence=result.confidence,
            list_type=list_type,
            camera_label=camera_label,
            timestamp=event.timestamp,
            duration_ms=self._cfg.popup_duration_ms,
            age=result.age,
            emotion=result.emotion,
            is_child=result.is_child,
            screenshot_path=screenshot_path,
        )

    def log_unknown_visitor(
        self,
        camera_label: str = "Camera",
        screenshot_path: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Log an unknown (unrecognised) visitor — no popup, just audit trail.
        Throttled by its own 60-second cooldown per camera.
        Set force=True for user-initiated captures (C key) to bypass cooldown.
        """
        key = ("UNKNOWN", LIST_UNKNOWN, camera_label)
        if not force and not self._unknown_cooldown.is_allowed(*key):
            return
        self._unknown_cooldown.record(*key)

        ts  = datetime.now()
        msg = (
            f"{ts.strftime('%Y-%m-%d %H:%M:%S')}  "
            f"{'UNKNOWN':<12}  [{camera_label}]  UNKNOWN VISITOR"
        )
        if screenshot_path:
            msg += f"  screenshot={screenshot_path}"
        match_log.info(msg)
        log.info("UNKNOWN VISITOR detected at [%s]", camera_label)

        if self._json_log is not None:
            self._json_log.log_event(
                name="UNKNOWN VISITOR",
                list_type=LIST_UNKNOWN,
                confidence=0.0,
                camera_label=camera_label,
                screenshot_path=screenshot_path,
            )
