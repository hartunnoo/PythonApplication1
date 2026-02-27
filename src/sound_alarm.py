"""
Sound alarm — plays beep patterns for blacklist/whitelist detections.

Uses winsound.Beep on Windows (built-in, no extra dependencies).
Runs in a daemon thread so detection is never blocked.

Blacklist : three rapid high-low beeps (urgent)
Whitelist : one soft confirmation beep (optional)
"""

from __future__ import annotations

import threading

from src.config import SoundConfig
from src.logger_setup import get_logger

log = get_logger(__name__)


class SoundAlarm:
    def __init__(self, cfg: SoundConfig) -> None:
        self._cfg = cfg
        if cfg.enabled:
            log.info("SoundAlarm ready (blacklist=%s, whitelist=%s).",
                     "on", "on" if cfg.whitelist_beep else "off")

    def alert_blacklist(self) -> None:
        if not self._cfg.enabled:
            return
        threading.Thread(target=self._play_blacklist, daemon=True,
                         name="alarm-bl").start()

    def alert_whitelist(self) -> None:
        if not self._cfg.enabled or not self._cfg.whitelist_beep:
            return
        threading.Thread(target=self._play_whitelist, daemon=True,
                         name="alarm-wl").start()

    # ── Private ───────────────────────────────────────────────────────

    def _play_blacklist(self) -> None:
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(1400, 180)
                winsound.Beep(700,  180)
        except Exception as exc:
            log.debug("Sound alarm failed: %s", exc)

    def _play_whitelist(self) -> None:
        try:
            import winsound
            winsound.Beep(880, 250)
        except Exception as exc:
            log.debug("Sound alarm failed: %s", exc)
