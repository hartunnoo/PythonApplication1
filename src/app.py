"""
Multi-camera application orchestrator — whitelist/blacklist edition.

One CameraWorker thread per camera.
Main thread composites all feeds into a tiled grid window.

Keyboard shortcuts:
  Q / ESC  — quit
  P        — pause / resume
  R        — hot-reload both face lists from disk
  I        — IC scan (OCR identity card in current frame)
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from typing import List, Optional

from src.alerter import Alerter
from src.config import AppConfig
from src.dashboard import DashboardServer
from src.display import create_window, destroy_windows, make_grid, read_key, show_frame
from src.face_db import FaceDatabaseManager, load_face_database
from src.ic_scanner import ICScanner
from src.json_logger import JsonEventLogger
from src.logger_setup import get_logger
from src.matcher import FaceMatcher
from src.report_generator import ReportGenerator

log = get_logger(__name__)

_KEY_QUIT_Q   = ord("q")
_KEY_QUIT_ESC = 27
_KEY_PAUSE    = ord("p")
_KEY_RELOAD   = ord("r")
_KEY_SCAN     = ord("i")
_KEY_CAPTURE  = ord("c")


class FaceRecognitionApp:
    def __init__(self, cfg: AppConfig) -> None:
        self._cfg     = cfg
        self._running = False
        self._paused  = False

        log.info(
            "Initialising Face Recognition System v2.0 — %d camera(s)",
            len(cfg.cameras),
        )
        log.info("Backend: OpenCV Haar Cascade (detection) + deepface %s (recognition)", cfg.deepface.model)

        db = load_face_database(
            cfg.paths.whitelist_dir,
            cfg.paths.blacklist_dir,
            cfg.paths.cache_dir,
            deepface_model=cfg.deepface.model,
        )
        self._db_manager = FaceDatabaseManager(db)

        self._matcher = FaceMatcher(
            threshold=cfg.deepface.threshold,
            model_name=cfg.deepface.model,
        )

        # JSON event logger — writes to logs/events.jsonl for dashboard
        json_log_path   = os.path.join(cfg.logging.log_dir, "events.jsonl")
        self._json_log  = JsonEventLogger(json_log_path)
        self._alerter   = Alerter(
            cfg.alerts,
            json_logger=self._json_log,
            sound_cfg=cfg.sound,
            email_cfg=cfg.email,
        )

        # Dashboard web server
        self._dashboard = DashboardServer(
            cfg=cfg.dashboard,
            events_jsonl_path=json_log_path,
            screenshots_dir=cfg.paths.screenshots_dir,
            db_manager=self._db_manager,
            matcher=self._matcher,
            paths_cfg=cfg.paths,
        )
        self._dashboard.set_camera_count(len(cfg.cameras))
        self._dashboard.start()

        # Daily CSV report generator
        self._reporter = ReportGenerator(cfg.report, json_log_path)
        self._reporter.start()

        # IC scanner (Tesseract OCR — press I to scan)
        self._ic_scanner: Optional[ICScanner] = None
        if cfg.ic_scan.enabled:
            self._ic_scanner = ICScanner(cfg.ic_scan, json_logger=self._json_log)

        log.info("JSON event log: %s", json_log_path)

        log.info(
            "  Whitelist: %d person(s) | Blacklist: %d person(s)",
            db.whitelist.person_count,
            db.blacklist.person_count,
        )
        log.info(
            "  deepface threshold: %.2f | min_confidence: %.0f%%",
            cfg.deepface.threshold,
            cfg.matching.min_confidence * 100,
        )
        for i, cam in enumerate(cfg.cameras):
            log.info("  Camera %d: [%s] device=%d", i, cam.label, cam.device_index)

    # ------------------------------------------------------------------

    def run(self) -> None:
        self._running = True
        self._register_signals()

        from src.camera_worker import CameraWorker

        workers: List[CameraWorker] = []
        try:
            for cam_cfg in self._cfg.cameras:
                w = CameraWorker(
                    cam_cfg=cam_cfg,
                    det_cfg=self._cfg.detection,
                    mat_cfg=self._cfg.matching,
                    db_manager=self._db_manager,
                    matcher=self._matcher,
                    alerter=self._alerter,
                    display_cfg=self._cfg.display,
                    screenshots_dir=self._cfg.paths.screenshots_dir,
                    unknown_faces_dir=self._cfg.paths.unknown_faces_dir,
                )
                w.start()
                workers.append(w)

            create_window(self._cfg.display.window_title)
            self._grid_loop(workers)

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt.")
        finally:
            self._shutdown(workers)

    # ------------------------------------------------------------------

    def _grid_loop(self, workers) -> None:
        cfg = self._cfg
        log.info(
            "Grid display running. Q/ESC=quit | P=pause | R=reload DB | I=IC scan | C=capture unknown"
        )

        import time as _time
        _last_capture_press = 0.0   # debounce: ignore rapid key-repeat events

        while self._running:
            key = read_key(1)
            if key in (_KEY_QUIT_Q, _KEY_QUIT_ESC):
                self._running = False
                break
            elif key == _KEY_PAUSE:
                self._paused = not self._paused
                for w in workers:
                    w.paused = self._paused
                log.info("Capture %s.", "paused" if self._paused else "resumed")
            elif key == _KEY_RELOAD:
                self._db_manager.reload(
                    cfg.paths.whitelist_dir,
                    cfg.paths.blacklist_dir,
                    cfg.paths.cache_dir,
                )
            elif key == _KEY_SCAN and self._ic_scanner is not None:
                # Scan IC in a background thread so OCR never blocks the main loop
                for w in workers:
                    frame = w.latest_raw_frame
                    if frame is not None:
                        label = w.label
                        threading.Thread(
                            target=self._ic_scanner.scan,
                            args=(frame, label),
                            daemon=True,
                            name="ic-scan",
                        ).start()
                        break
            elif key == _KEY_CAPTURE:
                # Toggle capture-mode — debounced to 0.5 s to suppress key-repeat
                now = _time.monotonic()
                if now - _last_capture_press >= 0.5:
                    _last_capture_press = now
                    for w in workers:
                        w.toggle_capture_mode()
                        break

            frames = [w.latest_frame for w in workers]
            grid   = make_grid(
                frames,
                cell_w=cfg.display.grid_cell_width,
                cell_h=cfg.display.grid_cell_height,
            )
            show_frame(cfg.display.window_title, grid)

    def _shutdown(self, workers) -> None:
        log.info("Shutting down %d worker(s)...", len(workers))
        self._running = False
        self._reporter.stop()
        for w in workers:
            w.stop()
        for w in workers:
            w.join(timeout=3.0)
        destroy_windows()
        log.info("Clean shutdown.")

    def _register_signals(self) -> None:
        def _h(sig, _): self._running = False
        signal.signal(signal.SIGINT, _h)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _h)
