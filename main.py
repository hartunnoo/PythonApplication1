"""
Entry point — Face Recognition System.

Run:
    python main.py                  Start the system normally
    python main.py --rebuild-db     Force rebuild face embeddings from photos
    python main.py --config PATH    Use a custom YAML config file
    python main.py --scan-only      Rebuild DB and exit (no camera)
"""

from __future__ import annotations

import argparse
import os
import sys

# ── DPI awareness (Windows) ────────────────────────────────────────────────
# On high-DPI displays (125%+), Windows silently upscales OpenCV windows
# using blurry nearest-neighbor interpolation.  Declaring DPI awareness
# tells Windows to hand us native-resolution pixels so text/lines stay
# crisp on 2K/4K screens.  Must be called before any window creation.
def _enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        # PROCESS_PER_MONITOR_DPI_AWARE = 2  (best — per-monitor scaling)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            # Fallback for older Windows (8.0)
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass

_enable_dpi_awareness()

# Fix tkinter/tcl path for Python installs where the venv doesn't inherit
# the TCL_LIBRARY / TK_LIBRARY environment variables automatically.
def _fix_tcl_paths() -> None:
    if os.environ.get("TCL_LIBRARY"):
        return   # already set
    # sys.base_prefix is the real Python install dir (not the venv)
    for prefix in (sys.base_prefix, sys.prefix):
        tcl_dir = os.path.join(prefix, "tcl")
        if not os.path.isdir(tcl_dir):
            continue
        for entry in os.listdir(tcl_dir):
            full = os.path.join(tcl_dir, entry)
            if entry.startswith("tcl8") and os.path.isdir(full):
                if os.path.isfile(os.path.join(full, "init.tcl")):
                    os.environ.setdefault("TCL_LIBRARY", full)
            elif entry.startswith("tk8") and os.path.isdir(full):
                os.environ.setdefault("TK_LIBRARY", full)
        if os.environ.get("TCL_LIBRARY"):
            break   # found it

_fix_tcl_paths()

from src.config import load_config
from src.logger_setup import setup_logging, get_logger
from src.app import FaceRecognitionApp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time face detection and recognition system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        metavar="PATH",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--rebuild-db",
        action="store_true",
        help="Delete cached embeddings and rebuild the face database from photos.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Rebuild the face database and exit without starting the camera.",
    )
    parser.add_argument(
        "--export-csv",
        metavar="DATE",
        nargs="?",
        const="today",
        help="Export events to CSV and exit. DATE = YYYY-MM-DD (default: today).",
    )
    return parser.parse_args()


def _rebuild_cache(cfg, log) -> None:
    """Delete old deepface embedding caches so they are rebuilt from scratch."""
    import glob as _glob
    cache_dir = cfg.paths.cache_dir
    targets   = [
        os.path.join(cache_dir, "deepface_whitelist_embeddings.pkl"),
        os.path.join(cache_dir, "deepface_blacklist_embeddings.pkl"),
        os.path.join(cache_dir, "deepface_whitelist_meta.txt"),
        os.path.join(cache_dir, "deepface_blacklist_meta.txt"),
    ]
    deleted = 0
    for path in targets:
        for f in _glob.glob(path):
            os.remove(f)
            log.info("  Deleted cache: %s", f)
            deleted += 1
    if deleted == 0:
        log.info("  No cache files found — will build fresh from photos.")


def main() -> None:
    args = _parse_args()

    # 1. Load configuration
    cfg = load_config(args.config)

    # 2. Set up logging (must be first before any log.xxx calls)
    setup_logging(cfg.logging)
    log = get_logger("main")

    log.info("=" * 60)
    log.info("Face Recognition System — starting up")
    log.info("Config: %s", args.config)
    log.info("=" * 60)

    # 3. Optional: rebuild cache
    if args.rebuild_db or args.scan_only:
        log.info("Clearing embedding cache before rebuild...")
        _rebuild_cache(cfg, log)

    # 4. CSV export mode
    if args.export_csv is not None:
        from src.report_generator import ReportGenerator
        import datetime as _dt
        json_log_path = os.path.join(cfg.logging.log_dir, "events.jsonl")
        reporter = ReportGenerator(cfg.report, json_log_path)
        if args.export_csv == "today":
            target = _dt.date.today()
        else:
            target = _dt.date.fromisoformat(args.export_csv)
        path = reporter.export_now(target)
        if path:
            log.info("Exported: %s", path)
        sys.exit(0)

    # 5. Scan-only mode: build DB and exit
    if args.scan_only:
        from src.face_db import load_face_database
        log.info("Scanning all face photos — this may take a moment...")
        db = load_face_database(
            whitelist_dir=cfg.paths.whitelist_dir,
            blacklist_dir=cfg.paths.blacklist_dir,
            cache_dir=cfg.paths.cache_dir,
            deepface_model=cfg.deepface.model,
        )
        log.info("Scan complete. Whitelist: %d person(s) / %d sample(s) | Blacklist: %d person(s) / %d sample(s)",
                 db.whitelist.person_count, db.whitelist.sample_count,
                 db.blacklist.person_count, db.blacklist.sample_count)
        sys.exit(0)

    # 5. Launch application
    app = FaceRecognitionApp(cfg)
    app.run()

    log.info("Goodbye.")
    sys.exit(0)


if __name__ == "__main__":
    main()
