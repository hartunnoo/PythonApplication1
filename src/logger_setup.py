"""
Logging configuration.

Two handlers per logger:
  - Console: human-readable, coloured level prefix
  - File (rotating): structured text, auto-rotated by size

A separate "match" logger writes every confirmed match event
to its own file for easy auditing.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

from src.config import LoggingConfig


# ---------------------------------------------------------------------------
# Colour formatter (console only)
# ---------------------------------------------------------------------------

_LEVEL_COLOURS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
}
_RESET = "\033[0m"


class _ColourFormatter(logging.Formatter):
    FMT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname}{_RESET}"
        return super().format(record)

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT)


class _PlainFormatter(logging.Formatter):
    FMT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logging(cfg: LoggingConfig) -> None:
    """
    Configure root logger and the dedicated 'match' logger.
    Call once at application startup.
    """
    os.makedirs(cfg.log_dir, exist_ok=True)

    numeric_level = getattr(logging, cfg.level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Avoid adding handlers if already configured (e.g., during tests)
    if root.handlers:
        return

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(_ColourFormatter())
    root.addHandler(console_handler)

    # --- Rotating file handler (main log) ---
    main_log_path = os.path.join(cfg.log_dir, cfg.log_file)
    file_handler = RotatingFileHandler(
        main_log_path,
        maxBytes=cfg.max_bytes,
        backupCount=cfg.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(_PlainFormatter())
    root.addHandler(file_handler)

    # --- Dedicated match event logger ---
    match_logger = logging.getLogger("match")
    match_logger.propagate = False  # Don't double-write to root handlers

    match_log_path = os.path.join(cfg.log_dir, cfg.match_log_file)
    match_handler = RotatingFileHandler(
        match_log_path,
        maxBytes=cfg.max_bytes,
        backupCount=cfg.backup_count,
        encoding="utf-8",
    )
    match_handler.setLevel(logging.INFO)
    match_handler.setFormatter(_PlainFormatter())

    match_console = logging.StreamHandler()
    match_console.setLevel(logging.INFO)
    match_console.setFormatter(_ColourFormatter())

    match_logger.addHandler(match_handler)
    match_logger.addHandler(match_console)
    match_logger.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (child of root)."""
    return logging.getLogger(name)


def get_match_logger() -> logging.Logger:
    """Return the dedicated match-event logger."""
    return logging.getLogger("match")
