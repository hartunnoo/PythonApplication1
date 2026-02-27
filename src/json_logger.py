"""
JSON Lines event logger — ASP.NET Core dashboard integration.

Appends one JSON object per detection event to a .jsonl file.
Each line is a valid JSON object for easy consumption by .NET's
System.Text.Json / Newtonsoft.Json.

File location (default): logs/events.jsonl

Example record:
  {
    "timestamp": "2026-02-26T14:30:00",
    "event": "detected",
    "camera": "Entrance",
    "name": "Tuno",
    "list_type": "whitelist",
    "confidence": 0.8731,
    "screenshot": "screenshots/whitelist_Tuno_20260226_143000_Entrance.jpg"
  }
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Optional


class JsonEventLogger:
    """Thread-safe JSONL writer for face detection events."""

    def __init__(self, path: str) -> None:
        dir_part = os.path.dirname(path)
        if dir_part:
            os.makedirs(dir_part, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()

    def log_event(
        self,
        name: str,
        list_type: str,
        confidence: float,
        camera_label: str,
        screenshot_path: Optional[str] = None,
        event: str = "detected",
    ) -> None:
        """Append one event record to the JSONL file."""
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "camera": camera_label,
            "name": name,
            "list_type": list_type,
            "confidence": round(confidence, 4),
            "screenshot": (screenshot_path or "").replace("\\", "/"),
        }
        try:
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
        except OSError:
            pass  # Non-fatal — don't crash the detection pipeline

    def log_ic_scan_event(
        self,
        name: str,
        doc_number: str,
        camera_label: str,
        snapshot_path: str = "",
        ic_dob: str = "",
        ic_sex: str = "",
        ic_pob: str = "",
        ic_nationality: str = "",
        ic_country: str = "",
        ic_expiry: str = "",
        ic_mrz_format: str = "",
        ic_checks_ok: bool = False,
    ) -> None:
        """Append a unified IC scan record covering front-IC and MRZ fields."""
        record = {
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "event":          "ic_scan",
            "camera":         camera_label,
            "name":           name,
            "list_type":      "ic_scan",
            "confidence":     1.0 if doc_number else 0.0,
            "screenshot":     snapshot_path.replace("\\", "/"),
            "ic_doc_number":  doc_number,
            "ic_dob":         ic_dob,
            "ic_sex":         ic_sex,
            "ic_pob":         ic_pob,
            "ic_nationality": ic_nationality,
            "ic_country":     ic_country,
            "ic_expiry":      ic_expiry,
            "ic_mrz_format":  ic_mrz_format,
            "ic_checks_ok":   ic_checks_ok,
        }
        try:
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
        except OSError:
            pass
