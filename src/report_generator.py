"""
Daily CSV report generator.

Runs as a background daemon thread — exports events.jsonl to a dated
CSV file once per day at midnight (configurable time).

Manual export: python main.py --export-csv

CSV columns:
  timestamp, camera, name, list_type, confidence, age, emotion, screenshot
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from datetime import datetime, date
from typing import Optional

from src.config import ReportConfig
from src.logger_setup import get_logger

log = get_logger(__name__)


class ReportGenerator:
    def __init__(self, cfg: ReportConfig, events_jsonl_path: str) -> None:
        self._cfg        = cfg
        self._jsonl_path = events_jsonl_path
        self._running    = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self._cfg.enabled:
            return
        os.makedirs(self._cfg.reports_dir, exist_ok=True)
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="report-gen"
        )
        self._thread.start()
        log.info("ReportGenerator started — daily CSV at %s → %s/",
                 self._cfg.export_time, self._cfg.reports_dir)

    def stop(self) -> None:
        self._running = False

    def export_now(self, target_date: Optional[date] = None) -> Optional[str]:
        """Export all events for target_date (default: today) to CSV.
        Returns the output path or None on failure."""
        if target_date is None:
            target_date = date.today()
        return self._export(target_date)

    # ── Private ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        """Sleep until the configured export time each day, then export."""
        last_exported: Optional[date] = None

        while self._running:
            now       = datetime.now()
            today     = now.date()
            h, m      = map(int, self._cfg.export_time.split(":"))
            export_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)

            # If today's export time has passed and we haven't exported yet today
            if now >= export_dt and last_exported != today:
                # Export yesterday's complete data
                yesterday = date.fromordinal(today.toordinal() - 1)
                path = self._export(yesterday)
                if path:
                    last_exported = today

            time.sleep(60)   # Check every minute

    def _export(self, target_date: date) -> Optional[str]:
        """Read events.jsonl, filter by date, write CSV."""
        try:
            os.makedirs(self._cfg.reports_dir, exist_ok=True)
            out_path = os.path.join(
                self._cfg.reports_dir,
                f"report_{target_date.strftime('%Y-%m-%d')}.csv",
            )

            rows = []
            if os.path.isfile(self._jsonl_path):
                with open(self._jsonl_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            ts  = rec.get("timestamp", "")
                            if ts.startswith(target_date.isoformat()):
                                rows.append(rec)
                        except json.JSONDecodeError:
                            continue

            fieldnames = [
                "timestamp", "camera", "name", "list_type",
                "confidence", "age", "emotion", "screenshot",
            ]
            with open(out_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                        extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    # Normalize confidence to percentage string
                    row["confidence"] = f"{float(row.get('confidence', 0)):.1%}"
                    writer.writerow(row)

            log.info(
                "CSV report exported: %s (%d event(s))",
                out_path, len(rows),
            )
            return out_path

        except Exception as exc:
            log.warning("CSV export failed: %s", exc)
            return None
