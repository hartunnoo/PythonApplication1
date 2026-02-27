"""
IC (Identity Card) scanner — Tesseract OCR-based number extraction.

Press 'I' while the system is running to scan the current camera frame.

Processing pipeline:
  1. Save raw snapshot to scans/
  2. Upscale 2× + grayscale + CLAHE contrast enhancement
  3. Adaptive threshold → clean black-on-white card image
  4. Tesseract OCR (PSM 11 — sparse text, finds numbers anywhere)
  5. Regex extracts IC/passport number
  6. Append structured record to scans/ic_scans.txt
  7. Write event to logs/events.jsonl for dashboard

Supported number formats (tried in order):
  Malaysian MyKad  : YYMMDD-SS-XXXX  e.g. 901231-10-5821
  12-digit IC      : YYMMDDSSXXXX    e.g. 901231105821
  Brunei IC        : XX-XXXXXXX      e.g. 00-1234567
  Brunei IC alt    : single letter + 6–8 digits  e.g. B123456
  Passport (any)   : 1–2 letters + 6–9 digits    e.g. BN1234567
  Generic fallback : any 7–20 digit run
"""

from __future__ import annotations

import os
import re
import threading
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import ICScanConfig
from src.logger_setup import get_logger
from src.mrz_scanner import MRZResult, MRZScanner

log = get_logger(__name__)

# ── Regex patterns — first match wins ────────────────────────────────────────
_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("MyKad (dashes)",   re.compile(r"\b\d{6}-\d{2}-\d{4}\b")),       # 901231-10-5821
    ("MyKad (compact)",  re.compile(r"\b\d{12}\b")),                   # 901231105821
    ("Brunei IC",        re.compile(r"\b\d{2}[\s\-\.]\d{6,7}\b")),    # 00-288800 (6d) / 00-1234567 (7d)
    ("Brunei IC compact",re.compile(r"\b0\d{7,9}\b")),                 # 00288800 / 002888001 — no separator
    ("Passport / IC",    re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")),       # BN1234567 / B123456
    ("Generic (7–20d)",  re.compile(r"\b\d{7,20}\b")),                 # 7+ digit fallback
]

# ── Front-IC field patterns (Brunei / Malaysian IC) ──────────────────────────
# TARIKH LAHIR  09-08-1981  (dd-mm-yyyy, dd/mm/yyyy, dd.mm.yyyy)
_FRONT_DOB_RE = re.compile(
    r"TARIKH\s+LAHIR[^0-9]{0,12}(\d{2}[\-\/\.]\d{2}[\-\/\.]\d{4})",
    re.IGNORECASE,
)
# JANTINA  LELAKI / PEREMPUAN
_FRONT_SEX_RE = re.compile(
    r"JANTINA[^A-Z]{0,8}(LELAKI|PEREMPUAN)",
    re.IGNORECASE,
)
# NEGERI TEMPAT LAHIR  BRUNEI DARUSSALAM  (max 2-word place name)
_FRONT_POB_RE = re.compile(
    r"TEMPAT\s+LAHIR[^A-Z]{0,12}([A-Z]{3,}(?:\s+[A-Z]{3,})?)",
    re.IGNORECASE,
)

# ── Tesseract configs ─────────────────────────────────────────────────────────
# PSM 11 = sparse text — finds text anywhere, ideal for ID cards
# PSM  6 = uniform block of text — catches structured card fields
# PSM  3 = fully automatic — best general fallback
# OEM  3 = default (LSTM engine)
_TSS_CFGS = [
    r"--oem 3 --psm 11",   # sparse text — finds numbers anywhere
    r"--oem 3 --psm 6",    # uniform block — better for structured card fields
]

_SCAN_COUNTER_LOCK = threading.Lock()
_scan_counter = 0


def _next_scan_number() -> int:
    global _scan_counter
    with _SCAN_COUNTER_LOCK:
        _scan_counter += 1
        return _scan_counter


class ICScanResult:
    def __init__(
        self,
        scan_no: int,
        timestamp: datetime,
        camera_label: str,
        ic_number: Optional[str],
        pattern_name: Optional[str],
        raw_text: str,
        snapshot_path: str,
        name_on_card: Optional[str] = None,
        mrz_result: Optional[MRZResult] = None,
        ic_dob: Optional[str] = None,   # YYYY-MM-DD from front-IC "TARIKH LAHIR"
        ic_sex: Optional[str] = None,   # "M" or "F" from front-IC "JANTINA"
        ic_pob: Optional[str] = None,   # place of birth from "TEMPAT LAHIR"
    ) -> None:
        self.scan_no       = scan_no
        self.timestamp     = timestamp
        self.camera_label  = camera_label
        self.ic_number     = ic_number
        self.pattern_name  = pattern_name
        self.raw_text      = raw_text
        self.snapshot_path = snapshot_path
        self.name_on_card  = name_on_card
        self.mrz_result    = mrz_result
        self.ic_dob        = ic_dob
        self.ic_sex        = ic_sex
        self.ic_pob        = ic_pob
        self.success       = ic_number is not None or mrz_result is not None

        # Effective DOB / sex: prefer MRZ (validated) over front-IC OCR
        self.dob = (mrz_result.dob if mrz_result and mrz_result.dob else ic_dob) or ""
        self.sex = (
            mrz_result.sex
            if mrz_result and mrz_result.sex not in ("<", "", None)
            else ic_sex
        ) or ""

    def __str__(self) -> str:
        status = f"FOUND ({self.pattern_name})" if self.success else "NOT FOUND"
        name   = f"  Name: {self.name_on_card}" if self.name_on_card else ""
        return (
            f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Scan #{self.scan_no:04d}  Camera: {self.camera_label}  "
            f"IC: {self.ic_number or 'N/A'}{name}  Status: {status}"
        )


class ICScanner:
    def __init__(
        self,
        cfg: ICScanConfig,
        json_logger=None,          # Optional[JsonEventLogger]
    ) -> None:
        self._cfg      = cfg
        self._json_log = json_logger

        # Set Tesseract binary path
        try:
            import pytesseract
            if cfg.tesseract_path and os.path.isfile(cfg.tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_path
            self._tess = pytesseract
            ver = pytesseract.get_tesseract_version()
            log.info("ICScanner ready — Tesseract %s | scans → %s/",
                     ver, cfg.scans_dir)
        except Exception as exc:
            self._tess = None
            log.error("ICScanner: Tesseract not available — %s", exc)

        os.makedirs(cfg.scans_dir, exist_ok=True)

        # MRZ scanner — reads ICAO 9303 zones (passports + ID cards)
        self._mrz_scanner = MRZScanner(tesseract_path=cfg.tesseract_path)

    # ── Public API ────────────────────────────────────────────────────

    def scan(
        self,
        frame: np.ndarray,
        camera_label: str = "Camera",
    ) -> ICScanResult:
        """
        Scan the frame for an IC number.
        Saves snapshot, runs OCR, writes to log.
        Always returns an ICScanResult (success=False if not found).
        """
        scan_no = _next_scan_number()
        ts      = datetime.now()

        # 1. Save raw snapshot
        snapshot_path = self._save_snapshot(frame, ts, camera_label, scan_no)

        if self._tess is None:
            result = ICScanResult(scan_no, ts, camera_label, None,
                                  None, "", snapshot_path)
            log.warning("ICScanner: Tesseract not available — scan skipped.")
            return result

        # 2. MRZ scan (runs first — most reliable for modern IC/passports)
        mrz_result = self._mrz_scanner.scan(frame)
        if mrz_result is not None:
            log.info("MRZ decoded: %s", mrz_result)
        else:
            log.debug("MRZ: no valid zone found — falling back to regex OCR")

        # 3. Regular OCR for IC number / name (fallback when no MRZ zone)
        raw_text = self._ocr(frame)
        ic_number, pattern_name = self._extract(raw_text)
        name_on_card            = self._extract_name(raw_text)

        # 3a. Extract front-IC structured fields (DOB, sex, place of birth)
        ic_dob, ic_sex, ic_pob = self._extract_front_fields(raw_text)
        if ic_dob:
            log.info("Front-IC DOB: %s  Sex: %s  POB: %s", ic_dob, ic_sex, ic_pob)

        # 3b. If dedicated MRZ scan failed, try parsing MRZ from regular OCR text.
        if mrz_result is None:
            mrz_result = self._mrz_scanner.scan_from_text(raw_text)
            if mrz_result is not None:
                log.info("MRZ decoded from OCR text: %s", mrz_result)

        # 4. MRZ data (check-digit validated) always wins over regex-matched values.
        #    This prevents MRZ digit sequences from being mis-labelled as IC numbers.
        if mrz_result is not None:
            if mrz_result.doc_number:
                ic_number    = mrz_result.doc_number
                pattern_name = f"MRZ {mrz_result.format}"
            if mrz_result.full_name:
                name_on_card = mrz_result.full_name

        result = ICScanResult(
            scan_no, ts, camera_label, ic_number,
            pattern_name, raw_text, snapshot_path,
            name_on_card=name_on_card,
            mrz_result=mrz_result,
            ic_dob=ic_dob,
            ic_sex=ic_sex,
            ic_pob=ic_pob,
        )

        # 5. Write to audit log
        self._write_log(result)

        # 6. Write unified IC event to events.jsonl for dashboard
        if self._json_log is not None:
            self._json_log.log_ic_scan_event(
                name=name_on_card or ic_number or "SCAN_NO_MATCH",
                doc_number=ic_number or "",
                camera_label=camera_label,
                snapshot_path=snapshot_path,
                ic_dob=result.dob,
                ic_sex=result.sex,
                ic_pob=ic_pob or "",
                ic_nationality=mrz_result.nationality if mrz_result else "",
                ic_country=mrz_result.country if mrz_result else "",
                ic_expiry=mrz_result.expiry if mrz_result else "",
                ic_mrz_format=mrz_result.format if mrz_result else "",
                ic_checks_ok=mrz_result.check_digits_ok if mrz_result else False,
            )

        if result.success:
            log.info("IC SCAN: %s", result)
        else:
            log.warning("IC SCAN: No IC number found — show MRZ zone or try better angle/lighting.")

        return result

    # ── Front-IC field extraction ─────────────────────────────────────

    @staticmethod
    def _extract_front_fields(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract DOB, sex, and place-of-birth from front-IC OCR text.
        Returns (dob_iso, sex_code, pob) — all may be None if not found.
          dob_iso  : "YYYY-MM-DD"
          sex_code : "M" or "F"
          pob      : e.g. "BRUNEI DARUSSALAM"
        """
        dob = sex = pob = None

        m = _FRONT_DOB_RE.search(text)
        if m:
            raw = m.group(1)
            parts = re.split(r"[\-\/\.]", raw)
            if len(parts) == 3 and len(parts[2]) == 4:
                dob = f"{parts[2]}-{parts[1]}-{parts[0]}"

        m = _FRONT_SEX_RE.search(text)
        if m:
            sex = "M" if m.group(1).upper() == "LELAKI" else "F"

        m = _FRONT_POB_RE.search(text)
        if m:
            pob = m.group(1).strip()

        return dob, sex, pob

    # ── Image preprocessing ───────────────────────────────────────────

    @staticmethod
    def _preprocess_variants(frame: np.ndarray) -> List[np.ndarray]:
        """
        Return preprocessing variants to maximise OCR hit-rate:
          1. Adaptive threshold         — handles uneven lighting (most cards)
          2. Adaptive threshold INVERTED — light/embossed text on gold/dark surface
          3. Otsu global threshold      — uniform backgrounds (laminated cards)
          4. Sharpened + adaptive       — recovers detail from blurry captures
        All variants are 3× upscaled with CLAHE contrast enhancement first.
        """
        h, w = frame.shape[:2]
        img  = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

        # Variant 1: adaptive — dark text on light/patterned background
        v1 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10,
        )
        v1 = cv2.morphologyEx(v1, cv2.MORPH_CLOSE, kernel)

        # Variant 2: inverted adaptive — light/embossed text on gold/dark background
        v2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10,
        )
        v2 = cv2.morphologyEx(v2, cv2.MORPH_CLOSE, kernel)

        # Variant 3: Otsu — uniform background (laminated cards, plain white ICs)
        _, v3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Variant 4: sharpen then adaptive — recovers blurry card text
        sharpen_kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]], dtype=np.float32)
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        v4 = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8,
        )
        v4 = cv2.morphologyEx(v4, cv2.MORPH_CLOSE, kernel)

        return [v1, v2, v3, v4]

    # ── OCR ──────────────────────────────────────────────────────────

    def _ocr(self, frame: np.ndarray) -> str:
        """
        Run Tesseract on all preprocessing variants × all PSM configs.
        Combines unique results to maximise number extraction coverage.
        """
        variants = self._preprocess_variants(frame)
        combined = set()
        for img in variants:
            for cfg in _TSS_CFGS:
                try:
                    raw  = self._tess.image_to_string(img, config=cfg)
                    text = re.sub(r"[^\w\s\-]", " ", raw)
                    text = re.sub(r"\s+", " ", text).strip().upper()
                    if text:
                        combined.add(text)
                except Exception as exc:
                    log.debug("OCR variant error: %s", exc)

        merged = " ".join(combined)
        log.info("OCR raw text: %s", merged[:400])
        return merged

    # ── Number extraction ─────────────────────────────────────────────

    @staticmethod
    def _extract(text: str) -> Tuple[Optional[str], Optional[str]]:
        """Try each pattern and return (ic_number, pattern_name) or (None, None)."""
        for name, pattern in _PATTERNS:
            m = pattern.search(text)
            if m:
                return m.group(0), name
        return None, None

    @staticmethod
    def _extract_name(text: str) -> Optional[str]:
        """
        Extract a person's full name from OCR text.
        Handles Brunei/Malay IC format: 'FIRSTNAME ... BIN/BINTE/BINTI SURNAME ...'
        Falls back to any sequence of 3+ capitalised words.
        """
        # Malay name with BIN / BINTE / BINTI keyword
        m = re.search(
            r'\b([A-Z]{2,}(?:\s+[A-Z]{2,}){0,5})\s+(BINTE?I?|BIN)\s+([A-Z]{2,}(?:\s+[A-Z]{2,}){0,3})\b',
            text,
        )
        if m:
            return m.group(0).strip()
        # Fallback: longest run of 3+ all-caps words (≥ 8 chars total)
        candidates = re.findall(r'\b(?:[A-Z]{2,}\s+){2,}[A-Z]{2,}\b', text)
        if candidates:
            return max(candidates, key=len).strip()
        return None

    # ── Snapshot ──────────────────────────────────────────────────────

    def _save_snapshot(
        self,
        frame: np.ndarray,
        ts: datetime,
        camera_label: str,
        scan_no: int,
    ) -> str:
        try:
            cam_safe = camera_label.replace(" ", "_")
            filename = f"scan_{ts.strftime('%Y%m%d_%H%M%S')}_{cam_safe}_{scan_no:04d}.jpg"
            path     = os.path.join(self._cfg.scans_dir, filename)
            snap     = frame.copy()

            # Overlay banner
            cv2.rectangle(snap, (0, 0), (snap.shape[1], 34), (10, 10, 30), cv2.FILLED)
            cv2.putText(snap, f"  IC SCAN #{scan_no:04d}  |  {ts.strftime('%Y-%m-%d %H:%M:%S')}  |  [{camera_label}]",
                        (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 200, 255), 1, cv2.LINE_AA)

            cv2.imwrite(path, snap)
            return path
        except Exception as exc:
            log.warning("Snapshot save failed: %s", exc)
            return ""

    # ── Audit log ─────────────────────────────────────────────────────

    def _write_log(self, r: ICScanResult) -> None:
        """Append a structured record to scans/ic_scans.txt."""
        try:
            log_path = self._cfg.scans_log
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

            sep    = "=" * 72
            status = f"FOUND  —  {r.pattern_name}" if r.success else "NOT FOUND"
            lines  = [
                sep,
                f"SCAN #{r.scan_no:04d}",
                f"  Date/Time   : {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                f"  Camera      : {r.camera_label}",
                f"  IC Number   : {r.ic_number or 'N/A'}",
                f"  Name        : {r.name_on_card or 'N/A'}",
                f"  Status      : {status}",
            ]

            # MRZ section — only when MRZ was successfully decoded
            if r.mrz_result is not None:
                lines.append("  ── MRZ Data ─────────────────────────────────")
                lines.extend(r.mrz_result.summary_lines())
                lines.append("  MRZ Raw:")
                for mrz_line in r.mrz_result.raw_lines:
                    lines.append(f"    {mrz_line}")

            lines += [
                f"  Raw OCR     : {r.raw_text[:120]}{'...' if len(r.raw_text) > 120 else ''}",
                f"  Snapshot    : {r.snapshot_path}",
                sep,
                "",
            ]
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")

        except Exception as exc:
            log.warning("IC scan log write failed: %s", exc)
