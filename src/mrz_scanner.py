"""
MRZ (Machine Readable Zone) scanner — ICAO 9303 compliant.

Supports:
  TD1 — 3 lines × 30 chars  (ID cards: Brunei IC, EU national ID, etc.)
  TD3 — 2 lines × 44 chars  (passports)

Processing pipeline:
  1. Four frame regions tried (full + progressively smaller bottom crops)
  2. Each region: 4× upscale + CLAHE + adaptive / Otsu threshold
  3. Tesseract OCR with A-Z/0-9/< char whitelist (PSM 6 + PSM 11)
  4. Regex to locate and normalise MRZ line groups
  5. ICAO 9303 field decode + check-digit validation
  6. Fallback: pattern-match line 2 alone to extract DOB/sex/expiry/nat

Typical results for Brunei TD1 IC:
  Line 1 : I<BRN00288800<<<<<<<<<<<<<<<<
  Line 2 : 8108094M3112180BRN<<<<<<<<<<<6
  Line 3 : MOHDSOUD<<AHMMAD<SUHARDDY<<<<
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.logger_setup import get_logger

log = get_logger(__name__)

# ── Tesseract configs ─────────────────────────────────────────────────────────
_MRZ_WL  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
_MRZ_CFGS = [
    f"--oem 3 --psm 6 -c tessedit_char_whitelist={_MRZ_WL}",    # uniform block
    f"--oem 3 --psm 11 -c tessedit_char_whitelist={_MRZ_WL}",   # sparse text
]

# ── Line lengths ──────────────────────────────────────────────────────────────
_TD1_LEN = 30
_TD3_LEN = 44

# ── ICAO 9303 check-digit table ───────────────────────────────────────────────
_WEIGHTS  = [7, 3, 1]
_CHAR_VAL: dict = {c: i for i, c in enumerate("0123456789")}
_CHAR_VAL.update({c: 10 + i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})
_CHAR_VAL["<"] = 0


def _check_digit(data: str) -> str:
    total = sum(_CHAR_VAL.get(c, 0) * _WEIGHTS[i % 3] for i, c in enumerate(data))
    return str(total % 10)


def _parse_date(yymmdd: str) -> str:
    """YYMMDD → YYYY-MM-DD using the ICAO 9303 ±50-year window from today.
    Example (current year 2026): cutoff = 76 → yy ≤ 76 → 2000s, yy > 76 → 1900s.
    This correctly maps DOB 81 → 1981 and Expiry 31 → 2031."""
    import datetime
    try:
        yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
        cutoff = (datetime.date.today().year % 100 + 50) % 100
        year = (2000 + yy) if yy <= cutoff else (1900 + yy)
        return f"{year:04d}-{mm:02d}-{dd:02d}"
    except Exception:
        return yymmdd


def _decode_name(name_field: str) -> Tuple[str, str]:
    """'SURNAME<<GIVEN<NAMES' → (surname, given_names)."""
    parts = name_field.split("<<", 1)
    surname = parts[0].replace("<", " ").strip()
    given   = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
    return surname, given


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MRZResult:
    format:          str        # "TD1" or "TD3"
    raw_lines:       List[str]
    doc_type:        str  = ""
    country:         str  = ""  # issuing state (3-letter ICAO code)
    doc_number:      str  = ""
    surname:         str  = ""
    given_names:     str  = ""
    nationality:     str  = ""
    dob:             str  = ""  # YYYY-MM-DD
    sex:             str  = ""  # M / F / <
    expiry:          str  = ""  # YYYY-MM-DD
    personal_number: str  = ""
    check_digits_ok: bool = False

    @property
    def full_name(self) -> str:
        return " ".join(p for p in [self.given_names, self.surname] if p).strip()

    def summary_lines(self) -> List[str]:
        """Human-readable lines for scan logs and dashboard display."""
        checks = "OK" if self.check_digits_ok else "FAIL"
        lines = [
            f"  MRZ Format  : {self.format}",
            f"  Doc Type    : {self.doc_type}",
            f"  Country     : {self.country}",
            f"  Doc Number  : {self.doc_number}",
            f"  Full Name   : {self.full_name}",
            f"  DOB         : {self.dob}",
            f"  Sex         : {self.sex}",
            f"  Nationality : {self.nationality}",
            f"  Expiry      : {self.expiry}",
        ]
        if self.personal_number:
            lines.append(f"  Personal No.: {self.personal_number}")
        lines.append(f"  Check Digits: {checks}")
        return lines

    def __str__(self) -> str:
        checks = "✓" if self.check_digits_ok else "✗"
        return (
            f"MRZ[{self.format}] {self.doc_type}/{self.country} "
            f"Doc:{self.doc_number} Name:{self.full_name} "
            f"DOB:{self.dob} Exp:{self.expiry} "
            f"Sex:{self.sex} Nat:{self.nationality} Checks:{checks}"
        )


# ── Field decoders ────────────────────────────────────────────────────────────

def decode_td1(lines: List[str]) -> MRZResult:
    """Decode 3-line TD1 MRZ (ID card)."""
    l1 = lines[0].ljust(_TD1_LEN, "<")[:_TD1_LEN]
    l2 = lines[1].ljust(_TD1_LEN, "<")[:_TD1_LEN]
    l3 = lines[2].ljust(_TD1_LEN, "<")[:_TD1_LEN]

    doc_type    = l1[0:2].replace("<", "").strip()
    country     = l1[2:5].replace("<", "").strip()
    doc_raw     = l1[5:14]
    doc_cd      = l1[14]

    dob_raw     = l2[0:6]
    dob_cd      = l2[6]
    sex         = l2[7] if l2[7] in "MF<" else "<"
    exp_raw     = l2[8:14]
    exp_cd      = l2[14]
    nationality = l2[15:18].replace("<", "").strip()

    surname, given = _decode_name(l3)
    doc_number  = doc_raw.replace("<", "").strip()

    checks_ok = (
        _check_digit(doc_raw) == doc_cd
        and _check_digit(dob_raw) == dob_cd
        and _check_digit(exp_raw) == exp_cd
    )

    return MRZResult(
        format="TD1",
        raw_lines=list(lines[:3]),
        doc_type=doc_type,
        country=country,
        doc_number=doc_number,
        surname=surname,
        given_names=given,
        nationality=nationality,
        dob=_parse_date(dob_raw),
        sex=sex,
        expiry=_parse_date(exp_raw),
        check_digits_ok=checks_ok,
    )


def decode_td3(lines: List[str]) -> MRZResult:
    """Decode 2-line TD3 MRZ (passport)."""
    l1 = lines[0].ljust(_TD3_LEN, "<")[:_TD3_LEN]
    l2 = lines[1].ljust(_TD3_LEN, "<")[:_TD3_LEN]

    doc_type    = l1[0:2].replace("<", "").strip()
    country     = l1[2:5].replace("<", "").strip()
    surname, given = _decode_name(l1[5:44])

    doc_raw     = l2[0:9]
    doc_cd      = l2[9]
    nationality = l2[10:13].replace("<", "").strip()
    dob_raw     = l2[13:19]
    dob_cd      = l2[19]
    sex         = l2[20] if l2[20] in "MF<" else "<"
    exp_raw     = l2[21:27]
    exp_cd      = l2[27]
    personal    = l2[28:42].replace("<", "").strip()
    doc_number  = doc_raw.replace("<", "").strip()

    checks_ok = (
        _check_digit(doc_raw) == doc_cd
        and _check_digit(dob_raw) == dob_cd
        and _check_digit(exp_raw) == exp_cd
    )

    return MRZResult(
        format="TD3",
        raw_lines=list(lines[:2]),
        doc_type=doc_type,
        country=country,
        doc_number=doc_number,
        surname=surname,
        given_names=given,
        nationality=nationality,
        dob=_parse_date(dob_raw),
        sex=sex,
        expiry=_parse_date(exp_raw),
        personal_number=personal,
        check_digits_ok=checks_ok,
    )


# ── MRZ Scanner ───────────────────────────────────────────────────────────────

# TD1 line-2 signature: 6-digit DOB + check + sex + 6-digit expiry + check + 3-letter nationality
_TD1_L2_RE = re.compile(
    r"[0-9]{6}[0-9][MF<][0-9]{6}[0-9][A-Z]{3}[A-Z0-9<]{11}[0-9]"
)
# TD3 line-2 signature: 9-char doc + check + 3-letter nat + 6-digit DOB + check + sex
_TD3_L2_RE = re.compile(
    r"[A-Z0-9<]{9}[0-9][A-Z]{3}[0-9]{6}[0-9][MF<][0-9]{6}[0-9]"
)


class MRZScanner:
    """
    Detects and decodes MRZ zones from camera frames.
    Thread-safe — all mutable state is local to scan() / scan_from_text().

    Two modes:
      scan(frame)        — dedicated Tesseract pass with MRZ char whitelist
      scan_from_text(t)  — parse MRZ directly from regular OCR output using
                           check-digit validation (more reliable fallback)
    """

    def __init__(self, tesseract_path: str = "") -> None:
        try:
            import os
            import pytesseract
            if tesseract_path and os.path.isfile(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            self._tess = pytesseract
            log.info("MRZScanner ready.")
        except ImportError:
            self._tess = None
            log.warning("MRZScanner: pytesseract not available — MRZ disabled.")

    # ── Public API ────────────────────────────────────────────────────

    def scan(self, frame: np.ndarray) -> Optional[MRZResult]:
        """
        Try to detect and decode an MRZ in the frame.
        Tries multiple frame regions; returns the first valid MRZ found.
        """
        if self._tess is None:
            return None

        h = frame.shape[0]
        regions = [
            frame,
            frame[int(h * 0.45):],
            frame[int(h * 0.55):],
            frame[int(h * 0.65):],
        ]

        for region in regions:
            if region.shape[0] < 20:
                continue
            for img in self._preprocess(region):
                text = self._ocr(img)
                result = self._parse(text)
                if result is not None:
                    return result
        return None

    # ── Preprocessing ─────────────────────────────────────────────────

    @staticmethod
    def _preprocess(region: np.ndarray) -> List[np.ndarray]:
        h, w = region.shape[:2]
        img  = cv2.resize(region, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # Variant 1: adaptive threshold (handles uneven card lighting)
        v1 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15
        )
        # Variant 2: Otsu (clean high-contrast MRZ zones)
        _, v2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return [v1, v2]

    # ── OCR ──────────────────────────────────────────────────────────

    def _ocr(self, img: np.ndarray) -> str:
        """Run Tesseract with MRZ whitelist; return normalised text."""
        parts: List[str] = []
        for cfg in _MRZ_CFGS:
            try:
                raw   = self._tess.image_to_string(img, config=cfg)
                lines = []
                for line in raw.splitlines():
                    clean = re.sub(r"[^A-Z0-9<]", "", line.upper())
                    if len(clean) >= 15:   # MRZ lines are 30-44 chars; 15 catches partial reads
                        lines.append(clean)
                if lines:
                    log.debug("MRZ OCR candidate lines: %s", lines)
                    parts.append("\n".join(lines))
            except Exception as exc:
                log.warning("MRZ OCR error: %s", exc)
        return "\n".join(parts)

    # ── Parsing ───────────────────────────────────────────────────────

    def _parse(self, text: str) -> Optional[MRZResult]:
        lines = [ln for ln in text.splitlines() if len(ln) >= 15]
        if not lines:
            return None

        # ── Try TD3 (passport, 2 lines × 44) ─────────────────────────
        for i in range(len(lines) - 1):
            l1, l2 = lines[i], lines[i + 1]
            if 42 <= len(l1) <= 46 and 42 <= len(l2) <= 46:
                l1p = _pad(l1, _TD3_LEN)
                l2p = _pad(l2, _TD3_LEN)
                if _td3_sanity(l2p):
                    try:
                        r = decode_td3([l1p, l2p])
                        log.info("MRZ TD3 decoded: %s", r)
                        return r
                    except Exception:
                        pass

        # ── Try TD1 (ID card, 3 lines × 30) ──────────────────────────
        for i in range(len(lines) - 2):
            l1, l2, l3 = lines[i], lines[i + 1], lines[i + 2]
            if (28 <= len(l1) <= 32 and 28 <= len(l2) <= 32 and 28 <= len(l3) <= 32):
                l1p = _pad(l1, _TD1_LEN)
                l2p = _pad(l2, _TD1_LEN)
                l3p = _pad(l3, _TD1_LEN)
                if _td1_sanity(l1p, l2p):
                    try:
                        r = decode_td1([l1p, l2p, l3p])
                        log.info("MRZ TD1 decoded: %s", r)
                        return r
                    except Exception:
                        pass

        # ── Fallback: regex-match TD1 line 2 pattern ──────────────────
        # Useful when OCR line breaks don't align with MRZ lines
        m = _TD1_L2_RE.search(text)
        if m:
            l2p = _pad(m.group(0), _TD1_LEN)
            try:
                dummy_l1 = ("I<<<<<<<<<<<<<<" + "<" * 15)[:_TD1_LEN]
                dummy_l3 = "<" * _TD1_LEN
                r = decode_td1([dummy_l1, l2p, dummy_l3])
                r.doc_number = ""  # unknown without line 1
                log.info("MRZ TD1 line-2 fallback: %s", r)
                return r
            except Exception:
                pass

        # ── Fallback: regex-match TD3 line 2 pattern ──────────────────
        m = _TD3_L2_RE.search(text)
        if m:
            l2p = _pad(m.group(0), _TD3_LEN)
            try:
                dummy_l1 = ("P<<<<<<<<<<<<<<" + "<" * 29)[:_TD3_LEN]
                r = decode_td3([dummy_l1, l2p])
                r.doc_number = ""
                log.info("MRZ TD3 line-2 fallback: %s", r)
                return r
            except Exception:
                pass

        return None


    def scan_from_text(self, text: str) -> Optional[MRZResult]:
        """
        Parse MRZ data from regular (non-whitelist) OCR output using check-digit
        validation.  Regular OCR strips '<' chars but keeps digits and letters, so
        TD1/TD3 line-2 patterns remain recognisable.
        Called as a fallback when scan() finds nothing.
        """
        # Normalise: uppercase, keep only alphanumeric + spaces
        norm = re.sub(r"[^A-Z0-9\s]", " ", text.upper())
        norm = re.sub(r"\s+", " ", norm).strip()
        # Also try a condensed version (spaces removed) in case OCR split digits
        condensed = norm.replace(" ", "")

        for src in (norm, condensed):
            r = self._try_td1_from_text(src, norm)
            if r is not None:
                return r
            r = self._try_td3_from_text(src, norm)
            if r is not None:
                return r

        return None

    def _try_td1_from_text(self, src: str, full_norm: str) -> Optional[MRZResult]:
        """Search src for a valid TD1 line-2 pattern; use full_norm for name/doc."""
        # TD1 L2: YYMMDDCSYYMMDDCNNN  (C=check digit, S=sex M/F, N=nationality)
        td1_re = re.compile(r"(\d{6})(\d)([MF])(\d{6})(\d)([A-Z]{3})")
        for m in td1_re.finditer(src):
            dob_raw, dob_cd, sex, exp_raw, exp_cd, nat = m.groups()
            if _check_digit(dob_raw) != dob_cd:
                continue
            if _check_digit(exp_raw) != exp_cd:
                continue
            log.info(
                "MRZ scan_from_text TD1: DOB=%s Sex=%s Exp=%s Nat=%s checks=OK",
                dob_raw, sex, exp_raw, nat,
            )
            doc_number = _find_doc_number(full_norm, nat, {dob_raw, exp_raw})
            surname, given_names = _find_name(full_norm)
            return MRZResult(
                format="TD1",
                raw_lines=[full_norm[:120]],
                doc_type="I",
                country=nat,
                doc_number=doc_number,
                surname=surname,
                given_names=given_names,
                nationality=nat,
                dob=_parse_date(dob_raw),
                sex=sex,
                expiry=_parse_date(exp_raw),
                check_digits_ok=True,
            )
        return None

    def _try_td3_from_text(self, src: str, full_norm: str) -> Optional[MRZResult]:
        """Search src for a valid TD3 line-2 pattern; use full_norm for name."""
        # TD3 L2: doc(9)C nat(3) YYMMDDCSYYMMDDCC
        td3_re = re.compile(r"([A-Z0-9]{7,9})(\d)([A-Z]{3})(\d{6})(\d)([MF])(\d{6})(\d)")
        for m in td3_re.finditer(src):
            doc_raw, doc_cd, nat, dob_raw, dob_cd, sex, exp_raw, exp_cd = m.groups()
            if _check_digit(dob_raw) != dob_cd:
                continue
            if _check_digit(exp_raw) != exp_cd:
                continue
            doc_number = doc_raw.replace("<", "").strip()
            log.info(
                "MRZ scan_from_text TD3: Doc=%s DOB=%s Sex=%s Exp=%s Nat=%s checks=OK",
                doc_number, dob_raw, sex, exp_raw, nat,
            )
            surname, given_names = _find_name(full_norm)
            return MRZResult(
                format="TD3",
                raw_lines=[full_norm[:120]],
                doc_type="P",
                country=nat,
                doc_number=doc_number,
                surname=surname,
                given_names=given_names,
                nationality=nat,
                dob=_parse_date(dob_raw),
                sex=sex,
                expiry=_parse_date(exp_raw),
                check_digits_ok=True,
            )
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pad(line: str, length: int) -> str:
    return line[:length].ljust(length, "<")


def _td3_sanity(l2: str) -> bool:
    return (
        len(l2) >= 28
        and l2[9]  in "0123456789"
        and l2[19] in "0123456789"
        and l2[20] in "MF<"
        and l2[27] in "0123456789"
    )


def _td1_sanity(l1: str, l2: str) -> bool:
    return (
        len(l1) >= 5
        and l1[0] in "AICPV"
        and len(l2) >= 15
        and l2[6]  in "0123456789"
        and l2[7]  in "MF<"
        and l2[14] in "0123456789"
    )


def _find_doc_number(text: str, nat: str, exclude: set) -> str:
    """
    Locate IC/passport document number in normalised OCR text.

    Strategy 1 — country code adjacent to digits (MRZ line 1 e.g. BRN00288800)
    Strategy 2 — leading-zero sequence (Brunei IDs start with 00)
    Strategy 3 — any 7–9 digit run not matching DOB / expiry fields
    """
    # Strategy 1a: nat code immediately followed by digits (5-9 digits; OCR may miss trailing chars)
    m = re.search(r"\b" + re.escape(nat) + r"\s*(\d{5,9})\b", text)
    if m:
        return m.group(1)
    # Strategy 1b: digits immediately before nat code
    m = re.search(r"\b(\d{5,9})\s*" + re.escape(nat) + r"\b", text)
    if m:
        return m.group(1)
    # Strategy 2: leading zeros (Brunei 00XXXXXX format, OCR may partially read it)
    m = re.search(r"\b0{1,2}\d{4,8}\b", text)
    if m and m.group(0) not in exclude:
        return m.group(0)
    # Strategy 3: any 7-9 digit run not in DOB/expiry (7+ avoids spurious short numbers)
    for dm in re.finditer(r"\b\d{7,9}\b", text):
        cand = dm.group(0)
        if cand not in exclude:
            return cand
    return ""


def _find_name(text: str) -> Tuple[str, str]:
    """
    Extract (surname, given_names) from normalised OCR text.

    Prioritises Malay BIN/BINTI patronymic pattern.
    Falls back to the longest run of all-caps words (≥ 3 words).
    Returns (surname, given_names) — for Malay names surname = patronymic.
    """
    m = re.search(
        r"\b([A-Z]{2,}(?:\s+[A-Z]{2,}){0,4})\s+(BINT[EI]?|BIN)\s+([A-Z]{2,}(?:\s+[A-Z]{2,}){0,3})\b",
        text,
    )
    if m:
        given_names = m.group(1).strip()
        surname     = (m.group(2) + " " + m.group(3)).strip()
        return surname, given_names
    # Fallback: longest all-caps sequence (≥ 3 words)
    caps = re.findall(r"\b(?:[A-Z]{2,}\s+){2,}[A-Z]{2,}\b", text)
    if caps:
        return max(caps, key=len).strip(), ""
    return "", ""
