"""
ClashVision Vision Subsystem — Numeric OCR Sanitizer & Value Parser.
Converts standard formatted numbers ("800,000"), compact notations ("800K", "1.2M"),
European dot notations ("800.000"), and battle timers ("2:45") into clean integers.
Guarantees ZERO hallucination on unrecognized or corrupted characters.
Derived directly from Requirement R2 and Feature 12.
"""

from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class ParsedNumericResult:
    """
    Structured outcome of a numeric OCR parsing operation.
    """
    value: Optional[int]
    raw_text: str
    success: bool
    is_timer: bool = False
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "raw_text": self.raw_text,
            "success": self.success,
            "is_timer": self.is_timer,
            "confidence": self.confidence,
        }


def sanitize_ocr_text(text: Any) -> str:
    """
    Normalizes common OCR artifact characters (e.g. smart quotes, fullwidth commas).
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    # Normalize unicode spaces and common fullwidth characters
    cleaned = text.strip()
    cleaned = cleaned.replace("，", ",").replace("．", ".").replace("：", ":")
    cleaned = cleaned.replace("\u202f", " ").replace("\xa0", " ")
    return cleaned


def parse_resource_number(text: Any) -> Optional[int]:
    """
    Authoritative reference parser implementing R2 numeric conversion.
    Converts standard, compact (K/M), and European dot strings to clean integers.
    Returns None for unread/malformed text (zero hallucination).
    """
    if text is None:
        return None

    if not isinstance(text, str):
        if isinstance(text, int) and not isinstance(text, bool):
            return text if text >= 0 else None
        return None

    cleaned = sanitize_ocr_text(text)
    if not cleaned:
        return None

    # Strip standard resource label prefixes
    cleaned = re.sub(
        r"^(Gold|Elixir|Dark Elixir|Dark\s*Elixir|DE)\s*:\s*",
        "",
        cleaned,
        flags=re.I,
    ).strip()

    if not cleaned:
        return None

    # Check for compact 'M' notation (e.g. "1.2M", "800M", "1.25m")
    m_match = re.search(r"^(\d+(?:[.,]\d+)?)\s*[Mm]$", cleaned)
    if m_match:
        val_str = m_match.group(1).replace(",", ".")
        try:
            return int(round(float(val_str) * 1_000_000))
        except (ValueError, OverflowError):
            return None

    # Check for compact 'K' notation (e.g. "800K", "350k", "5.5K")
    k_match = re.search(r"^(\d+(?:[.,]\d+)?)\s*[Kk]$", cleaned)
    if k_match:
        val_str = k_match.group(1).replace(",", ".")
        try:
            return int(round(float(val_str) * 1_000))
        except (ValueError, OverflowError):
            return None

    # Reject invalid characters, negative numbers, consecutive separators, mixed separators, or trailing punctuation
    if (
        ("," in cleaned and "." in cleaned)
        or ",," in cleaned
        or ".." in cleaned
        or "-" in cleaned
        or cleaned.endswith(",")
        or cleaned.endswith(".")
        or cleaned.startswith(",")
        or cleaned.startswith(".")
    ):
        return None

    # Comma or dot separated numbers (standard 3-digit grouping or pure integer)
    # Examples: "800,000", "1,250,000", "800.000", "12", "0"
    if re.fullmatch(r"\d{1,3}(?:[,.]\d{3})+|\d+", cleaned):
        digits_only = re.sub(r"[,.\s]", "", cleaned)
        try:
            return int(digits_only)
        except (ValueError, OverflowError):
            return None

    return None


def parse_timer_seconds(text: Any) -> Optional[int]:
    """
    Converts battle timer 'M:SS' or 'MM:SS' into total integer seconds.
    Returns None for unread, malformed, or empty strings.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        return None

    cleaned = sanitize_ocr_text(text)
    match = re.fullmatch(r"(\d+):(\d{2})", cleaned)
    if match:
        try:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            return minutes * 60 + seconds
        except (ValueError, OverflowError):
            return None

    return None


def parse_numeric(text: Any) -> ParsedNumericResult:
    """
    Universal entry point: attempts timer parsing first if colon is present,
    otherwise parses as resource amount.
    """
    if text is None:
        return ParsedNumericResult(value=None, raw_text="", success=False)

    if not isinstance(text, str):
        if isinstance(text, int) and not isinstance(text, bool):
            success = text >= 0
            return ParsedNumericResult(
                value=text if success else None,
                raw_text=str(text),
                success=success,
                confidence=1.0 if success else 0.0,
            )
        return ParsedNumericResult(value=None, raw_text=str(text), success=False, confidence=0.0)

    raw_clean = text.strip()

    if ":" in raw_clean and not any(pref in raw_clean.upper() for pref in ["GOLD", "ELIXIR", "DE"]):
        timer_val = parse_timer_seconds(raw_clean)
        if timer_val is not None:
            return ParsedNumericResult(
                value=timer_val,
                raw_text=raw_clean,
                success=True,
                is_timer=True,
                confidence=0.98,
            )

    resource_val = parse_resource_number(raw_clean)
    if resource_val is not None:
        return ParsedNumericResult(
            value=resource_val,
            raw_text=raw_clean,
            success=True,
            is_timer=False,
            confidence=0.95,
        )

    return ParsedNumericResult(
        value=None,
        raw_text=raw_clean,
        success=False,
        is_timer=False,
        confidence=0.0,
    )


def parse_resource_dict(raw_dict: Dict[str, str]) -> Dict[str, Optional[int]]:
    """
    Batch parses a dictionary of raw OCR strings (e.g. {'gold': '800,000', 'elixir': '1.2M'}).
    """
    return {k: parse_resource_number(v) for k, v in raw_dict.items()}
