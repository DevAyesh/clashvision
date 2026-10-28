"""
ClashVision Vision Subsystem.
Provides 12-state game recognition, pluggable OCR engines (Pytesseract, RapidOCR, Mock),
numeric and timer value parsing, image preprocessing, and base layout feature extraction.
"""

from clashvision.vision.preprocessor import (
    VisionPreprocessor,
    OCRDebugData,
)
from clashvision.vision.numeric_parser import (
    ParsedNumericResult,
    parse_resource_number,
    parse_timer_seconds,
    parse_numeric,
    parse_resource_dict,
    sanitize_ocr_text,
)
from clashvision.vision.ocr_engine import (
    BaseOCREngine,
    MockOCREngine,
    PytesseractEngine,
    RapidOCREngine,
    get_ocr_engine,
)
from clashvision.vision.state_classifier import (
    StateClassifier,
    StateDebouncer,
)
from clashvision.vision.layout_analyzer import (
    BaseLayoutAnalyzer,
)

__all__ = [
    "VisionPreprocessor",
    "OCRDebugData",
    "ParsedNumericResult",
    "parse_resource_number",
    "parse_timer_seconds",
    "parse_numeric",
    "parse_resource_dict",
    "sanitize_ocr_text",
    "BaseOCREngine",
    "MockOCREngine",
    "PytesseractEngine",
    "RapidOCREngine",
    "get_ocr_engine",
    "StateClassifier",
    "StateDebouncer",
    "BaseLayoutAnalyzer",
]
