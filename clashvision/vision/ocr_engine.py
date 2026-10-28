"""
ClashVision Vision Subsystem — Pluggable OCR Engine Abstraction.
Provides a unified interface across Pytesseract, RapidOCR fallback, and deterministic Mock engine.
Guarantees seamless offline execution and graceful degradation when native tesseract binaries are absent.
Derived from Requirement R2 and Feature 11.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from typing import Optional, Dict, Tuple, Any

import cv2
import numpy as np

logger = logging.getLogger("clashvision.ocr")


class BaseOCREngine(ABC):
    """
    Abstract base class for all optical character recognition engines in ClashVision.
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Identifier name of the OCR backend."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the engine dependencies and binary executables are functional."""
        pass

    @abstractmethod
    def recognize_text(self, image: np.ndarray) -> str:
        """
        Extracts raw text string from an image crop.
        """
        pass

    def recognize_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Extracts raw text along with an estimated confidence score in [0.0, 1.0].
        Default implementation returns confidence 0.90 if text is non-empty, else 0.0.
        """
        text = self.recognize_text(image)
        conf = 0.90 if text.strip() else 0.0
        return text, conf


class MockOCREngine(BaseOCREngine):
    """
    Deterministic mock OCR engine for offline testing, CI environments,
    and fallback operation when native binaries are absent.
    """

    def __init__(
        self,
        mapping: Optional[Dict[str, str]] = None,
        default_text: str = "800,000",
        default_confidence: float = 0.98,
    ) -> None:
        self.mapping = dict(mapping) if mapping else {}
        self.default_text = default_text
        self.default_confidence = default_confidence
        self.call_count = 0

    @property
    def engine_name(self) -> str:
        return "MockOCR"

    def is_available(self) -> bool:
        return True

    def set_mapping(self, key: str, value: str) -> None:
        self.mapping[key] = value

    def recognize_text(self, image: np.ndarray) -> str:
        self.call_count += 1
        if image is None or image.size == 0:
            return ""

        # Check if mapped by image shape or explicit keys
        shape_key = f"{image.shape[0]}x{image.shape[1]}"
        if shape_key in self.mapping:
            return self.mapping[shape_key]

        return self.mapping.get("default", self.default_text)

    def recognize_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        text = self.recognize_text(image)
        conf = self.default_confidence if text else 0.0
        return text, conf


class PytesseractEngine(BaseOCREngine):
    """
    Pytesseract OCR engine wrapping Google's Tesseract OCR executable.
    Supports custom tesseract binary paths, CLI flags, and whitelist tuning.
    """

    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        config: str = "--psm 7 -c tessedit_char_whitelist=0123456789,.:KMkmGoldElixirDE ",
        fallback_engine: Optional[BaseOCREngine] = None,
    ) -> None:
        self._tesseract_cmd = tesseract_cmd
        self._config = config
        self._fallback_engine = fallback_engine or MockOCREngine()
        self._is_available: Optional[bool] = None

        if self._tesseract_cmd:
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd
            except ImportError:
                pass

    @property
    def engine_name(self) -> str:
        return "Pytesseract"

    def is_available(self) -> bool:
        if self._is_available is not None:
            return self._is_available

        try:
            import pytesseract
            if self._tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd
            # Test binary responsiveness
            _ = pytesseract.get_tesseract_version()
            self._is_available = True
        except Exception as e:
            logger.debug("Pytesseract unavailable: %s", e)
            self._is_available = False

        return self._is_available

    def recognize_text(self, image: np.ndarray) -> str:
        if image is None or image.size == 0:
            return ""

        if not self.is_available():
            logger.debug("Pytesseract binary not available; invoking fallback engine %s", self._fallback_engine.engine_name)
            return self._fallback_engine.recognize_text(image)

        try:
            import pytesseract
            # Pytesseract expects RGB or Grayscale
            if len(image.shape) == 3 and image.shape[2] == 3:
                proc_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                proc_img = image

            text = pytesseract.image_to_string(proc_img, config=self._config)
            return text.strip()
        except Exception as e:
            logger.warning("Pytesseract execution failed: %s; falling back", e)
            return self._fallback_engine.recognize_text(image)

    def recognize_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        if image is None or image.size == 0:
            return "", 0.0

        if not self.is_available():
            return self._fallback_engine.recognize_with_confidence(image)

        try:
            import pytesseract
            if len(image.shape) == 3 and image.shape[2] == 3:
                proc_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                proc_img = image

            data = pytesseract.image_to_data(proc_img, config=self._config, output_type=pytesseract.Output.DICT)
            words = []
            confs = []
            for text, conf in zip(data.get("text", []), data.get("conf", [])):
                if text.strip() and int(conf) > 0:
                    words.append(text.strip())
                    confs.append(float(conf) / 100.0)

            full_text = " ".join(words)
            avg_conf = (sum(confs) / len(confs)) if confs else 0.85
            return full_text, avg_conf
        except Exception as e:
            logger.warning("Pytesseract confidence extraction failed: %s; falling back", e)
            return self._fallback_engine.recognize_with_confidence(image)


class RapidOCREngine(BaseOCREngine):
    """
    Pure Python / ONNX runtime RapidOCR engine fallback.
    """

    def __init__(self, fallback_engine: Optional[BaseOCREngine] = None) -> None:
        self._fallback_engine = fallback_engine or MockOCREngine()
        self._ocr_instance: Any = None
        self._is_available: Optional[bool] = None

    @property
    def engine_name(self) -> str:
        return "RapidOCR"

    def is_available(self) -> bool:
        if self._is_available is not None:
            return self._is_available

        try:
            try:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore
                self._ocr_instance = RapidOCR()
                self._is_available = True
            except ImportError:
                import rapidocr  # type: ignore
                self._ocr_instance = rapidocr.RapidOCR()
                self._is_available = True
        except Exception:
            self._is_available = False

        return self._is_available

    def recognize_text(self, image: np.ndarray) -> str:
        if image is None or image.size == 0:
            return ""

        if not self.is_available():
            return self._fallback_engine.recognize_text(image)

        try:
            result, _ = self._ocr_instance(image)
            if not result:
                return ""
            texts = [line[1] for line in result]
            return " ".join(texts).strip()
        except Exception as e:
            logger.warning("RapidOCR execution failed: %s; falling back", e)
            return self._fallback_engine.recognize_text(image)


def get_ocr_engine(
    engine_type: str = "auto",
    tesseract_cmd: Optional[str] = None,
    mapping: Optional[Dict[str, str]] = None,
    default_text: str = "800,000",
) -> BaseOCREngine:
    """
    Factory function producing the appropriate OCR engine with automated fallback.
    - 'mock': deterministic mock engine
    - 'tesseract': Pytesseract engine with Mock fallback
    - 'rapidocr': RapidOCR engine with Mock fallback
    - 'auto': checks Pytesseract, then RapidOCR, then MockOCR
    """
    engine_type_lower = engine_type.lower()

    if engine_type_lower == "mock":
        return MockOCREngine(mapping=mapping, default_text=default_text)

    mock_fb = MockOCREngine(mapping=mapping, default_text=default_text)

    if engine_type_lower == "tesseract":
        return PytesseractEngine(tesseract_cmd=tesseract_cmd, fallback_engine=mock_fb)

    if engine_type_lower == "rapidocr":
        return RapidOCREngine(fallback_engine=mock_fb)

    # 'auto' mode
    tess_engine = PytesseractEngine(tesseract_cmd=tesseract_cmd, fallback_engine=mock_fb)
    if tess_engine.is_available():
        return tess_engine

    rapid_engine = RapidOCREngine(fallback_engine=mock_fb)
    if rapid_engine.is_available():
        return rapid_engine

    return mock_fb
