"""
Unit tests for ClashVision Pluggable OCR Engine Architecture.
Verifies Requirement R2 and Feature 11:
- BaseOCREngine abstract interface
- Deterministic MockOCREngine with call counting and shape mapping
- PytesseractEngine with configurable binary path and graceful offline fallback
- RapidOCREngine fallback interface
- Factory function get_ocr_engine
- Visual OCR debug packaging
"""

import unittest
from typing import Optional, Dict
import numpy as np
import pytest

from clashvision.vision.ocr_engine import (
    BaseOCREngine,
    MockOCREngine,
    PytesseractEngine,
    RapidOCREngine,
    get_ocr_engine,
)
from clashvision.vision.preprocessor import VisionPreprocessor, OCRDebugData
from tests.fixtures.synthetic_generator import SyntheticGenerator


class TestBaseOCREngineInterface(unittest.TestCase):
    """Verifies ABC constraints on BaseOCREngine."""

    def test_cannot_instantiate_abstract_engine(self):
        with self.assertRaises(TypeError):
            BaseOCREngine()  # type: ignore

    def test_engine_subclass_must_implement_methods(self):
        class IncompleteEngine(BaseOCREngine):
            pass

        with self.assertRaises(TypeError):
            IncompleteEngine()  # type: ignore


class TestMockOCREngine(unittest.TestCase):
    """Verifies deterministic MockOCREngine behavior."""

    def test_mock_default_recognition(self):
        engine = MockOCREngine(default_text="1,250,000")
        img = np.zeros((40, 160, 3), dtype=np.uint8)
        text = engine.recognize_text(img)
        self.assertEqual(text, "1,250,000")
        self.assertEqual(engine.call_count, 1)

    def test_mock_mapping_by_shape(self):
        engine = MockOCREngine(
            mapping={
                "40x160": "800,000",
                "50x200": "1.2M",
            }
        )
        crop1 = np.zeros((40, 160, 3), dtype=np.uint8)
        crop2 = np.zeros((50, 200, 3), dtype=np.uint8)
        self.assertEqual(engine.recognize_text(crop1), "800,000")
        self.assertEqual(engine.recognize_text(crop2), "1.2M")

    def test_mock_call_counter_increments(self):
        engine = MockOCREngine()
        for i in range(10):
            self.assertEqual(engine.call_count, i)
            engine.recognize_text(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(engine.call_count, 10)

    def test_mock_recognize_with_confidence(self):
        engine = MockOCREngine(default_text="950,000", default_confidence=0.96)
        text, conf = engine.recognize_with_confidence(np.zeros((20, 20), dtype=np.uint8))
        self.assertEqual(text, "950,000")
        self.assertEqual(conf, 0.96)

    def test_mock_is_available(self):
        engine = MockOCREngine()
        self.assertTrue(engine.is_available())
        self.assertEqual(engine.engine_name, "MockOCR")


class TestPytesseractEngine(unittest.TestCase):
    """Verifies Pytesseract engine wrapper and graceful fallback handling."""

    def test_engine_name(self):
        engine = PytesseractEngine()
        self.assertEqual(engine.engine_name, "Pytesseract")

    def test_custom_binary_path_configuration(self):
        engine = PytesseractEngine(tesseract_cmd="C:\\NonExistent\\tesseract.exe")
        self.assertEqual(engine._tesseract_cmd, "C:\\NonExistent\\tesseract.exe")
        # Should report False for non-existent path
        self.assertFalse(engine.is_available())

    def test_graceful_fallback_when_binary_missing(self):
        mock_fb = MockOCREngine(default_text="FallbackResult")
        engine = PytesseractEngine(
            tesseract_cmd="C:\\NonExistent\\tesseract.exe",
            fallback_engine=mock_fb,
        )
        crop = np.zeros((40, 160, 3), dtype=np.uint8)
        # Execution must NOT raise an unhandled exception
        result = engine.recognize_text(crop)
        self.assertEqual(result, "FallbackResult")
        self.assertEqual(mock_fb.call_count, 1)

    def test_empty_image_handling(self):
        engine = PytesseractEngine()
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(engine.recognize_text(empty_img), "")


class TestRapidOCREngine(unittest.TestCase):
    """Verifies RapidOCR engine abstraction and fallback."""

    def test_engine_name(self):
        engine = RapidOCREngine()
        self.assertEqual(engine.engine_name, "RapidOCR")

    def test_rapidocr_fallback_when_uninstalled(self):
        mock_fb = MockOCREngine(default_text="RapidFallback")
        engine = RapidOCREngine(fallback_engine=mock_fb)
        crop = np.zeros((40, 160, 3), dtype=np.uint8)
        result = engine.recognize_text(crop)
        self.assertEqual(result, "RapidFallback")


class TestGetOCREngineFactory(unittest.TestCase):
    """Verifies get_ocr_engine factory method."""

    def test_get_mock_engine(self):
        engine = get_ocr_engine("mock", mapping={"default": "500K"})
        self.assertIsInstance(engine, MockOCREngine)
        self.assertEqual(engine.recognize_text(np.zeros((10, 10, 3), dtype=np.uint8)), "500K")

    def test_get_tesseract_engine(self):
        engine = get_ocr_engine("tesseract")
        self.assertIsInstance(engine, PytesseractEngine)

    def test_get_rapidocr_engine(self):
        engine = get_ocr_engine("rapidocr")
        self.assertIsInstance(engine, RapidOCREngine)

    def test_get_auto_engine_never_crashes(self):
        engine = get_ocr_engine("auto")
        self.assertIsInstance(engine, BaseOCREngine)
        # Should be functional
        text = engine.recognize_text(np.zeros((20, 20, 3), dtype=np.uint8))
        self.assertIsInstance(text, str)


class TestOCRDebugPackaging(unittest.TestCase):
    """Verifies packaging of raw crops with preprocessed threshold and parsed int."""

    def test_ocr_debug_data_creation(self):
        raw = np.full((40, 160, 3), 50, dtype=np.uint8)
        preprocessed = VisionPreprocessor.preprocess_for_ocr(raw)
        debug = OCRDebugData(
            raw_crop=raw,
            preprocessed_crop=preprocessed,
            raw_text="850,000",
            parsed_value=850_000,
            confidence=0.98,
            roi_name="gold_meter",
        )
        self.assertEqual(debug.parsed_value, 850_000)
        self.assertEqual(debug.roi_name, "gold_meter")
        d = debug.to_dict()
        self.assertEqual(d["parsed_value"], 850_000)
        self.assertEqual(d["roi_name"], "gold_meter")
        self.assertEqual(d["raw_crop_shape"], [40, 160, 3])


if __name__ == "__main__":
    unittest.main()
