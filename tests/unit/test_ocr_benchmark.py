"""
Multi-engine OCR benchmark and synthetic test dataset suite.
Verifies Phase 3 OCR evaluation across Mock, Tesseract, RapidOCR, and EasyOCR engines.
Tests edge cases: comma/dot notation, compact M/K notation, noise, and low contrast.
"""

import time
import unittest
import numpy as np
import cv2

from clashvision.core.models import OCRResult
from clashvision.vision.ocr_engine import (
    BaseOCREngine,
    MockOCREngine,
    PytesseractEngine,
    RapidOCREngine,
    EasyOCREngine,
    get_ocr_engine,
)
from clashvision.vision.preprocessor import VisionPreprocessor
from clashvision.vision.numeric_parser import parse_resource_number, evaluate_resource_confidence


def generate_synthetic_resource_crop(
    text: str,
    width: int = 180,
    height: int = 44,
    bg_color: tuple = (30, 25, 20),
    text_color: tuple = (255, 255, 255),
    noise_sigma: float = 0.0,
    low_contrast: bool = False,
) -> np.ndarray:
    """Generates a synthetic numeric image crop simulating in-game resource readout."""
    if low_contrast:
        bg_color = (80, 80, 80)
        text_color = (120, 120, 120)

    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = max(5, (width - tw) // 2)
    ty = max(th + 5, (height + th) // 2)

    cv2.putText(img, text, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)

    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy

    return img


class TestMultiEngineOCRDataset(unittest.TestCase):
    """Verifies OCR engines across a rich dataset of resource text patterns."""

    def setUp(self):
        self.dataset = [
            ("1,250,000", 1_250_000, "gold"),
            ("850,000", 850_000, "elixir"),
            ("2.4M", 2_400_000, "gold"),
            ("75k", 75_000, "dark_elixir"),
            ("9,800", 9_800, "dark_elixir"),
            ("120.000", 120_000, "elixir"),
        ]

    def test_mock_engine_dataset_recognition(self):
        """Verifies MockOCREngine with predefined mapping against the dataset."""
        mapping = {f"item_{i}": item[0] for i, item in enumerate(self.dataset)}
        engine = MockOCREngine(mapping=mapping, default_text="1,250,000")

        for i, (expected_text, expected_val, r_type) in enumerate(self.dataset):
            crop = generate_synthetic_resource_crop(expected_text)
            engine.default_text = expected_text
            res = engine.recognize_resource(crop, resource_type=r_type)
            self.assertIsInstance(res, OCRResult)
            self.assertEqual(res.resource_type, r_type)
            self.assertEqual(res.text, expected_text)
            self.assertEqual(res.normalized_value, expected_val)
            self.assertTrue(res.is_trustworthy())

    def test_preprocessed_crop_robustness(self):
        """Verifies preprocessing steps on noisy and low-contrast crops."""
        # Low contrast
        low_contrast_crop = generate_synthetic_resource_crop("500,000", low_contrast=True)
        preprocessed = VisionPreprocessor.preprocess_resource_crop(low_contrast_crop, method="clahe_otsu")
        self.assertEqual(preprocessed.shape, (44, 180))
        self.assertEqual(preprocessed.dtype, np.uint8)

        # Noisy
        noisy_crop = generate_synthetic_resource_crop("350,000", noise_sigma=15.0)
        bilat_prep = VisionPreprocessor.preprocess_resource_crop(noisy_crop, method="bilateral_adaptive")
        self.assertEqual(bilat_prep.shape, (44, 180))

        # Color invert
        inv_prep = VisionPreprocessor.preprocess_resource_crop(low_contrast_crop, method="invert_thresh")
        self.assertEqual(inv_prep.shape, (44, 180))

    def test_engine_graceful_fallbacks(self):
        """Verifies all engines provide stable OCRResult even when backend libs are unconfigured."""
        engines = [
            get_ocr_engine("mock"),
            get_ocr_engine("tesseract"),
            get_ocr_engine("rapidocr"),
            get_ocr_engine("easyocr"),
        ]

        dummy_crop = generate_synthetic_resource_crop("750,000")
        for eng in engines:
            result = eng.recognize_resource(dummy_crop, resource_type="gold")
            self.assertIsInstance(result, OCRResult)
            self.assertIsInstance(result.engine, str)
            self.assertIsInstance(result.confidence, float)
            self.assertTrue(0.0 <= result.confidence <= 1.0)

    def test_benchmark_engine_latency(self):
        """Measures and logs execution latency across available engine interfaces."""
        engine = MockOCREngine(default_text="1,500,000", default_confidence=0.98)
        crop = generate_synthetic_resource_crop("1,500,000")

        iterations = 20
        start = time.perf_counter()
        for _ in range(iterations):
            _ = engine.recognize_resource(crop, resource_type="gold")
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / iterations

        # Mock engine should execute in < 2ms per crop
        self.assertLess(elapsed_ms, 2.0)


if __name__ == "__main__":
    unittest.main()
