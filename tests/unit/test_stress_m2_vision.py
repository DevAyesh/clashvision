"""
Milestone 2 Adversarial Stress Test Suite for ClashVision Vision & OCR Subsystems.
Adversarially challenges:
- clashvision/vision/numeric_parser.py
- clashvision/vision/preprocessor.py

Requirements: R2 (Vision Recognition & Tesseract OCR Engines), Feature 10, Feature 12.
Acceptance Criteria: Zero unhandled exceptions, zero hallucination on unrecognized/corrupted text.
"""

import math
import unittest
import numpy as np
import pytest

from clashvision.core.models import Rect, NormalizedRect
from clashvision.vision.numeric_parser import (
    parse_resource_number,
    parse_timer_seconds,
    parse_numeric,
    parse_resource_dict,
    sanitize_ocr_text,
    ParsedNumericResult,
)
from clashvision.vision.preprocessor import (
    VisionPreprocessor,
    OCRDebugData,
)


class TestNumericParserStressPassing(unittest.TestCase):
    """Verifies robustness of numeric_parser on boundary, extreme-length, and sanitized inputs."""

    def test_nominal_and_boundary_numeric_parsing(self):
        cases = [
            ("0", 0),
            ("00", 0),
            ("0K", 0),
            ("0k", 0),
            ("0M", 0),
            ("0m", 0),
            ("1", 1),
            ("800,000", 800_000),
            ("800.000", 800_000),
            ("1.2M", 1_200_000),
            ("800K", 800_000),
            ("350k", 350_000),
            ("999,999,999", 999_999_999),
        ]
        for raw, expected in cases:
            self.assertEqual(parse_resource_number(raw), expected, f"Failed on '{raw}'")

    def test_extreme_integer_string_length(self):
        """Long digit strings (up to 5,000 digits) tested against Python 3.11+ 4300-digit conversion limits."""
        large_strings = [
            "9" * 1000,
            "9" * 4300,
            "9" * 5000,
            ",".join(["999"] * 1500),
        ]
        for s in large_strings:
            try:
                res = parse_resource_number(s)
                self.assertTrue(res is None or isinstance(res, int))
            except Exception as e:
                self.fail(f"Unhandled exception on extreme string length ({len(s)} chars): {type(e).__name__}: {e}")

    def test_unicode_noise_and_confusables(self):
        """Homoglyphs, fullwidth chars, zero-width chars, emojis, combining characters."""
        adversarial_strings = [
            "\u200b800,000",
            "800,000\u200b",
            "800\u200b,000",
            "\ufeff800,000",
            "\u202e800,000",
            "💰 800,000",
            "800,000 ⚔️",
            "VIII C",
            "800\u041c",
            "1.2\u043c",
            "800\u043a",
            "١٢٣٤٥٦",
            "一二三四五",
            "800K\x00",
            "\x00800,000",
            "800\x00,000",
        ]
        for s in adversarial_strings:
            try:
                res = parse_resource_number(s)
                self.assertTrue(res is None or isinstance(res, int))
            except Exception as e:
                self.fail(f"Unhandled exception on Unicode string {repr(s)}: {type(e).__name__}: {e}")

    def test_negative_integers_and_unicode_dashes(self):
        """All negative resource numbers must return None (resources cannot be negative)."""
        negative_strings = [
            "-800,000",
            "-1",
            "-0",
            "-1.2M",
            "-800K",
            "\u2212800,000",
            "\u2013800,000",
            "\u2014800,000",
            "\ufe63800,000",
            "\uff0d800,000",
        ]
        for s in negative_strings:
            res = parse_resource_number(s)
            self.assertIsNone(res, f"Negative string '{s}' must return None, got {res}")

    def test_special_float_tokens_zero_hallucination(self):
        """String representations of NaN, Infinity, scientific notation."""
        tokens = [
            "NaN",
            "nan",
            "NAN",
            "Infinity",
            "infinity",
            "inf",
            "-inf",
            "-Infinity",
            "1e6",
            "1E6",
            "1e+6",
            "1e-6",
            "NaNM",
            "nanM",
            "infM",
            "InfinityM",
            "NaNK",
            "nanK",
            "infK",
        ]
        for t in tokens:
            res = parse_resource_number(t)
            self.assertIsNone(res, f"Special token '{t}' must return None, got {res}")

    def test_battle_timer_stress_cases(self):
        """Battle timer parser under boundary and malformed conditions."""
        timer_stress = [
            ("3:00", 180),
            ("2:45", 165),
            ("00:00", 0),
            ("0:00", 0),
            ("-1:30", None),
            ("1:-30", None),
            ("0:0", None),
            ("0:000", None),
            ("1:2:30", None),
            ("::", None),
            ("1:30:45", None),
            ("Time: 2:45", None),
        ]
        for raw, expected in timer_stress:
            res = parse_timer_seconds(raw)
            self.assertEqual(res, expected, f"Failed on timer string '{raw}'")


class TestNumericParserVulnerabilities(unittest.TestCase):
    """
    Empirically reproduces confirmed vulnerabilities in clashvision/vision/numeric_parser.py.
    Each test asserts the required specification contract and demonstrates the failure.
    """

    def test_float_overflow_m_notation_crash(self):
        """
        VULNERABILITY 1: Unhandled OverflowError in compact 'M' notation.
        Input with >= 305 digits before 'M' causes float() or float*1e6 to overflow
        IEEE 754 limit (~1.8e308), raising OverflowError which is unhandled.
        """
        overflow_inputs = [
            "9" * 310 + "M",
            "9" * 400 + "m",
            "1" + "0" * 308 + "M",
            "1" + "0" * 305 + "M",
            "9" * 500 + "M",
        ]
        for s in overflow_inputs:
            with self.subTest(input_len=len(s)):
                try:
                    res = parse_resource_number(s)
                    self.assertIsNone(res, f"Expected None for overflow input, got {res}")
                except OverflowError as e:
                    self.fail(f"VULNERABILITY CONFIRMED: Unhandled OverflowError on '{s[:15]}...': {e}")

    def test_float_overflow_k_notation_crash(self):
        """
        VULNERABILITY 1 (cont): Unhandled OverflowError in compact 'K' notation.
        """
        overflow_inputs = [
            "9" * 310 + "K",
            "9" * 400 + "k",
            "1" + "0" * 308 + "K",
            "1" + "0" * 306 + "K",
        ]
        for s in overflow_inputs:
            with self.subTest(input_len=len(s)):
                try:
                    res = parse_resource_number(s)
                    self.assertIsNone(res, f"Expected None for overflow input, got {res}")
                except OverflowError as e:
                    self.fail(f"VULNERABILITY CONFIRMED: Unhandled OverflowError on '{s[:15]}...': {e}")

    def test_universal_parse_numeric_overflow_crash(self):
        """
        VULNERABILITY 1 (cont): Universal parse_numeric crashes on float overflow input.
        """
        overflow_str = "9" * 400 + "M"
        try:
            res = parse_numeric(overflow_str)
            self.assertFalse(res.success)
            self.assertIsNone(res.value)
        except OverflowError as e:
            self.fail(f"VULNERABILITY CONFIRMED: parse_numeric crashed with OverflowError: {e}")

    def test_mixed_separators_zero_hallucination(self):
        """
        VULNERABILITY 2: Zero-hallucination violation on mixed comma/dot grouping separators.
        '800.000,000.000' mixes dots and commas in thousands groupings; it must be rejected as
        corrupted OCR, but currently hallucinates 800,000,000,000 (800 billion).
        """
        corrupted_mixed = "800.000,000.000"
        res = parse_resource_number(corrupted_mixed)
        self.assertIsNone(
            res,
            f"VULNERABILITY CONFIRMED: Mixed separator string '{corrupted_mixed}' hallucinated value {res} instead of None",
        )

    def test_non_string_type_safety_crash(self):
        """
        VULNERABILITY 3: Unhandled AttributeError when non-string (int/float/dict) is passed.
        """
        non_strings = [123, 800000, 1.25]
        for val in non_strings:
            with self.subTest(val=val):
                try:
                    res = parse_resource_number(val)  # type: ignore
                    self.assertTrue(res is None or isinstance(res, int))
                except AttributeError as e:
                    self.fail(f"VULNERABILITY CONFIRMED: Unhandled AttributeError on {type(val)} input: {e}")


class TestVisionPreprocessorStressPassing(unittest.TestCase):
    """Verifies robustness of VisionPreprocessor under extreme dimensions, noise, and operations."""

    def test_extreme_image_dimensions_handling(self):
        extreme_shapes = [
            (0, 0),
            (0, 0, 3),
            (1, 1),
            (1, 1, 3),
            (1, 1, 4),
            (2, 2),
            (2, 2, 3),
            (7, 7),
            (8, 8),
            (9, 9),
            (100, 1),
            (1, 100),
        ]
        for shape in extreme_shapes:
            with self.subTest(shape=shape):
                img = np.zeros(shape, dtype=np.uint8)
                gray = VisionPreprocessor.to_grayscale(img)
                self.assertEqual(len(gray.shape), 2)
                if shape[0] == 0 or shape[1] == 0:
                    self.assertEqual(gray.size, 0)
                else:
                    self.assertEqual(gray.shape, (shape[0], shape[1]))

    def test_clahe_contrast_enhancement_extremes(self):
        sizes = [(1, 1), (2, 2), (4, 4), (7, 7), (8, 8), (100, 1), (1, 100)]
        for h, w in sizes:
            gray = np.full((h, w), 128, dtype=np.uint8)
            res = VisionPreprocessor.enhance_contrast(gray)
            self.assertEqual(res.shape, (h, w))

    def test_adaptive_threshold_all_uniform_and_noise(self):
        methods = ["otsu", "gaussian", "mean", "fixed"]
        test_images = {
            "all_black": np.zeros((40, 160), dtype=np.uint8),
            "all_white": np.full((40, 160), 255, dtype=np.uint8),
            "uniform_gray": np.full((40, 160), 128, dtype=np.uint8),
            "random_noise": np.random.randint(0, 256, (40, 160), dtype=np.uint8),
            "tiny_1x1_black": np.zeros((1, 1), dtype=np.uint8),
            "tiny_1x1_white": np.full((1, 1), 255, dtype=np.uint8),
            "tiny_2x2": np.full((2, 2), 128, dtype=np.uint8),
        }
        for img_name, img in test_images.items():
            for method in methods:
                with self.subTest(img=img_name, method=method):
                    thresh = VisionPreprocessor.adaptive_threshold(img, method=method)
                    self.assertEqual(thresh.shape, img.shape)
                    self.assertEqual(thresh.dtype, np.uint8)

    def test_morphological_noise_removal_extreme_kernels(self):
        img = np.zeros((5, 5), dtype=np.uint8)
        kernels = [0, -1, 1, 2, 5, 10, 50, 100]
        ops = ["open", "close", "dilate", "erode", "invalid_op"]
        for k in kernels:
            for op in ops:
                with self.subTest(kernel=k, op=op):
                    res = VisionPreprocessor.remove_noise(img, kernel_size=k, operation=op)
                    self.assertEqual(res.shape, (5, 5))

    def test_scale_for_ocr_extreme_factors(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        factors = [0.0, -1.0, -0.5, 0.01, 0.5, 1.0, 2.0, 5.0]
        for f in factors:
            with self.subTest(factor=f):
                res = VisionPreprocessor.scale_for_ocr(img, scale_factor=f)
                self.assertGreaterEqual(res.shape[0], 1)
                self.assertGreaterEqual(res.shape[1], 1)

    def test_extreme_crops_and_out_of_bounds_roi(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        extreme_rois = [
            (-50, -50, 20, 20),
            (150, 150, 50, 50),
            (-50, 50, 100, 20),
            (50, -50, 20, 100),
            (0, 0, 0, 0),
            (50, 50, -20, -20),
            (50, 50, 0, 20),
            (50, 50, 20, 0),
            (0, 0, 1000, 1000),
            Rect(-10, -10, 5, 5),
            Rect(200, 200, 50, 50),
            Rect(10, 10, -5, -5),
            NormalizedRect(-0.5, -0.5, 0.2, 0.2),
            NormalizedRect(1.5, 1.5, 0.5, 0.5),
            NormalizedRect(0.1, 0.1, -0.2, -0.2),
        ]
        for roi in extreme_rois:
            with self.subTest(roi=str(roi)):
                crop = VisionPreprocessor.crop_roi(img, roi)
                self.assertIsInstance(crop, np.ndarray)
                self.assertLessEqual(crop.shape[0], 100)
                self.assertLessEqual(crop.shape[1], 100)

    def test_end_to_end_preprocess_pipeline_matrix(self):
        crops = [
            np.zeros((0, 0), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
            np.zeros((30, 120, 3), dtype=np.uint8),
            np.full((30, 120, 3), 255, dtype=np.uint8),
            np.random.randint(0, 256, (30, 120, 3), dtype=np.uint8),
        ]
        options = [
            {"enhance": True, "binarize": True, "invert": False, "scale": 1.0, "noise_removal": True},
            {"enhance": False, "binarize": True, "invert": True, "scale": 2.0, "noise_removal": False},
            {"enhance": True, "binarize": False, "invert": False, "scale": 2.0, "noise_removal": False},
            {"enhance": False, "binarize": False, "invert": False, "scale": 1.0, "noise_removal": False},
        ]
        for crop in crops:
            for opt in options:
                res = VisionPreprocessor.preprocess_for_ocr(crop, **opt)
                self.assertIsInstance(res, np.ndarray)


class TestVisionPreprocessorVulnerabilities(unittest.TestCase):
    """
    Empirically reproduces confirmed vulnerabilities in clashvision/vision/preprocessor.py.
    """

    def test_preprocess_for_ocr_float32_crash(self):
        """
        VULNERABILITY 4: cv2.error in CLAHE when float32 3-channel image is passed to preprocess_for_ocr.
        to_grayscale produces float32 grayscale, which CLAHE rejects.
        """
        img_f32 = np.zeros((40, 160, 3), dtype=np.float32)
        try:
            res = VisionPreprocessor.preprocess_for_ocr(img_f32)
            self.assertIsInstance(res, np.ndarray)
            self.assertEqual(res.dtype, np.uint8)
        except Exception as e:
            self.fail(f"VULNERABILITY CONFIRMED: preprocess_for_ocr crashed on float32 image: {type(e).__name__}: {e}")

    def test_to_grayscale_unsupported_dtypes(self):
        """
        VULNERABILITY 5: to_grayscale crashes or fails to emit uint8 on non-uint8 dtypes.
        """
        dtypes = [np.float32, np.int32, np.float64, np.uint16, bool]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                img = np.zeros((10, 10, 3), dtype=dt)
                try:
                    gray = VisionPreprocessor.to_grayscale(img)
                    self.assertEqual(gray.dtype, np.uint8, f"Expected uint8 output, got {gray.dtype}")
                except Exception as e:
                    self.fail(f"VULNERABILITY CONFIRMED: to_grayscale crashed on dtype {dt}: {type(e).__name__}: {e}")

    def test_to_grayscale_1d_array_crash(self):
        """
        VULNERABILITY 6: IndexError in to_grayscale on 1D array.
        """
        img_1d = np.zeros(100, dtype=np.uint8)
        try:
            gray = VisionPreprocessor.to_grayscale(img_1d)
            self.assertIsInstance(gray, np.ndarray)
        except IndexError as e:
            self.fail(f"VULNERABILITY CONFIRMED: to_grayscale raised IndexError on 1D array: {e}")

    def test_crop_roi_nan_coordinates_crash(self):
        """
        VULNERABILITY 7: ValueError/OverflowError in crop_roi when ROI contains NaN or Inf.
        """
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        nan_rois = [
            (float("nan"), 0, 10, 10),
            (0, float("nan"), 10, 10),
            (0, 0, float("nan"), 10),
            (float("inf"), 0, 10, 10),
            (0, 0, float("inf"), 10),
        ]
        for r in nan_rois:
            with self.subTest(nan_roi=r):
                try:
                    crop = VisionPreprocessor.crop_roi(img, r)
                    self.assertIsInstance(crop, np.ndarray)
                except (ValueError, OverflowError) as e:
                    self.fail(f"VULNERABILITY CONFIRMED: crop_roi crashed on NaN/Inf ROI {r}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    unittest.main()
