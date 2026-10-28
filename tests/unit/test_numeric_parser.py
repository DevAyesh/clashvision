"""
Unit tests for ClashVision numeric and timer OCR parsing.
Verifies Requirement R2:
- Standard formatting ("800,000")
- Compact suffix notations ("800K", "1.2M")
- European dot notation ("800.000")
- Battle timer conversion ("2:45" -> 165s)
- Absolute zero hallucination on unrecognized or corrupted inputs
"""

import unittest
from typing import Optional, List, Tuple
import pytest

from clashvision.vision.numeric_parser import (
    parse_resource_number,
    parse_timer_seconds,
    parse_numeric,
    parse_resource_dict,
    sanitize_ocr_text,
    ParsedNumericResult,
)
from tests.fixtures.recorded_samples import OCR_CORPUS, TIMER_CORPUS


class TestNumericParserStandard(unittest.TestCase):
    """Verifies parsing of standard and European formatted numbers."""

    def test_standard_comma_separated_integers(self):
        cases = [
            ("800,000", 800_000),
            ("1,250,000", 1_250_000),
            ("50,000", 50_000),
            ("3,500", 3_500),
            ("0", 0),
            ("10,000,000", 10_000_000),
            ("12", 12),
        ]
        for raw, expected in cases:
            self.assertEqual(parse_resource_number(raw), expected, f"Failed on '{raw}'")

    def test_european_dot_separated_integers(self):
        cases = [
            ("800.000", 800_000),
            ("1.250.000", 1_250_000),
            ("50.000", 50_000),
            ("10.000.000", 10_000_000),
        ]
        for raw, expected in cases:
            self.assertEqual(parse_resource_number(raw), expected, f"Failed on '{raw}'")


class TestNumericParserCompactNotation(unittest.TestCase):
    """Verifies compact 'K' and 'M' suffix notation."""

    def test_compact_k_notation(self):
        cases = [
            ("800K", 800_000),
            ("800k", 800_000),
            ("350K", 350_000),
            ("5.5K", 5_500),
            ("5,5K", 5_500),
            ("0K", 0),
            ("1K", 1_000),
        ]
        for raw, expected in cases:
            self.assertEqual(parse_resource_number(raw), expected, f"Failed on '{raw}'")

    def test_compact_m_notation(self):
        cases = [
            ("1.2M", 1_200_000),
            ("1.25M", 1_250_000),
            ("2M", 2_000_000),
            ("0.8M", 800_000),
            ("1.2m", 1_200_000),
            ("1,25m", 1_250_000),
        ]
        for raw, expected in cases:
            self.assertEqual(parse_resource_number(raw), expected, f"Failed on '{raw}'")


class TestNumericParserPrefixesAndPunctuation(unittest.TestCase):
    """Verifies label prefix stripping and whitespace handling."""

    def test_label_prefixes(self):
        cases = [
            ("Gold: 850,000", 850_000),
            ("Elixir: 1.5M", 1_500_000),
            ("DE: 4,500", 4_500),
            ("Dark Elixir: 5,000", 5_000),
            ("Gold:850,000", 850_000),
            ("ELIXIR: 720K", 720_000),
        ]
        for raw, expected in cases:
            self.assertEqual(parse_resource_number(raw), expected, f"Failed on '{raw}'")

    def test_whitespace_padding(self):
        cases = [
            ("  720,000  ", 720_000),
            ("\t800K\n", 800_000),
            (" 1.2M ", 1_200_000),
        ]
        for raw, expected in cases:
            self.assertEqual(parse_resource_number(raw), expected, f"Failed on '{raw}'")


class TestTimerParser(unittest.TestCase):
    """Verifies battle timer parsing into total seconds."""

    def test_timer_nominal_cases(self):
        cases = [
            ("3:00", 180),
            ("2:45", 165),
            ("1:15", 75),
            ("0:59", 59),
            ("0:01", 1),
            ("0:00", 0),
            ("99:99", 6039),
        ]
        for raw, expected in cases:
            self.assertEqual(parse_timer_seconds(raw), expected, f"Failed on timer '{raw}'")

    def test_timer_malformed_returns_none(self):
        malformed = [
            "",
            "   ",
            "invalid",
            "1:2",  # single digit seconds
            "2:455",  # 3 digit seconds
            "timer",
            "??:??",
            "None",
        ]
        for raw in malformed:
            self.assertIsNone(parse_timer_seconds(raw), f"Expected None for timer '{raw}'")


class TestZeroHallucination(unittest.TestCase):
    """Guarantees malformed or corrupted inputs return None with zero hallucination."""

    def test_malformed_inputs_return_none(self):
        malformed_strings = [
            "???",
            "---",
            "N/A",
            "Not Found",
            "",
            "   ",
            "abcxyz",
            "$$$",
            ",,",
            "..",
            "-800,000",
            "800,",
            ",800",
            ".800",
            "800.",
            "800,,000",
            "800..000",
            "Unknown",
            "Error 404",
            "123abc456",
        ]
        for raw in malformed_strings:
            res = parse_resource_number(raw)
            self.assertIsNone(res, f"Expected None for '{raw}', got {res}")


class TestParseNumericModel(unittest.TestCase):
    """Verifies ParsedNumericResult dataclass and universal parse_numeric."""

    def test_parse_numeric_resource(self):
        res = parse_numeric("850,000")
        self.assertTrue(res.success)
        self.assertEqual(res.value, 850_000)
        self.assertFalse(res.is_timer)
        self.assertGreater(res.confidence, 0.90)

    def test_parse_numeric_timer(self):
        res = parse_numeric("2:45")
        self.assertTrue(res.success)
        self.assertEqual(res.value, 165)
        self.assertTrue(res.is_timer)

    def test_parse_numeric_unrecognized(self):
        res = parse_numeric("???")
        self.assertFalse(res.success)
        self.assertIsNone(res.value)
        self.assertEqual(res.confidence, 0.0)

    def test_parse_resource_dict(self):
        raw_dict = {
            "gold": "850,000",
            "elixir": "1.2M",
            "dark_elixir": "DE: 4.5K",
            "empty": "???",
        }
        parsed = parse_resource_dict(raw_dict)
        self.assertEqual(parsed["gold"], 850_000)
        self.assertEqual(parsed["elixir"], 1_200_000)
        self.assertEqual(parsed["dark_elixir"], 4_500)
        self.assertIsNone(parsed["empty"])

    def test_sanitize_ocr_text_unicode(self):
        self.assertEqual(sanitize_ocr_text("800，000"), "800,000")
        self.assertEqual(sanitize_ocr_text("1．2M"), "1.2M")
        self.assertEqual(sanitize_ocr_text("Gold：800K"), "Gold:800K")


class TestRecordedCorpusValidation(unittest.TestCase):
    """Cross-validates against recorded ground-truth corpus."""

    def test_ocr_corpus(self):
        for raw, expected, is_valid, desc in OCR_CORPUS:
            val = parse_resource_number(raw)
            self.assertEqual(val, expected, f"Corpus item failed: '{raw}' ({desc})")

    def test_timer_corpus(self):
        for raw, expected, is_valid in TIMER_CORPUS:
            val = parse_timer_seconds(raw)
            self.assertEqual(val, expected, f"Timer corpus item failed: '{raw}'")


if __name__ == "__main__":
    unittest.main()
