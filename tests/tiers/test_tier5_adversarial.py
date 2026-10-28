"""
Tier 5: Adversarial Stress Tests & Empirical Verification for Capture, Vision, and Decision Subsystems.
Milestone 6 Adversarial Coverage Hardening Suite.

Target Modules:
1. clashvision/capture/ (window_detector.py, mss_grabber.py, safety_guard.py)
2. clashvision/vision/ (state_classifier.py, numeric_parser.py, ocr_engine.py, preprocessor.py, layout_analyzer.py)
3. clashvision/decision/ (evaluator.py, recommendation_engine.py, 5 strategies)

Covers:
- Extreme resolutions (4K, 8K, 320x240, 1x1, non-standard aspect ratios like 32:9, 9:16).
- Malformed/corrupted image arrays (NaNs, Infs, non-contiguous, all-zeros, all-255, channel mutations).
- OCR strings with unicode fuzzing, bizarre separators, and zero-width spaces.
- Extreme loot combinations (billions, negatives, zero-loot, missing fields, NaNs/Infs).
- Strategy scoring invariants and tie-breaking stability.
- Thread-safety, concurrency stress, and emergency stop halt guarantees.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

from clashvision.capture.mss_grabber import MSSGrabber
from clashvision.capture.safety_guard import SafetyGuard, get_safety_guard
from clashvision.capture.window_detector import WindowDetector
from clashvision.core.config import ResourceThresholds
from clashvision.core.constants import (
    DEFAULT_CAPTURE_FPS,
    GameState,
    ResourcePriority,
    RiskLevel,
    StrategyType,
    VIRTUAL_REFERENCE_HEIGHT,
    VIRTUAL_REFERENCE_WIDTH,
)
from clashvision.core.coordinates import (
    clamp_point_to_bounds,
    normalized_to_point,
    point_to_normalized,
)
from clashvision.core.exceptions import (
    EmergencyStopTriggered,
    LiveInputSafetyViolation,
    WindowNotFoundError,
)
from clashvision.core.models import (
    BaseLayoutFeatures,
    DeploymentStep,
    FarmingRecommendation,
    FramePacket,
    NormalizedPoint,
    NormalizedRect,
    Point,
    Rect,
    ResourceReading,
    StrategyEvaluation,
    VisionAnalysisResult,
    WindowInfo,
)
from clashvision.decision.barch import BarchStrategy
from clashvision.decision.base_strategy import BaseStrategy
from clashvision.decision.custom_strategy import CustomStrategy, CustomStrategyConfig
from clashvision.decision.evaluator import BaseEvaluationResult, BaseEvaluator
from clashvision.decision.mass_dragon import MassDragonStrategy
from clashvision.decision.mass_miner import MassMinerStrategy
from clashvision.decision.recommendation_engine import RecommendationEngine
from clashvision.decision.sneaky_goblin import SneakyGoblinStrategy
from clashvision.vision.layout_analyzer import BaseLayoutAnalyzer
from clashvision.vision.numeric_parser import (
    ParsedNumericResult,
    parse_numeric,
    parse_resource_dict,
    parse_resource_number,
    parse_timer_seconds,
    sanitize_ocr_text,
)
from clashvision.vision.ocr_engine import (
    BaseOCREngine,
    MockOCREngine,
    get_ocr_engine,
)
from clashvision.vision.preprocessor import (
    OCRDebugData,
    VisionPreprocessor,
)
from clashvision.vision.state_classifier import StateClassifier, StateDebouncer


class TestTier5CaptureVisionDecisionAdversarial:
    """
    Comprehensive Tier 5 adversarial stress testing suite covering Capture, Vision, and Decision.
    """

    # =========================================================================
    # 1. Extreme Resolutions & Non-Standard Aspect Ratios
    # =========================================================================

    @pytest.mark.parametrize(
        "width,height,label",
        [
            (3840, 2160, "4K_UHD"),
            (7680, 4320, "8K_FUHD"),
            (320, 240, "QVGA_Retro"),
            (1, 1, "Minimal_1x1"),
            (5120, 1440, "32_9_SuperUltrawide"),
            (1080, 1920, "9_16_PortraitEmulator"),
            (1080, 1080, "1_1_Square"),
        ],
    )
    def test_extreme_resolutions_capture_window_detector(
        self, width: int, height: int, label: str
    ) -> None:
        """WindowDetector must accurately compute geometry and aspect ratios across extreme resolutions."""
        detector = WindowDetector(mock_mode=True)
        win = WindowInfo(
            hwnd=7000 + (width % 100),
            title=f"Google Play Games on PC - {label}",
            process_name="PlayGamesPC.exe",
            client_rect=Rect(x=0, y=0, width=width, height=height),
            window_rect=Rect(x=0, y=0, width=width, height=height),
            is_visible=True,
            is_minimized=False,
        )
        detector.register_mock_window(win)

        queried = detector.get_window_info(win.hwnd)
        assert queried is not None
        assert queried.client_rect.width == width
        assert queried.client_rect.height == height
        assert math.isclose(queried.aspect_ratio, width / height, rel_tol=1e-3)
        assert queried.is_capturable is True

    @pytest.mark.parametrize(
        "width,height",
        [
            (3840, 2160),
            (7680, 4320),
            (320, 240),
            (1, 1),
            (5120, 1440),
            (1080, 1920),
        ],
    )
    def test_extreme_resolutions_mss_grabber(self, width: int, height: int) -> None:
        """MSSGrabber mock mode must produce valid FramePackets across all extreme dimensions."""
        grabber = MSSGrabber(target_fps=60, mock_mode=True)
        win = WindowInfo(
            hwnd=8001,
            client_rect=Rect(x=0, y=0, width=width, height=height),
            is_visible=True,
            is_minimized=False,
        )
        grabber.start()
        try:
            packet = grabber.grab_frame(win)
            assert packet is not None
            assert isinstance(packet, FramePacket)
            assert packet.frame is not None
            assert packet.frame.ndim == 3
            assert packet.frame.shape[2] == 3
            # MSSGrabber clamps synthetic generation minimum dimension to 32
            expected_h = max(32, height)
            expected_w = max(32, width)
            assert packet.frame.shape[0] == expected_h
            assert packet.frame.shape[1] == expected_w
            assert packet.frame_id == 1
        finally:
            grabber.stop()

    @pytest.mark.parametrize(
        "width,height",
        [
            (3840, 2160),
            (7680, 4320),
            (320, 240),
            (1, 1),
            (5120, 1440),
            (1080, 1920),
        ],
    )
    def test_extreme_resolutions_state_classifier(self, width: int, height: int) -> None:
        """StateClassifier must produce bounded confidence [0.0, 1.0] and valid result on all resolutions."""
        sc = StateClassifier()
        # Generate synthetic frame with grassy background tone
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (45, 130, 60)

        res = sc.classify(img)
        assert isinstance(res, VisionAnalysisResult)
        assert isinstance(res.state, GameState)
        assert 0.0 <= res.confidence <= 1.0
        assert "scores" in res.evidence or "error" in res.evidence

        if width < 20 or height < 20:
            assert res.state == GameState.UNKNOWN
            assert res.confidence == 0.0
            assert res.evidence.get("error") == "frame_too_small"

    @pytest.mark.parametrize(
        "width,height",
        [
            (3840, 2160),
            (7680, 4320),
            (320, 240),
            (1, 1),
            (5120, 1440),
            (1080, 1920),
        ],
    )
    def test_extreme_resolutions_layout_analyzer(self, width: int, height: int) -> None:
        """BaseLayoutAnalyzer must safely analyze images of extreme dimensions without exceptions."""
        bla = BaseLayoutAnalyzer()
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (45, 130, 60)

        layout = bla.analyze_layout(img)
        assert isinstance(layout, BaseLayoutFeatures)
        assert 0.0 <= layout.exposed_collector_ratio <= 1.0
        assert isinstance(layout.th_accessible, bool)
        assert isinstance(layout.air_defense_locations, list)
        assert layout.compartment_count >= 0

    @pytest.mark.parametrize(
        "cw,ch",
        [
            (3840, 2160),
            (7680, 4320),
            (5120, 1440),
            (1080, 1920),
            (320, 240),
        ],
    )
    def test_extreme_resolutions_strategy_deployment_plans_normalized(
        self, cw: int, ch: int
    ) -> None:
        """All 5 strategies must generate bounded deployment coordinates across resolutions under normalized layout."""
        strats: List[BaseStrategy] = [
            SneakyGoblinStrategy(),
            BarchStrategy(),
            MassMinerStrategy(),
            MassDragonStrategy(),
            CustomStrategy(),
        ]
        # Layout with normalized coordinates [0.0, 1.0]
        layout = BaseLayoutFeatures(
            exposed_collector_ratio=0.75,
            th_accessible=True,
            air_defense_locations=[(0.3, 0.4), (0.7, 0.6)],
            compartment_count=4,
            wall_perimeter_intact=True,
        )

        for strat in strats:
            plan = strat.generate_deployment_plan(layout, canvas_width=cw, canvas_height=ch)
            assert len(plan) > 0, f"{strat.name} produced empty plan on ({cw}, {ch})"
            for step in plan:
                assert isinstance(step, DeploymentStep)
                assert 0 <= step.target_x <= cw, (
                    f"{strat.name}: target_x {step.target_x} out of [0, {cw}] on canvas ({cw}, {ch})"
                )
                assert 0 <= step.target_y <= ch, (
                    f"{strat.name}: target_y {step.target_y} out of [0, {ch}] on canvas ({cw}, {ch})"
                )
                assert 0.0 <= step.normalized_x <= 1.0, (
                    f"{strat.name}: normalized_x {step.normalized_x} out of [0.0, 1.0]"
                )
                assert 0.0 <= step.normalized_y <= 1.0, (
                    f"{strat.name}: normalized_y {step.normalized_y} out of [0.0, 1.0]"
                )
                assert step.delay_ms >= 0
                assert step.unit_count > 0

    def test_extreme_resolutions_coordinate_transforms(self) -> None:
        """Point and Rect coordinate transforms must preserve normalization bounds under extreme resolutions."""
        resolutions = [(1, 1), (320, 240), (3840, 2160), (7680, 4320), (5120, 1440)]
        for w, h in resolutions:
            pt = Point(w // 2, h // 2)
            norm = point_to_normalized(pt, w, h)
            assert 0.0 <= norm.x <= 1.0
            assert 0.0 <= norm.y <= 1.0

            reconstructed = normalized_to_point(norm, w, h)
            assert 0 <= reconstructed.x <= w
            assert 0 <= reconstructed.y <= h

            # Boundary points
            p_zero = Point(0, 0)
            n_zero = point_to_normalized(p_zero, w, h)
            assert n_zero.x == 0.0 and n_zero.y == 0.0

            p_max = Point(w, h)
            n_max = point_to_normalized(p_max, w, h)
            assert n_max.x == 1.0 and n_max.y == 1.0

    # =========================================================================
    # 2. Malformed / Corrupted Image Arrays & Type Mutations
    # =========================================================================

    def test_vision_classifier_nan_and_inf_arrays(self) -> None:
        """StateClassifier must handle float32 arrays contaminated with NaNs and Infs without crashing."""
        sc = StateClassifier()

        nan_frame = np.full((120, 120, 3), np.nan, dtype=np.float32)
        res_nan = sc.classify(nan_frame)
        assert isinstance(res_nan, VisionAnalysisResult)
        assert 0.0 <= res_nan.confidence <= 1.0

        inf_frame = np.full((120, 120, 3), np.inf, dtype=np.float32)
        res_inf = sc.classify(inf_frame)
        assert isinstance(res_inf, VisionAnalysisResult)
        assert 0.0 <= res_inf.confidence <= 1.0

        mixed_frame = np.zeros((100, 100, 3), dtype=np.float32)
        mixed_frame[10:30, :, :] = np.nan
        mixed_frame[50:70, :, :] = np.inf
        res_mixed = sc.classify(mixed_frame)
        assert isinstance(res_mixed, VisionAnalysisResult)
        assert 0.0 <= res_mixed.confidence <= 1.0

    def test_vision_preprocessor_nan_inf_crops(self) -> None:
        """VisionPreprocessor methods must safely process NaN/Inf image arrays and extreme ROIs."""
        vp = VisionPreprocessor()

        # To grayscale with NaN / Inf
        nan_img = np.full((60, 60, 3), np.nan, dtype=np.float32)
        gray_nan = vp.to_grayscale(nan_img)
        assert gray_nan.dtype == np.uint8
        assert gray_nan.shape == (60, 60)

        inf_img = np.full((60, 60, 3), np.inf, dtype=np.float32)
        gray_inf = vp.to_grayscale(inf_img)
        assert gray_inf.dtype == np.uint8
        assert gray_inf.shape == (60, 60)

        # Adaptive threshold
        thresh = vp.adaptive_threshold(gray_nan, method="otsu")
        assert thresh.shape == (60, 60)

        # Crop with NaN/Inf ROI
        base_img = np.zeros((100, 100, 3), dtype=np.uint8)
        nan_roi = Rect(x=int(float("nan")) if not math.isnan(float("nan")) else 0, y=0, width=20, height=20)
        # Using NormalizedRect with NaNs
        n_nan_roi = NormalizedRect(x=float("nan"), y=0.0, width=0.2, height=0.2)
        crop = vp.crop_roi(base_img, n_nan_roi)
        assert isinstance(crop, np.ndarray)
        assert crop.shape == (0, 0, 3)

    def test_vision_layout_analyzer_nan_inf_arrays(self) -> None:
        """BaseLayoutAnalyzer must handle NaN and Inf arrays gracefully without raising uncaught errors."""
        bla = BaseLayoutAnalyzer()
        nan_img = np.full((120, 120, 3), np.nan, dtype=np.float32)
        layout_nan = bla.analyze_layout(nan_img)
        assert isinstance(layout_nan, BaseLayoutFeatures)
        assert 0.0 <= layout_nan.exposed_collector_ratio <= 1.0

        inf_img = np.full((120, 120, 3), np.inf, dtype=np.float32)
        layout_inf = bla.analyze_layout(inf_img)
        assert isinstance(layout_inf, BaseLayoutFeatures)
        assert 0.0 <= layout_inf.exposed_collector_ratio <= 1.0

    def test_vision_non_contiguous_and_fortran_arrays(self) -> None:
        """Non-contiguous (sliced, strided, Fortran-ordered) arrays must be processed without layout errors."""
        sc = StateClassifier()
        vp = VisionPreprocessor()

        # Step 1: Fortran contiguous array
        c_frame = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
        f_frame = np.asfortranarray(c_frame)
        assert not f_frame.flags["C_CONTIGUOUS"]
        assert f_frame.flags["F_CONTIGUOUS"]

        res_f = sc.classify(f_frame)
        assert isinstance(res_f, VisionAnalysisResult)
        gray_f = vp.to_grayscale(f_frame)
        assert gray_f.shape == (120, 160)

        # Step 2: Sliced non-contiguous array (negative strides)
        sliced_frame = c_frame[::-1, ::-1, :]
        assert not sliced_frame.flags["C_CONTIGUOUS"]
        assert not sliced_frame.flags["F_CONTIGUOUS"]

        res_sliced = sc.classify(sliced_frame)
        assert isinstance(res_sliced, VisionAnalysisResult)
        gray_sliced = vp.to_grayscale(sliced_frame)
        assert gray_sliced.shape == (120, 160)

    def test_vision_uniform_extreme_arrays(self) -> None:
        """Solid black (all-0) and solid white (all-255) frames must not cause divide-by-zero or crashes."""
        sc = StateClassifier()
        vp = VisionPreprocessor()
        bla = BaseLayoutAnalyzer()

        # Pure black frame
        black = np.zeros((200, 200, 3), dtype=np.uint8)
        res_black = sc.classify(black)
        assert isinstance(res_black, VisionAnalysisResult)
        assert 0.0 <= res_black.confidence <= 1.0
        gray_black = vp.to_grayscale(black)
        assert gray_black.max() == 0
        layout_black = bla.analyze_layout(black)
        assert layout_black.exposed_collector_ratio == 0.0

        # Pure white frame
        white = np.full((200, 200, 3), 255, dtype=np.uint8)
        res_white = sc.classify(white)
        assert isinstance(res_white, VisionAnalysisResult)
        assert 0.0 <= res_white.confidence <= 1.0
        gray_white = vp.to_grayscale(white)
        assert gray_white.min() == 255
        layout_white = bla.analyze_layout(white)
        assert 0.0 <= layout_white.exposed_collector_ratio <= 1.0

    def test_vision_channel_and_dimension_mutations(self) -> None:
        """StateClassifier and VisionPreprocessor must handle 1D, 2D single-channel, 4-channel BGRA, and 0-size."""
        sc = StateClassifier()
        vp = VisionPreprocessor()

        # 1D array
        arr_1d = np.zeros(500, dtype=np.uint8)
        res_1d = sc.classify(arr_1d)
        assert res_1d.state == GameState.UNKNOWN
        gray_1d = vp.to_grayscale(arr_1d)
        assert gray_1d.shape == (0, 0)

        # 2D single channel
        arr_2d = np.zeros((100, 100), dtype=np.uint8)
        res_2d = sc.classify(arr_2d)
        assert isinstance(res_2d, VisionAnalysisResult)
        gray_2d = vp.to_grayscale(arr_2d)
        assert gray_2d.shape == (100, 100)

        # 4-channel BGRA
        arr_bgra = np.zeros((100, 100, 4), dtype=np.uint8)
        res_bgra = sc.classify(arr_bgra)
        assert isinstance(res_bgra, VisionAnalysisResult)
        gray_bgra = vp.to_grayscale(arr_bgra)
        assert gray_bgra.shape == (100, 100)

        # 0-size array
        arr_0 = np.zeros((0, 0, 3), dtype=np.uint8)
        res_0 = sc.classify(arr_0)
        assert res_0.state == GameState.UNKNOWN
        assert res_0.confidence == 0.0
        gray_0 = vp.to_grayscale(arr_0)
        assert gray_0.shape == (0, 0)

    def test_vision_unsupported_dtypes_handling(self) -> None:
        """Verify dtype handling: VisionPreprocessor normalizes all dtypes; StateClassifier rejects non-CV depths."""
        vp = VisionPreprocessor()
        sc = StateClassifier()

        dtypes_to_test = [np.float32, np.float64, np.int32, np.uint16, bool]
        for dt in dtypes_to_test:
            arr = np.zeros((40, 40, 3), dtype=dt)
            # VisionPreprocessor guarantees conversion to uint8
            gray = vp.to_grayscale(arr)
            assert gray.dtype == np.uint8
            assert gray.shape == (40, 40)

        # Verified hardening: StateClassifier directly on float32/float64/int32/uint16/bool
        # Must gracefully sanitize non-uint8 depths without raising cv2.error
        for dt in [np.float32, np.float64, np.int32, np.uint16, bool]:
            arr_dt = np.zeros((50, 50, 3), dtype=dt)
            res = sc.classify(arr_dt)
            assert isinstance(res, VisionAnalysisResult)
            assert 0.0 <= res.confidence <= 1.0

    # =========================================================================
    # 3. OCR Strings with Unicode Fuzzing & Confusables
    # =========================================================================

    def test_numeric_parser_unicode_fuzzing(self) -> None:
        """Numeric parser must exhibit zero hallucination when presented with zero-width characters and confusables."""
        fuzzed_inputs = [
            "\u200b800,000",       # Leading zero-width space
            "800,000\u200b",       # Trailing zero-width space
            "800\u200b000",        # Infix zero-width space replacing comma
            "\ufeff800,000",       # Byte Order Mark (BOM)
            "\u200c800000",        # Zero-width non-joiner
            "800000\u200d",        # Zero-width joiner
            "\u200e800,000",       # Left-to-right mark
            "\u200f800,000",       # Right-to-left mark
            "\u2060800000",        # Word joiner
            "💰 800,000 🪙",        # Emojis surrounding numbers
            "⚔️ 1.2M 🛡️",          # Combat emojis
            "VIII C",              # Roman numerals
            "1.2\u043c",           # Cyrillic small letter em
            "800\u041c",           # Cyrillic capital EM
            "800\u043a",           # Cyrillic small ka
            "١٢٣٤٥٦",              # Arabic-Indic digits
            "一二三四五",          # Chinese ideographic numerals
            "\x00800,000",         # Embedded NULL byte
            "800\x00000",          # Null byte separator
        ]
        for raw in fuzzed_inputs:
            res = parse_resource_number(raw)
            # Confusable strings, emojis, zero-width spaces, and roman numerals must NOT hallucinate arbitrary numbers
            if "VIII" in raw or "💰" in raw or "⚔️" in raw or "\u200b" in raw or "\ufeff" in raw or "\u4e00" in raw:
                assert res is None, f"Zero-hallucination violated on {repr(raw)}: got {res}"
            elif raw == "\u0661\u0662\u0663\u0664\u0665\u0666":
                # Python's int() natively converts Arabic-Indic digits to 123456
                assert res in (123456, None)
            else:
                assert res is None or isinstance(res, int)

            parsed = parse_numeric(raw)
            if res is None:
                assert parsed.success is False
                assert parsed.value is None
            else:
                assert parsed.success is True
                assert parsed.value == res

    def test_numeric_parser_bizarre_separators(self) -> None:
        """Numeric parser must reject mixed or consecutive separators to prevent false OCR readings."""
        invalid_separator_strings = [
            "800,,000",
            "800...000",
            "800..000",
            "800.,000",
            "800,.000",
            "800.000,00",        # Mixed dots and commas
            "800,000.00",        # Mixed commas and decimal dots
            ",800,000",          # Leading separator
            ".800.000",          # Leading dot
            "800,000,",          # Trailing comma
            "800.000.",          # Trailing dot
            "800,,000,000",
            "1,2,3,4,5",         # Malformed grouping
        ]
        for s in invalid_separator_strings:
            res = parse_resource_number(s)
            assert res is None, f"Zero-hallucination failed: '{s}' parsed as {res}"

    def test_numeric_parser_compact_notation_and_billions(self) -> None:
        """Numeric parser must correctly translate compact K/M notations and multi-billion integer strings."""
        test_cases = [
            ("800K", 800_000),
            ("800k", 800_000),
            ("350K", 350_000),
            ("5.5K", 5_500),
            ("1.2M", 1_200_000),
            ("1.25m", 1_250_000),
            ("2.5M", 2_500_000),
            ("9999M", 9_999_000_000),
            ("2,000,000,000", 2_000_000_000),
            ("999,999,999,999", 999_999_999_999),
            ("0K", 0),
            ("0M", 0),
            ("0", 0),
            ("Gold: 800,000", 800_000),
            ("Elixir: 1.2M", 1_200_000),
            ("Dark Elixir: 15,000", 15_000),
            ("DE: 5,500", 5_500),
        ]
        for text, expected in test_cases:
            val = parse_resource_number(text)
            assert val == expected, f"Failed on '{text}': got {val}, expected {expected}"
            parsed = parse_numeric(text)
            assert parsed.success is True
            assert parsed.value == expected

    def test_numeric_parser_negative_and_zero_protection(self) -> None:
        """Negative numbers and unicode minus characters must strictly return None."""
        negatives = [
            "-800,000",
            "-1",
            "-0",
            "-1.2M",
            "-800K",
            "--800000",
            "\u2212800,000",     # Unicode minus U+2212
            "\u2013800,000",     # En-dash U+2013
            "\u2014800,000",     # Em-dash U+2014
        ]
        for s in negatives:
            res = parse_resource_number(s)
            assert res is None, f"Negative value permitted: '{s}' -> {res}"
            parsed = parse_numeric(s)
            assert parsed.success is False
            assert parsed.value is None

    def test_numeric_parser_non_string_type_safety(self) -> None:
        """Numeric parser must gracefully handle non-string types without raising exceptions."""
        non_strings = [
            None,
            12345,
            -500,
            0,
            True,
            False,
            float("nan"),
            float("inf"),
            float("-inf"),
            [800, 0],
            {"gold": "800,000"},
            object(),
        ]
        for item in non_strings:
            res = parse_resource_number(item)
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                assert res == item
            else:
                assert res is None, f"Non-string item {repr(item)} unexpectedly parsed as {res}"

            parsed = parse_numeric(item)
            assert isinstance(parsed, ParsedNumericResult)
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                assert parsed.success is True
                assert parsed.value == item
            else:
                assert parsed.success is False
                assert parsed.value is None

    def test_numeric_parser_battle_timer_fuzzing(self) -> None:
        """Battle timer parser must accurately convert M:SS formats and reject corrupted patterns."""
        valid_timers = [
            ("2:45", 165),
            ("0:05", 5),
            ("3:00", 180),
            ("0:00", 0),
            ("10:00", 600),
            ("120:00", 7200),
        ]
        for raw, expected in valid_timers:
            sec = parse_timer_seconds(raw)
            assert sec == expected, f"Timer '{raw}' expected {expected}, got {sec}"
            parsed = parse_numeric(raw)
            assert parsed.success is True
            assert parsed.is_timer is True
            assert parsed.value == expected

        invalid_timers = [
            ":45",
            "2:",
            "2:4",
            "2:455",
            "abc:def",
            "2:60:00",
            "::",
            "2:-10",
            "",
            None,
        ]
        for inv in invalid_timers:
            sec = parse_timer_seconds(inv)
            assert sec is None, f"Invalid timer '{inv}' parsed as {sec}"

    def test_ocr_engine_abstraction_and_fallback(self) -> None:
        """OCREngine factory must instantiate and fall back seamlessly when native tesseract is missing."""
        mock_engine = get_ocr_engine("mock", default_text="950,000")
        assert isinstance(mock_engine, MockOCREngine)
        assert mock_engine.is_available() is True
        assert mock_engine.recognize_text(np.zeros((10, 10, 3), dtype=np.uint8)) == "950,000"

        # Auto mode returns available engine or fallback
        auto_engine = get_ocr_engine("auto", default_text="1,200,000")
        assert isinstance(auto_engine, BaseOCREngine)
        assert auto_engine.is_available() is True
        text, conf = auto_engine.recognize_with_confidence(np.zeros((10, 10, 3), dtype=np.uint8))
        assert len(text) > 0
        assert 0.0 <= conf <= 1.0

    # =========================================================================
    # 4. Extreme Loot Combinations & Evaluator Thresholds
    # =========================================================================

    def test_evaluator_multi_billion_loot(self) -> None:
        """BaseEvaluator and RecommendationEngine must clamp scores to [0.0, 100.0] on multi-billion loot."""
        evaluator = BaseEvaluator()
        rec_engine = RecommendationEngine()

        reading = ResourceReading(
            gold=2_000_000_000,
            elixir=2_000_000_000,
            dark_elixir=10_000_000,
        )
        layout = BaseLayoutFeatures(exposed_collector_ratio=1.0, th_accessible=True)

        res = evaluator.evaluate(reading, layout)
        assert res.meets_thresholds is True
        assert 0.0 <= res.normalized_score <= 100.0
        assert math.isclose(res.normalized_score, 100.0, abs_tol=1e-2)

        rec = rec_engine.evaluate_base(reading, layout)
        assert 0.0 <= rec.suitability_score <= 100.0
        assert rec.primary_strategy in StrategyType
        assert len(rec.rankings) == 5
        assert len(rec.deployment_plan) > 0

    def test_evaluator_all_negative_and_zero_loot(self) -> None:
        """Zero-loot and all-negative readings must result in unviable status and 0.0 suitability score."""
        evaluator = BaseEvaluator()
        rec_engine = RecommendationEngine()

        zero_cases = [
            ResourceReading(gold=0, elixir=0, dark_elixir=0),
            ResourceReading(gold=-500_000, elixir=-500_000, dark_elixir=-5_000),
            ResourceReading(gold=-100, elixir=0, dark_elixir=0),
            ResourceReading(gold=None, elixir=None, dark_elixir=None),
        ]
        for reading in zero_cases:
            ev = evaluator.evaluate(reading)
            assert ev.meets_thresholds is False
            assert ev.normalized_score == 0.0
            assert ev.resource_score == 0.0

            rec = rec_engine.evaluate_base(reading)
            assert rec.suitability_score == 0.0
            assert len(rec.deployment_plan) == 0
            for r in rec.rankings:
                assert r.viable is False
                assert r.score == 0.0

    def test_evaluator_missing_and_partial_loot(self) -> None:
        """Missing individual resource fields (None) must be handled gracefully without exceptions."""
        evaluator = BaseEvaluator()
        rec_engine = RecommendationEngine()

        # Only gold present
        r_gold = ResourceReading(gold=1_000_000, elixir=None, dark_elixir=None)
        ev_gold = evaluator.evaluate(r_gold)
        assert ev_gold.meets_thresholds is True
        assert ev_gold.normalized_score > 0.0
        rec_gold = rec_engine.evaluate_base(r_gold)
        assert rec_gold.suitability_score > 0.0

        # Only elixir present
        r_elixir = ResourceReading(gold=None, elixir=1_000_000, dark_elixir=None)
        ev_elixir = evaluator.evaluate(r_elixir)
        assert ev_elixir.meets_thresholds is True
        assert ev_elixir.normalized_score > 0.0

        # Only dark elixir present
        r_de = ResourceReading(gold=None, elixir=None, dark_elixir=15_000)
        thresh_de = ResourceThresholds(priority=ResourcePriority.DARK_ELIXIR_FIRST)
        ev_de = evaluator.evaluate(r_de, thresholds=thresh_de)
        assert ev_de.meets_thresholds is True
        assert ev_de.normalized_score > 0.0

    def test_evaluator_non_finite_loot_numbers(self) -> None:
        """NaNs, Infs, and extreme floats in ResourceReading must be sanitized without ValueError/OverflowError."""
        evaluator = BaseEvaluator()
        rec_engine = RecommendationEngine()

        nan_reading = ResourceReading(
            gold=float("nan"),  # type: ignore
            elixir=float("inf"),  # type: ignore
            dark_elixir=float("-inf"),  # type: ignore
        )
        ev = evaluator.evaluate(nan_reading)
        assert ev.meets_thresholds is False
        assert ev.normalized_score == 0.0

        rec = rec_engine.evaluate_base(nan_reading)
        assert rec.suitability_score == 0.0
        assert len(rec.deployment_plan) == 0

    def test_evaluator_priority_threshold_satisfaction(self) -> None:
        """Threshold satisfaction must strictly respect active ResourcePriority enum rules."""
        evaluator = BaseEvaluator()

        reading = ResourceReading(gold=600_000, elixir=200_000, dark_elixir=1_000)
        t_gold = ResourceThresholds(
            min_gold=500_000, min_elixir=400_000, min_dark_elixir=3_000,
            priority=ResourcePriority.GOLD_FIRST,
        )
        t_elixir = ResourceThresholds(
            min_gold=500_000, min_elixir=400_000, min_dark_elixir=3_000,
            priority=ResourcePriority.ELIXIR_FIRST,
        )
        t_dark = ResourceThresholds(
            min_gold=500_000, min_elixir=400_000, min_dark_elixir=3_000,
            priority=ResourcePriority.DARK_ELIXIR_FIRST,
        )

        assert evaluator.evaluate(reading, thresholds=t_gold).meets_thresholds is True
        assert evaluator.evaluate(reading, thresholds=t_elixir).meets_thresholds is False
        assert evaluator.evaluate(reading, thresholds=t_dark).meets_thresholds is False

    # =========================================================================
    # 5. Strategy Scoring Invariants and Tie-Breaking Stability
    # =========================================================================

    def test_strategy_scoring_bounds_all_5_strategies(self) -> None:
        """All 5 strategies must evaluate strictly within [0.0, 100.0] score bounds for any valid or boundary input."""
        strats = [
            SneakyGoblinStrategy(),
            BarchStrategy(),
            MassMinerStrategy(),
            MassDragonStrategy(),
            CustomStrategy(),
        ]
        readings = [
            ResourceReading(gold=100_000, elixir=100_000, dark_elixir=500),
            ResourceReading(gold=500_000, elixir=500_000, dark_elixir=3_000),
            ResourceReading(gold=1_500_000, elixir=1_500_000, dark_elixir=10_000),
            ResourceReading(gold=50_000_000, elixir=50_000_000, dark_elixir=500_000),
            ResourceReading(gold=0, elixir=0, dark_elixir=0),
        ]
        layouts = [
            None,
            BaseLayoutFeatures(exposed_collector_ratio=0.0, th_accessible=False),
            BaseLayoutFeatures(exposed_collector_ratio=0.5, th_accessible=True),
            BaseLayoutFeatures(exposed_collector_ratio=1.0, th_accessible=True, metadata={"dead_base": True}),
        ]

        for strat in strats:
            for r in readings:
                for l in layouts:
                    ev = strat.evaluate(r, l)
                    assert isinstance(ev, StrategyEvaluation)
                    assert 0.0 <= ev.score <= 100.0, f"{strat.name}: score {ev.score} out of [0, 100]"
                    assert 0.0 <= ev.resource_suitability <= 100.0
                    assert 0.0 <= ev.layout_suitability <= 100.0
                    assert ev.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.EXTREME)
                    assert isinstance(ev.viable, bool)

    def test_strategy_tie_breaking_deterministic_stability(self) -> None:
        """When multiple strategies tie at identical scores, ranking and selection must be 100% deterministic."""
        class FixedMockStrategy(BaseStrategy):
            def __init__(self, stype: StrategyType, fixed_score: float) -> None:
                self._stype = stype
                self._score = fixed_score

            @property
            def strategy_type(self) -> StrategyType:
                return self._stype

            @property
            def name(self) -> str:
                return self._stype.name

            @property
            def description(self) -> str:
                return ""

            @property
            def army_composition(self) -> Dict[str, int]:
                return {"Troop": 10}

            def evaluate(self, resources: Optional[ResourceReading], layout: Optional[BaseLayoutFeatures], thresholds: Optional[ResourceThresholds] = None) -> StrategyEvaluation:
                return StrategyEvaluation(
                    strategy_type=self._stype,
                    score=self._score,
                    viable=True,
                    risk_level=RiskLevel.LOW,
                    resource_suitability=self._score,
                    layout_suitability=self._score,
                )

            def generate_deployment_plan(self, layout: Optional[BaseLayoutFeatures], canvas_width: int = 1920, canvas_height: int = 1080) -> List[DeploymentStep]:
                return []

        rec_engine = RecommendationEngine()
        # Register 5 strategies all fixed to identical score 80.0
        for stype in StrategyType:
            rec_engine.register_strategy(FixedMockStrategy(stype, 80.0))

        reading = ResourceReading(gold=500_000, elixir=500_000, dark_elixir=3_000)

        # Execute 100 consecutive evaluations
        observed_winners: List[StrategyType] = []
        for _ in range(100):
            res = rec_engine.evaluate_base(reading)
            observed_winners.append(res.primary_strategy)

        # Every single run must yield the exact same deterministic primary strategy
        assert len(set(observed_winners)) == 1, f"Tie-breaking unstable: observed {set(observed_winners)}"
        assert observed_winners[0] == StrategyType.SNEAKY_GOBLIN

    def test_strategy_all_unviable_zero_loot_invariant(self) -> None:
        """When loot is zero, all strategies must report unviable and recommendation must be zero."""
        rec_engine = RecommendationEngine()
        zero_reading = ResourceReading(gold=0, elixir=0, dark_elixir=0)

        rec = rec_engine.evaluate_base(zero_reading)
        assert rec.suitability_score == 0.0
        assert len(rec.deployment_plan) == 0
        for ev in rec.rankings:
            assert ev.viable is False
            assert ev.score == 0.0

    def test_strategy_deployment_plan_structure(self) -> None:
        """Deployment steps generated by strategies must have strictly valid structure."""
        strats = [
            SneakyGoblinStrategy(),
            BarchStrategy(),
            MassMinerStrategy(),
            MassDragonStrategy(),
            CustomStrategy(),
        ]
        layout = BaseLayoutFeatures(
            exposed_collector_ratio=0.8,
            th_accessible=True,
            air_defense_locations=[(0.4, 0.4), (0.6, 0.6)],
            compartment_count=4,
        )

        for strat in strats:
            plan = strat.generate_deployment_plan(layout, canvas_width=1920, canvas_height=1080)
            assert len(plan) > 0
            for idx, step in enumerate(plan):
                assert step.step_id == idx + 1
                assert len(step.troop_type) > 0
                assert step.unit_count > 0
                assert step.delay_ms >= 0
                assert 0 <= step.target_x <= 1920
                assert 0 <= step.target_y <= 1080
                assert 0.0 <= step.normalized_x <= 1.0
                assert 0.0 <= step.normalized_y <= 1.0

    def test_custom_strategy_extreme_configs(self) -> None:
        """CustomStrategy must survive extreme user configurations without crashing."""
        extreme_configs = [
            CustomStrategyConfig(name="Empty Army", army_composition={}),
            CustomStrategyConfig(
                name="Massive Army",
                army_composition={"Super Barbarian": 10_000, "Super Archer": 10_000},
                training_time_seconds=99_999,
                elixir_cost=50_000_000,
            ),
            CustomStrategyConfig(
                name="Strict TH Requirement",
                requires_accessible_th=True,
                max_acceptable_risk=RiskLevel.LOW,
            ),
            CustomStrategyConfig(
                name="Custom Steps",
                custom_steps=[
                    {"troop_type": "Hero", "unit_count": 1, "target_x": 100, "target_y": 100, "normalized_x": 0.1, "normalized_y": 0.1},
                    {"troop_type": "Zero Unit", "unit_count": 0},  # Filtered out
                ],
            ),
        ]
        reading = ResourceReading(gold=500_000, elixir=500_000, dark_elixir=3_000)
        layout = BaseLayoutFeatures(exposed_collector_ratio=0.5, th_accessible=False)

        for cfg in extreme_configs:
            strat = CustomStrategy(cfg)
            ev = strat.evaluate(reading, layout)
            assert isinstance(ev, StrategyEvaluation)
            assert 0.0 <= ev.score <= 100.0

            plan = strat.generate_deployment_plan(layout, canvas_width=1920, canvas_height=1080)
            assert isinstance(plan, list)

    def test_strategy_layout_coordinate_anomalies_and_boundaries(self) -> None:
        """Empirically document behavior when layout contains anomalous or out-of-bounds Air Defense coordinates."""
        # Case A: Negative coordinates in Air Defense locations
        neg_layout = BaseLayoutFeatures(air_defense_locations=[(-100, -200)])
        dragon = MassDragonStrategy()
        plan_neg = dragon.generate_deployment_plan(neg_layout, canvas_width=1920, canvas_height=1080)
        assert len(plan_neg) > 0

        # Case B: Layout ADs in reference space (e.g. 500, 500) on a 320x240 canvas
        # Verified hardening: coordinates must strictly clamp to canvas dimensions [0, cw] and [0, ch]
        scaled_layout = BaseLayoutFeatures(air_defense_locations=[(500, 500)])
        plan_scaled = dragon.generate_deployment_plan(scaled_layout, canvas_width=320, canvas_height=240)
        assert len(plan_scaled) > 0
        for step in plan_scaled:
            assert isinstance(step.target_x, int)
            assert isinstance(step.target_y, int)
            assert 0 <= step.target_x <= 320, f"target_x out of bounds: {step.target_x}"
            assert 0 <= step.target_y <= 240, f"target_y out of bounds: {step.target_y}"
            assert 0.0 <= step.normalized_x <= 1.0, f"normalized_x out of bounds: {step.normalized_x}"
            assert 0.0 <= step.normalized_y <= 1.0, f"normalized_y out of bounds: {step.normalized_y}"

        # Case C: Non-finite NaN and Inf coordinates in layout Air Defenses
        nan_layout = BaseLayoutFeatures(
            air_defense_locations=[(float("nan"), float("nan")), (float("inf"), float("-inf"))]
        )
        plan_nan = dragon.generate_deployment_plan(nan_layout, canvas_width=320, canvas_height=240)
        assert len(plan_nan) > 0
        for step in plan_nan:
            assert math.isfinite(step.target_x) and 0 <= step.target_x <= 320
            assert math.isfinite(step.target_y) and 0 <= step.target_y <= 240
            assert math.isfinite(step.normalized_x) and 0.0 <= step.normalized_x <= 1.0
            assert math.isfinite(step.normalized_y) and 0.0 <= step.normalized_y <= 1.0

    def test_custom_strategy_nan_and_boundary_hardening(self) -> None:
        """CustomStrategy deployment plan generation must sanitize NaNs, Infs, and clamp to canvas bounds."""
        cfg = CustomStrategyConfig(
            custom_steps=[
                {"troop_type": "Barbarian", "unit_count": 1, "normalized_x": float("nan"), "normalized_y": float("inf")},
                {"troop_type": "Archer", "unit_count": 2, "target_x": float("nan"), "target_y": 5000},
                {"troop_type": "Giant", "unit_count": 1, "target_x": -100, "target_y": -200},
                {"troop_type": "Goblin", "unit_count": 5, "normalized_x": 1.5, "normalized_y": -0.2},
            ]
        )
        strat = CustomStrategy(cfg)
        plan = strat.generate_deployment_plan(None, canvas_width=320, canvas_height=240)
        assert len(plan) == 4
        for step in plan:
            assert isinstance(step.target_x, int)
            assert isinstance(step.target_y, int)
            assert math.isfinite(step.target_x) and 0 <= step.target_x <= 320
            assert math.isfinite(step.target_y) and 0 <= step.target_y <= 240
            assert math.isfinite(step.normalized_x) and 0.0 <= step.normalized_x <= 1.0
            assert math.isfinite(step.normalized_y) and 0.0 <= step.normalized_y <= 1.0

    def test_layout_analyzer_non_uint8_dtypes_hardening(self) -> None:
        """BaseLayoutAnalyzer must sanitize non-uint8 and non-finite image arrays without crashing."""
        analyzer = BaseLayoutAnalyzer()
        for dt in [np.float32, np.float64, np.int32, np.uint16, bool]:
            arr = np.zeros((100, 100, 3), dtype=dt)
            features = analyzer.analyze_layout(arr)
            assert isinstance(features, BaseLayoutFeatures)

        # Floating point with NaNs and Infs
        arr_nan = np.full((100, 100, 3), float("nan"), dtype=np.float64)
        arr_nan[10:20, 10:20] = float("inf")
        features_nan = analyzer.analyze_layout(arr_nan)
        assert isinstance(features_nan, BaseLayoutFeatures)

    # =========================================================================
    # 6. Capture & Safety Guard Concurrency & Invariants
    # =========================================================================

    def test_safety_guard_concurrent_access_stress(self) -> None:
        """SafetyGuard must maintain thread safety under concurrent verification, halts, and window updates."""
        sg = SafetyGuard()
        sg.clear_all()
        sg.register_simulator_window(9001)
        sg.register_live_window(5001, "Google Play Games", "PlayGamesPC.exe")

        errors: List[Exception] = []
        violations_caught: List[str] = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(50):
                    # Attempt simulator action
                    sg.verify_and_execute_input(action_type="click", target_x=100, target_y=100, is_simulator=True)
                    # Attempt live action (must be blocked)
                    try:
                        sg.verify_and_execute_input(action_type="click", target_x=100, target_y=100, is_simulator=False)
                    except LiveInputSafetyViolation:
                        violations_caught.append(f"T{thread_id}")

                    # Emergency stop toggling
                    if thread_id == 0 and i == 25:
                        sg.trigger_emergency_stop("Stress test halt")
                        time.sleep(0.01)
                        sg.reset_emergency_stop()

            except EmergencyStopTriggered:
                # Allowed when emergency stop is active
                pass
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "SafetyGuard stress thread hung"

        assert len(errors) == 0, f"Errors in concurrent SafetyGuard: {errors}"
        assert len(violations_caught) > 0, "No violations intercepted"

    def test_safety_guard_live_window_tamper_resistance(self) -> None:
        """Registered live HWNDs must strictly reject any automated input attempt regardless of parameters."""
        sg = SafetyGuard()
        sg.clear_all()
        sg.register_live_window(6666, "Live Game Window", "PlayGamesPC.exe")

        # 1. Direct verify_input_allowed
        with pytest.raises(LiveInputSafetyViolation):
            sg.verify_input_allowed(target_hwnd=6666, action_name="pyautogui.click")

        # 2. verify_and_execute_input outside simulator
        with pytest.raises(LiveInputSafetyViolation):
            sg.verify_and_execute_input(action_type="click", target_x=500, target_y=500, is_simulator=False)

        # 3. Global desktop untargeted action while live window is registered
        with pytest.raises(LiveInputSafetyViolation):
            sg.verify_input_allowed(target_hwnd=None, action_name="pyautogui.moveTo")

    def test_safety_guard_default_deny_mode(self) -> None:
        """SafetyGuard default-deny mode must reject uninitialized or non-simulator inputs."""
        sg = SafetyGuard(default_deny=True)
        sg.clear_all()
        sg.set_default_deny(True)

        with pytest.raises(LiveInputSafetyViolation):
            sg.verify_input_allowed(target_hwnd=1234, action_name="click")

        # Registering simulator HWND allows input
        sg.register_simulator_window(8888)
        sg.verify_input_allowed(target_hwnd=8888, action_name="simulator_click")

    def test_mss_grabber_rapid_lifecycle(self) -> None:
        """MSSGrabber must survive rapid sequential start/stop cycles and auto-pause when window is minimized."""
        grabber = MSSGrabber(target_fps=30, mock_mode=True, auto_pause_on_minimized=True)

        for _ in range(25):
            grabber.start()
            assert grabber.is_running is True
            grabber.stop()
            assert grabber.is_running is False

        # Minimized window auto-pause
        min_win = WindowInfo(
            hwnd=4001,
            client_rect=Rect(0, 0, 800, 600),
            is_visible=True,
            is_minimized=True,
        )
        grabber.start()
        try:
            packet = grabber.grab_frame(min_win)
            assert packet is None
            assert grabber.is_paused is True
            grabber.resume()
            assert grabber.is_paused is False
        finally:
            grabber.stop()
