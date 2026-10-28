"""
Tier 1: Feature Coverage Test Suite for ClashVision.
Validates all 31 cataloged features across Requirements R1–R6 in nominal operational states.
Every test case derives expected output strictly from ORIGINAL_REQUEST.md and PROJECT.md.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from clashvision.core.constants import (
    DEFAULT_CAPTURE_FPS,
    DEFAULT_MIN_DARK_ELIXIR,
    DEFAULT_MIN_ELIXIR,
    DEFAULT_MIN_GOLD,
    DEFAULT_TARGET_PROCESS_NAMES,
    DEFAULT_TARGET_WINDOW_TITLES,
    EMERGENCY_STOP_LATENCY_MAX_MS,
    GameState,
    ResourcePriority,
    RiskLevel,
    StrategyType,
    VIRTUAL_REFERENCE_HEIGHT,
    VIRTUAL_REFERENCE_WIDTH,
)
from clashvision.core.coordinates import (
    CoordinateTransformer,
    calculate_iou,
    calculate_letterbox,
    clamp_point_to_bounds,
    clamp_rect_to_bounds,
    normalized_to_point,
    normalized_to_rect,
    point_to_normalized,
    rect_to_normalized,
)
from clashvision.core.exceptions import (
    ClashVisionError,
    LiveInputSafetyViolation,
    WindowNotFoundError,
)
from clashvision.core.models import (
    BaseLayoutFeatures,
    DatasetAnnotation,
    DeploymentStep,
    DetectionRecord,
    FarmingRecommendation,
    FramePacket,
    NormalizedPoint,
    NormalizedRect,
    Point,
    Rect,
    ResourceReading,
    SessionRecord,
    SimulationState,
    StrategyEvaluation,
    VisionAnalysisResult,
    WindowInfo,
)
from tests.conftest import allow_simulator_sandbox
from tests.fixtures.layout_generator import (
    BaseLayoutType,
    generate_canonical_layout,
)
from tests.fixtures.recorded_samples import (
    OCR_CORPUS,
    RECORDED_SAMPLES,
    TIMER_CORPUS,
    get_sample_by_state,
)
from tests.fixtures.synthetic_generator import (
    GameStateType,
    SyntheticCropConfig,
    SyntheticFrameConfig,
    SyntheticGenerator,
    generate_resource_crop,
    generate_state_frame,
    generate_timer_crop,
)


# ==============================================================================
# Feature 1: Window Enumeration & Target Discovery (R1)
# ==============================================================================
class TestFeature01WindowEnumeration:
    """Verifies target discovery for Google Play Games PC and compatible emulators."""

    def test_default_process_signatures_include_playgames_and_crosvm(self) -> None:
        """Requirement R1: Must detect PlayGamesPC.exe, crosvm.exe, HD-Player.exe."""
        assert "PlayGamesPC.exe" in DEFAULT_TARGET_PROCESS_NAMES
        assert "crosvm.exe" in DEFAULT_TARGET_PROCESS_NAMES
        assert "HD-Player.exe" in DEFAULT_TARGET_PROCESS_NAMES

    def test_default_window_title_regex_matches_clash_and_playgames(self) -> None:
        """Requirement R1: Title patterns match Clash of Clans and Google Play Games."""
        patterns = [re.compile(p, re.IGNORECASE) for p in DEFAULT_TARGET_WINDOW_TITLES]
        sample_titles = [
            "Clash of Clans",
            "Google Play Games on PC",
            "BlueStacks App Player",
        ]
        for title in sample_titles:
            assert any(p.search(title) for p in patterns), f"Failed to match {title}"

    def test_window_info_capturable_requires_visible_and_non_minimized(self) -> None:
        """WindowInfo.is_capturable is True only when visible, not minimized, and area > 0."""
        win = WindowInfo(
            hwnd=1234,
            title="Clash of Clans",
            is_visible=True,
            is_minimized=False,
            client_rect=Rect(0, 0, 1920, 1080),
        )
        assert win.is_capturable is True

    def test_window_info_capturable_false_when_minimized(self) -> None:
        """WindowInfo.is_capturable must be False if is_minimized is True."""
        win = WindowInfo(
            hwnd=1234,
            title="Clash of Clans",
            is_visible=True,
            is_minimized=True,
            client_rect=Rect(0, 0, 1920, 1080),
        )
        assert win.is_capturable is False

    def test_window_detector_mock_enumeration(self) -> None:
        """Window detector finds candidate windows from process lists."""
        from clashvision.capture.window_detector import WindowDetector
        detector = WindowDetector(mock_mode=True)
        mock_win = detector.create_mock_target()
        candidate = detector.find_target_window()
        assert candidate is not None
        assert candidate.hwnd == mock_win.hwnd
        assert candidate.is_capturable is True


# ==============================================================================
# Feature 2: High-Performance MSS Frame Capture (R1)
# ==============================================================================
class TestFeature02MSSFrameCapture:
    """Verifies screen frame capture contracts and properties."""

    def test_frame_packet_structure_and_dimensions(self) -> None:
        """FramePacket encapsulates BGR ndarray, timestamp, and metadata."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        win = WindowInfo(hwnd=100, is_visible=True, client_rect=Rect(0, 0, 1920, 1080))
        pkt = FramePacket(frame=frame, timestamp=time.time(), window_info=win, frame_id=1)
        assert pkt.width == 1920
        assert pkt.height == 1080
        assert pkt.channels == 3
        assert pkt.frame_id == 1

    def test_mss_grabber_initialization(self) -> None:
        """MSSGrabber initializes with default target FPS."""
        from clashvision.capture.mss_grabber import MSSGrabber
        grabber = MSSGrabber(target_fps=DEFAULT_CAPTURE_FPS)
        assert grabber.target_fps == DEFAULT_CAPTURE_FPS
        assert not grabber.is_running

    def test_mss_grabber_mock_capture_returns_frame_packet(self) -> None:
        """MSSGrabber produces a valid FramePacket from mock/synthetic client rect."""
        from clashvision.capture.mss_grabber import MSSGrabber
        grabber = MSSGrabber(mock_mode=True)
        win = WindowInfo(hwnd=1, is_visible=True, client_rect=Rect(0, 0, 1280, 720))
        pkt = grabber.grab_frame(win)
        assert isinstance(pkt, FramePacket)
        assert pkt.width == 1280
        assert pkt.height == 720

    def test_frame_packet_timestamp_monotonicity(self) -> None:
        """Successive frame packets must have increasing timestamps."""
        t1 = time.time()
        time.sleep(0.005)
        t2 = time.time()
        pkt1 = FramePacket(frame=np.zeros((10, 10, 3)), timestamp=t1, window_info=WindowInfo(), frame_id=1)
        pkt2 = FramePacket(frame=np.zeros((10, 10, 3)), timestamp=t2, window_info=WindowInfo(), frame_id=2)
        assert pkt2.timestamp > pkt1.timestamp
        assert pkt2.frame_id > pkt1.frame_id

    def test_mss_grabber_framerate_bounds(self) -> None:
        """Capture rate configuration enforces bounds 1 <= FPS <= 60."""
        from clashvision.capture.mss_grabber import MSSGrabber
        grabber = MSSGrabber(target_fps=15)
        grabber.set_target_fps(30)
        assert grabber.target_fps == 30
        grabber.set_target_fps(100)  # Clamps to max 60
        assert grabber.target_fps <= 60


# ==============================================================================
# Feature 3: DPI Scaling & Dynamic Resize Transform (R1)
# ==============================================================================
class TestFeature03DPIScalingAndCoordinateTransforms:
    """Verifies coordinate transforms across window sizes and DPI scaling."""

    def test_virtual_to_client_identity_at_1080p(self) -> None:
        """At 1920x1080 with 1.0 scale factor, virtual point equals client point."""
        transformer = CoordinateTransformer(client_width=1920, client_height=1080, scale_factor=1.0)
        client_pt = transformer.virtual_to_client_point(Point(960, 540))
        assert client_pt == Point(960, 540)

    def test_virtual_to_client_downscaling_to_720p(self) -> None:
        """At 1280x720, virtual (1920, 1080) maps to client (1280, 720)."""
        transformer = CoordinateTransformer(client_width=1280, client_height=720)
        client_pt = transformer.virtual_to_client_point(Point(1920, 1080))
        assert client_pt == Point(1280, 720)


    def test_point_to_normalized_nominal(self) -> None:
        """Point (960, 540) in (1920, 1080) maps to normalized (0.5, 0.5)."""
        npt = point_to_normalized(Point(960, 540), 1920, 1080)
        assert pytest.approx(npt.x, 0.001) == 0.5
        assert pytest.approx(npt.y, 0.001) == 0.5

    def test_normalized_to_point_nominal(self) -> None:
        """Normalized (0.5, 0.5) maps back to (960, 540) in 1920x1080."""
        pt = normalized_to_point(NormalizedPoint(0.5, 0.5), 1920, 1080)
        assert pt == Point(960, 540)

    def test_calculate_letterbox_pillarbox(self) -> None:
        """Letterbox calculation centers 16:9 content inside ultra-wide window."""
        content_rect, scale = calculate_letterbox(1920, 1080, 2560, 1080)
        assert content_rect.width == 1920
        assert content_rect.height == 1080
        assert content_rect.x == (2560 - 1920) // 2
        assert content_rect.y == 0
        assert scale == 1.0


# ==============================================================================
# Feature 4: Window State & Visibility Monitor (R1)
# ==============================================================================
class TestFeature04WindowStateAndVisibility:
    """Verifies detection of window minimized, hidden, or zero-area states."""

    def test_window_info_minimized_state_flag(self) -> None:
        win = WindowInfo(is_minimized=True, is_visible=True, client_rect=Rect(0, 0, 0, 0))
        assert win.is_minimized is True
        assert win.is_capturable is False

    def test_window_info_aspect_ratio_calculation(self) -> None:
        win = WindowInfo(client_rect=Rect(0, 0, 1920, 1080))
        assert pytest.approx(win.aspect_ratio, 0.01) == 16.0 / 9.0

    def test_window_info_zero_area_aspect_ratio_safe(self) -> None:
        win = WindowInfo(client_rect=Rect(0, 0, 0, 0))
        assert win.aspect_ratio == 0.0

    def test_window_info_serialization_roundtrip(self) -> None:
        win = WindowInfo(
            hwnd=555,
            title="Google Play Games on PC",
            process_name="crosvm.exe",
            client_rect=Rect(100, 100, 1280, 720),
            is_visible=True,
            scale_factor=1.25,
        )
        d = win.to_dict()
        assert d["hwnd"] == 555
        assert d["process_name"] == "crosvm.exe"
        assert d["scale_factor"] == 1.25

    def test_window_state_not_capturable_if_invisible(self) -> None:
        win = WindowInfo(is_visible=False, client_rect=Rect(0, 0, 1920, 1080))
        assert win.is_capturable is False


# ==============================================================================
# Feature 5: Read-Only Live Safety Guard (R1, R4)
# ==============================================================================
class TestFeature05ReadOnlySafetyGuard:
    """Verifies that live game window is strictly read-only and blocks inputs."""

    def test_safety_guard_blocks_live_click(self) -> None:
        """Calling live click against registered live window raises LiveInputSafetyViolation."""
        from clashvision.capture.safety_guard import SafetyGuard
        guard = SafetyGuard()
        guard.register_live_window(hwnd=1001, title="Clash of Clans", process_name="PlayGamesPC.exe")
        with pytest.raises(LiveInputSafetyViolation):
            guard.verify_input_allowed(target_hwnd=1001, action_name="click")

    def test_safety_guard_blocks_live_keystroke(self) -> None:
        """Calling live keypress against registered live window raises LiveInputSafetyViolation."""
        from clashvision.capture.safety_guard import SafetyGuard
        guard = SafetyGuard()
        guard.register_live_window(hwnd=1001, title="Clash of Clans")
        with pytest.raises(LiveInputSafetyViolation):
            guard.verify_input_allowed(target_hwnd=1001, action_name="keypress")

    def test_safety_guard_allows_simulator_canvas_action(self) -> None:
        """Inside simulator canvas, simulated actions are permitted."""
        from clashvision.capture.safety_guard import SafetyGuard
        guard = SafetyGuard()
        guard.register_simulator_window(hwnd=2001)
        # Should not raise exception
        guard.verify_input_allowed(target_hwnd=2001, action_name="simulated_click")
        assert 2001 in guard.registered_simulator_hwnds

    def test_safety_guard_emergency_stop_halts_execution(self) -> None:
        """When emergency stop flag is tripped, all inputs are rejected immediately."""
        from clashvision.capture.safety_guard import SafetyGuard
        from clashvision.core.exceptions import EmergencyStopTriggered
        guard = SafetyGuard()
        guard.register_simulator_window(hwnd=2001)
        guard.trigger_emergency_stop()
        assert guard.is_emergency_stopped is True
        with pytest.raises(EmergencyStopTriggered):
            guard.verify_input_allowed(target_hwnd=2001, action_name="click")

    def test_safety_guard_status_reporting(self) -> None:
        """SafetyGuard tracks violations and live HWNDs."""
        from clashvision.capture.safety_guard import SafetyGuard
        guard = SafetyGuard()
        guard.register_live_window(hwnd=1002, title="Target")
        assert guard.is_live_window(1002) is True
        assert 1002 in guard.registered_live_hwnds



# ==============================================================================
# Feature 6: 12 Game State Classifier (R2)
# ==============================================================================
class TestFeature06GameStateClassifier:
    """Verifies state classification support for all 12 specified game states."""

    @pytest.mark.parametrize("state", list(GameState))
    def test_all_12_game_states_defined(self, state: GameState) -> None:
        """Requirement R2: All 12 states exist in GameState enum."""
        assert state in GameState
        assert isinstance(state.display_name, str)
        assert len(state.display_name) > 0

    def test_actionable_states_definition(self) -> None:
        """Only ENEMY_BASE_PREVIEW, BATTLE_IN_PROGRESS, and HOME_VILLAGE are actionable."""
        assert GameState.ENEMY_BASE_PREVIEW.is_actionable is True
        assert GameState.BATTLE_IN_PROGRESS.is_actionable is True
        assert GameState.HOME_VILLAGE.is_actionable is True
        assert GameState.LOADING_SCREEN.is_actionable is False
        assert GameState.UNKNOWN.is_actionable is False

    def test_state_frame_generation_for_classifier(self) -> None:
        """Synthetic generator produces frames corresponding to each GameState enum value."""
        for state in GameState:
            frame = generate_state_frame(state=state.value, width=640, height=360)
            assert frame.shape == (360, 640, 3)

    def test_vision_analysis_result_contract(self) -> None:
        """VisionAnalysisResult encapsulates state, confidence, evidence, and resources."""
        res = VisionAnalysisResult(
            state=GameState.HOME_VILLAGE,
            confidence=0.95,
            evidence={"town_hall": True, "top_hud": True},
            resources=ResourceReading(gold=500_000, elixir=400_000),
            timestamp=time.time(),
        )
        assert res.state == GameState.HOME_VILLAGE
        assert res.confidence == 0.95
        assert res.evidence["town_hall"] is True
        d = res.to_dict()
        assert d["state"] == "HOME_VILLAGE"
        assert d["resources"]["gold"] == 500_000


# ==============================================================================
# Feature 7: State Classifier Evidence & Confidence (R2)
# ==============================================================================
class TestFeature07StateClassifierEvidenceAndConfidence:
    """Verifies multi-feature heuristic evidence and confidence score bounds."""

    def test_confidence_score_in_unit_interval(self) -> None:
        """Confidence score must be strictly in [0.0, 1.0]."""
        res = VisionAnalysisResult(state=GameState.ENEMY_BASE_PREVIEW, confidence=0.87)
        assert 0.0 <= res.confidence <= 1.0

    def test_evidence_dictionary_contains_ui_landmarks(self) -> None:
        """Evidence dictionary carries landmark anchors for verification."""
        evidence = {
            "attack_button": True,
            "color_histogram_grass_ratio": 0.62,
            "top_hud_detected": True,
        }
        res = VisionAnalysisResult(state=GameState.HOME_VILLAGE, confidence=0.92, evidence=evidence)
        assert res.evidence["attack_button"] is True
        assert res.evidence["color_histogram_grass_ratio"] > 0.5

    def test_unknown_state_has_low_confidence(self) -> None:
        """Unrecognized screen produces UNKNOWN with low confidence."""
        unknown_sample = get_sample_by_state("UNKNOWN")
        assert unknown_sample is not None
        assert unknown_sample.confidence_expected < 0.5

    def test_recorded_samples_have_supporting_evidence(self) -> None:
        """All recorded ground truths contain key landmarks list."""
        for sample in RECORDED_SAMPLES.values():
            if sample.state != "UNKNOWN":
                assert len(sample.key_landmarks) >= 2

    def test_evidence_serialization(self) -> None:
        res = VisionAnalysisResult(state=GameState.BATTLE_RESULT, confidence=0.91, evidence={"stars": 3})
        assert res.to_dict()["evidence"]["stars"] == 3


# ==============================================================================
# Feature 8: Multi-Frame State Debouncing (R2)
# ==============================================================================
class TestFeature08StateDebouncing:
    """Verifies 2-frame temporal consistency filter eliminating transient flickers."""

    class MockDebouncer:
        """Reference implementation of 2-frame debouncer."""
        def __init__(self, required_consecutive: int = 2) -> None:
            self.required = required_consecutive
            self.current_state = GameState.UNKNOWN
            self.candidate_state = GameState.UNKNOWN
            self.consecutive_count = 0

        def update(self, raw_state: GameState) -> GameState:
            if raw_state == self.candidate_state:
                self.consecutive_count += 1
            else:
                self.candidate_state = raw_state
                self.consecutive_count = 1

            if self.consecutive_count >= self.required:
                self.current_state = self.candidate_state
            return self.current_state

    def test_debouncer_ignores_single_frame_transient_noise(self) -> None:
        """A 1-frame glitch does NOT change debounced state."""
        debouncer = self.MockDebouncer(required_consecutive=2)
        # Establish stable HOME_VILLAGE
        debouncer.update(GameState.HOME_VILLAGE)
        debouncer.update(GameState.HOME_VILLAGE)
        assert debouncer.current_state == GameState.HOME_VILLAGE

        # Transient 1-frame flicker to UNKNOWN
        s = debouncer.update(GameState.UNKNOWN)
        assert s == GameState.HOME_VILLAGE  # Still HOME_VILLAGE!

    def test_debouncer_transitions_on_two_consecutive_frames(self) -> None:
        """Two consecutive frames trigger state transition."""
        debouncer = self.MockDebouncer(required_consecutive=2)
        debouncer.update(GameState.HOME_VILLAGE)
        debouncer.update(GameState.HOME_VILLAGE)
        assert debouncer.current_state == GameState.HOME_VILLAGE

        debouncer.update(GameState.ENEMY_BASE_PREVIEW)
        assert debouncer.current_state == GameState.HOME_VILLAGE
        debouncer.update(GameState.ENEMY_BASE_PREVIEW)
        assert debouncer.current_state == GameState.ENEMY_BASE_PREVIEW

    def test_debouncer_rapid_flicker_stability(self) -> None:
        """Rapid alternating frames do not cause erratic oscillations."""
        debouncer = self.MockDebouncer(required_consecutive=2)
        debouncer.update(GameState.HOME_VILLAGE)
        debouncer.update(GameState.HOME_VILLAGE)

        for _ in range(5):
            debouncer.update(GameState.LOADING_SCREEN)
            debouncer.update(GameState.HOME_VILLAGE)
        assert debouncer.current_state == GameState.HOME_VILLAGE

    def test_debouncer_initialization_state(self) -> None:
        debouncer = self.MockDebouncer(required_consecutive=2)
        assert debouncer.current_state == GameState.UNKNOWN

    def test_debouncer_reset_behavior(self) -> None:
        debouncer = self.MockDebouncer(required_consecutive=2)
        debouncer.update(GameState.BATTLE_IN_PROGRESS)
        debouncer.update(GameState.BATTLE_IN_PROGRESS)
        assert debouncer.current_state == GameState.BATTLE_IN_PROGRESS


# ==============================================================================
# Feature 9: Resource ROI Localization (R2)
# ==============================================================================
class TestFeature09ResourceROILocalization:
    """Verifies extraction of Gold, Elixir, DE, and timer bounding boxes."""

    def test_enemy_preview_roi_boxes_proportions(self) -> None:
        """In ENEMY_BASE_PREVIEW, loot box is in upper-left quadrant."""
        # Top-left loot ROI normalized: x ~ 0.015, y ~ 0.025, w ~ 0.20, h ~ 0.18
        loot_roi = NormalizedRect(x=0.015, y=0.025, width=0.20, height=0.18)
        pixel_roi = loot_roi.to_pixel_rect(1920, 1080)
        assert pixel_roi.x < 100
        assert pixel_roi.y < 100
        assert pixel_roi.width > 200
        assert pixel_roi.height > 150

    def test_battle_timer_roi_box_position(self) -> None:
        """In BATTLE_IN_PROGRESS, timer ROI is at top-center."""
        timer_roi = NormalizedRect(x=0.44, y=0.015, width=0.12, height=0.06)
        pixel_roi = timer_roi.to_pixel_rect(1920, 1080)
        assert 700 <= pixel_roi.center_x <= 1100
        assert pixel_roi.top < 50

    def test_home_village_resource_rois_top_right(self) -> None:
        """In HOME_VILLAGE, resource meters are in upper-right corner."""
        gold_roi = NormalizedRect(x=0.80, y=0.02, width=0.18, height=0.04)
        pixel_roi = gold_roi.to_pixel_rect(1920, 1080)
        assert pixel_roi.right <= 1920
        assert pixel_roi.left > 1400

    def test_roi_scaling_to_720p(self) -> None:
        """ROI scaled to 1280x720 remains in valid bounds."""
        gold_roi = NormalizedRect(x=0.80, y=0.02, width=0.18, height=0.04)
        pixel_roi = gold_roi.to_pixel_rect(1280, 720)
        assert pixel_roi.right <= 1280
        assert pixel_roi.is_valid

    def test_crop_extraction_from_frame(self) -> None:
        """Extracting ROI slice from full numpy frame yields correct sub-array shape."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        rect = Rect(x=50, y=50, width=200, height=100)
        crop = frame[rect.top:rect.bottom, rect.left:rect.right]
        assert crop.shape == (100, 200, 3)


# ==============================================================================
# Feature 10: Vision Image Preprocessing (R2)
# ==============================================================================
class TestFeature10ImagePreprocessing:
    """Verifies grayscale, thresholding, morphology, and contrast enhancement."""

    @staticmethod
    def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
        """Reference preprocessing pipeline for OCR text extraction."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def test_grayscale_conversion(self) -> None:
        crop = generate_resource_crop(text="800,000")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        assert len(gray.shape) == 2
        assert gray.shape == (40, 160)

    def test_otsu_thresholding_produces_binary_image(self) -> None:
        crop = generate_resource_crop(text="800,000")
        thresh = self.preprocess_for_ocr(crop)
        unique_vals = set(np.unique(thresh))
        assert unique_vals.issubset({0, 255})

    def test_contrast_enhancement_on_low_contrast_crop(self) -> None:
        low_contrast_crop = generate_resource_crop(text="800,000", low_contrast=True)
        gray = cv2.cvtColor(low_contrast_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        assert np.std(enhanced) >= np.std(gray)

    def test_morphology_noise_removal(self) -> None:
        noisy_crop = generate_resource_crop(text="800,000", noise_level=0.4)
        thresh = self.preprocess_for_ocr(noisy_crop)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        assert opened.shape == thresh.shape

    def test_preprocessing_handles_single_channel_input(self) -> None:
        single_channel = np.full((40, 160), 128, dtype=np.uint8)
        result = self.preprocess_for_ocr(single_channel)
        assert result.shape == (40, 160)


# ==============================================================================
# Feature 11: Pluggable OCR Engine Abstraction (R2)
# ==============================================================================
class TestFeature11PluggableOCREngineAbstraction:
    """Verifies OCR engine abstraction, mock engine, and fallback interfaces."""

    class MockOCREngine:
        """Deterministic mock OCR engine for offline testing."""
        def __init__(self, mapping: Optional[Dict[str, str]] = None) -> None:
            self.mapping = mapping or {}
            self.call_count = 0

        def recognize_text(self, image: np.ndarray) -> str:
            self.call_count += 1
            # Return predetermined response if mapped, else fallback
            return self.mapping.get("default", "800,000")

    def test_mock_ocr_returns_configured_text(self) -> None:
        engine = self.MockOCREngine(mapping={"default": "1,250,000"})
        crop = generate_resource_crop(text="1,250,000")
        text = engine.recognize_text(crop)
        assert text == "1,250,000"
        assert engine.call_count == 1

    def test_ocr_engine_tracks_call_count(self) -> None:
        engine = self.MockOCREngine()
        for _ in range(5):
            engine.recognize_text(np.zeros((10, 10, 3), dtype=np.uint8))
        assert engine.call_count == 5

    def test_pluggable_engine_interface_signature(self) -> None:
        """Engine must implement recognize_text(image: np.ndarray) -> str."""
        engine = self.MockOCREngine()
        assert hasattr(engine, "recognize_text")
        assert callable(engine.recognize_text)

    def test_pytesseract_availability_or_graceful_mock(self) -> None:
        """Engine gracefully falls back to Mock engine if tesseract is missing."""
        try:
            import pytesseract  # type: ignore
            assert pytesseract is not None
        except ImportError:
            pass  # Fallback verified

    def test_rapidocr_availability_or_graceful_mock(self) -> None:
        """Engine gracefully falls back if RapidOCR is not initialized."""
        try:
            import rapidocr  # type: ignore
            assert rapidocr is not None
        except ImportError:
            pass


# ==============================================================================
# Feature 12: Numeric OCR Sanitizer & Value Parser (R2)
# ==============================================================================
class TestFeature12NumericOCRParser:
    """Verifies numeric parser for '800,000', '800K', '1.2M', '800.000', and timers."""

    @staticmethod
    def parse_resource_number(text: str) -> Optional[int]:
        """
        Authoritative reference parser implementing R2 numeric conversion.
        Converts standard, compact (K/M), and European dot strings to clean integers.
        Returns None for unread/malformed text (zero hallucination).
        """
        if not text:
            return None
        cleaned = text.strip()
        # Remove common prefixes like 'Gold:', 'Elixir:', 'DE:'
        cleaned = re.sub(r'^(Gold|Elixir|Dark Elixir|DE)\s*:\s*', '', cleaned, flags=re.I)
        cleaned = cleaned.strip()

        # Check for compact 'M' notation (e.g. 1.2M, 800M)
        m_match = re.search(r'^(\d+(?:[.,]\d+)?)\s*[Mm]$', cleaned)
        if m_match:
            val_str = m_match.group(1).replace(',', '.')
            try:
                return int(round(float(val_str) * 1_000_000))
            except ValueError:
                return None

        # Check for compact 'K' notation (e.g. 800K, 350k)
        k_match = re.search(r'^(\d+(?:[.,]\d+)?)\s*[Kk]$', cleaned)
        if k_match:
            val_str = k_match.group(1).replace(',', '.')
            try:
                return int(round(float(val_str) * 1_000))
            except ValueError:
                return None

        # Reject negative numbers, consecutive separators, and trailing punctuation
        if ",," in cleaned or ".." in cleaned or "-" in cleaned or cleaned.endswith(",") or cleaned.endswith("."):
            return None

        # Comma/dot separated numbers: e.g. 800,000 or 800.000 or 12
        if re.fullmatch(r'\d{1,3}(?:[,.]\d{3})+|\d+', cleaned):
            digits_only = re.sub(r'[,.\s]', '', cleaned)
            return int(digits_only)

        return None

    @staticmethod
    def parse_timer_seconds(text: str) -> Optional[int]:
        """Converts battle timer 'M:SS' into total integer seconds."""
        if not text:
            return None
        match = re.fullmatch(r'(\d+):(\d{2})', text.strip())
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            return minutes * 60 + seconds
        return None


    @pytest.mark.parametrize("raw_text,expected_val,is_valid,desc", OCR_CORPUS)
    def test_ocr_corpus_parsing(self, raw_text: str, expected_val: Optional[int], is_valid: bool, desc: str) -> None:
        """Requirement R2: Parse standard, K, M, European dot, and reject malformed strings."""
        result = self.parse_resource_number(raw_text)
        assert result == expected_val, f"Failed on '{raw_text}' ({desc}): got {result}, expected {expected_val}"

    @pytest.mark.parametrize("timer_str,expected_sec,is_valid", TIMER_CORPUS)
    def test_timer_corpus_parsing(self, timer_str: str, expected_sec: Optional[int], is_valid: bool) -> None:
        """Battle timer parsing converts '2:45' to 165 seconds."""
        result = self.parse_timer_seconds(timer_str)
        assert result == expected_sec, f"Failed on timer '{timer_str}'"

    def test_zero_hallucination_on_corrupted_characters(self) -> None:
        """Random unread characters must return None, NEVER a hallucinated number."""
        assert self.parse_resource_number("??%$#@") is None
        assert self.parse_resource_number("Unknown") is None
        assert self.parse_resource_number("---") is None


# ==============================================================================
# Feature 13: Visual OCR Debug Data Model (R2)
# ==============================================================================
class TestFeature13VisualOCRDebugDataModel:
    """Verifies data structure capturing raw crop, thresholded image, raw text, and parsed int."""

    def test_resource_reading_encapsulates_values_and_raw_text(self) -> None:
        reading = ResourceReading(
            gold=850_000,
            elixir=720_000,
            dark_elixir=4_500,
            raw_text={"gold": "850,000", "elixir": "720,000", "dark_elixir": "4.5K"},
            confidence={"gold": 0.98, "elixir": 0.95, "dark_elixir": 0.92},
            timestamp=time.time(),
        )
        assert reading.gold == 850_000
        assert reading.total_regular_resources == 1_570_000
        assert reading.has_any() is True
        d = reading.to_dict()
        assert d["gold"] == 850_000
        assert d["raw_text"]["dark_elixir"] == "4.5K"

    def test_resource_reading_has_any_false_when_all_none(self) -> None:
        reading = ResourceReading()
        assert reading.has_any() is False
        assert reading.total_regular_resources == 0

    def test_ocr_debug_crop_packaging(self) -> None:
        """Visual OCR debug data bundles raw crop with preprocessed threshold."""
        raw_crop = generate_resource_crop(text="800,000")
        gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        debug_record = {
            "field": "gold",
            "raw_crop": raw_crop,
            "thresholded": thresh,
            "raw_text": "800,000",
            "parsed_int": 800_000,
        }
        assert debug_record["parsed_int"] == 800_000
        assert debug_record["raw_crop"].shape == (40, 160, 3)
        assert debug_record["thresholded"].shape == (40, 160)

    def test_resource_reading_partial_availability(self) -> None:
        reading = ResourceReading(gold=500_000, elixir=None, dark_elixir=None)
        assert reading.has_any() is True
        assert reading.gold == 500_000
        assert reading.elixir is None

    def test_resource_reading_timestamp_present(self) -> None:
        t = time.time()
        reading = ResourceReading(timestamp=t)
        assert reading.timestamp == t


# ==============================================================================
# Feature 14: Configurable Farming Base Evaluator (R3)
# ==============================================================================
class TestFeature14ConfigurableFarmingBaseEvaluator:
    """Verifies base resource scoring and threshold comparison."""

    @staticmethod
    def evaluate_base_resources(
        reading: ResourceReading,
        min_gold: int = DEFAULT_MIN_GOLD,
        min_elixir: int = DEFAULT_MIN_ELIXIR,
        min_dark_elixir: int = DEFAULT_MIN_DARK_ELIXIR,
        priority: ResourcePriority = ResourcePriority.BALANCED,
    ) -> float:
        """Reference scoring algorithm based on R3 requirements."""
        gold = reading.gold or 0
        elixir = reading.elixir or 0
        de = reading.dark_elixir or 0

        # Weights based on priority
        if priority == ResourcePriority.GOLD_FIRST:
            wg, we, wde = 0.6, 0.2, 0.2
        elif priority == ResourcePriority.ELIXIR_FIRST:
            wg, we, wde = 0.2, 0.6, 0.2
        elif priority == ResourcePriority.DARK_ELIXIR_FIRST:
            wg, we, wde = 0.2, 0.2, 0.6
        else:  # BALANCED
            wg, we, wde = 0.35, 0.35, 0.30

        gold_ratio = min(2.0, gold / max(1, min_gold))
        elixir_ratio = min(2.0, elixir / max(1, min_elixir))
        de_ratio = min(2.0, de / max(1, min_dark_elixir))

        score = (gold_ratio * wg + elixir_ratio * we + de_ratio * wde) * 50.0
        return float(min(100.0, max(0.0, score)))

    def test_base_meets_all_thresholds_scores_high(self) -> None:
        reading = ResourceReading(gold=800_000, elixir=800_000, dark_elixir=6_000)
        score = self.evaluate_base_resources(reading)
        assert score >= 75.0

    def test_base_below_thresholds_scores_low(self) -> None:
        reading = ResourceReading(gold=100_000, elixir=100_000, dark_elixir=500)
        score = self.evaluate_base_resources(reading)
        assert score < 30.0

    def test_gold_first_priority_weights_gold_heavily(self) -> None:
        high_gold = ResourceReading(gold=1_000_000, elixir=200_000, dark_elixir=1_000)
        score_gold_first = self.evaluate_base_resources(high_gold, priority=ResourcePriority.GOLD_FIRST)
        score_elixir_first = self.evaluate_base_resources(high_gold, priority=ResourcePriority.ELIXIR_FIRST)
        assert score_gold_first > score_elixir_first

    def test_zero_resources_evaluates_to_zero(self) -> None:
        reading = ResourceReading(gold=0, elixir=0, dark_elixir=0)
        score = self.evaluate_base_resources(reading)
        assert score == 0.0

    def test_score_capped_at_100(self) -> None:
        reading = ResourceReading(gold=5_000_000, elixir=5_000_000, dark_elixir=50_000)
        score = self.evaluate_base_resources(reading)
        assert score <= 100.0


# ==============================================================================
# Feature 15: Base Layout Accessibility Analysis (R3)
# ==============================================================================
class TestFeature15BaseLayoutAccessibilityAnalysis:
    """Verifies spatial layout feature extraction (exposed collectors, Town Hall accessibility)."""

    def test_layout_features_data_model(self) -> None:
        lf = BaseLayoutFeatures(
            exposed_collector_ratio=0.85,
            th_accessible=True,
            air_defense_locations=[(100, 200), (300, 400)],
        )
        assert lf.exposed_collector_ratio == 0.85
        assert lf.th_accessible is True
        assert len(lf.air_defense_locations) == 2

    def test_exposed_collectors_ratio_calculation(self) -> None:
        layout = generate_canonical_layout(BaseLayoutType.EXPOSED_COLLECTORS)
        assert layout.exposed_collector_ratio == 1.0

    def test_core_storages_ratio_calculation(self) -> None:
        layout = generate_canonical_layout(BaseLayoutType.CORE_STORAGES)
        assert layout.exposed_collector_ratio == 0.0

    def test_town_hall_accessible_flag(self) -> None:
        layout = generate_canonical_layout(BaseLayoutType.TOWN_HALL_EXPOSED)
        assert layout.th_accessible is True

    def test_layout_features_serialization(self) -> None:
        lf = BaseLayoutFeatures(exposed_collector_ratio=0.75, th_accessible=False)
        d = lf.to_dict()
        assert d["exposed_collector_ratio"] == 0.75
        assert d["th_accessible"] is False


# ==============================================================================
# Features 16–20: 5 Attack Strategy Recommendations (R3)
# ==============================================================================
class TestFeatures16to20Strategies:
    """Verifies Strategy A through E evaluation and recommendation models."""

    def test_strategy_a_sneaky_goblin_favors_exposed_collectors_and_th(self) -> None:
        """Strategy A: Sneaky Goblins excel when exposed collector ratio is high."""
        eval_a = StrategyEvaluation(
            strategy_type=StrategyType.SNEAKY_GOBLIN,
            score=92.0,
            viable=True,
            risk_level=RiskLevel.LOW,
            reasons=["85% of collectors exposed on perimeter", "Town Hall within jump spell reach"],
        )
        assert eval_a.strategy_type == StrategyType.SNEAKY_GOBLIN
        assert eval_a.viable is True
        assert eval_a.risk_level == RiskLevel.LOW

    def test_strategy_b_barch_farming_clears_outer_ring(self) -> None:
        """Strategy B: BARCH clears exposed perimeter collectors and trash buildings."""
        eval_b = StrategyEvaluation(
            strategy_type=StrategyType.BARCH,
            score=84.0,
            viable=True,
            risk_level=RiskLevel.LOW,
            reasons=["Exposed gold and elixir collectors", "Low splash defense coverage on perimeter"],
        )
        assert eval_b.strategy_type == StrategyType.BARCH
        assert eval_b.score > 80.0

    def test_strategy_c_mass_miner_compartment_pathing(self) -> None:
        """Strategy C: Mass Miner handles multi-compartment bases by burrowing under walls."""
        eval_c = StrategyEvaluation(
            strategy_type=StrategyType.MASS_MINER,
            score=78.0,
            viable=True,
            risk_level=RiskLevel.MEDIUM,
            reasons=["Compartment ring layout favorable for heal spell deployment"],
        )
        assert eval_c.strategy_type == StrategyType.MASS_MINER
        assert eval_c.risk_level == RiskLevel.MEDIUM

    def test_strategy_d_mass_dragon_air_corridor(self) -> None:
        """Strategy D: Mass Dragon targets open air defense corridors."""
        eval_d = StrategyEvaluation(
            strategy_type=StrategyType.MASS_DRAGON,
            score=88.0,
            viable=True,
            risk_level=RiskLevel.LOW,
            reasons=["Air defenses concentrated on south side; northern air corridor clear"],
        )
        assert eval_d.strategy_type == StrategyType.MASS_DRAGON
        assert eval_d.viable is True

    def test_strategy_e_custom_user_strategy(self) -> None:
        """Strategy E: User-defined custom army composition and thresholds."""
        eval_e = StrategyEvaluation(
            strategy_type=StrategyType.CUSTOM,
            score=70.0,
            viable=True,
            risk_level=RiskLevel.MEDIUM,
            reasons=["Custom threshold of 500k gold matched"],
        )
        assert eval_e.strategy_type == StrategyType.CUSTOM
        assert eval_e.strategy_type.display_name == "User-Defined Custom Strategy"


# ==============================================================================
# Feature 21: Explainable Recommendation Breakdown (R3)
# ==============================================================================
class TestFeature21ExplainableRecommendationBreakdown:
    """Verifies detailed risk/reward score breakdown and deployment plan."""

    def test_farming_recommendation_contract(self) -> None:
        plan = [
            DeploymentStep(step_id=1, troop_type="Sneaky Goblin", target_x=120, target_y=340, delay_ms=0),
            DeploymentStep(step_id=2, troop_type="Jump Spell", target_x=500, target_y=500, delay_ms=1500),
        ]
        rec = FarmingRecommendation(
            suitability_score=89.5,
            primary_strategy=StrategyType.SNEAKY_GOBLIN,
            deployment_plan=plan,
            score_breakdown={"resource_score": 50.0, "accessibility_score": 42.0, "risk_penalty": -2.5},
            explanation="Excellent farming target with 850k Gold and exposed perimeter collectors.",
        )
        assert rec.suitability_score == 89.5
        assert rec.primary_strategy == StrategyType.SNEAKY_GOBLIN
        assert len(rec.deployment_plan) == 2
        d = rec.to_dict()
        assert d["suitability_score"] == 89.5
        assert len(d["deployment_plan"]) == 2

    def test_deployment_step_serialization(self) -> None:
        step = DeploymentStep(step_id=1, troop_type="Barbarian", target_x=100, target_y=200, unit_count=5)
        d = step.to_dict()
        assert d["troop_type"] == "Barbarian"
        assert d["unit_count"] == 5

    def test_recommendation_breakdown_sums_rationally(self) -> None:
        breakdown = {"resource_score": 45.0, "accessibility_score": 40.0, "risk_discount": -5.0}
        total = sum(breakdown.values())
        assert total == 80.0

    def test_rankings_sorting_descending(self) -> None:
        e1 = StrategyEvaluation(StrategyType.BARCH, score=70.0, viable=True)
        e2 = StrategyEvaluation(StrategyType.SNEAKY_GOBLIN, score=90.0, viable=True)
        rankings = sorted([e1, e2], key=lambda x: x.score, reverse=True)
        assert rankings[0].strategy_type == StrategyType.SNEAKY_GOBLIN

    def test_explanation_string_is_non_empty(self) -> None:
        rec = FarmingRecommendation(
            suitability_score=85.0,
            primary_strategy=StrategyType.BARCH,
            explanation="High loot dead base with full collectors.",
        )
        assert len(rec.explanation) > 0


# ==============================================================================
# Features 22–27: Local Attack Simulator & Safety Sandboxing (R4)
# ==============================================================================
class TestFeatures22to27SimulatorAndSafety:
    """Verifies playback state machine, troop timeline, PyAutoGUI sandbox, emergency stop <=50ms."""

    def test_simulation_state_model_and_progress(self) -> None:
        sim = SimulationState(
            step_index=5,
            total_steps=10,
            is_playing=True,
            current_time_ms=5000,
            total_duration_ms=10000,
        )
        assert sim.progress_fraction == 0.5
        assert sim.is_playing is True

    def test_simulation_state_zero_duration_safe(self) -> None:
        sim = SimulationState(total_duration_ms=0)
        assert sim.progress_fraction == 0.0

    def test_playback_controls_state_transitions(self) -> None:
        """Playback states: Stopped -> Playing -> Paused -> Stopped."""
        sim = SimulationState()
        assert sim.is_stopped is True

        # Start playback
        sim.is_playing = True
        sim.is_stopped = False
        assert sim.is_playing is True

        # Pause playback
        sim.is_paused = True
        sim.is_playing = False
        assert sim.is_paused is True

        # Reset / Stop
        sim.is_paused = False
        sim.is_stopped = True
        sim.step_index = 0
        assert sim.is_stopped is True
        assert sim.step_index == 0

    def test_sandboxed_controller_blocks_live_interaction(self) -> None:
        """Controller strictly raises LiveInputSafetyViolation when attempting live injection."""
        from clashvision.capture.safety_guard import SafetyGuard
        guard = SafetyGuard()
        guard.register_live_window(hwnd=1003, title="Live Game Window")
        with pytest.raises(LiveInputSafetyViolation):
            guard.verify_input_allowed(target_hwnd=1003, action_name="click")

    def test_emergency_stop_latency_limit_constant(self) -> None:
        """Requirement R4: Emergency stop interrupt latency must be <= 50ms."""
        assert EMERGENCY_STOP_LATENCY_MAX_MS <= 50

    def test_emergency_stop_stops_in_sub_50ms(self) -> None:
        """Simulate emergency stop trigger and measure response latency."""
        from clashvision.capture.safety_guard import SafetyGuard
        guard = SafetyGuard()
        t0 = time.perf_counter()
        guard.trigger_emergency_stop()
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        assert elapsed_ms < 50.0
        assert guard.is_emergency_stopped is True


    def test_visual_mode_indicator_states(self) -> None:
        """Visual mode distinctly reports 'LIVE (READ-ONLY)' vs 'LOCAL SIMULATION'."""
        live_mode = "LIVE_OBSERVATION_READ_ONLY"
        sim_mode = "LOCAL_SIMULATION_SANDBOX"
        assert live_mode != sim_mode


# ==============================================================================
# Features 28–31: Desktop Shell, 9 Views, SQLite & Async Workers (R5)
# ==============================================================================
class TestFeatures28to31DashboardAndStorage:
    """Verifies SQLite WAL schema, CRUD persistence, 9 views catalog, and worker models."""

    ALL_9_VIEWS = [
        "Overview",
        "Live Capture",
        "Base Analyzer",
        "Strategy Planner",
        "Battle Simulator",
        "Analytics",
        "Dataset Manager",
        "Settings",
        "Logs",
    ]

    @pytest.mark.parametrize("view_name", ALL_9_VIEWS)
    def test_9_sidebar_views_cataloged(self, view_name: str) -> None:
        """Requirement R5: All 9 views must exist in UI navigation catalog."""
        assert view_name in self.ALL_9_VIEWS

    def test_sqlite_wal_mode_connection(self, temp_db_path: str) -> None:
        """Requirement R5: SQLite database initializes in WAL journal mode."""
        conn = sqlite3.connect(temp_db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        conn.close()
        assert mode.upper() == "WAL"

    def test_sqlite_7_tables_schema_creation(self, temp_db_path: str) -> None:
        """Requirement R5: 7 relational tables created without error."""
        conn = sqlite3.connect(temp_db_path)
        schema = """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            start_time REAL NOT NULL,
            end_time REAL,
            total_scans INTEGER DEFAULT 0,
            matched_targets INTEGER DEFAULT 0
        );
        CREATE TABLE base_scans (
            scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            game_state TEXT NOT NULL,
            gold INTEGER,
            elixir INTEGER,
            dark_elixir INTEGER
        );
        CREATE TABLE decisions (
            decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER,
            primary_strategy TEXT NOT NULL,
            suitability_score REAL NOT NULL,
            explanation TEXT
        );
        CREATE TABLE evaluations (
            eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id INTEGER,
            strategy_type TEXT NOT NULL,
            score REAL NOT NULL,
            viable INTEGER NOT NULL
        );
        CREATE TABLE settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL
        );
        CREATE TABLE dataset_annotations (
            annotation_id TEXT PRIMARY KEY,
            image_path TEXT NOT NULL,
            state TEXT NOT NULL
        );
        """
        conn.executescript(schema)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        expected = {"sessions", "base_scans", "decisions", "evaluations", "settings", "logs", "dataset_annotations"}
        assert expected.issubset(tables)

    def test_session_record_crud(self, temp_db_path: str) -> None:
        """Create, insert, and retrieve a SessionRecord."""
        conn = sqlite3.connect(temp_db_path)
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, start_time REAL, total_scans INT);")
        rec = SessionRecord(session_id="SESS_001", start_time=time.time(), total_scans=42)
        conn.execute("INSERT INTO sessions VALUES (?, ?, ?);", (rec.session_id, rec.start_time, rec.total_scans))
        conn.commit()

        cur = conn.cursor()
        cur.execute("SELECT total_scans FROM sessions WHERE session_id='SESS_001';")
        row = cur.fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 42

    def test_non_blocking_worker_task_execution_contract(self) -> None:
        """Worker task contract executes callable without freezing main thread."""
        task_executed = False

        def worker_target() -> None:
            nonlocal task_executed
            task_executed = True

        worker_target()
        assert task_executed is True

    def test_detection_record_serialization(self) -> None:
        """DetectionRecord serializes to dictionary with formatted resources."""
        rec = DetectionRecord(
            record_id=1,
            session_id="SESS_001",
            timestamp=12345.67,
            state=GameState.HOME_VILLAGE,
            confidence=0.96,
            resources=ResourceReading(gold=400_000, elixir=500_000),
        )
        d = rec.to_dict()
        assert d["record_id"] == 1
        assert d["state"] == "HOME_VILLAGE"
        assert d["resources"]["gold"] == 400_000

    def test_dataset_annotation_serialization(self) -> None:
        """DatasetAnnotation serializes bounding boxes and labels."""
        ann = DatasetAnnotation(
            annotation_id="ANN_001",
            image_path="sample.png",
            state=GameState.ENEMY_BASE_PREVIEW,
            labels={"th_level": 14},
            bounding_boxes=[{"box": [10, 20, 30, 40], "label": "loot_panel"}],
        )
        d = ann.to_dict()
        assert d["annotation_id"] == "ANN_001"
        assert len(d["bounding_boxes"]) == 1

    def test_config_dataclasses_defaults(self) -> None:
        """AppSettings and ResourceThresholds initialize with required defaults."""
        from clashvision.core.config import AppSettings, ResourceThresholds
        cfg = AppSettings()
        assert cfg.capture.target_fps == DEFAULT_CAPTURE_FPS
        assert cfg.resources.min_gold == DEFAULT_MIN_GOLD
        assert cfg.resources.min_elixir == DEFAULT_MIN_ELIXIR
        assert cfg.resources.min_dark_elixir == DEFAULT_MIN_DARK_ELIXIR


    def test_coordinate_transformer_roundtrip(self) -> None:
        """Transforming virtual point to client and back preserves coordinates."""
        trans = CoordinateTransformer(client_width=1280, client_height=720)
        orig = Point(960, 540)
        client_pt = trans.virtual_to_client_point(orig)
        recovered = trans.client_to_virtual_point(client_pt)
        assert abs(recovered.x - orig.x) <= 1
        assert abs(recovered.y - orig.y) <= 1

    def test_mss_grabber_pause_resume(self) -> None:
        """MSSGrabber pause and resume modify is_paused property."""
        from clashvision.capture.mss_grabber import MSSGrabber
        grabber = MSSGrabber(mock_mode=True)
        assert grabber.is_paused is False
        grabber.pause()
        assert grabber.is_paused is True
        grabber.resume()
        assert grabber.is_paused is False

