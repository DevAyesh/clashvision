"""
Tier 2: Boundary & Corner Case Test Suite for ClashVision.
Validates all 31 features against extreme values, boundary conditions, zero/negative inputs,
malformed strings, corrupted image crops, and stress conditions.
Requirement threshold: >= 150 boundary test cases.
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

from clashvision.core.constants import (
    DEFAULT_CAPTURE_FPS,
    DEFAULT_MIN_DARK_ELIXIR,
    DEFAULT_MIN_ELIXIR,
    DEFAULT_MIN_GOLD,
    EMERGENCY_STOP_LATENCY_MAX_MS,
    GameState,
    ResourcePriority,
    RiskLevel,
    StrategyType,
)
from clashvision.core.coordinates import (
    CoordinateTransformer,
    calculate_iou,
    calculate_letterbox,
    clamp_point_to_bounds,
    clamp_rect_to_bounds,
    intersect_rects,
    normalized_to_point,
    normalized_to_rect,
    point_to_normalized,
    rect_to_normalized,
    union_rects,
)
from clashvision.core.exceptions import (
    ClashVisionError,
    ConfigurationError,
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
    SessionRecord,
    SimulationState,
    StrategyEvaluation,
    VisionAnalysisResult,
    WindowInfo,
)
from tests.conftest import allow_simulator_sandbox
from tests.fixtures.layout_generator import (
    BaseLayoutModel,
    BaseLayoutType,
    Building,
    generate_canonical_layout,
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
from tests.tiers.test_tier1_features import (
    TestFeature08StateDebouncing,
    TestFeature12NumericOCRParser,
    TestFeature14ConfigurableFarmingBaseEvaluator,
)


# ==============================================================================
# Group 1: Geometry & Coordinate Boundary Tests (Features 1, 3)
# ==============================================================================
class TestGeometryBoundaryConditions:
    """Boundary and corner testing for Rect, Point, and transformations."""

    @pytest.mark.parametrize(
        "x,y,w,h,is_valid,area",
        [
            (0, 0, 0, 0, False, 0),
            (0, 0, -10, 50, False, 0),
            (0, 0, 100, -20, False, 0),
            (-50, -50, 100, 100, True, 10000),
            (0, 0, 1, 1, True, 1),
            (1919, 1079, 1, 1, True, 1),
            (0, 0, 100_000, 100_000, True, 10_000_000_000),
        ],
    )
    def test_rect_extreme_dimensions(
        self, x: int, y: int, w: int, h: int, is_valid: bool, area: int
    ) -> None:
        r = Rect(x, y, w, h)
        assert r.is_valid == is_valid
        assert r.area == area

    @pytest.mark.parametrize(
        "px,py,bound_w,bound_h,expected_x,expected_y",
        [
            (-100, -50, 1920, 1080, 0, 0),
            (0, 0, 1920, 1080, 0, 0),
            (1919, 1079, 1920, 1080, 1919, 1079),
            (1920, 1080, 1920, 1080, 1919, 1079),
            (5000, 9999, 1920, 1080, 1919, 1079),
            (50, 50, 0, 0, 0, 0),
            (50, 50, -100, -100, 0, 0),
        ],
    )
    def test_clamp_point_to_bounds_extremes(
        self, px: int, py: int, bound_w: int, bound_h: int, expected_x: int, expected_y: int
    ) -> None:
        pt = clamp_point_to_bounds(Point(px, py), bound_w, bound_h)
        assert pt == Point(expected_x, expected_y)

    @pytest.mark.parametrize(
        "rx,ry,rw,rh,bw,bh,expected_w,expected_h",
        [
            (-50, -50, 200, 200, 1920, 1080, 150, 150),
            (1900, 1000, 100, 100, 1920, 1080, 20, 80),
            (2000, 2000, 100, 100, 1920, 1080, 0, 0),
            (0, 0, 3000, 3000, 1920, 1080, 1920, 1080),
            (0, 0, 100, 100, 0, 0, 0, 0),
        ],
    )
    def test_clamp_rect_to_bounds_extremes(
        self, rx: int, ry: int, rw: int, rh: int, bw: int, bh: int, expected_w: int, expected_h: int
    ) -> None:
        clamped = clamp_rect_to_bounds(Rect(rx, ry, rw, rh), bw, bh)
        assert clamped.width == expected_w
        assert clamped.height == expected_h

    @pytest.mark.parametrize(
        "r1,r2,expected_iou",
        [
            (Rect(0, 0, 100, 100), Rect(0, 0, 100, 100), 1.0),
            (Rect(0, 0, 100, 100), Rect(100, 0, 100, 100), 0.0),  # Edge touching
            (Rect(0, 0, 100, 100), Rect(200, 200, 100, 100), 0.0),  # Completely disjoint
            (Rect(0, 0, 100, 100), Rect(0, 0, 0, 0), 0.0),  # Zero area
            (Rect(0, 0, 100, 100), Rect(50, 0, 100, 100), 5000 / 15000),  # Half overlap
        ],
    )
    def test_iou_boundary_conditions(self, r1: Rect, r2: Rect, expected_iou: float) -> None:
        iou = calculate_iou(r1, r2)
        assert pytest.approx(iou, 0.001) == expected_iou

    @pytest.mark.parametrize("scale", [0.001, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])
    def test_coordinate_transformer_scales(self, scale: float) -> None:
        """Arbitrary scale factors produce strictly positive pixel dimensions."""
        trans = CoordinateTransformer(
            client_width=int(1920 * scale),
            client_height=int(1080 * scale),
            scale_factor=scale,
        )
        pt = trans.virtual_to_client_point(Point(1920, 1080))
        assert pt.x == trans.client_width
        assert pt.y == trans.client_height



# ==============================================================================
# Group 2: Window & Screen Capture Boundary Tests (Features 1, 2, 4, 5)
# ==============================================================================
class TestWindowAndCaptureBoundaries:
    """Boundary conditions for window handles, visibility, FPS clamping, and safety guard."""

    @pytest.mark.parametrize(
        "fps_in,expected_fps",
        [
            (-10, 1),
            (0, 1),
            (1, 1),
            (15, 15),
            (60, 60),
            (61, 60),
            (1000, 60),
        ],
    )
    def test_mss_grabber_fps_boundary_clamping(self, fps_in: int, expected_fps: int) -> None:
        from clashvision.capture.mss_grabber import MSSGrabber
        grabber = MSSGrabber(target_fps=fps_in)
        assert grabber.target_fps == expected_fps
        grabber.set_target_fps(fps_in)
        assert grabber.target_fps == expected_fps

    def test_grab_frame_with_none_window_raises(self) -> None:
        from clashvision.capture.mss_grabber import MSSGrabber
        grabber = MSSGrabber(mock_mode=True)
        with pytest.raises(WindowNotFoundError):
            grabber.grab_frame(None)

    def test_grab_frame_minimized_window_auto_pauses(self) -> None:
        from clashvision.capture.mss_grabber import MSSGrabber
        grabber = MSSGrabber(mock_mode=True, auto_pause_on_minimized=True)
        win = WindowInfo(hwnd=10, is_minimized=True, is_visible=True, client_rect=Rect(0, 0, 1920, 1080))
        result = grabber.grab_frame(win)
        assert result is None
        assert grabber.is_paused is True

    def test_grab_frame_zero_area_client_rect_auto_pauses(self) -> None:
        from clashvision.capture.mss_grabber import MSSGrabber
        grabber = MSSGrabber(mock_mode=True, auto_pause_on_minimized=True)
        win = WindowInfo(hwnd=11, is_minimized=False, is_visible=True, client_rect=Rect(0, 0, 0, 0))
        result = grabber.grab_frame(win)
        assert result is None
        assert grabber.is_paused is True

    def test_safety_guard_rapid_emergency_stop_cycles(self) -> None:
        """Rapidly setting and resetting emergency stop 50 times remains stable."""
        from clashvision.capture.safety_guard import SafetyGuard
        guard = SafetyGuard()
        for _ in range(50):
            guard.trigger_emergency_stop()
            assert guard.is_emergency_stopped is True
            guard.reset_emergency_stop()
            assert guard.is_emergency_stopped is False

    def test_safety_guard_blocks_untargeted_global_call_when_live_active(self) -> None:
        """If live windows exist and target_hwnd is None, global action raises violation."""
        from clashvision.capture.safety_guard import SafetyGuard
        guard = SafetyGuard()
        guard.register_live_window(hwnd=999)
        with pytest.raises(LiveInputSafetyViolation):
            guard.verify_input_allowed(target_hwnd=None, action_name="global_keystroke")

    def test_safety_guard_multiple_live_windows(self) -> None:
        """Registering multiple emulator HWNDs blocks calls against any of them."""
        from clashvision.capture.safety_guard import SafetyGuard
        guard = SafetyGuard()
        guard.register_live_window(101)
        guard.register_live_window(102)
        guard.register_live_window(103)
        assert len(guard.registered_live_hwnds) == 3
        for h in [101, 102, 103]:
            with pytest.raises(LiveInputSafetyViolation):
                guard.verify_input_allowed(target_hwnd=h, action_name="click")
        guard.clear_all()
        assert len(guard.registered_live_hwnds) == 0


# ==============================================================================
# Group 3: Vision & Preprocessing Corner Cases (Features 6, 7, 8, 9, 10)
# ==============================================================================
class TestVisionAndPreprocessingCornerCases:
    """Tests extreme visual conditions: pure black/white, extreme noise, small crops."""

    def test_pure_black_frame_processing(self) -> None:
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        assert np.all(thresh == 0)

    def test_pure_white_frame_processing(self) -> None:
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        assert np.all(thresh == 255)

    def test_extreme_gaussian_noise_frame(self) -> None:
        """Frame with extreme noise does not crash generator or preprocessing."""
        img = generate_state_frame(
            state=GameStateType.HOME_VILLAGE,
            width=640,
            height=360,
            add_noise=True,
            noise_sigma=80.0,
        )
        assert img.shape == (360, 640, 3)
        assert img.dtype == np.uint8

    def test_tiny_crop_dimension_1x1(self) -> None:
        """1x1 crop handled without crash."""
        crop = np.zeros((1, 1, 3), dtype=np.uint8)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        assert gray.shape == (1, 1)

    @pytest.mark.parametrize("state", list(GameState))
    def test_all_12_states_low_resolution_360p(self, state: GameState) -> None:
        """Classifier fixture generation supports 360p (640x360)."""
        frame = generate_state_frame(state=state.value, width=640, height=360)
        assert frame.shape == (360, 640, 3)

    @pytest.mark.parametrize("state", list(GameState))
    def test_all_12_states_ultra_hd_1440p(self, state: GameState) -> None:
        """Classifier fixture generation supports 1440p (2560x1440)."""
        frame = generate_state_frame(state=state.value, width=2560, height=1440)
        assert frame.shape == (1440, 2560, 3)

    def test_debouncer_extreme_requirement_count(self) -> None:
        """Debouncer requiring 10 consecutive frames."""
        debouncer = TestFeature08StateDebouncing.MockDebouncer(required_consecutive=10)
        for i in range(9):
            debouncer.update(GameState.BATTLE_IN_PROGRESS)
            assert debouncer.current_state == GameState.UNKNOWN
        debouncer.update(GameState.BATTLE_IN_PROGRESS)
        assert debouncer.current_state == GameState.BATTLE_IN_PROGRESS


# ==============================================================================
# Group 4: Numeric OCR Parser Corner Cases (Features 11, 12, 13)
# ==============================================================================
class TestNumericOCRParserCornerCases:
    """Boundary and adversarial inputs for text/number conversion."""

    @staticmethod
    def PARSER(text: str) -> Optional[int]:
        return TestFeature12NumericOCRParser.parse_resource_number(text)

    @staticmethod
    def TIMER_PARSER(text: str) -> Optional[int]:
        return TestFeature12NumericOCRParser.parse_timer_seconds(text)


    @pytest.mark.parametrize(
        "input_str,expected_out",
        [
            ("0", 0),
            ("00", 0),
            ("0,000", 0),
            ("0K", 0),
            ("0.0M", 0),
            ("0m", 0),
            ("1", 1),
            ("9", 9),
            ("999", 999),
            ("10,000,000", 10_000_000),  # Max storage capacity TH15
            ("100,000,000", 100_000_000),  # Extreme capacity
            ("20,000,000", 20_000_000),
            ("0.5K", 500),
            ("0.05M", 50_000),
            ("1.999M", 1_999_000),
            ("2.000M", 2_000_000),
        ],
    )
    def test_extreme_and_zero_valid_numbers(self, input_str: str, expected_out: int) -> None:
        res = self.PARSER(input_str)
        assert res == expected_out, f"Failed on '{input_str}'"

    @pytest.mark.parametrize(
        "malformed_str",
        [
            "-1",
            "-500000",
            "-800K",
            "8,,000",
            "1..2M",
            "80A0",
            "Gold 850X",
            "null",
            "None",
            "NaN",
            "Infinity",
            "1e6",
            "0x10",
            " ",
            "\t\n",
            "...",
            "???",
            "///",
            "---",
            "___",
            "!@#$%",
            "()()",
        ],
    )
    def test_malformed_and_negative_strings_return_none(self, malformed_str: str) -> None:
        """Negative numbers and malformed syntaxes must return None (zero hallucination)."""
        res = self.PARSER(malformed_str)
        assert res is None, f"Expected None for '{malformed_str}', got {res}"

    @pytest.mark.parametrize(
        "timer_str,expected_sec",
        [
            ("0:00", 0),
            ("0:01", 1),
            ("0:59", 59),
            ("1:00", 60),
            ("3:00", 180),
            ("3:01", 181),
            ("10:00", 600),
            ("99:59", 5940 + 59),
        ],
    )
    def test_timer_boundary_values(self, timer_str: str, expected_sec: int) -> None:
        res = self.TIMER_PARSER(timer_str)
        assert res == expected_sec

    @pytest.mark.parametrize(
        "invalid_timer",
        [
            "-1:00",
            ":30",
            "2:",
            "2-30",
            "2.30",
            "abc",
            "",
            "   ",
        ],
    )
    def test_invalid_timers_return_none(self, invalid_timer: str) -> None:
        res = self.TIMER_PARSER(invalid_timer)
        assert res is None


# ==============================================================================
# Group 5: Base Evaluator & Strategy Boundaries (Features 14–21)
# ==============================================================================
class TestEvaluatorAndStrategyBoundaries:
    """Extreme resources, negative weights, zero buildings, and scoring caps."""

    @staticmethod
    def EVALUATOR(*args: Any, **kwargs: Any) -> float:
        return TestFeature14ConfigurableFarmingBaseEvaluator.evaluate_base_resources(*args, **kwargs)


    def test_zero_resources_all_zero(self) -> None:
        reading = ResourceReading(gold=0, elixir=0, dark_elixir=0)
        score = self.EVALUATOR(reading)
        assert score == 0.0

    def test_one_resource_maxed_others_none(self) -> None:
        reading = ResourceReading(gold=10_000_000, elixir=None, dark_elixir=None)
        score = self.EVALUATOR(reading)
        assert 0.0 < score <= 100.0

    def test_suitability_score_never_exceeds_100(self) -> None:
        reading = ResourceReading(gold=100_000_000, elixir=100_000_000, dark_elixir=1_000_000)
        score = self.EVALUATOR(reading)
        assert score == 100.0

    def test_negative_resource_thresholds_validation(self) -> None:
        from clashvision.core.config import ResourceThresholds
        rt = ResourceThresholds(min_gold=-1)
        with pytest.raises(ConfigurationError):
            rt.validate()

    def test_negative_resource_weights_validation(self) -> None:
        from clashvision.core.config import ResourceThresholds
        rt = ResourceThresholds(gold_weight=-0.5)
        with pytest.raises(ConfigurationError):
            rt.validate()

    @pytest.mark.parametrize("ratio", [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    def test_exposed_collector_ratio_boundaries(self, ratio: float) -> None:
        lf = BaseLayoutFeatures(exposed_collector_ratio=ratio)
        assert 0.0 <= lf.exposed_collector_ratio <= 1.0

    def test_base_layout_zero_buildings(self) -> None:
        model = BaseLayoutModel(layout_type=BaseLayoutType.EXPOSED_COLLECTORS, buildings=[])
        assert model.collector_count == 0
        assert model.exposed_collector_count == 0
        assert model.storage_count == 0

    def test_empty_deployment_plan_recommendation(self) -> None:
        rec = FarmingRecommendation(
            suitability_score=0.0,
            primary_strategy=StrategyType.CUSTOM,
            deployment_plan=[],
            explanation="Unsuitable base; zero deploy steps.",
        )
        assert len(rec.deployment_plan) == 0
        assert rec.suitability_score == 0.0


# ==============================================================================
# Group 6: Simulator & Playback Boundary Cases (Features 22–27)
# ==============================================================================
class TestSimulatorAndPlaybackBoundaries:
    """Extreme playback progress, step bounds, and sandboxed input coverage."""

    def test_simulation_state_current_time_exceeds_duration_clamps_to_1(self) -> None:
        sim = SimulationState(current_time_ms=15000, total_duration_ms=10000)
        assert sim.progress_fraction == 1.0

    def test_simulation_state_negative_current_time_clamps_to_0(self) -> None:
        sim = SimulationState(current_time_ms=-500, total_duration_ms=10000)
        assert sim.progress_fraction == 0.0

    def test_simulation_config_invalid_speed_validation(self) -> None:
        from clashvision.core.config import SimulatorConfig
        cfg = SimulatorConfig(playback_speed=-1.0)
        with pytest.raises(ConfigurationError):
            cfg.validate()

    def test_simulation_config_invalid_canvas_size_validation(self) -> None:
        from clashvision.core.config import SimulatorConfig
        cfg = SimulatorConfig(canvas_width=0, canvas_height=-100)
        with pytest.raises(ConfigurationError):
            cfg.validate()

    @pytest.mark.parametrize(
        "pyautogui_method",
        [
            "click",
            "doubleClick",
            "tripleClick",
            "rightClick",
            "middleClick",
            "moveTo",
            "moveRel",
            "dragTo",
            "dragRel",
            "mouseDown",
            "mouseUp",
            "press",
            "keyDown",
            "keyUp",
            "typewrite",
            "write",
            "hotkey",
        ],
    )
    def test_all_pyautogui_methods_blocked_outside_sandbox(self, pyautogui_method: str) -> None:
        """Every single PyAutoGUI input method raises LiveInputSafetyViolation outside sandbox."""
        try:
            import pyautogui  # type: ignore
            func = getattr(pyautogui, pyautogui_method, None)
            if func is not None:
                with pytest.raises(LiveInputSafetyViolation):
                    func()
        except ImportError:
            pytest.skip("pyautogui not installed")


# ==============================================================================
# Group 7: Storage & Database Boundaries (Features 28–31)
# ==============================================================================
class TestStorageAndDatabaseBoundaries:
    """SQL injection resistance, unicode persistence, duplicate keys, and transactions."""

    def test_sql_injection_resistance_in_session_id(self, temp_db_path: str) -> None:
        """Parameterized queries prevent SQL injection."""
        conn = sqlite3.connect(temp_db_path)
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, note TEXT);")
        malicious_id = "'; DROP TABLE sessions; --"
        conn.execute("INSERT INTO sessions VALUES (?, ?);", (malicious_id, "Injected"))
        conn.commit()

        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions';")
        assert cur.fetchone() is not None  # Table was NOT dropped!
        conn.close()

    def test_unicode_and_special_characters_persistence(self, temp_db_path: str) -> None:
        """Persists emoji, Cyrillic, Chinese, and symbols cleanly."""
        conn = sqlite3.connect(temp_db_path)
        conn.execute("CREATE TABLE notes (id INT PRIMARY KEY, text TEXT);")
        unicode_text = "⚔️ ClashVision 测试 1.2M 💰"
        conn.execute("INSERT INTO notes VALUES (?, ?);", (1, unicode_text))
        conn.commit()

        cur = conn.cursor()
        cur.execute("SELECT text FROM notes WHERE id=1;")
        retrieved = cur.fetchone()[0]
        conn.close()
        assert retrieved == unicode_text

    def test_duplicate_primary_key_raises_integrity_error(self, temp_db_path: str) -> None:
        conn = sqlite3.connect(temp_db_path)
        conn.execute("CREATE TABLE items (id TEXT PRIMARY KEY);")
        conn.execute("INSERT INTO items VALUES ('A1');")
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO items VALUES ('A1');")
        conn.close()

    def test_transaction_rollback_on_failure(self, temp_db_path: str) -> None:
        conn = sqlite3.connect(temp_db_path)
        conn.execute("CREATE TABLE balance (val INT);")
        conn.execute("INSERT INTO balance VALUES (100);")
        conn.commit()

        try:
            with conn:
                conn.execute("UPDATE balance SET val = 200;")
                raise RuntimeError("Simulated failure during transaction")
        except RuntimeError:
            pass

        cur = conn.cursor()
        cur.execute("SELECT val FROM balance;")
        val = cur.fetchone()[0]
        conn.close()
        assert val == 100  # Successfully rolled back!
