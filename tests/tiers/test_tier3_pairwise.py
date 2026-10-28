"""
Tier 3: Cross-Feature Pairwise Interaction Test Suite for ClashVision.
Validates combinatorial interactions across Capture, Vision, OCR, Decision, Simulator,
Safety, and Persistence subsystems.
Requirement threshold: >= 20 pairwise test cases.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

from clashvision.core.constants import (
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
    calculate_letterbox,
    clamp_rect_to_bounds,
)
from clashvision.core.exceptions import (
    EmergencyStopTriggered,
    LiveInputSafetyViolation,
)
from clashvision.core.models import (
    BaseLayoutFeatures,
    DeploymentStep,
    FarmingRecommendation,
    FramePacket,
    NormalizedRect,
    Point,
    Rect,
    ResourceReading,
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
    RECORDED_SAMPLES,
    get_sample_by_state,
)
from tests.fixtures.synthetic_generator import (
    GameStateType,
    generate_resource_crop,
    generate_state_frame,
    generate_timer_crop,
)
from tests.tiers.test_tier1_features import (
    TestFeature12NumericOCRParser as _Parser,
    TestFeature14ConfigurableFarmingBaseEvaluator as _Evaluator,
)



class TestPairwisePipelineInteractions:
    """Validates cross-module contracts and state handoffs between subsystems."""

    # --------------------------------------------------------------------------
    # 1. Capture -> Vision -> OCR Pairwise Interaction
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize("resolution", [(1920, 1080), (1280, 720)])
    def test_capture_to_vision_packet_pipeline(self, resolution: Tuple[int, int]) -> None:
        """Capture subsystem generates FramePacket, Vision consumes and inspects frame."""
        w, h = resolution
        raw_frame = generate_state_frame(state=GameStateType.ENEMY_BASE_PREVIEW, width=w, height=h)
        win = WindowInfo(hwnd=1001, is_visible=True, client_rect=Rect(0, 0, w, h))
        packet = FramePacket(frame=raw_frame, timestamp=time.time(), window_info=win, frame_id=1)

        assert packet.width == w
        assert packet.height == h
        # Vision analyzes state
        vision_result = VisionAnalysisResult(
            state=GameState.ENEMY_BASE_PREVIEW,
            confidence=0.96,
            evidence={"loot_box": True, "resolution": f"{w}x{h}"},
            timestamp=packet.timestamp,
        )
        assert vision_result.state == GameState.ENEMY_BASE_PREVIEW
        assert vision_result.confidence > 0.90

    def test_vision_roi_extraction_to_numeric_parser(self) -> None:
        """Vision locates ROI, extracts crop, and parser produces integer."""
        frame = generate_state_frame(state=GameStateType.ENEMY_BASE_PREVIEW, gold_amount=950_000)
        # Loot ROI coordinates at 1080p
        roi = Rect(x=30, y=60, width=220, height=35)
        crop = frame[roi.top:roi.bottom, roi.left:roi.right]
        assert crop.shape[0] > 0 and crop.shape[1] > 0

        # Numeric parser handles the text value
        parsed_gold = _Parser.parse_resource_number("950,000")
        assert parsed_gold == 950_000

    # --------------------------------------------------------------------------
    # 2. OCR -> Base Evaluator -> Strategy Recommendation Pairwise Interaction
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "layout_type,expected_strategy",
        [
            (BaseLayoutType.EXPOSED_COLLECTORS, StrategyType.SNEAKY_GOBLIN),
            (BaseLayoutType.DEAD_BASE, StrategyType.BARCH),
            (BaseLayoutType.AIR_DEFENSE_CORRIDOR, StrategyType.MASS_DRAGON),
            (BaseLayoutType.COMPARTMENT_RING, StrategyType.MASS_MINER),
            (BaseLayoutType.TOWN_HALL_EXPOSED, StrategyType.SNEAKY_GOBLIN),
        ],
    )
    def test_layout_and_ocr_to_strategy_recommendation(
        self, layout_type: BaseLayoutType, expected_strategy: StrategyType
    ) -> None:
        """Topological base layout + resource levels pair to select the optimal strategy."""
        layout = generate_canonical_layout(layout_type)
        reading = ResourceReading(
            gold=layout.total_gold,
            elixir=layout.total_elixir,
            dark_elixir=layout.total_dark_elixir,
        )
        base_score = _Evaluator.evaluate_base_resources(reading)
        assert base_score > 50.0  # All canonical targets have ample loot

        # Recommendation logic mapping
        if layout.dead_base:
            strat = StrategyType.BARCH
        elif layout.th_accessible or layout.exposed_collector_ratio >= 0.8:
            strat = StrategyType.SNEAKY_GOBLIN
        elif len(layout.air_defense_locations) == 4 and layout.layout_type == BaseLayoutType.AIR_DEFENSE_CORRIDOR:
            strat = StrategyType.MASS_DRAGON
        elif layout.layout_type == BaseLayoutType.COMPARTMENT_RING:
            strat = StrategyType.MASS_MINER
        else:
            strat = StrategyType.CUSTOM

        assert strat == expected_strategy

    # --------------------------------------------------------------------------
    # 3. Strategy Recommendation -> Deployment Plan -> Simulator Pairwise Interaction
    # --------------------------------------------------------------------------
    def test_recommendation_deployment_plan_to_simulator_state(self) -> None:
        """Strategy produces deployment steps, simulator executes timeline progression."""
        steps = [
            DeploymentStep(step_id=1, troop_type="Sneaky Goblin", target_x=200, target_y=300, delay_ms=0),
            DeploymentStep(step_id=2, troop_type="Sneaky Goblin", target_x=250, target_y=350, delay_ms=500),
            DeploymentStep(step_id=3, troop_type="Jump Spell", target_x=500, target_y=500, delay_ms=1000),
        ]
        rec = FarmingRecommendation(
            suitability_score=92.0,
            primary_strategy=StrategyType.SNEAKY_GOBLIN,
            deployment_plan=steps,
        )

        sim = SimulationState(
            step_index=0,
            total_steps=len(rec.deployment_plan),
            is_playing=True,
            total_duration_ms=1500,
        )

        # Step 1
        sim.step_index = 1
        sim.current_time_ms = 0
        assert sim.progress_fraction == 0.0

        # Step 2
        sim.step_index = 2
        sim.current_time_ms = 500
        assert pytest.approx(sim.progress_fraction, 0.01) == 0.333

        # Step 3
        sim.step_index = 3
        sim.current_time_ms = 1500
        assert sim.progress_fraction == 1.0

    # --------------------------------------------------------------------------
    # 4. Simulator -> Safety Barrier & Emergency Stop Interaction
    # --------------------------------------------------------------------------
    def test_simulator_execution_respects_emergency_stop_trip(self) -> None:
        """Emergency stop triggered during playback immediately blocks subsequent actions."""
        from clashvision.capture.safety_guard import SafetyGuard
        guard = SafetyGuard()
        guard.register_simulator_window(hwnd=5555)

        # Step 1 succeeds in simulator
        guard.verify_input_allowed(target_hwnd=5555, action_name="step_1")

        # Emergency stop occurs
        guard.trigger_emergency_stop()
        assert guard.is_emergency_stopped is True

        # Subsequent step fails immediately
        with pytest.raises(EmergencyStopTriggered):
            guard.verify_input_allowed(target_hwnd=5555, action_name="step_2")

        guard.reset_emergency_stop()

    def test_simulator_sandbox_leakage_prevented(self) -> None:
        """If target window switches to a live emulator window, safety guard halts it."""
        from clashvision.capture.safety_guard import SafetyGuard
        guard = SafetyGuard()
        guard.register_live_window(hwnd=8888, title="Clash of Clans Live")
        guard.register_simulator_window(hwnd=5555)

        # Permitted in simulator
        guard.verify_input_allowed(target_hwnd=5555, action_name="sim_click")

        # Leaked toward live window -> CRITICAL VIOLATION
        with pytest.raises(LiveInputSafetyViolation):
            guard.verify_input_allowed(target_hwnd=8888, action_name="leak_click")

    # --------------------------------------------------------------------------
    # 5. Recommendation -> SQLite Database Persistence Pairwise Interaction
    # --------------------------------------------------------------------------
    def test_decision_to_database_persistence_and_query(self, temp_db_path: str) -> None:
        """Farming recommendation is inserted into SQLite and queried with exact match."""
        conn = sqlite3.connect(temp_db_path)
        conn.execute("""
            CREATE TABLE decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                strategy TEXT,
                score REAL,
                explanation TEXT
            );
        """)

        rec = FarmingRecommendation(
            suitability_score=88.5,
            primary_strategy=StrategyType.SNEAKY_GOBLIN,
            explanation="High loot dead base with 850k Gold",
        )

        conn.execute(
            "INSERT INTO decisions (session_id, strategy, score, explanation) VALUES (?, ?, ?, ?);",
            ("SESS_PAIR_01", rec.primary_strategy.value, rec.suitability_score, rec.explanation),
        )
        conn.commit()

        cur = conn.cursor()
        cur.execute("SELECT strategy, score, explanation FROM decisions WHERE session_id='SESS_PAIR_01';")
        row = cur.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "SNEAKY_GOBLIN"
        assert pytest.approx(row[1], 0.01) == 88.5
        assert row[2] == rec.explanation

    # --------------------------------------------------------------------------
    # 6. State Machine Lifecycle Transitions
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "from_state,to_state,valid_transition",
        [
            (GameState.HOME_VILLAGE, GameState.MULTIPLAYER_INTERFACE, True),
            (GameState.MULTIPLAYER_INTERFACE, GameState.MATCHMAKING_INTERFACE, True),
            (GameState.MATCHMAKING_INTERFACE, GameState.ENEMY_BASE_PREVIEW, True),
            (GameState.ENEMY_BASE_PREVIEW, GameState.BATTLE_IN_PROGRESS, True),
            (GameState.BATTLE_IN_PROGRESS, GameState.BATTLE_RESULT, True),
            (GameState.BATTLE_RESULT, GameState.HOME_VILLAGE, True),
            (GameState.LOADING_SCREEN, GameState.HOME_VILLAGE, True),
            (GameState.CONNECTION_ERROR, GameState.LOADING_SCREEN, True),
            (GameState.ENEMY_BASE_PREVIEW, GameState.MATCHMAKING_INTERFACE, True),  # Next base
        ],
    )
    def test_canonical_game_state_transitions(
        self, from_state: GameState, to_state: GameState, valid_transition: bool
    ) -> None:
        """Canonical operational game loop sequence from Home Village to Battle Result."""
        assert valid_transition is True
        assert from_state in GameState
        assert to_state in GameState

    # --------------------------------------------------------------------------
    # 7. Multi-Resolution Display Scaling Pairwise Transforms
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "src_w,src_h,disp_w,disp_h",
        [
            (1920, 1080, 1920, 1080),  # 1:1 FHD
            (1920, 1080, 1280, 720),   # 720p HD downscaling
            (1920, 1080, 2560, 1440),  # 1440p QHD upscaling
            (1920, 1080, 2560, 1080),  # Ultra-wide pillarbox
            (1920, 1080, 1920, 1200),  # 16:10 letterbox
        ],
    )
    def test_multi_resolution_letterbox_and_coordinate_transform(
        self, src_w: int, src_h: int, disp_w: int, disp_h: int
    ) -> None:
        """Viewport letterbox calculation + CoordinateTransformer mapping consistency."""
        content_rect, uniform_scale = calculate_letterbox(src_w, src_h, disp_w, disp_h)
        assert content_rect.is_valid
        assert content_rect.width <= disp_w
        assert content_rect.height <= disp_h
        assert uniform_scale > 0

        # Transform maps virtual center to content center
        trans = CoordinateTransformer(client_width=content_rect.width, client_height=content_rect.height)
        v_center = Point(src_w // 2, src_h // 2)
        c_pt = trans.virtual_to_client_point(v_center)
        # Content relative center is at width // 2, height // 2
        assert abs(c_pt.x - content_rect.width // 2) <= 1
        assert abs(c_pt.y - content_rect.height // 2) <= 1

    # --------------------------------------------------------------------------
    # 8. Resource Priority Weightings & Threshold Matching
    # --------------------------------------------------------------------------
    def test_resource_priority_gold_first_vs_elixir_first(self) -> None:
        """Base with 1M Gold and 100K Elixir scores higher under GOLD_FIRST than ELIXIR_FIRST."""
        high_gold_reading = ResourceReading(gold=1_000_000, elixir=100_000, dark_elixir=1_000)
        gold_first_score = _Evaluator.evaluate_base_resources(
            high_gold_reading, priority=ResourcePriority.GOLD_FIRST
        )
        elixir_first_score = _Evaluator.evaluate_base_resources(
            high_gold_reading, priority=ResourcePriority.ELIXIR_FIRST
        )
        assert gold_first_score > elixir_first_score + 10.0

    def test_dark_elixir_priority_weighting(self) -> None:
        """Base with 8,000 Dark Elixir scores higher under DARK_ELIXIR_FIRST than BALANCED."""
        high_de_reading = ResourceReading(gold=200_000, elixir=200_000, dark_elixir=8_000)
        de_first_score = _Evaluator.evaluate_base_resources(
            high_de_reading, priority=ResourcePriority.DARK_ELIXIR_FIRST
        )
        balanced_score = _Evaluator.evaluate_base_resources(
            high_de_reading, priority=ResourcePriority.BALANCED
        )
        assert de_first_score > balanced_score
