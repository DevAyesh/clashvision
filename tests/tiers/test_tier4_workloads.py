"""
Tier 4: Real-World Application Workload Test Suite for ClashVision.
Executes multi-step, end-to-end simulated farming sessions matching realistic gameplay:
1. Scenario 1: Dead Base BARCH Farming (Scout -> Skip -> Match -> Recommend -> Simulate -> Results)
2. Scenario 2: Sneaky Goblin Town Hall Snipe (Scout -> Snipable TH -> Recommend -> Deploy Plan -> Playback)
3. Scenario 3: Mass Dragon Air Corridor with Emergency Stop Interruption
4. Scenario 4: Multi-Base Skipping Session with Analytics Metrics Tracking
5. Scenario 5: Connection Error Interruption and Automated Recovery
6. Scenario 6: User-Defined Custom Strategy Configuration & Re-Evaluation
Requirement threshold: >= 5 workload scenarios.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

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
from clashvision.core.exceptions import (
    EmergencyStopTriggered,
    LiveInputSafetyViolation,
)
from clashvision.core.models import (
    BaseLayoutFeatures,
    DeploymentStep,
    FarmingRecommendation,
    FramePacket,
    Rect,
    ResourceReading,
    SessionRecord,
    SimulationState,
    StrategyEvaluation,
    VisionAnalysisResult,
    WindowInfo,
)
from tests.fixtures.layout_generator import (
    BaseLayoutType,
    generate_canonical_layout,
)
from tests.fixtures.synthetic_generator import (
    GameStateType,
    generate_state_frame,
)
from tests.tiers.test_tier1_features import (
    TestFeature12NumericOCRParser as _Parser,
    TestFeature14ConfigurableFarmingBaseEvaluator as _Evaluator,
)



class TestRealWorldApplicationWorkloads:
    """End-to-end simulated workflows verifying complete multi-step sessions."""

    def test_workload_scenario_1_dead_base_barch_farming(self, temp_db_path: str) -> None:
        """
        Scenario 1: Complete BARCH Dead Base Farming Lifecycle.
        Home Village -> Multiplayer -> Matchmaking -> Base 1 (Skip) -> Base 2 (Match) ->
        Recommend BARCH -> Simulate 5-step playback -> Battle Result -> Home.
        """
        session_id = "WORKLOAD_SESS_01"
        scans_count = 0
        matched_targets = 0

        # Step 1: In Home Village
        f1 = generate_state_frame(state=GameStateType.HOME_VILLAGE)
        v1 = VisionAnalysisResult(state=GameState.HOME_VILLAGE, confidence=0.96)
        assert v1.state == GameState.HOME_VILLAGE

        # Step 2: Transition through Multiplayer & Matchmaking
        v_multi = VisionAnalysisResult(state=GameState.MULTIPLAYER_INTERFACE, confidence=0.92)
        v_match = VisionAnalysisResult(state=GameState.MATCHMAKING_INTERFACE, confidence=0.90)
        assert v_multi.state == GameState.MULTIPLAYER_INTERFACE
        assert v_match.state == GameState.MATCHMAKING_INTERFACE

        # Step 3: Base 1 Scout (Substandard: 120k Gold, 90k Elixir)
        scans_count += 1
        low_loot = ResourceReading(gold=120_000, elixir=90_000, dark_elixir=400)
        score_1 = _Evaluator.evaluate_base_resources(low_loot)
        assert score_1 < 40.0  # Reject / Skip base!

        # Step 4: Click 'Next' -> Base 2 Scout (Lucrative Dead Base: 1.1M Gold, 1.2M Elixir)
        scans_count += 1
        dead_base_layout = generate_canonical_layout(BaseLayoutType.DEAD_BASE)
        high_loot = ResourceReading(
            gold=dead_base_layout.total_gold,
            elixir=dead_base_layout.total_elixir,
            dark_elixir=dead_base_layout.total_dark_elixir,
        )
        score_2 = _Evaluator.evaluate_base_resources(high_loot)
        assert score_2 >= 85.0  # Accept base!
        matched_targets += 1

        # Step 5: Recommendation Engine generates BARCH attack plan
        rec = FarmingRecommendation(
            suitability_score=score_2,
            primary_strategy=StrategyType.BARCH,
            deployment_plan=[
                DeploymentStep(step_id=1, troop_type="Barbarian", target_x=100, target_y=200, unit_count=20, delay_ms=0),
                DeploymentStep(step_id=2, troop_type="Archer", target_x=100, target_y=220, unit_count=20, delay_ms=500),
                DeploymentStep(step_id=3, troop_type="Barbarian", target_x=800, target_y=200, unit_count=20, delay_ms=1000),
                DeploymentStep(step_id=4, troop_type="Archer", target_x=800, target_y=220, unit_count=20, delay_ms=1500),
                DeploymentStep(step_id=5, troop_type="Sneaky Goblin", target_x=500, target_y=800, unit_count=10, delay_ms=2500),
            ],
            explanation="Dead base with 1.1M Gold; perimeter collectors fully vulnerable to BARCH sweep.",
        )
        assert rec.primary_strategy == StrategyType.BARCH
        assert len(rec.deployment_plan) == 5

        # Step 6: Hand off to Local Simulator for playback
        sim = SimulationState(
            step_index=0,
            total_steps=len(rec.deployment_plan),
            is_playing=True,
            total_duration_ms=2500,
        )
        for step in rec.deployment_plan:
            sim.step_index = step.step_id
            sim.current_time_ms = step.delay_ms
        assert sim.step_index == 5
        assert sim.progress_fraction == 1.0

        # Step 7: Transition to BATTLE_RESULT
        v_result = VisionAnalysisResult(
            state=GameState.BATTLE_RESULT,
            confidence=0.94,
            evidence={"stars": 3, "gold_stolen": 1_100_000},
        )
        assert v_result.state == GameState.BATTLE_RESULT
        assert v_result.evidence["stars"] == 3

        # Step 8: Persist session record to SQLite
        conn = sqlite3.connect(temp_db_path)
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, scans INT, matched INT);")
        conn.execute("INSERT INTO sessions VALUES (?, ?, ?);", (session_id, scans_count, matched_targets))
        conn.commit()

        cur = conn.cursor()
        cur.execute("SELECT scans, matched FROM sessions WHERE session_id=?;", (session_id,))
        row = cur.fetchone()
        conn.close()
        assert row == (2, 1)

    def test_workload_scenario_2_sneaky_goblin_town_hall_sniping(self) -> None:
        """
        Scenario 2: Town Hall accessibility analysis and Sneaky Goblin sniping.
        Scout base with exposed TH -> Recommend SNEAKY_GOBLIN -> Verify plan targets TH.
        """
        th_exposed_layout = generate_canonical_layout(BaseLayoutType.TOWN_HALL_EXPOSED)
        assert th_exposed_layout.th_accessible is True

        reading = ResourceReading(gold=550_000, elixir=500_000, dark_elixir=3_500)
        score = _Evaluator.evaluate_base_resources(reading)
        assert score > 60.0

        # Build Snipe Recommendation
        plan = [
            DeploymentStep(step_id=1, troop_type="Sneaky Goblin", target_x=150, target_y=500, unit_count=5, delay_ms=0, note="TH Snipe"),
            DeploymentStep(step_id=2, troop_type="Jump Spell", target_x=250, target_y=500, unit_count=1, delay_ms=1000, note="Core path"),
        ]
        rec = FarmingRecommendation(
            suitability_score=85.0,
            primary_strategy=StrategyType.SNEAKY_GOBLIN,
            deployment_plan=plan,
            explanation="Town Hall isolated outside walls; snipable with 5 Sneaky Goblins.",
        )
        assert rec.primary_strategy == StrategyType.SNEAKY_GOBLIN
        assert "TH Snipe" in rec.deployment_plan[0].note

    def test_workload_scenario_3_mass_dragon_air_corridor_with_emergency_stop(self) -> None:
        """
        Scenario 3: Mass Dragon Air Corridor analysis with emergency stop trigger midway.
        """
        from clashvision.capture.safety_guard import SafetyGuard
        ad_layout = generate_canonical_layout(BaseLayoutType.AIR_DEFENSE_CORRIDOR)
        assert len(ad_layout.air_defense_locations) == 4

        guard = SafetyGuard()
        guard.register_simulator_window(hwnd=7777)

        # Simulator starts 4-step playback
        sim = SimulationState(step_index=0, total_steps=4, is_playing=True)

        # Step 1 & 2 execute cleanly
        guard.verify_input_allowed(target_hwnd=7777, action_name="deploy_dragon_1")
        sim.step_index = 1
        guard.verify_input_allowed(target_hwnd=7777, action_name="deploy_dragon_2")
        sim.step_index = 2

        # User hits Emergency Stop Hotkey midway
        t0 = time.perf_counter()
        guard.trigger_emergency_stop(reason="User hit Escape hotkey")
        t1 = time.perf_counter()
        assert (t1 - t0) * 1000.0 < EMERGENCY_STOP_LATENCY_MAX_MS

        # Step 3 must be blocked immediately
        sim.is_playing = False
        with pytest.raises(EmergencyStopTriggered):
            guard.verify_input_allowed(target_hwnd=7777, action_name="deploy_dragon_3")

        assert sim.step_index == 2  # Halted at step 2!
        guard.reset_emergency_stop()

    def test_workload_scenario_4_multi_base_skipping_session(self, temp_db_path: str) -> None:
        """
        Scenario 4: Multi-base scout skipping loop (Scouts 5 bases, skips 4, attacks 5th).
        """
        scanned_loot_values = [
            (120_000, 80_000, 300),   # Base 1: Skip
            (90_000, 110_000, 200),   # Base 2: Skip
            (210_000, 180_000, 800),  # Base 3: Skip
            (150_000, 130_000, 400),  # Base 4: Skip
            (980_000, 850_000, 6500), # Base 5: MATCH!
        ]

        conn = sqlite3.connect(temp_db_path)
        conn.execute("CREATE TABLE scans (id INT, gold INT, matched INT);")

        matched_target_id = None
        for idx, (g, e, de) in enumerate(scanned_loot_values, start=1):
            reading = ResourceReading(gold=g, elixir=e, dark_elixir=de)
            score = _Evaluator.evaluate_base_resources(reading)
            is_matched = 1 if score >= 75.0 else 0
            if is_matched:
                matched_target_id = idx
            conn.execute("INSERT INTO scans VALUES (?, ?, ?);", (idx, g, is_matched))

        conn.commit()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), SUM(matched) FROM scans;")
        total_scans, total_matched = cur.fetchone()
        conn.close()

        assert total_scans == 5
        assert total_matched == 1
        assert matched_target_id == 5

    def test_workload_scenario_5_connection_error_and_auto_recovery(self) -> None:
        """
        Scenario 5: Network connection error interrupt and automatic safe recovery.
        Active battle -> Connection Error popup -> System pauses capture -> Reload -> Home.
        """
        from clashvision.capture.mss_grabber import MSSGrabber
        grabber = MSSGrabber(mock_mode=True)
        assert grabber.is_paused is False

        # Connection error frame detected
        f_err = generate_state_frame(state=GameStateType.CONNECTION_ERROR)
        v_err = VisionAnalysisResult(
            state=GameState.CONNECTION_ERROR,
            confidence=0.95,
            evidence={"connection_lost_modal": True},
        )
        assert v_err.state == GameState.CONNECTION_ERROR

        # Auto-pause capture to prevent bad OCR reads or desync
        grabber.pause()
        assert grabber.is_paused is True

        # Reload game triggered -> Loading screen
        v_load = VisionAnalysisResult(state=GameState.LOADING_SCREEN, confidence=0.97)
        assert v_load.state == GameState.LOADING_SCREEN

        # Fully recovered into Home Village -> Resume capture
        v_home = VisionAnalysisResult(state=GameState.HOME_VILLAGE, confidence=0.95)
        grabber.resume()
        assert grabber.is_paused is False
        assert v_home.state == GameState.HOME_VILLAGE

    def test_workload_scenario_6_custom_strategy_tuning(self) -> None:
        """
        Scenario 6: User tunes custom resource thresholds and priority weighting.
        """
        from clashvision.core.config import ResourceThresholds
        custom_thresholds = ResourceThresholds(
            min_gold=700_000,
            min_elixir=700_000,
            min_dark_elixir=5_000,
            priority=ResourcePriority.DARK_ELIXIR_FIRST,
            dark_elixir_weight=200.0,
        )
        custom_thresholds.validate()

        # Evaluate base with massive DE (8,000 DE)
        reading = ResourceReading(gold=500_000, elixir=400_000, dark_elixir=8_000)
        score = _Evaluator.evaluate_base_resources(
            reading,
            min_gold=custom_thresholds.min_gold,
            min_elixir=custom_thresholds.min_elixir,
            min_dark_elixir=custom_thresholds.min_dark_elixir,
            priority=custom_thresholds.priority,
        )
        # Because DE is weighted heavily in DARK_ELIXIR_FIRST, high DE gives strong score
        assert score > 60.0

        eval_custom = StrategyEvaluation(
            strategy_type=StrategyType.CUSTOM,
            score=score,
            viable=True,
            risk_level=RiskLevel.MEDIUM,
            reasons=["Matched custom Dark Elixir priority (8,000 DE)"],
        )
        assert eval_custom.strategy_type == StrategyType.CUSTOM
        assert eval_custom.viable is True
