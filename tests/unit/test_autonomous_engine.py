"""
Unit Test Suite for ClashVision Autonomous Farming Engine.
Verifies Master Prompt Requirements across State Machine, Loot Filtering,
Strategy Execution, Sandboxed Deployment, Battle Resolution, and Safety Controls.
"""

import time
import unittest
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from clashvision.core.constants import StrategyType, RiskLevel, GameState
from clashvision.core.models import (
    BaseLayoutFeatures,
    DeploymentStep,
    ResourceReading,
    FarmingRecommendation,
)
from clashvision.core.exceptions import (
    LiveInputSafetyViolation,
    EmergencyStopTriggered,
)
from clashvision.simulator.safety_barrier import SafetyBarrier
from clashvision.capture.safety_guard import SafetyGuard
from clashvision.simulator.autonomous_state_machine import (
    AutonomousStateMachine,
    FarmingState,
    InvalidStateTransitionError,
)
from clashvision.simulator.target_selector import (
    TargetSelector,
    TargetFilterConfig,
    FilterMode,
    TargetEvaluationResult,
)
from clashvision.simulator.deployment_executor import (
    DeploymentExecutor,
    ExecutedStepRecord,
)
from clashvision.simulator.battle_resolver import (
    BattleResolver,
    SimulatedBattleResult,
)
from clashvision.simulator.autonomous_engine import (
    AutonomousEngine,
    AutonomousEngineConfig,
    AutonomousSessionStats,
)


class TestAutonomousStateMachine(unittest.TestCase):
    """Verifies the 15 formal states and transition validation matrix."""

    def setUp(self):
        self.barrier = SafetyBarrier()
        self.barrier.reset_emergency_stop()
        self.sm = AutonomousStateMachine(barrier=self.barrier)

    def test_initial_state(self):
        self.assertEqual(self.sm.current_state, FarmingState.STOPPED)
        self.assertEqual(self.sm.error_count, 0)
        self.assertEqual(len(self.sm.history), 0)

    def test_complete_valid_cycle_transitions(self):
        # STOPPED -> INITIALIZING -> READY -> SEARCHING -> ANALYZING_BASE ->
        # EVALUATING_LOOT -> SELECTING_STRATEGY -> PREPARING_ATTACK ->
        # EXECUTING_ATTACK -> WAITING_FOR_RESULT -> RECORDING_RESULTS -> READY
        self.sm.transition_to(FarmingState.INITIALIZING)
        self.assertEqual(self.sm.current_state, FarmingState.INITIALIZING)

        self.sm.transition_to(FarmingState.READY)
        self.sm.transition_to(FarmingState.SEARCHING)
        self.sm.transition_to(FarmingState.ANALYZING_BASE)
        self.sm.transition_to(FarmingState.EVALUATING_LOOT)
        self.sm.transition_to(FarmingState.SELECTING_STRATEGY)
        self.sm.transition_to(FarmingState.PREPARING_ATTACK)
        self.sm.transition_to(FarmingState.EXECUTING_ATTACK)
        self.sm.transition_to(FarmingState.WAITING_FOR_RESULT)
        self.sm.transition_to(FarmingState.RECORDING_RESULTS)
        self.sm.transition_to(FarmingState.READY)

        self.assertEqual(self.sm.current_state, FarmingState.READY)
        self.assertEqual(len(self.sm.history), 11)

    def test_skipping_base_transition(self):
        self.sm.transition_to(FarmingState.INITIALIZING)
        self.sm.transition_to(FarmingState.READY)
        self.sm.transition_to(FarmingState.SEARCHING)
        self.sm.transition_to(FarmingState.ANALYZING_BASE)
        self.sm.transition_to(FarmingState.EVALUATING_LOOT)
        # Loot unviable -> SKIPPING_BASE -> SEARCHING
        self.sm.transition_to(FarmingState.SKIPPING_BASE)
        self.assertEqual(self.sm.current_state, FarmingState.SKIPPING_BASE)
        self.sm.transition_to(FarmingState.SEARCHING)
        self.assertEqual(self.sm.current_state, FarmingState.SEARCHING)

    def test_invalid_transition_raises_error(self):
        self.sm.transition_to(FarmingState.INITIALIZING)
        # Attempting direct jump from INITIALIZING to EXECUTING_ATTACK
        with self.assertRaises(InvalidStateTransitionError):
            self.sm.transition_to(FarmingState.EXECUTING_ATTACK)

    def test_duplicate_transition_suppression(self):
        self.sm.transition_to(FarmingState.INITIALIZING)
        initial_history_len = len(self.sm.history)
        # Attempting duplicate transition to same state
        self.sm.transition_to(FarmingState.INITIALIZING)
        self.assertEqual(len(self.sm.history), initial_history_len)

    def test_pause_and_resume_recovery(self):
        self.sm.transition_to(FarmingState.INITIALIZING)
        self.sm.transition_to(FarmingState.READY)
        self.sm.transition_to(FarmingState.SEARCHING)

        self.sm.pause(reason="user_button")
        self.assertEqual(self.sm.current_state, FarmingState.PAUSED)

        self.sm.resume(trigger="user_button")
        self.assertEqual(self.sm.current_state, FarmingState.SEARCHING)

    def test_emergency_stop_forces_error_state(self):
        self.sm.transition_to(FarmingState.INITIALIZING)
        self.sm.transition_to(FarmingState.READY)

        # Trip emergency barrier
        self.barrier.trigger_emergency_stop(reason="Unit test emergency halt")
        self.sm.transition_to(FarmingState.SEARCHING)

        self.assertEqual(self.sm.current_state, FarmingState.ERROR)
        self.assertGreater(self.sm.error_count, 0)


class TestTargetSelectorAndLootFiltering(unittest.TestCase):
    """Verifies target selection across ALL, ANY, WEIGHTED, and CUSTOM filtering modes."""

    def setUp(self):
        self.default_config = TargetFilterConfig(
            min_gold=600_000,
            min_elixir=600_000,
            min_dark_elixir=5_000,
            filter_mode=FilterMode.ALL,
            min_ocr_confidence=0.60,
        )
        self.selector = TargetSelector(self.default_config)

    def test_filter_mode_all_pass(self):
        res = ResourceReading(gold=750_000, elixir=650_000, dark_elixir=5_500)
        eval_result = self.selector.evaluate(res)
        self.assertEqual(eval_result.action, "ATTACK")
        self.assertTrue(eval_result.is_suitable)
        self.assertGreaterEqual(eval_result.farming_score, 50.0)

    def test_filter_mode_all_fail_when_one_below(self):
        # Dark Elixir below threshold (4,000 < 5,000)
        res = ResourceReading(gold=800_000, elixir=800_000, dark_elixir=4_000)
        eval_result = self.selector.evaluate(res)
        self.assertEqual(eval_result.action, "SKIP")
        self.assertFalse(eval_result.is_suitable)
        self.assertTrue(any("DE" in r for r in eval_result.reasons))

    def test_filter_mode_any_pass(self):
        self.selector.update_config(TargetFilterConfig(
            min_gold=600_000,
            min_elixir=600_000,
            min_dark_elixir=5_000,
            filter_mode=FilterMode.ANY,
        ))
        # Only Gold is high, Elixir and DE are low
        res = ResourceReading(gold=850_000, elixir=150_000, dark_elixir=1_000)
        eval_result = self.selector.evaluate(res)
        self.assertEqual(eval_result.action, "ATTACK")
        self.assertTrue(eval_result.is_suitable)

    def test_filter_mode_weighted(self):
        self.selector.update_config(TargetFilterConfig(
            filter_mode=FilterMode.WEIGHTED,
            weight_gold=1.0,
            weight_elixir=1.0,
            weight_dark_elixir=100.0,
            min_weighted_total=1_200_000.0,
        ))
        # 400k Gold + 300k Elixir + 6k DE * 100 = 400k + 300k + 600k = 1.3M >= 1.2M
        res = ResourceReading(gold=400_000, elixir=300_000, dark_elixir=6_000)
        eval_result = self.selector.evaluate(res)
        self.assertEqual(eval_result.action, "ATTACK")
        self.assertGreaterEqual(eval_result.weighted_value, 1_200_000.0)

    def test_filter_mode_custom_rule(self):
        # Custom rule: Must have Town Hall accessible AND Gold >= 500k
        custom_rule = lambda res, lay: (lay is not None and lay.th_accessible and res.gold >= 500_000)
        self.selector.update_config(TargetFilterConfig(
            filter_mode=FilterMode.CUSTOM,
            custom_predicate=custom_rule,
        ))

        res = ResourceReading(gold=650_000, elixir=100_000, dark_elixir=0)
        layout_exposed = BaseLayoutFeatures(th_accessible=True)
        layout_trapped = BaseLayoutFeatures(th_accessible=False)

        self.assertEqual(self.selector.evaluate(res, layout_exposed).action, "ATTACK")
        self.assertEqual(self.selector.evaluate(res, layout_trapped).action, "SKIP")

    def test_low_ocr_confidence_triggers_skip(self):
        # Numbers meet thresholds, but confidence is low (0.45 < 0.60)
        res = ResourceReading(
            gold=900_000, elixir=900_000, dark_elixir=8_000,
            confidence={"gold": 0.45, "elixir": 0.45, "dark_elixir": 0.45}
        )
        eval_result = self.selector.evaluate(res)
        self.assertEqual(eval_result.action, "SKIP")
        self.assertTrue(any("confidence" in r.lower() for r in eval_result.reasons))

    def test_none_resources_safe_handling(self):
        eval_result = self.selector.evaluate(None)
        self.assertEqual(eval_result.action, "SKIP")
        self.assertFalse(eval_result.is_suitable)


class TestDeploymentExecutorAndSafety(unittest.TestCase):
    """Verifies coordinate scaling, sandboxed dispatch, and live window safety blocks."""

    def setUp(self):
        self.guard = SafetyGuard()
        self.barrier = SafetyBarrier()
        self.barrier.reset_emergency_stop()
        self.executor = DeploymentExecutor(
            canvas_width=1920,
            canvas_height=1080,
            guard=self.guard,
            barrier=self.barrier,
        )

    def test_coordinate_mapping_normalized(self):
        step = DeploymentStep(
            step_id=1,
            troop_type="Sneaky Goblin",
            normalized_x=0.25,
            normalized_y=0.50,
            unit_count=5,
        )
        px, py, nx, ny = self.executor.map_coordinates(step)
        self.assertEqual(px, 480)
        self.assertEqual(py, 540)
        self.assertAlmostEqual(nx, 0.25, places=2)
        self.assertAlmostEqual(ny, 0.50, places=2)

    def test_execute_step_confirmed(self):
        step = DeploymentStep(
            step_id=1,
            troop_type="Sneaky Goblin",
            normalized_x=0.10,
            normalized_y=0.20,
            unit_count=3,
        )
        record = self.executor.execute_step(step)
        self.assertEqual(record.status, "CONFIRMED")
        self.assertEqual(record.step_id, 1)
        self.assertEqual(record.target_pixel_x, 192)
        self.assertEqual(record.target_pixel_y, 216)

    def test_emergency_stop_halts_execution(self):
        self.barrier.trigger_emergency_stop(reason="Test Stop")
        step = DeploymentStep(step_id=2, troop_type="Dragon")
        with self.assertRaises(EmergencyStopTriggered):
            self.executor.execute_step(step)

    def test_live_window_safety_violation(self):
        # Register live emulator HWND 88888
        self.guard.register_live_window(hwnd=88888, title="BlueStacks", process_name="HD-Player.exe")
        # Direct controller at the live emulator
        self.executor.controller.canvas_hwnd = 88888

        step = DeploymentStep(step_id=3, troop_type="Barbarian")
        with self.assertRaises(LiveInputSafetyViolation):
            self.executor.execute_step(step)


class TestBattleResolver(unittest.TestCase):
    """Verifies battle result calculation, star rating, and loot yields."""

    def test_sneaky_goblin_resolution(self):
        steps = [
            DeploymentStep(step_id=i, troop_type="Sneaky Goblin", unit_count=5)
            for i in range(10)
        ]
        avail = ResourceReading(gold=1_000_000, elixir=800_000, dark_elixir=6_000)
        layout = BaseLayoutFeatures(th_accessible=True, exposed_collector_ratio=0.8)

        res = BattleResolver.resolve(
            strategy_type=StrategyType.SNEAKY_GOBLIN,
            steps=steps,
            available_resources=avail,
            layout=layout,
            execution_fraction=1.0,
        )
        self.assertIsInstance(res, SimulatedBattleResult)
        self.assertGreaterEqual(res.stars, 1)  # Town Hall destroyed or 50%
        self.assertGreaterEqual(res.gold_looted, 850_000)  # > 85% harvest
        self.assertTrue(res.th_destroyed)

    def test_barch_resolution(self):
        steps = [
            DeploymentStep(step_id=i, troop_type="Barbarian", unit_count=10)
            for i in range(8)
        ]
        avail = ResourceReading(gold=600_000, elixir=600_000, dark_elixir=2_000)
        layout = BaseLayoutFeatures(th_accessible=False, exposed_collector_ratio=0.9)

        res = BattleResolver.resolve(
            strategy_type=StrategyType.BARCH,
            steps=steps,
            available_resources=avail,
            layout=layout,
            execution_fraction=1.0,
        )
        self.assertGreaterEqual(res.destruction_percent, 50)
        self.assertGreaterEqual(res.stars, 1)


class TestAutonomousEngineEndToEnd(unittest.TestCase):
    """Verifies autonomous session execution, cycle limits, and analytics tracking."""

    def test_autonomous_engine_two_cycles(self):
        config = AutonomousEngineConfig(
            max_cycles=2,
            max_search_attempts_per_cycle=15,
            target_filter=TargetFilterConfig(
                min_gold=400_000,
                min_elixir=400_000,
                min_dark_elixir=3_000,
                filter_mode=FilterMode.ALL,
            ),
            playback_speed=10.0,
            base_generation_seed=5005,
        )
        engine = AutonomousEngine(config=config)
        engine.start()

        timeout = 15.0
        t0 = time.time()
        while engine.is_running and (time.time() - t0) < timeout:
            time.sleep(0.3)

        self.assertFalse(engine.is_running)
        self.assertEqual(engine.stats.total_cycles_completed, 2)
        self.assertGreaterEqual(engine.stats.total_attacks_performed, 2)
        self.assertGreater(engine.stats.total_gold_acquired, 0)
        self.assertGreater(engine.stats.total_stars, 0)
        self.assertGreater(len(engine.last_cycle_actions), 0)

    def test_search_attempts_exhaustion_safety(self):
        # Impossible thresholds guarantee search exhaustion
        config = AutonomousEngineConfig(
            max_cycles=1,
            max_search_attempts_per_cycle=5,
            target_filter=TargetFilterConfig(
                min_gold=999_000_000,  # Unattainable
                filter_mode=FilterMode.ALL,
            ),
            playback_speed=10.0,
        )
        engine = AutonomousEngine(config=config)
        engine.start()

        timeout = 10.0
        t0 = time.time()
        while engine.is_running and (time.time() - t0) < timeout:
            time.sleep(0.3)

        self.assertFalse(engine.is_running)
        self.assertEqual(engine.stats.total_cycles_completed, 0)
        self.assertEqual(engine.stats.total_attacks_performed, 0)
        self.assertGreaterEqual(engine.stats.total_bases_skipped, 5)


if __name__ == "__main__":
    unittest.main()

