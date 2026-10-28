"""
Integration Test Suite for ClashVision Autonomous Farming Engine.
Verifies complete autonomous simulation workflows, multi-cycle endurance,
strategy transitions, database persistence, and fault tolerance.
"""

import time
import unittest
from typing import Dict, Any
from unittest.mock import MagicMock

import pytest
import numpy as np

from clashvision.core.constants import StrategyType, GameState
from clashvision.core.models import ResourceReading, BaseLayoutFeatures
from clashvision.simulator.safety_barrier import SafetyBarrier
from clashvision.capture.safety_guard import SafetyGuard
from clashvision.simulator.autonomous_state_machine import FarmingState
from clashvision.simulator.target_selector import TargetFilterConfig, FilterMode
from clashvision.simulator.autonomous_engine import (
    AutonomousEngine,
    AutonomousEngineConfig,
)
from clashvision.storage.repository import Repository
from clashvision.storage.db import DatabaseManager


class TestAutonomousFarmingLoopIntegration(unittest.TestCase):
    """End-to-end integration tests verifying autonomous farming cycles."""

    def setUp(self):
        self.db = DatabaseManager(":memory:")
        self.repo = Repository(db=self.db)
        self.barrier = SafetyBarrier()
        self.barrier.reset_emergency_stop()

    def test_full_farming_loop_three_consecutive_cycles(self):
        """Runs 3 uninterrupted farming cycles without user intervention."""
        config = AutonomousEngineConfig(
            max_cycles=3,
            max_search_attempts_per_cycle=25,
            target_filter=TargetFilterConfig(
                min_gold=400_000,
                min_elixir=400_000,
                min_dark_elixir=3_000,
                filter_mode=FilterMode.ALL,
            ),
            playback_speed=10.0,
            base_generation_seed=777,
        )

        engine = AutonomousEngine(config=config, repo=self.repo, barrier=self.barrier)

        cycle_events = []
        engine.add_cycle_listener(lambda data: cycle_events.append(data))

        engine.start()

        timeout = 20.0
        t0 = time.time()
        while engine.is_running and (time.time() - t0) < timeout:
            time.sleep(0.3)

        self.assertFalse(engine.is_running, "Engine should terminate cleanly upon reaching max_cycles")
        self.assertEqual(engine.stats.total_cycles_completed, 3)
        self.assertEqual(len(cycle_events), 3)
        self.assertGreaterEqual(engine.stats.total_attacks_performed, 3)
        self.assertGreater(engine.stats.total_gold_acquired, 1_000_000)
        self.assertGreater(engine.stats.total_elixir_acquired, 1_000_000)
        self.assertGreater(engine.stats.total_stars, 2)

        # Check database records
        scans = self.repo.get_base_scans(limit=10)
        self.assertGreaterEqual(len(scans), 3)

    def test_army_composition_strategy_enforcement(self):
        """Verifies strategy is constrained to available troop composition."""
        # Config with only Dragon available -> should force Mass Dragon
        config = AutonomousEngineConfig(
            max_cycles=1,
            max_search_attempts_per_cycle=10,
            target_filter=TargetFilterConfig(min_gold=50_000, min_elixir=50_000, filter_mode=FilterMode.ANY),
            available_army={"Dragon": 12, "Lightning Spell": 11},
            playback_speed=10.0,
            base_generation_seed=909,
        )

        engine = AutonomousEngine(config=config, repo=self.repo, barrier=self.barrier)
        engine.start()

        timeout = 10.0
        t0 = time.time()
        while engine.is_running and (time.time() - t0) < timeout:
            time.sleep(0.3)

        if engine._thread and engine._thread.is_alive():
            engine._thread.join(timeout=1.0)

        self.assertFalse(engine.is_running)
        self.assertEqual(engine.stats.total_cycles_completed, 1)
        self.assertIn("MASS_DRAGON", engine.stats.strategy_counts)

    def test_emergency_stop_during_active_deployment(self):
        """Emergency stop must immediately abort active deployment within 50ms."""
        config = AutonomousEngineConfig(
            max_cycles=5,
            max_search_attempts_per_cycle=10,
            playback_speed=0.5,  # Slower to ensure we intercept during execution
        )

        engine = AutonomousEngine(config=config, repo=self.repo, barrier=self.barrier)
        engine.start()

        # Wait until it starts executing or searching
        t0 = time.time()
        while engine.current_state in (FarmingState.STOPPED, FarmingState.INITIALIZING, FarmingState.READY):
            time.sleep(0.05)
            if time.time() - t0 > 5.0:
                break

        # Trip emergency stop
        t_trip = time.perf_counter()
        engine.emergency_stop()
        t_halt = (time.perf_counter() - t_trip) * 1000.0

        if engine._thread and engine._thread.is_alive():
            engine._thread.join(timeout=1.0)

        self.assertFalse(engine.is_running)
        self.assertEqual(engine.current_state, FarmingState.ERROR)
        self.assertLess(t_halt, 50.0, f"Emergency stop took too long: {t_halt:.2f}ms")

    def test_corrupted_ocr_graceful_recovery(self):
        """Simulates OCR errors or corrupted readings without crashing the engine."""
        config = AutonomousEngineConfig(
            max_cycles=1,
            max_search_attempts_per_cycle=10,
            target_filter=TargetFilterConfig(filter_mode=FilterMode.ALL, min_ocr_confidence=0.85),
            playback_speed=10.0,
        )

        engine = AutonomousEngine(config=config, repo=self.repo, barrier=self.barrier)
        # Mock OCR engine to raise exception on first 2 calls, then succeed
        mock_ocr = MagicMock()
        mock_ocr.recognize_text.side_effect = [
            RuntimeError("OCR segmentation fault"),
            ValueError("Corrupted pixel buffer"),
            "900,000",
            "900,000",
            "8,000",
        ]
        engine.ocr_engine = mock_ocr

        engine.start()

        timeout = 10.0
        t0 = time.time()
        while engine.is_running and (time.time() - t0) < timeout:
            time.sleep(0.3)

        self.assertFalse(engine.is_running)
        self.assertEqual(engine.stats.total_cycles_completed, 1)
        self.assertGreaterEqual(engine.stats.ocr_failures, 2)


if __name__ == "__main__":
    unittest.main()
