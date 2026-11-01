"""
End-to-End Autonomous Simulator & Safety Boundary Verification Test Suite.
Verifies Phase 9 Requirements:
- Multi-cycle autonomous farming execution in complete sandbox isolation
- Zero-live-game-input invariant (no unattended PyAutoGUI/ADB/keyboard commands)
- Emergency stop latency bound (<= 50ms) and state latching
- Adversarial base evaluation resilience and SQLite persistence
"""

import os
import tempfile
import time
import unittest

from clashvision.core.constants import StrategyType
from clashvision.core.exceptions import EmergencyStopTriggered, LiveInputSafetyViolation
from clashvision.core.models import WindowInfo, Rect
from clashvision.capture.safety_guard import SafetyGuard
from clashvision.simulator.safety_barrier import SafetyBarrier
from clashvision.simulator.autonomous_state_machine import FarmingState
from clashvision.simulator.target_selector import TargetFilterConfig, FilterMode
from clashvision.simulator.autonomous_engine import AutonomousEngine, AutonomousEngineConfig
from clashvision.storage.db import DatabaseManager
from clashvision.storage.repository import Repository


class TestSimulatorE2ESafety(unittest.TestCase):
    """End-to-end simulation and safety boundary tests."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "e2e_sim.db")
        self.db = DatabaseManager(db_path=self.db_path)
        self.repo = Repository(db=self.db)

        self.guard = SafetyGuard()
        self.barrier = SafetyBarrier()
        self.barrier.reset_emergency_stop()

        # Configure fast autonomous simulation config for testing
        self.config = AutonomousEngineConfig(
            max_cycles=3,
            max_search_attempts_per_cycle=20,
            session_time_limit_seconds=30.0,
            playback_speed=10.0,
            base_generation_seed=2026,
            target_filter=TargetFilterConfig(
                min_gold=400_000,
                min_elixir=400_000,
                min_dark_elixir=2_000,
                filter_mode=FilterMode.ANY,
                min_ocr_confidence=0.50,
            ),
        )

        self.engine = AutonomousEngine(
            config=self.config,
            repo=self.repo,
            barrier=self.barrier,
            guard=self.guard,
        )

    def tearDown(self):
        if self.engine.is_running:
            self.engine.stop()
            time.sleep(0.1)
        self.barrier.reset_emergency_stop()
        self.db.close()
        self.temp_dir.cleanup()

    def test_e2e_multi_cycle_autonomous_farming_run(self):
        """Verifies multi-cycle continuous autonomous execution and persistence."""
        cycles_recorded = []
        self.engine.add_cycle_listener(lambda c: cycles_recorded.append(c))

        self.engine.start()

        # Wait for 3 cycles to complete (playback_speed=10x makes this very fast)
        start_wait = time.perf_counter()
        while len(cycles_recorded) < 3 and (time.perf_counter() - start_wait) < 15.0:
            time.sleep(0.05)

        self.engine.stop()

        self.assertGreaterEqual(len(cycles_recorded), 2)
        stats = self.engine.stats
        self.assertGreater(stats.total_bases_evaluated, 0)
        self.assertGreater(stats.total_attacks_performed, 0)
        self.assertGreater(stats.total_gold_acquired, 0)
        self.assertGreater(stats.total_elixir_acquired, 0)

        # Verify DB persistence
        scans = self.repo.get_base_scans(limit=50)
        self.assertGreater(len(scans), 0)

    def test_strict_safety_boundary_zero_live_clicks(self):
        """Verifies that attempting to target a live game window raises LiveInputSafetyViolation."""
        live_window = WindowInfo(
            hwnd=99999,
            title="Clash of Clans",
            client_rect=Rect(0, 0, 1920, 1080),
            window_rect=Rect(0, 0, 1920, 1080),
            is_visible=True,
            is_minimized=False,
            is_focused=True,
        )
        self.guard.register_live_window(live_window.hwnd, "Clash of Clans")

        # The SafetyGuard must strictly block dispatching any input to Clash of Clans
        with self.assertRaises(LiveInputSafetyViolation):
            self.guard.verify_input_allowed(live_window.hwnd, "click")

    def test_emergency_stop_sub_50ms_latency_and_lock(self):
        """Verifies emergency stop halts execution immediately and locks state."""
        self.engine.start()
        time.sleep(0.1)
        self.assertTrue(self.engine.is_running)

        t0 = time.perf_counter()
        self.engine.emergency_stop()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Must trip and halt in under 50 milliseconds
        self.assertLess(latency_ms, 50.0)
        self.assertTrue(self.barrier.is_emergency_stopped)
        self.assertEqual(self.engine.current_state, FarmingState.ERROR)

        # Subsequent execution attempts must be rejected
        with self.assertRaises(EmergencyStopTriggered):
            self.barrier.check_emergency_stop()


if __name__ == "__main__":
    unittest.main()
