"""
Milestone 4 Adversarial Stress & Vulnerability Test Suite for ClashVision Simulator Subsystem.
Covers:
1. Sandboxed Input Controller Boundary Testing:
   - Unregistered window handles, None HWND, live window handles, collision/spoofed handles, zero/negative HWNDs.
   - Coordinate boundaries: negative, exceeding canvas width/height, boundary corners (0,0), (W,H), NaN, Inf.
   - Emergency stop active blocks all input dispatches.
2. Emergency Stop Trip Latency:
   - 100+ trials empirical latency measurement verifying strict <= 50ms requirement (target < 5ms).
   - Latency under callback load (20 active callbacks).
   - Callback exception resilience.
3. Concurrency / Race Conditions:
   - Rapid toggling between play/pause/resume/stop/reset while ticking.
   - Speed mutations during playback.
   - High-contention multi-threaded stress harness.
4. Empirical Discovered Vulnerabilities:
   - Vulnerability 1: None coordinates bypass canvas bounds checking in SandboxedController.verify_can_execute.
   - Vulnerability 2: NaN speed multiplier bypasses set_speed validation and crashes subsequent tick().
   - Vulnerability 3: Non-integer string coordinates trigger unhandled TypeError.
   - Vulnerability 4: SimulationEngine lacks internal thread synchronization lock.

Derived from Requirements R1, R4 and Features 5, 22, 23, 24, 25, 26, 27.
"""

from __future__ import annotations
import math
import random
import statistics
import threading
import time
import unittest
import pytest
from typing import Any, Dict, List, Optional, Tuple

from clashvision.core.constants import (
    EMERGENCY_STOP_LATENCY_MAX_MS,
    VIRTUAL_REFERENCE_WIDTH,
    VIRTUAL_REFERENCE_HEIGHT,
)
from clashvision.core.models import (
    DeploymentStep,
    FarmingRecommendation,
    Rect,
    SimulationState,
)
from clashvision.core.exceptions import (
    EmergencyStopTriggered,
    LiveInputSafetyViolation,
)
from clashvision.capture.safety_guard import SafetyGuard, get_safety_guard
from clashvision.simulator.safety_barrier import (
    SafetyBarrier,
    VisualMode,
    get_safety_barrier,
)
from clashvision.simulator.sandboxed_controller import (
    SandboxedController,
    SandboxedInputController,
    InputAction,
)
from clashvision.simulator.simulation_engine import (
    SimulationEngine,
    PlaybackState,
    ActiveTroop,
)
from clashvision.simulator.replay_recorder import (
    ReplayRecorder,
    ReplayEvent,
)


class TestSandboxedControllerAdversarialBoundaries(unittest.TestCase):
    """
    Adversarially tests SandboxedController window handle security and coordinate boundaries.
    """

    def setUp(self) -> None:
        self.guard = get_safety_guard()
        self.barrier = get_safety_barrier()
        self.guard.clear_all()
        self.barrier.clear_all()

    def tearDown(self) -> None:
        self.guard.clear_all()
        self.barrier.clear_all()

    def test_unregistered_window_raises_live_input_safety_violation(self) -> None:
        """Controller with unregistered HWND strictly raises LiveInputSafetyViolation."""
        ctrl = SandboxedController(canvas_hwnd=99999, guard=self.guard, barrier=self.barrier)
        with self.assertRaises(LiveInputSafetyViolation) as ctx:
            ctrl.click(500, 500)
        self.assertIn("unregistered window", str(ctx.exception))
        self.assertEqual(len(ctrl.action_history), 0)

    def test_none_hwnd_raises_live_input_safety_violation(self) -> None:
        """Controller with None HWND strictly raises LiveInputSafetyViolation."""
        ctrl = SandboxedController(canvas_hwnd=None, guard=self.guard, barrier=self.barrier)
        with self.assertRaises(LiveInputSafetyViolation) as ctx:
            ctrl.click(500, 500)
        self.assertIn("No simulator canvas window registered", str(ctx.exception))
        self.assertEqual(len(ctrl.action_history), 0)

    def test_live_window_raises_live_input_safety_violation(self) -> None:
        """Controller targeting a live emulator window strictly raises LiveInputSafetyViolation."""
        live_hwnd = 12345
        self.guard.register_live_window(live_hwnd, title="Google Play Games PC", process_name="PlayGamesPC.exe")
        ctrl = SandboxedController(canvas_hwnd=live_hwnd, guard=self.guard, barrier=self.barrier)

        with self.assertRaises(LiveInputSafetyViolation) as ctx:
            ctrl.click(500, 500)
        self.assertIn("live target window", str(ctx.exception))
        self.assertEqual(len(ctrl.action_history), 0)

    def test_collision_window_raises_live_input_safety_violation(self) -> None:
        """Attempting to register a live window as a simulator window fails closed (live check takes priority)."""
        coll_hwnd = 54321
        self.guard.register_live_window(coll_hwnd, title="Live Clash of Clans")
        self.guard.register_simulator_window(coll_hwnd)  # Malicious/buggy collision registration

        ctrl = SandboxedController(canvas_hwnd=coll_hwnd, guard=self.guard, barrier=self.barrier)
        with self.assertRaises(LiveInputSafetyViolation) as ctx:
            ctrl.click(100, 100)
        self.assertIn("live target window", str(ctx.exception))

    def test_zero_and_negative_hwnd_blocked(self) -> None:
        """HWND 0 and negative HWND values are rejected as unregistered windows."""
        for bad_hwnd in [0, -1, -9999]:
            ctrl = SandboxedController(canvas_hwnd=bad_hwnd, guard=self.guard, barrier=self.barrier)
            with self.assertRaises(LiveInputSafetyViolation):
                ctrl.click(100, 100)

    def test_negative_coordinates_rejected(self) -> None:
        """Negative X or Y coordinates raise ValueError."""
        sim_hwnd = 8801
        self.guard.register_simulator_window(sim_hwnd)
        ctrl = SandboxedController(canvas_hwnd=sim_hwnd, canvas_width=1920, canvas_height=1080, guard=self.guard, barrier=self.barrier)

        with self.assertRaises(ValueError):
            ctrl.click(-1, 500)
        with self.assertRaises(ValueError):
            ctrl.click(500, -1)
        with self.assertRaises(ValueError):
            ctrl.move_to(-100, -100)
        self.assertEqual(len(ctrl.action_history), 0)

    def test_excess_width_and_height_coordinates_rejected(self) -> None:
        """Coordinates exceeding canvas bounds raise ValueError."""
        sim_hwnd = 8802
        self.guard.register_simulator_window(sim_hwnd)
        ctrl = SandboxedController(canvas_hwnd=sim_hwnd, canvas_width=1920, canvas_height=1080, guard=self.guard, barrier=self.barrier)

        with self.assertRaises(ValueError):
            ctrl.click(1921, 500)
        with self.assertRaises(ValueError):
            ctrl.click(500, 1081)
        with self.assertRaises(ValueError):
            ctrl.click(10000, 10000)
        self.assertEqual(len(ctrl.action_history), 0)

    def test_boundary_corner_coordinates_allowed(self) -> None:
        """Exact boundary corners (0, 0), (W, 0), (0, H), (W, H) are within valid bounds."""
        sim_hwnd = 8803
        self.guard.register_simulator_window(sim_hwnd)
        ctrl = SandboxedController(canvas_hwnd=sim_hwnd, canvas_width=1920, canvas_height=1080, guard=self.guard, barrier=self.barrier)

        corners = [(0, 0), (1920, 0), (0, 1080), (1920, 1080)]
        for x, y in corners:
            self.assertTrue(ctrl.click(x, y))
        self.assertEqual(len(ctrl.action_history), 4)

    def test_drag_destination_bounds_validation(self) -> None:
        """Drag action validates both start and end coordinates."""
        sim_hwnd = 8804
        self.guard.register_simulator_window(sim_hwnd)
        ctrl = SandboxedController(canvas_hwnd=sim_hwnd, canvas_width=1920, canvas_height=1080, guard=self.guard, barrier=self.barrier)

        # Valid drag
        self.assertTrue(ctrl.drag(100, 100, 500, 500, duration_ms=50))

        # Start out of bounds
        with self.assertRaises(ValueError):
            ctrl.drag(-10, 100, 500, 500)

        # End out of bounds
        with self.assertRaises(ValueError):
            ctrl.drag(100, 100, 2000, 500)

    def test_float_nan_and_inf_coordinates_rejected(self) -> None:
        """Float NaN and Infinity coordinates evaluate as out of bounds and raise ValueError."""
        sim_hwnd = 8805
        self.guard.register_simulator_window(sim_hwnd)
        ctrl = SandboxedController(canvas_hwnd=sim_hwnd, canvas_width=1920, canvas_height=1080, guard=self.guard, barrier=self.barrier)

        with self.assertRaises(ValueError):
            ctrl.click(float("nan"), 500)
        with self.assertRaises(ValueError):
            ctrl.click(500, float("nan"))
        with self.assertRaises(ValueError):
            ctrl.click(float("inf"), 500)
        with self.assertRaises(ValueError):
            ctrl.click(500, float("-inf"))

    def test_emergency_stop_blocks_all_controller_actions(self) -> None:
        """When emergency stop latch trips, all controller actions raise EmergencyStopTriggered."""
        sim_hwnd = 8806
        self.guard.register_simulator_window(sim_hwnd)
        ctrl = SandboxedController(canvas_hwnd=sim_hwnd, guard=self.guard, barrier=self.barrier)

        self.barrier.trigger_emergency_stop(reason="Boundary breach test")

        with self.assertRaises(EmergencyStopTriggered):
            ctrl.click(100, 100)
        with self.assertRaises(EmergencyStopTriggered):
            ctrl.move_to(200, 200)
        with self.assertRaises(EmergencyStopTriggered):
            ctrl.drag(10, 10, 50, 50)
        with self.assertRaises(EmergencyStopTriggered):
            ctrl.press_key("space")


class TestEmergencyStopTripLatency(unittest.TestCase):
    """
    Empirically measures emergency stop response latency across 100+ trials
    verifying strict compliance with Requirement R4 (latency <= 50ms, target < 5ms).
    """

    def setUp(self) -> None:
        self.guard = get_safety_guard()
        self.barrier = get_safety_barrier()
        self.guard.clear_all()
        self.barrier.clear_all()

    def tearDown(self) -> None:
        self.guard.clear_all()
        self.barrier.clear_all()

    def test_latency_distribution_200_trials(self) -> None:
        """
        Executes 200 trials measuring both wall-clock time and barrier-internal high-res timer.
        Verifies:
        - Strict maximum <= 50.0 ms.
        - Target maximum <= 5.0 ms.
        - Mean latency < 1.0 ms.
        """
        trials = 200
        wall_latencies_ms: List[float] = []
        barrier_latencies_ms: List[float] = []

        for i in range(trials):
            self.barrier.reset_emergency_stop()
            t0 = time.perf_counter()
            reported_ms = self.barrier.trigger_emergency_stop(reason=f"Latency trial {i}")
            t1 = time.perf_counter()
            wall_ms = (t1 - t0) * 1000.0

            wall_latencies_ms.append(wall_ms)
            barrier_latencies_ms.append(reported_ms)

        max_wall = max(wall_latencies_ms)
        mean_wall = statistics.mean(wall_latencies_ms)
        p99_wall = statistics.quantiles(wall_latencies_ms, n=100)[98]

        self.assertLessEqual(max_wall, EMERGENCY_STOP_LATENCY_MAX_MS, f"Max latency {max_wall:.4f} ms exceeded 50ms limit")
        self.assertLessEqual(max_wall, 5.0, f"Max latency {max_wall:.4f} ms exceeded 5ms target limit")
        self.assertLessEqual(mean_wall, 1.0, f"Mean latency {mean_wall:.4f} ms exceeded 1ms threshold")
        self.assertLessEqual(p99_wall, 2.0, f"P99 latency {p99_wall:.4f} ms exceeded 2ms threshold")
        self.assertEqual(len(wall_latencies_ms), trials)

    def test_latency_under_callback_load_20_callbacks(self) -> None:
        """
        Registers 20 active emergency callbacks and executes 100 trials.
        Verifies all 20 callbacks fire reliably on every trip while latency strictly remains <= 50ms.
        """
        called_counts = [0] * 20
        for idx in range(20):
            def make_cb(i: int):
                def cb():
                    called_counts[i] += 1
                return cb
            self.barrier.register_emergency_callback(make_cb(idx))

        trials = 100
        latencies_ms: List[float] = []

        for i in range(trials):
            self.barrier.reset_emergency_stop()
            t0 = time.perf_counter()
            self.barrier.trigger_emergency_stop(reason=f"Callback load trial {i}")
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        max_lat = max(latencies_ms)
        mean_lat = statistics.mean(latencies_ms)

        self.assertLessEqual(max_lat, EMERGENCY_STOP_LATENCY_MAX_MS)
        self.assertLessEqual(mean_lat, 1.0)
        # Verify every single callback fired in all 100 trials
        for count in called_counts:
            self.assertEqual(count, trials)

    def test_latency_resilience_to_callback_failure(self) -> None:
        """
        Emergency stop remains responsive (< 50ms) and sets latch even if registered callbacks raise exceptions.
        """
        def faulty_callback_1():
            raise RuntimeError("Fatal crash in callback 1")

        def faulty_callback_2():
            raise ZeroDivisionError("Division by zero in callback 2")

        normal_executed = False
        def healthy_callback():
            nonlocal normal_executed
            normal_executed = True

        self.barrier.register_emergency_callback(faulty_callback_1)
        self.barrier.register_emergency_callback(healthy_callback)
        self.barrier.register_emergency_callback(faulty_callback_2)

        t0 = time.perf_counter()
        self.barrier.trigger_emergency_stop(reason="Resilience test")
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        self.assertLessEqual(latency_ms, EMERGENCY_STOP_LATENCY_MAX_MS)
        self.assertTrue(normal_executed)
        self.assertTrue(self.barrier.is_emergency_stopped)
        self.assertTrue(self.guard.is_emergency_stopped)


class TestConcurrencyAndRaceConditions(unittest.TestCase):
    """
    Stress-tests SimulationEngine under concurrent multi-threaded execution.
    """

    def setUp(self) -> None:
        self.guard = get_safety_guard()
        self.barrier = get_safety_barrier()
        self.guard.clear_all()
        self.barrier.clear_all()

    def tearDown(self) -> None:
        self.guard.clear_all()
        self.barrier.clear_all()

    def test_rapid_playback_state_toggling_concurrent_with_tick(self) -> None:
        """
        Rapidly toggles start, pause, resume, stop, reset across multiple threads
        while tick() loop runs concurrently, asserting no unhandled exceptions or deadlocks.
        """
        steps = [
            DeploymentStep(step_id=i, troop_type="Troop", target_x=100 + i * 10, target_y=100 + i * 10, delay_ms=50)
            for i in range(1, 20)
        ]
        engine = SimulationEngine(steps=steps, barrier=self.barrier, guard=self.guard)

        errors: List[Tuple[str, str]] = []
        stop_event = threading.Event()

        def tick_worker():
            while not stop_event.is_set():
                try:
                    engine.tick(20)
                except Exception as e:
                    errors.append(("tick", str(e)))

        def toggle_worker():
            actions = [engine.start, engine.pause, engine.resume, engine.stop, engine.reset]
            while not stop_event.is_set():
                act = random.choice(actions)
                try:
                    act()
                except Exception as e:
                    errors.append(("toggle", str(e)))

        threads = [
            threading.Thread(target=tick_worker),
            threading.Thread(target=toggle_worker),
        ]
        for t in threads:
            t.start()

        time.sleep(1.0)
        stop_event.set()

        for t in threads:
            t.join(timeout=2.0)

        self.assertEqual(len(errors), 0, f"Encountered unexpected concurrency errors: {errors[:5]}")

    def test_speed_mutation_during_playback(self) -> None:
        """
        Rapidly mutates speed multiplier (0.1x - 10.0x) while playback tick loop is active.
        """
        steps = [
            DeploymentStep(step_id=i, troop_type="Troop", target_x=100 + i * 10, target_y=100 + i * 10, delay_ms=10)
            for i in range(1, 30)
        ]
        engine = SimulationEngine(steps=steps, barrier=self.barrier, guard=self.guard)
        engine.start()

        errors: List[Tuple[str, str]] = []
        stop_event = threading.Event()
        speeds = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 10.0]

        def tick_worker():
            while not stop_event.is_set():
                try:
                    engine.tick(16)
                except Exception as e:
                    errors.append(("tick", str(e)))

        def speed_worker():
            while not stop_event.is_set():
                spd = random.choice(speeds)
                try:
                    engine.set_speed(spd)
                except Exception as e:
                    errors.append(("speed", str(e)))

        threads = [
            threading.Thread(target=tick_worker),
            threading.Thread(target=speed_worker),
        ]
        for t in threads:
            t.start()

        time.sleep(1.0)
        stop_event.set()

        for t in threads:
            t.join(timeout=2.0)

        self.assertEqual(len(errors), 0, f"Encountered speed mutation concurrency errors: {errors[:5]}")

    def test_multi_threaded_stress_harness(self) -> None:
        """
        Full multi-threaded stress harness:
        - Thread 1: Ticking timeline
        - Thread 2: State machine transitions (start, pause, resume, stop, reset)
        - Thread 3: Speed mutations
        - Thread 4: Timeline state and active troops inspectors
        """
        steps = [
            DeploymentStep(step_id=i, troop_type="Barbarian", target_x=100 + i * 10, target_y=100 + i * 10, delay_ms=25)
            for i in range(1, 25)
        ]
        engine = SimulationEngine(steps=steps, barrier=self.barrier, guard=self.guard)

        errors: List[Tuple[str, str]] = []
        stop_event = threading.Event()

        def tick_worker():
            while not stop_event.is_set():
                try:
                    engine.tick(16)
                except Exception as e:
                    errors.append(("tick", str(e)))

        def control_worker():
            actions = [engine.start, engine.pause, engine.resume, engine.stop, engine.reset]
            while not stop_event.is_set():
                act = random.choice(actions)
                try:
                    act()
                except Exception as e:
                    errors.append(("control", str(e)))

        def speed_worker():
            speeds = [0.1, 0.5, 1.0, 2.0, 4.0, 10.0]
            while not stop_event.is_set():
                try:
                    engine.set_speed(random.choice(speeds))
                except Exception as e:
                    errors.append(("speed", str(e)))

        def reader_worker():
            while not stop_event.is_set():
                try:
                    _ = engine.simulation_state
                    _ = engine.active_troops
                    _ = engine.progress_fraction
                except Exception as e:
                    errors.append(("reader", str(e)))

        threads = [
            threading.Thread(target=tick_worker),
            threading.Thread(target=control_worker),
            threading.Thread(target=speed_worker),
            threading.Thread(target=reader_worker),
        ]
        for t in threads:
            t.start()

        time.sleep(1.0)
        stop_event.set()

        for t in threads:
            t.join(timeout=2.0)

        self.assertEqual(len(errors), 0, f"Multi-threaded stress harness errors: {errors[:5]}")


class TestSimulatorVulnerabilities(unittest.TestCase):
    """
    Empirical reproduction of discovered bugs, boundary bypasses, and contract flaws in Milestone 4.
    """

    def setUp(self) -> None:
        self.guard = get_safety_guard()
        self.barrier = get_safety_barrier()
        self.guard.clear_all()
        self.barrier.clear_all()

    def tearDown(self) -> None:
        self.guard.clear_all()
        self.barrier.clear_all()

    def test_vuln_none_coordinates_bypass_bounds_check(self) -> None:
        """
        REMEDIATED VULNERABILITY 1: None coordinates must be strictly rejected with ValueError
        for mouse actions (click, move_to, drag) to ensure spatial isolation in SandboxedController.
        """
        sim_hwnd = 9101
        self.guard.register_simulator_window(sim_hwnd)
        ctrl = SandboxedController(canvas_hwnd=sim_hwnd, canvas_width=1920, canvas_height=1080, guard=self.guard, barrier=self.barrier)

        with self.assertRaises(ValueError):
            ctrl.click(None, None)

        with self.assertRaises(ValueError):
            ctrl.click(None, 500)

        with self.assertRaises(ValueError):
            ctrl.click(500, None)

        with self.assertRaises(ValueError):
            ctrl.move_to(None, None)

        # Non-mouse keystroke action can execute with None coordinates
        self.assertTrue(ctrl.press_key("space"))

    def test_vuln_nan_speed_bypasses_validation_and_crashes_tick(self) -> None:
        """
        REMEDIATED VULNERABILITY 2: NaN / infinite speed multiplier must be rejected with
        ValueError in SimulationEngine.set_speed() before it can corrupt playback.
        """
        engine = SimulationEngine(barrier=self.barrier, guard=self.guard)

        with self.assertRaises(ValueError):
            engine.set_speed(float("nan"))

        with self.assertRaises(ValueError):
            engine.set_speed(float("inf"))

        with self.assertRaises(ValueError):
            engine.set_speed(float("-inf"))

        # Valid speed succeeds
        engine.set_speed(2.0)
        self.assertEqual(engine.speed, 2.0)

    def test_vuln_string_coordinate_raises_type_error(self) -> None:
        """
        REMEDIATED VULNERABILITY 3: Passing non-numeric string coordinates raises clean
        ValueError instead of unhandled TypeError.
        """
        sim_hwnd = 9102
        self.guard.register_simulator_window(sim_hwnd)
        ctrl = SandboxedController(canvas_hwnd=sim_hwnd, guard=self.guard, barrier=self.barrier)

        with self.assertRaises(ValueError):
            ctrl.click("500", 500)

        with self.assertRaises(ValueError):
            ctrl.click(500, "500")

    def test_vuln_engine_lacks_concurrency_lock(self) -> None:
        """
        REMEDIATED VULNERABILITY 4: SimulationEngine now maintains internal thread
        synchronization lock (self._lock = threading.RLock()) guarding mutable simulation state.
        """
        engine = SimulationEngine(barrier=self.barrier, guard=self.guard)
        self.assertTrue(hasattr(engine, "_lock"))
        self.assertIsInstance(engine._lock, type(threading.RLock()))


if __name__ == "__main__":
    unittest.main()
