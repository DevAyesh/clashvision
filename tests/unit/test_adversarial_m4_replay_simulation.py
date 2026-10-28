"""
Milestone 4 Adversarial Stress & Vulnerability Test Suite for ClashVision Simulator Subsystem.
Focus:
- Replay recorder & timeline scrubbing: extreme scrub targets (negative, exceeding, float, malformed JSON, fidelity).
- Troop timeline & lifetime simulation: empty plans, 10,000+ rapid steps, overlapping timestamps, zero-delay steps, marker cleanup.
- State machine mutation resilience: invalid speed multipliers (< 0.1, > 10.0, NaN, Inf), step forward when stopped, step backward at step 0.

Derived from Requirements R3, R4, Features 22-26, and Milestone 4 Dispatch Instructions.
"""

from __future__ import annotations
import json
import math
import os
import tempfile
import unittest
import pytest
from typing import Any, Dict, List, Optional, Tuple

from clashvision.core.constants import (
    EMERGENCY_STOP_LATENCY_MAX_MS,
    StrategyType,
    VIRTUAL_REFERENCE_WIDTH,
    VIRTUAL_REFERENCE_HEIGHT,
)
from clashvision.core.models import (
    DeploymentStep,
    FarmingRecommendation,
    SimulationState,
)
from clashvision.core.exceptions import (
    EmergencyStopTriggered,
    LiveInputSafetyViolation,
)
from clashvision.simulator.simulation_engine import (
    ActiveTroop,
    PlaybackState,
    SimulationEngine,
)
from clashvision.simulator.replay_recorder import (
    ReplayEvent,
    ReplayRecorder,
)


class TestSimulatorRobustnessPassing(unittest.TestCase):
    """
    Verifies that the simulator and replay subsystems maintain invariants and contracts
    under boundary inputs, stress loads, and rapid state mutations.
    """

    def setUp(self) -> None:
        self.sample_steps = [
            DeploymentStep(step_id=1, troop_type="Sneaky Goblin", target_x=200, target_y=300, delay_ms=0, unit_count=5),
            DeploymentStep(step_id=2, troop_type="Sneaky Goblin", target_x=250, target_y=350, delay_ms=500, unit_count=5),
            DeploymentStep(step_id=3, troop_type="Jump Spell", target_x=400, target_y=400, delay_ms=1000, unit_count=1),
            DeploymentStep(step_id=4, troop_type="Sneaky Goblin", target_x=600, target_y=600, delay_ms=500, unit_count=10),
        ]

    # --------------------------------------------------------------------------
    # Replay Recorder & Timeline Scrubbing (Focus 1)
    # --------------------------------------------------------------------------
    def test_timeline_scrub_negative_time(self) -> None:
        """Negative scrub target (e.g. -500ms) returns 0 highest step and empty event list."""
        rec = ReplayRecorder()
        rec.record_step(self.sample_steps[0], timestamp_ms=0)
        rec.record_step(self.sample_steps[1], timestamp_ms=500)

        step_idx, events = rec.seek_to_time(-500)
        self.assertEqual(step_idx, 0)
        self.assertEqual(len(events), 0)

        step_idx, events = rec.seek_to_time(-1)
        self.assertEqual(step_idx, 0)
        self.assertEqual(len(events), 0)

    def test_timeline_scrub_float_time(self) -> None:
        """Float scrub target (e.g. 500.5ms) correctly matches events up to timestamp."""
        rec = ReplayRecorder()
        rec.record_step(self.sample_steps[0], timestamp_ms=0)
        rec.record_step(self.sample_steps[1], timestamp_ms=500)
        rec.record_step(self.sample_steps[2], timestamp_ms=1500)

        step_idx, events = rec.seek_to_time(500.5)
        self.assertEqual(step_idx, 2)
        self.assertEqual(len(events), 2)

        step_idx, events = rec.seek_to_time(0.0)
        self.assertEqual(step_idx, 1)
        self.assertEqual(len(events), 1)

    def test_timeline_scrub_exceeding_duration(self) -> None:
        """Scrub target far exceeding total duration (e.g. 100,000,000ms) returns all events."""
        rec = ReplayRecorder()
        for i, step in enumerate(self.sample_steps):
            rec.record_step(step, timestamp_ms=i * 500)

        step_idx, events = rec.seek_to_time(100_000_000)
        self.assertEqual(step_idx, len(self.sample_steps))
        self.assertEqual(len(events), len(self.sample_steps))

    def test_timeline_scrub_nan_time(self) -> None:
        """Float NaN scrub target returns (0, []) safely without raising exceptions."""
        rec = ReplayRecorder()
        rec.record_step(self.sample_steps[0], timestamp_ms=0)
        rec.record_step(self.sample_steps[1], timestamp_ms=500)

        step_idx, events = rec.seek_to_time(float("nan"))
        self.assertEqual(step_idx, 0)
        self.assertEqual(len(events), 0)

    def test_timeline_scrub_inf_time(self) -> None:
        """Float Inf scrub target returns all recorded events safely."""
        rec = ReplayRecorder()
        rec.record_step(self.sample_steps[0], timestamp_ms=0)
        rec.record_step(self.sample_steps[1], timestamp_ms=500)

        step_idx, events = rec.seek_to_time(float("inf"))
        self.assertEqual(step_idx, 2)
        self.assertEqual(len(events), 2)

    def test_step_scrub_negative_step(self) -> None:
        """Negative target step index returns timestamp 0 and empty event list."""
        rec = ReplayRecorder()
        rec.record_step(self.sample_steps[0], timestamp_ms=0)
        rec.record_step(self.sample_steps[1], timestamp_ms=500)

        ts, events = rec.seek_to_step(-10)
        self.assertEqual(ts, 0)
        self.assertEqual(len(events), 0)

    def test_step_scrub_nominal_steps(self) -> None:
        """Step scrubbing returns matching events and step execution timestamp."""
        rec = ReplayRecorder()
        rec.record_step(self.sample_steps[0], timestamp_ms=0)
        rec.record_step(self.sample_steps[1], timestamp_ms=500)
        rec.record_step(self.sample_steps[2], timestamp_ms=1500)

        ts, events = rec.seek_to_step(1)
        self.assertEqual(ts, 0)
        self.assertEqual(len(events), 1)

        ts, events = rec.seek_to_step(2)
        self.assertEqual(ts, 500)
        self.assertEqual(len(events), 2)

    def test_replay_event_filtering_and_clearing(self) -> None:
        """ReplayRecorder correctly filters events by event_type and clears buffer."""
        rec = ReplayRecorder()
        rec.record_step(self.sample_steps[0], timestamp_ms=0)
        rec.record_action("CLICK", 100, 200, timestamp_ms=10)
        rec.record_emergency_stop(timestamp_ms=20, reason="Test interrupt")

        self.assertEqual(len(rec.get_events("STEP")), 1)
        self.assertEqual(len(rec.get_events("ACTION")), 1)
        self.assertEqual(len(rec.get_events("EMERGENCY_STOP")), 1)
        self.assertEqual(len(rec.get_events("NON_EXISTENT")), 0)

        rec.clear()
        self.assertEqual(rec.event_count, 0)
        self.assertEqual(len(rec.events), 0)

    def test_replay_roundtrip_serialization_fidelity(self) -> None:
        """Replay export and import maintains full event and metadata fidelity."""
        rec = ReplayRecorder(metadata={"base_name": "WarBase_1", "author": "Tester"})
        rec.record_step(self.sample_steps[0], timestamp_ms=0, troop_marker={"unit": "goblin"})
        rec.record_state_change("IDLE", "PLAYING", timestamp_ms=0)
        rec.record_action("CLICK", 200, 300, timestamp_ms=10, button="left")
        rec.record_emergency_stop(timestamp_ms=50, reason="Safety trip")

        exported = rec.export_replay()
        self.assertIn("events", exported)
        self.assertEqual(exported["total_events"], 4)
        self.assertEqual(exported["step_count"], 1)
        self.assertEqual(exported["metadata"]["base_name"], "WarBase_1")

        rec2 = ReplayRecorder()
        rec2.import_replay(exported)
        self.assertEqual(rec2.event_count, 4)
        self.assertEqual(rec2.metadata["base_name"], "WarBase_1")
        self.assertEqual(rec2.events[0].event_type, "STEP")
        self.assertEqual(rec2.events[0].data["step_id"], 1)
        self.assertEqual(rec2.events[1].event_type, "STATE_CHANGE")
        self.assertEqual(rec2.events[2].event_type, "ACTION")
        self.assertEqual(rec2.events[3].event_type, "EMERGENCY_STOP")

    def test_replay_file_save_and_load(self) -> None:
        """ReplayRecorder saves to JSON file and loads back with perfect fidelity."""
        rec = ReplayRecorder({"scenario": "adversarial"})
        rec.record_step(self.sample_steps[0], timestamp_ms=0)
        rec.record_step(self.sample_steps[1], timestamp_ms=500)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "replay.json")
            rec.save_to_file(file_path)
            self.assertTrue(os.path.exists(file_path))

            rec_loaded = ReplayRecorder()
            rec_loaded.load_from_file(file_path)
            self.assertEqual(rec_loaded.event_count, 2)
            self.assertEqual(rec_loaded.metadata["scenario"], "adversarial")
            self.assertEqual(rec_loaded.events[1].timestamp_ms, 500)

    def test_replay_extra_fields_ignored_in_from_dict(self) -> None:
        """ReplayEvent.from_dict gracefully ignores unknown extra JSON fields."""
        raw_event = {
            "timestamp_ms": 1234,
            "event_type": "STEP",
            "step_index": 5,
            "data": {"unit": "P.E.K.K.A"},
            "unknown_extra_field_1": "corrupt",
            "unknown_extra_field_2": 99999,
        }
        event = ReplayEvent.from_dict(raw_event)
        self.assertEqual(event.timestamp_ms, 1234)
        self.assertEqual(event.event_type, "STEP")
        self.assertEqual(event.step_index, 5)
        self.assertEqual(event.data["unit"], "P.E.K.K.A")

    # --------------------------------------------------------------------------
    # Troop Timeline & Lifetime Simulation (Focus 2)
    # --------------------------------------------------------------------------
    def test_empty_deployment_plan_step_forward_behavior(self) -> None:
        """Calling step_forward() on an empty plan immediately marks COMPLETED and returns None."""
        eng = SimulationEngine(steps=[])
        eng.start()
        res = eng.step_forward()
        self.assertIsNone(res)
        self.assertEqual(eng.state, PlaybackState.COMPLETED)

    def test_ten_thousand_rapid_steps_scaling(self) -> None:
        """10,000 rapid steps load, tick, deploy, and complete without crashing."""
        large_steps = [
            DeploymentStep(
                step_id=i,
                troop_type="Barbarian",
                target_x=i % 1000,
                target_y=i % 800,
                delay_ms=1,
            )
            for i in range(10_000)
        ]
        eng = SimulationEngine(steps=large_steps)
        self.assertEqual(eng.total_duration_ms, 15_000)
        self.assertEqual(eng.total_steps, 10_000)

        eng.start()
        # Tick in 1000ms increments
        for _ in range(15):
            eng.tick(1000)

        self.assertEqual(eng.state, PlaybackState.COMPLETED)
        self.assertEqual(eng.step_index, 10_000)
        self.assertEqual(len(eng.all_troops), 10_000)

    def test_zero_delay_simultaneous_steps(self) -> None:
        """Multiple deployment steps with delay_ms=0 deploy concurrently at timestamp 0."""
        zero_steps = [
            DeploymentStep(step_id=1, troop_type="Wall Breaker", target_x=100, target_y=100, delay_ms=0),
            DeploymentStep(step_id=2, troop_type="Sneaky Goblin", target_x=200, target_y=200, delay_ms=0),
            DeploymentStep(step_id=3, troop_type="Sneaky Goblin", target_x=300, target_y=300, delay_ms=0),
        ]
        eng = SimulationEngine(steps=zero_steps)
        eng.start()
        eng.tick(0)

        self.assertEqual(eng.step_index, 3)
        self.assertEqual(len(eng.active_troops), 3)
        for t in eng.active_troops:
            self.assertEqual(t["spawn_time_ms"], 0)

    def test_overlapping_timestamps(self) -> None:
        """Steps scheduled at identical cumulative timestamps deploy together without step loss."""
        overlap_steps = [
            DeploymentStep(step_id=1, troop_type="Golem", target_x=100, target_y=100, delay_ms=500),
            DeploymentStep(step_id=2, troop_type="Wizard 1", target_x=120, target_y=120, delay_ms=0),
            DeploymentStep(step_id=3, troop_type="Wizard 2", target_x=140, target_y=140, delay_ms=0),
            DeploymentStep(step_id=4, troop_type="P.E.K.K.A", target_x=200, target_y=200, delay_ms=500),
        ]
        eng = SimulationEngine(steps=overlap_steps)
        eng.start()

        eng.tick(500)
        # Steps 1, 2, 3 should all be deployed at 500ms
        self.assertEqual(eng.step_index, 3)
        self.assertEqual(len(eng.active_troops), 3)

        eng.tick(500)
        # Step 4 should now be deployed at 1000ms
        self.assertEqual(eng.step_index, 4)
        self.assertEqual(len(eng.active_troops), 4)

    def test_active_troop_ttl_expiration_nominal(self) -> None:
        """Active troop marker expires and transitions to EXPIRED when TTL is reached."""
        eng = SimulationEngine(default_troop_ttl_ms=3000)
        eng.load_steps([DeploymentStep(step_id=1, troop_type="Goblin", delay_ms=0)], total_duration_ms=6000)
        eng.start()
        eng.tick(0)
        self.assertEqual(len(eng.active_troops), 1)
        self.assertEqual(eng.all_troops[0]["status"], "ACTIVE")

        eng.tick(1500)
        self.assertEqual(len(eng.active_troops), 1)
        self.assertEqual(eng.all_troops[0]["remaining_lifetime_ms"], 1500)

        eng.tick(1600)
        self.assertEqual(len(eng.active_troops), 0)
        self.assertEqual(eng.all_troops[0]["status"], "EXPIRED")
        self.assertEqual(eng.all_troops[0]["remaining_lifetime_ms"], 0)

    # --------------------------------------------------------------------------
    # State Machine Mutation Resilience (Focus 3)
    # --------------------------------------------------------------------------
    def test_speed_multiplier_valid_range(self) -> None:
        """Valid playback speeds (0.1x to 10.0x) correctly scale timeline tick delta."""
        eng = SimulationEngine()
        eng.load_steps(self.sample_steps, total_duration_ms=10000)
        eng.start()

        # 0.5x speed
        eng.set_speed(0.5)
        self.assertEqual(eng.speed, 0.5)
        eng.tick(1000)
        self.assertEqual(eng.current_time_ms, 500)

        # 2.0x speed
        eng.set_speed(2.0)
        self.assertEqual(eng.speed, 2.0)
        eng.tick(1000)
        self.assertEqual(eng.current_time_ms, 2500)

        # Maximum boundary (10.0x)
        eng.set_speed(10.0)
        self.assertEqual(eng.speed, 10.0)

        # Minimum boundary (0.1x)
        eng.set_speed(0.1)
        self.assertEqual(eng.speed, 0.1)

    def test_speed_multiplier_out_of_bounds_rejected(self) -> None:
        """Speeds < 0.1, > 10.0, and Infinite values raise ValueError."""
        eng = SimulationEngine()
        invalid_speeds = [0.09, 0.0, -1.0, -100.0, 10.01, 50.0, float("inf"), float("-inf")]
        for s in invalid_speeds:
            with self.assertRaises(ValueError):
                eng.set_speed(s)

    def test_step_forward_when_stopped_idle_paused(self) -> None:
        """step_forward() executes next step while preserving STOPPED, IDLE, or PAUSED state."""
        eng = SimulationEngine(steps=self.sample_steps)

        # Step forward while IDLE
        self.assertEqual(eng.state, PlaybackState.IDLE)
        step = eng.step_forward()
        self.assertIsNotNone(step)
        self.assertEqual(step.step_id, 1)
        self.assertEqual(eng.state, PlaybackState.IDLE)

        # Pause and step forward
        eng.start()
        eng.pause()
        self.assertEqual(eng.state, PlaybackState.PAUSED)
        step = eng.step_forward()
        self.assertIsNotNone(step)
        self.assertEqual(step.step_id, 2)
        self.assertEqual(eng.state, PlaybackState.PAUSED)

        # Stop and step forward
        eng.stop()
        self.assertEqual(eng.state, PlaybackState.STOPPED)
        step = eng.step_forward()
        self.assertIsNotNone(step)
        self.assertEqual(step.step_id, 3)
        self.assertEqual(eng.state, PlaybackState.STOPPED)

    def test_step_forward_past_end(self) -> None:
        """Stepping forward past the end transitions to COMPLETED and repeatedly returns None."""
        eng = SimulationEngine(steps=[self.sample_steps[0]])
        eng.step_forward()
        res = eng.step_forward()
        self.assertIsNone(res)
        self.assertEqual(eng.state, PlaybackState.COMPLETED)

        # Calling 10 more times safely returns None
        for _ in range(10):
            self.assertIsNone(eng.step_forward())
            self.assertEqual(eng.state, PlaybackState.COMPLETED)

    def test_step_backward_at_step_zero(self) -> None:
        """Calling step_backward() at step 0 is a safe no-op across all states."""
        eng = SimulationEngine(steps=self.sample_steps)
        self.assertEqual(eng.step_index, 0)

        for _ in range(50):
            eng.step_backward()
            self.assertEqual(eng.step_index, 0)
            self.assertEqual(eng.current_time_ms, 0)

    def test_step_backward_rewind_reconstructs_troops(self) -> None:
        """step_backward() decrements step index, rewinds time, and reconstructs deployed troops."""
        eng = SimulationEngine()
        eng.load_steps(self.sample_steps, total_duration_ms=5000)
        eng.start()
        eng.step_forward() # step 1
        eng.step_forward() # step 2
        eng.step_forward() # step 3

        self.assertEqual(eng.step_index, 3)
        self.assertEqual(len(eng.all_troops), 3)

        eng.step_backward()
        self.assertEqual(eng.step_index, 2)
        self.assertEqual(len(eng.all_troops), 2)
        self.assertEqual(eng.current_time_ms, 500)

        eng.step_backward()
        self.assertEqual(eng.step_index, 1)
        self.assertEqual(len(eng.all_troops), 1)
        self.assertEqual(eng.current_time_ms, 0)

    def test_negative_tick_delta_clamped_to_zero(self) -> None:
        """Ticking with negative delta_ms (e.g. -500) clamps to 0 and does not rewind time or crash."""
        eng = SimulationEngine()
        eng.load_steps(self.sample_steps, total_duration_ms=5000)
        eng.start()
        eng.tick(1000)
        self.assertEqual(eng.current_time_ms, 1000)

        eng.tick(-500)
        self.assertEqual(eng.current_time_ms, 1000)

    def test_callback_dispatch_and_error_isolation(self) -> None:
        """Callbacks fire on tick/step/complete/state_change and exceptions inside callbacks are isolated."""
        eng = SimulationEngine()
        eng.load_steps([self.sample_steps[0]], total_duration_ms=100)
        ticks: list[int] = []
        steps_fired: list[int] = []
        states_fired: list[str] = []

        eng.register_on_tick(lambda s: ticks.append(s.current_time_ms))
        eng.register_on_step(lambda step, marker: steps_fired.append(step.step_id))
        eng.register_on_state_change(lambda old_s, new_s: states_fired.append(f"{old_s.value}->{new_s.value}"))

        # Register a malicious / buggy callback that raises an unhandled exception
        def failing_callback(*args: Any) -> None:
            raise RuntimeError("Malicious callback failure")

        eng.register_on_tick(failing_callback)
        eng.register_on_step(failing_callback)
        eng.register_on_state_change(failing_callback)

        eng.start()
        eng.tick(100)

        # Simulation continues operating despite callback exceptions
        self.assertGreater(len(ticks), 0)
        self.assertIn(1, steps_fired)
        self.assertIn("IDLE->PLAYING", states_fired)

    def test_rapid_state_churn_stress(self) -> None:
        """500+ iterations of alternating playback state switches maintain consistent state machine."""
        eng = SimulationEngine(steps=self.sample_steps)
        for _ in range(500):
            eng.start()
            eng.pause()
            eng.resume()
            eng.step_forward()
            eng.step_backward()
            eng.stop()
            eng.reset()

        self.assertEqual(eng.state, PlaybackState.IDLE)
        self.assertEqual(eng.step_index, 0)
        self.assertEqual(eng.current_time_ms, 0)


class TestSimulatorVulnerabilitiesDiscovered(unittest.TestCase):
    """
    Verification of remediated vulnerabilities and contract integrity in Milestone 4 (`clashvision/simulator/`).
    """

    def test_vuln_1_nan_speed_multiplier_crashes_tick(self) -> None:
        """
        REMEDIATED DEFECT 1: Float NaN speed multiplier is rejected immediately by
        set_speed() with ValueError, preventing playback corruption and thread crashes.
        """
        eng = SimulationEngine(steps=[DeploymentStep(step_id=1, troop_type="Barb", delay_ms=100)])
        eng.start()

        nan = float("nan")
        with self.assertRaises(ValueError) as ctx:
            eng.set_speed(nan)
        self.assertIn("finite number", str(ctx.exception))

        # Speed remains clean and valid; tick executes without crash
        self.assertEqual(eng.speed, 1.0)
        state = eng.tick(100)
        self.assertEqual(state.current_time_ms, 100)

    def test_vuln_2_empty_deployment_plan_hangs_tick_indefinitely(self) -> None:
        """
        REMEDIATED DEFECT 2: When an empty deployment plan (steps=[]) is loaded,
        tick() cleanly transitions self._state = PlaybackState.COMPLETED,
        matching step_forward() and preventing infinite playback hangs.
        """
        eng = SimulationEngine(steps=[])
        eng.start()
        self.assertEqual(eng.state, PlaybackState.PLAYING)
        self.assertEqual(eng.total_duration_ms, 0)

        eng.tick(100)

        # Cleanly completed on tick
        self.assertEqual(eng.state, PlaybackState.COMPLETED)
        self.assertTrue(eng.is_completed)

    def test_vuln_3_premature_completion_freezes_troop_ttl(self) -> None:
        """
        REMEDIATED DEFECT 3: Total battle duration includes default_troop_ttl_ms,
        and tick() allows deployed troops to live out their lifetimes and transition to
        EXPIRED rather than freezing active markers at spawn time.
        """
        eng = SimulationEngine(
            steps=[DeploymentStep(step_id=1, troop_type="Sneaky Goblin", delay_ms=100)],
            default_troop_ttl_ms=5000,
        )
        self.assertEqual(eng.total_duration_ms, 5100)
        eng.start()

        # At 100ms, step is deployed; battle is still PLAYING (not prematurely completed)
        eng.tick(100)
        self.assertEqual(eng.state, PlaybackState.PLAYING)
        self.assertEqual(len(eng.active_troops), 1)
        self.assertEqual(eng.all_troops[0]["status"], "ACTIVE")

        # Tick forward 5 more seconds (5000ms) -> Troop lifetime expires at 5100ms
        for _ in range(5):
            eng.tick(1000)

        # Engine transitions to COMPLETED once battle timeline and troop TTL elapses
        self.assertEqual(eng.state, PlaybackState.COMPLETED)
        self.assertEqual(len(eng.active_troops), 0)
        self.assertEqual(eng.all_troops[0]["status"], "EXPIRED")
        self.assertEqual(eng.all_troops[0]["remaining_lifetime_ms"], 0)

    def test_vuln_4_seek_to_step_out_of_bounds_resets_timestamp_to_zero(self) -> None:
        """
        REMEDIATED DEFECT 4: ReplayRecorder.seek_to_step(target_step_index) resolves
        target_timestamp to the timestamp of the last available step/event instead of
        resetting to 0ms when scrubbing beyond the recorded step count.
        """
        rec = ReplayRecorder()
        rec.record_step(DeploymentStep(step_id=1, troop_type="Barb", delay_ms=0), timestamp_ms=0)
        rec.record_step(DeploymentStep(step_id=2, troop_type="Arch", delay_ms=500), timestamp_ms=500)

        # Scrubbing to step 999 matches all events and resolves to last step timestamp (500ms)
        ts, events = rec.seek_to_step(999)
        self.assertEqual(len(events), 2)
        self.assertEqual(ts, 500)

    def test_vuln_5_replay_deserialization_crashes_on_none_values(self) -> None:
        """
        REMEDIATED DEFECT 5: ReplayEvent.from_dict() and ReplayRecorder.import_replay()
        defensively sanitize None values in dictionary keys without raising TypeError.
        """
        rec = ReplayRecorder()

        # None timestamp_ms does not crash; defaults to 0
        rec.import_replay({"events": [{"timestamp_ms": None}]})
        self.assertEqual(len(rec.events), 1)
        self.assertEqual(rec.events[0].timestamp_ms, 0)

        # None data does not crash; defaults to empty dict
        rec.import_replay({"events": [{"data": None}]})
        self.assertEqual(len(rec.events), 1)
        self.assertEqual(rec.events[0].data, {})

        # None events does not crash; defaults to empty list
        rec.import_replay({"events": None})
        self.assertEqual(len(rec.events), 0)

        # None metadata does not crash; defaults to empty dict
        rec.import_replay({"metadata": None})
        self.assertEqual(rec.metadata, {})


if __name__ == "__main__":
    unittest.main()
