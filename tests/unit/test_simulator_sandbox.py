"""
Comprehensive Unit Tests for ClashVision Local Attack Simulator & Safety Sandboxing.
Covers:
- SafetyBarrier & VisualMode (Emergency stop latch <= 50ms, visual indicator, callbacks)
- SandboxedController & InputAction (Canvas bounding, safety isolation, click/drag/press)
- SimulationEngine & PlaybackState (Timeline progression, speed control, troop lifecycle, step scrubbing)
- ReplayRecorder & ReplayEvent (Event capture, scrubbing, JSON import/export)
- End-to-end tactical simulation scenarios with emergency stop interruption
Derived from Requirements R1, R3, R4 and Features 5, 22, 23, 24, 25, 26, 27.
"""

from __future__ import annotations
import json
import os
import tempfile
import time
import pytest

from clashvision.core.constants import (
    EMERGENCY_STOP_LATENCY_MAX_MS,
    StrategyType,
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
    TroopMarker,
)
from clashvision.simulator.replay_recorder import (
    ReplayRecorder,
    ReplayEvent,
)
from tests.conftest import allow_simulator_sandbox


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
def clean_safety():
    """Ensures clean safety barrier and safety guard state for each test."""
    guard = get_safety_guard()
    barrier = get_safety_barrier()
    guard.clear_all()
    barrier.clear_all()
    yield barrier, guard
    guard.clear_all()
    barrier.clear_all()


@pytest.fixture
def sample_steps() -> list[DeploymentStep]:
    """Sample sequence of deployment steps."""
    return [
        DeploymentStep(step_id=1, troop_type="Sneaky Goblin", target_x=300, target_y=400, normalized_x=0.156, normalized_y=0.370, delay_ms=0, unit_count=5, note="North collector"),
        DeploymentStep(step_id=2, troop_type="Sneaky Goblin", target_x=350, target_y=450, normalized_x=0.182, normalized_y=0.417, delay_ms=500, unit_count=5, note="East collector"),
        DeploymentStep(step_id=3, troop_type="Jump Spell", target_x=600, target_y=600, normalized_x=0.312, normalized_y=0.555, delay_ms=1000, unit_count=1, note="Core wall bridge"),
        DeploymentStep(step_id=4, troop_type="Sneaky Goblin", target_x=800, target_y=800, normalized_x=0.416, normalized_y=0.740, delay_ms=800, unit_count=10, note="Town Hall snipe"),
    ]


@pytest.fixture
def sample_recommendation(sample_steps: list[DeploymentStep]) -> FarmingRecommendation:
    """Sample FarmingRecommendation container."""
    return FarmingRecommendation(
        suitability_score=94.5,
        primary_strategy=StrategyType.SNEAKY_GOBLIN,
        deployment_plan=sample_steps,
        timestamp=time.time(),
        explanation="High exposed collector ratio favorable for Sneaky Goblin sweep.",
    )


# ==============================================================================
# Test Suite 1: SafetyBarrier & VisualMode (Features 26, 27)
# ==============================================================================
class TestSafetyBarrier:
    """Verifies visual mode indicators, emergency stop latch latency, and callbacks."""

    def test_visual_mode_enum_values_and_display_names(self) -> None:
        assert VisualMode.LIVE_OBSERVATION_READ_ONLY.value == "LIVE_OBSERVATION_READ_ONLY"
        assert VisualMode.LOCAL_SIMULATION_SANDBOX.value == "LOCAL_SIMULATION_SANDBOX"
        assert VisualMode.LIVE_OBSERVATION_READ_ONLY.display_name == "LIVE (READ-ONLY)"
        assert VisualMode.LOCAL_SIMULATION_SANDBOX.display_name == "LOCAL SIMULATION (SANDBOX)"
        assert VisualMode.LIVE_OBSERVATION_READ_ONLY.is_read_only is True
        assert VisualMode.LOCAL_SIMULATION_SANDBOX.is_read_only is False

    def test_default_mode_is_live_read_only(self, clean_safety) -> None:
        barrier, _ = clean_safety
        assert barrier.visual_mode == VisualMode.LIVE_OBSERVATION_READ_ONLY
        assert barrier.is_live_mode is True
        assert barrier.is_simulator_mode is False

    def test_mode_switching(self, clean_safety) -> None:
        barrier, _ = clean_safety
        barrier.set_simulator_mode()
        assert barrier.visual_mode == VisualMode.LOCAL_SIMULATION_SANDBOX
        assert barrier.is_simulator_mode is True
        assert barrier.is_live_mode is False

        barrier.set_live_mode()
        assert barrier.visual_mode == VisualMode.LIVE_OBSERVATION_READ_ONLY
        assert barrier.is_live_mode is True

        barrier.set_mode("LOCAL_SIMULATION_SANDBOX")
        assert barrier.is_simulator_mode is True

    def test_emergency_stop_response_latency_under_50ms(self, clean_safety) -> None:
        """Requirement R4: Emergency stop interrupt response latency <= 50ms."""
        barrier, guard = clean_safety
        assert barrier.is_emergency_stopped is False

        t0 = time.perf_counter()
        latency_ms = barrier.trigger_emergency_stop(reason="User hit Escape hotkey")
        t1 = time.perf_counter()
        measured_elapsed_ms = (t1 - t0) * 1000.0

        assert latency_ms < EMERGENCY_STOP_LATENCY_MAX_MS
        assert measured_elapsed_ms < EMERGENCY_STOP_LATENCY_MAX_MS
        assert barrier.is_emergency_stopped is True
        assert guard.is_emergency_stopped is True
        assert barrier.last_trigger_reason == "User hit Escape hotkey"
        assert barrier.last_trigger_latency_ms > 0.0

    def test_emergency_stop_check_raises_exception(self, clean_safety) -> None:
        barrier, _ = clean_safety
        # Initially clear, check does not raise
        barrier.check_emergency_stop()

        barrier.trigger_emergency_stop(reason="Test Stop")
        with pytest.raises(EmergencyStopTriggered) as exc_info:
            barrier.check_emergency_stop()
        assert "Test Stop" in str(exc_info.value)

    def test_emergency_stop_reset(self, clean_safety) -> None:
        barrier, guard = clean_safety
        barrier.trigger_emergency_stop()
        assert barrier.is_emergency_stopped is True
        assert guard.is_emergency_stopped is True

        barrier.reset_emergency_stop()
        assert barrier.is_emergency_stopped is False
        assert guard.is_emergency_stopped is False
        barrier.check_emergency_stop()  # Should not raise

    def test_emergency_stop_callbacks_execution(self, clean_safety) -> None:
        barrier, _ = clean_safety
        callback_called_1 = False
        callback_called_2 = False

        def cb1():
            nonlocal callback_called_1
            callback_called_1 = True

        def cb2():
            nonlocal callback_called_2
            callback_called_2 = True

        barrier.register_emergency_callback(cb1)
        barrier.register_emergency_callback(cb2)

        barrier.trigger_emergency_stop()
        assert callback_called_1 is True
        assert callback_called_2 is True

        # Unregister cb1
        barrier.reset_emergency_stop()
        callback_called_1 = False
        callback_called_2 = False
        barrier.unregister_emergency_callback(cb1)

        barrier.trigger_emergency_stop()
        assert callback_called_1 is False
        assert callback_called_2 is True

    def test_emergency_stop_callback_error_resilience(self, clean_safety) -> None:
        barrier, _ = clean_safety
        second_called = False

        def faulty_cb():
            raise RuntimeError("Failure in callback handler")

        def normal_cb():
            nonlocal second_called
            second_called = True

        barrier.register_emergency_callback(faulty_cb)
        barrier.register_emergency_callback(normal_cb)

        # Trigger should not crash despite faulty callback
        barrier.trigger_emergency_stop()
        assert second_called is True
        assert barrier.is_emergency_stopped is True

    def test_window_verification_delegation(self, clean_safety) -> None:
        barrier, _ = clean_safety
        live_hwnd = 1001
        sim_hwnd = 2002
        unregistered_hwnd = 3003

        barrier.register_live_window(hwnd=live_hwnd, title="Clash of Clans Live")
        barrier.register_simulator_window(hwnd=sim_hwnd)

        assert barrier.is_live_window(live_hwnd) is True
        assert barrier.is_live_window(sim_hwnd) is False

        # Live window rejected
        with pytest.raises(LiveInputSafetyViolation):
            barrier.verify_window_allowed(target_hwnd=live_hwnd, action_name="test_click")

        # Unregistered window rejected
        with pytest.raises(LiveInputSafetyViolation):
            barrier.verify_window_allowed(target_hwnd=unregistered_hwnd, action_name="test_click")

        # Registered simulator window allowed
        barrier.verify_window_allowed(target_hwnd=sim_hwnd, action_name="test_click")

        # When emergency stop trips, even simulator window is blocked
        barrier.trigger_emergency_stop()
        with pytest.raises(EmergencyStopTriggered):
            barrier.verify_window_allowed(target_hwnd=sim_hwnd, action_name="test_click")


# ==============================================================================
# Test Suite 2: SandboxedController (Features 5, 25)
# ==============================================================================
class TestSandboxedController:
    """Verifies quarantined PyAutoGUI controller restricted strictly to simulator canvas."""

    def test_alias_and_initialization(self, clean_safety) -> None:
        barrier, guard = clean_safety
        guard.register_simulator_window(hwnd=4001)
        controller = SandboxedController(
            canvas_hwnd=4001,
            canvas_width=1920,
            canvas_height=1080,
            guard=guard,
            barrier=barrier,
        )
        assert controller.canvas_hwnd == 4001
        assert controller.canvas_width == 1920
        assert controller.canvas_height == 1080
        assert controller.strict_sandbox is True
        assert len(controller.action_history) == 0

        # Verify alias
        assert SandboxedInputController is SandboxedController

    def test_rejects_action_when_no_canvas_window_registered(self, clean_safety) -> None:
        barrier, guard = clean_safety
        controller = SandboxedController(canvas_hwnd=None, guard=guard, barrier=barrier)
        with pytest.raises(LiveInputSafetyViolation) as exc_info:
            controller.click(500, 500)
        assert "No simulator canvas window registered" in str(exc_info.value)

    def test_rejects_action_targeting_live_window(self, clean_safety) -> None:
        barrier, guard = clean_safety
        live_hwnd = 9001
        guard.register_live_window(hwnd=live_hwnd, title="Live Emulator")
        controller = SandboxedController(canvas_hwnd=live_hwnd, guard=guard, barrier=barrier)

        with pytest.raises(LiveInputSafetyViolation):
            controller.click(300, 300)

    def test_rejects_action_targeting_unregistered_window(self, clean_safety) -> None:
        barrier, guard = clean_safety
        unregistered_hwnd = 8001
        controller = SandboxedController(canvas_hwnd=unregistered_hwnd, guard=guard, barrier=barrier)

        with pytest.raises(LiveInputSafetyViolation):
            controller.click(300, 300)

    def test_rejects_action_when_emergency_stop_active(self, clean_safety) -> None:
        barrier, guard = clean_safety
        sim_hwnd = 5001
        guard.register_simulator_window(hwnd=sim_hwnd)
        controller = SandboxedController(canvas_hwnd=sim_hwnd, guard=guard, barrier=barrier)

        barrier.trigger_emergency_stop()
        with pytest.raises(EmergencyStopTriggered):
            controller.click(200, 200)

    def test_click_action_dispatched_in_sandbox(self, clean_safety) -> None:
        barrier, guard = clean_safety
        sim_hwnd = 6001
        guard.register_simulator_window(hwnd=sim_hwnd)
        controller = SandboxedController(canvas_hwnd=sim_hwnd, guard=guard, barrier=barrier)

        result = controller.click(450, 600, button="left")
        assert result is True
        assert len(controller.action_history) == 1
        act = controller.action_history[0]
        assert act.action_type == "click"
        assert act.x == 450
        assert act.y == 600
        assert act.button == "left"

    def test_move_to_action_dispatched_in_sandbox(self, clean_safety) -> None:
        barrier, guard = clean_safety
        sim_hwnd = 6002
        guard.register_simulator_window(hwnd=sim_hwnd)
        controller = SandboxedController(canvas_hwnd=sim_hwnd, guard=guard, barrier=barrier)

        result = controller.move_to(800, 450)
        assert result is True
        assert len(controller.action_history) == 1
        assert controller.action_history[0].action_type == "move_to"
        assert controller.action_history[0].x == 800
        assert controller.action_history[0].y == 450

    def test_drag_action_dispatched_in_sandbox(self, clean_safety) -> None:
        barrier, guard = clean_safety
        sim_hwnd = 6003
        guard.register_simulator_window(hwnd=sim_hwnd)
        controller = SandboxedController(canvas_hwnd=sim_hwnd, guard=guard, barrier=barrier)

        result = controller.drag(100, 100, 500, 500, duration_ms=250)
        assert result is True
        assert len(controller.action_history) == 1
        act = controller.action_history[0]
        assert act.action_type == "drag"
        assert act.x == 100
        assert act.y == 100
        assert act.x2 == 500
        assert act.y2 == 500
        assert act.duration_ms == 250

    def test_press_key_action_dispatched_in_sandbox(self, clean_safety) -> None:
        barrier, guard = clean_safety
        sim_hwnd = 6004
        guard.register_simulator_window(hwnd=sim_hwnd)
        controller = SandboxedController(canvas_hwnd=sim_hwnd, guard=guard, barrier=barrier)

        result = controller.press_key("space")
        assert result is True
        assert len(controller.action_history) == 1
        assert controller.action_history[0].action_type == "press_key"
        assert controller.action_history[0].key == "space"

    def test_coordinate_bounds_validation(self, clean_safety) -> None:
        barrier, guard = clean_safety
        sim_hwnd = 6005
        guard.register_simulator_window(hwnd=sim_hwnd)
        controller = SandboxedController(
            canvas_hwnd=sim_hwnd,
            canvas_width=1920,
            canvas_height=1080,
            guard=guard,
            barrier=barrier,
        )

        # Negative X
        with pytest.raises(ValueError):
            controller.click(-1, 500)

        # Negative Y
        with pytest.raises(ValueError):
            controller.click(500, -1)

        # X exceeding width
        with pytest.raises(ValueError):
            controller.click(1921, 500)

        # Y exceeding height
        with pytest.raises(ValueError):
            controller.click(500, 1081)

        # Drag out of bounds end coordinate
        with pytest.raises(ValueError):
            controller.drag(100, 100, 2000, 500)

    def test_verify_and_execute_dispatch_contract(self, clean_safety) -> None:
        barrier, guard = clean_safety
        sim_hwnd = 6006
        guard.register_simulator_window(hwnd=sim_hwnd)
        controller = SandboxedController(canvas_hwnd=sim_hwnd, guard=guard, barrier=barrier)

        click_act = InputAction(action_type="click", x=200, y=300)
        move_act = InputAction(action_type="move_to", x=300, y=400)
        drag_act = InputAction(action_type="drag", x=100, y=100, x2=200, y2=200, duration_ms=100)
        key_act = InputAction(action_type="press_key", key="escape")

        assert controller.verify_and_execute(click_act) is True
        assert controller.verify_and_execute(move_act) is True
        assert controller.verify_and_execute(drag_act) is True
        assert controller.verify_and_execute(key_act) is True
        assert len(controller.action_history) == 4

        # Unknown action raises ValueError
        invalid_act = InputAction(action_type="unsupported_op")
        with pytest.raises(ValueError):
            controller.verify_and_execute(invalid_act)

    def test_clear_action_history(self, clean_safety) -> None:
        barrier, guard = clean_safety
        sim_hwnd = 6007
        guard.register_simulator_window(hwnd=sim_hwnd)
        controller = SandboxedController(canvas_hwnd=sim_hwnd, guard=guard, barrier=barrier)

        controller.click(100, 100)
        controller.click(200, 200)
        assert len(controller.action_history) == 2
        controller.clear_history()
        assert len(controller.action_history) == 0


# ==============================================================================
# Test Suite 3: SimulationEngine (Features 22, 23, 24)
# ==============================================================================
class TestSimulationEngine:
    """Verifies playback state machine, timeline controls, speed, and active troops."""

    def test_initial_state_idle(self, clean_safety) -> None:
        barrier, guard = clean_safety
        engine = SimulationEngine(barrier=barrier, guard=guard)
        assert engine.state == PlaybackState.IDLE
        assert engine.is_stopped is True
        assert engine.is_playing is False
        assert engine.is_paused is False
        assert engine.step_index == 0
        assert engine.total_steps == 0
        assert engine.current_time_ms == 0
        assert engine.progress_fraction == 0.0

    def test_load_steps_and_total_duration(self, clean_safety, sample_steps) -> None:
        barrier, guard = clean_safety
        engine = SimulationEngine(barrier=barrier, guard=guard)
        engine.load_steps(sample_steps)

        assert engine.total_steps == 4
        # Delays: 0 + 500 + 1000 + 800 = 2300 ms + 5000 ms default troop TTL = 7300 ms
        assert engine.total_duration_ms == 7300
        assert engine.current_time_ms == 0
        assert engine.step_index == 0

    def test_load_recommendation(self, clean_safety, sample_recommendation) -> None:
        barrier, guard = clean_safety
        engine = SimulationEngine(barrier=barrier, guard=guard)
        engine.load_recommendation(sample_recommendation)

        assert engine.total_steps == 4
        assert engine.total_duration_ms == 7300

    def test_state_transitions_start_pause_resume_stop(self, clean_safety, sample_steps) -> None:
        barrier, guard = clean_safety
        engine = SimulationEngine(steps=sample_steps, barrier=barrier, guard=guard)

        # Start -> PLAYING
        engine.start()
        assert engine.state == PlaybackState.PLAYING
        assert engine.is_playing is True
        assert engine.is_stopped is False

        # Pause -> PAUSED
        engine.pause()
        assert engine.state == PlaybackState.PAUSED
        assert engine.is_paused is True
        assert engine.is_playing is False

        # Resume -> PLAYING
        engine.resume()
        assert engine.state == PlaybackState.PLAYING
        assert engine.is_playing is True

        # Stop -> STOPPED
        engine.stop()
        assert engine.state == PlaybackState.STOPPED
        assert engine.is_stopped is True
        assert engine.is_playing is False

    def test_reset_clears_timeline_and_troops(self, clean_safety, sample_steps) -> None:
        barrier, guard = clean_safety
        engine = SimulationEngine(steps=sample_steps, barrier=barrier, guard=guard)
        engine.start()
        engine.tick(600)  # Deploys steps 1 and 2
        assert engine.step_index == 2
        assert len(engine.active_troops) == 2

        engine.reset()
        assert engine.state == PlaybackState.IDLE
        assert engine.step_index == 0
        assert engine.current_time_ms == 0
        assert len(engine.active_troops) == 0

    def test_speed_multipliers_and_boundaries(self, clean_safety) -> None:
        barrier, guard = clean_safety
        engine = SimulationEngine(barrier=barrier, guard=guard)

        # Standard supported speeds
        for spd in [0.5, 1.0, 2.0, 4.0]:
            engine.set_speed(spd)
            assert engine.speed == spd

        # Boundary checks
        engine.set_speed(0.1)
        assert engine.speed == 0.1
        engine.set_speed(10.0)
        assert engine.speed == 10.0

        # Out-of-bounds raises ValueError
        with pytest.raises(ValueError):
            engine.set_speed(0.05)

        with pytest.raises(ValueError):
            engine.set_speed(10.5)

    def test_timeline_progression_and_step_deployment(self, clean_safety, sample_steps) -> None:
        barrier, guard = clean_safety
        engine = SimulationEngine(steps=sample_steps, barrier=barrier, guard=guard)
        engine.start()

        # Step 1 is scheduled at time 0
        engine.tick(0)
        assert engine.step_index == 1
        assert len(engine.active_troops) == 1
        assert engine.active_troops[0]["troop_type"] == "Sneaky Goblin"

        # Advance to 500 ms -> Step 2 deployed
        engine.tick(500)
        assert engine.current_time_ms == 500
        assert engine.step_index == 2
        assert len(engine.active_troops) == 2

        # Advance to 1500 ms (delta 1000) -> Step 3 deployed (Jump Spell)
        engine.tick(1000)
        assert engine.current_time_ms == 1500
        assert engine.step_index == 3
        assert len(engine.active_troops) == 3

        # Advance to 2300 ms (delta 800) -> Step 4 deployed, still PLAYING while troops active
        engine.tick(800)
        assert engine.current_time_ms == 2300
        assert engine.step_index == 4
        assert engine.state == PlaybackState.PLAYING

        # Advance to 7300 ms (delta 5000) -> Troops expire, playback completes
        engine.tick(5000)
        assert engine.current_time_ms == 7300
        assert engine.progress_fraction == 1.0
        assert engine.state == PlaybackState.COMPLETED
        assert len(engine.active_troops) == 0

    def test_speed_accelerated_tick(self, clean_safety, sample_steps) -> None:
        barrier, guard = clean_safety
        engine = SimulationEngine(steps=sample_steps, speed=2.0, barrier=barrier, guard=guard)
        engine.start()

        # At 2.0x speed, tick(250) advances current_time by 500 ms
        engine.tick(250)
        assert engine.current_time_ms == 500
        assert engine.step_index == 2

    def test_troop_marker_lifetime_and_expiration(self, clean_safety) -> None:
        barrier, guard = clean_safety
        steps = [
            DeploymentStep(step_id=1, troop_type="Barbarian", target_x=100, target_y=100, delay_ms=0),
        ]
        # TTL of 1000 ms
        engine = SimulationEngine(steps=steps, default_troop_ttl_ms=1000, barrier=barrier, guard=guard)
        engine.start()

        engine.tick(0)
        assert len(engine.active_troops) == 1
        assert engine.active_troops[0]["status"] == "ACTIVE"
        assert engine.active_troops[0]["remaining_lifetime_ms"] == 1000

        # Advance by 500 ms
        engine.tick(500)
        assert len(engine.active_troops) == 1
        assert engine.active_troops[0]["remaining_lifetime_ms"] == 500

        # Advance by 600 ms -> Exceeds 1000 ms TTL
        engine.tick(600)
        assert len(engine.active_troops) == 0  # Filtered out of active
        assert len(engine.all_troops) == 1
        assert engine.all_troops[0]["status"] == "EXPIRED"

    def test_step_forward_manual_control(self, clean_safety, sample_steps) -> None:
        barrier, guard = clean_safety
        engine = SimulationEngine(steps=sample_steps, barrier=barrier, guard=guard)

        s1 = engine.step_forward()
        assert s1 is not None and s1.step_id == 1
        assert engine.step_index == 1
        assert engine.current_time_ms == 0

        s2 = engine.step_forward()
        assert s2 is not None and s2.step_id == 2
        assert engine.step_index == 2
        assert engine.current_time_ms == 500

        s3 = engine.step_forward()
        assert s3 is not None and s3.step_id == 3
        assert engine.step_index == 3
        assert engine.current_time_ms == 1500

        s4 = engine.step_forward()
        assert s4 is not None and s4.step_id == 4
        assert engine.step_index == 4
        assert engine.current_time_ms == 2300

        # Beyond last step returns None
        s5 = engine.step_forward()
        assert s5 is None

    def test_step_backward_rewind(self, clean_safety, sample_steps) -> None:
        barrier, guard = clean_safety
        engine = SimulationEngine(steps=sample_steps, barrier=barrier, guard=guard)

        # Step forward 3 times
        engine.step_forward()
        engine.step_forward()
        engine.step_forward()
        assert engine.step_index == 3
        assert engine.current_time_ms == 1500

        # Step backward to 2
        engine.step_backward()
        assert engine.step_index == 2
        assert engine.current_time_ms == 500
        assert len(engine.active_troops) == 2

        # Step backward to 1
        engine.step_backward()
        assert engine.step_index == 1
        assert engine.current_time_ms == 0
        assert len(engine.active_troops) == 1

        # Step backward to 0
        engine.step_backward()
        assert engine.step_index == 0
        assert engine.current_time_ms == 0
        assert len(engine.active_troops) == 0

    def test_callbacks_registration_and_triggering(self, clean_safety, sample_steps) -> None:
        barrier, guard = clean_safety
        engine = SimulationEngine(steps=sample_steps, barrier=barrier, guard=guard)

        ticks_count = 0
        steps_count = 0
        completed = False
        state_transitions: list[tuple[PlaybackState, PlaybackState]] = []

        engine.register_on_tick(lambda state: nonlocal_inc_tick())
        engine.register_on_step(lambda step, marker: nonlocal_inc_step())
        engine.register_on_complete(lambda state: nonlocal_set_complete())
        engine.register_on_state_change(lambda o, n: state_transitions.append((o, n)))

        def nonlocal_inc_tick():
            nonlocal ticks_count
            ticks_count += 1

        def nonlocal_inc_step():
            nonlocal steps_count
            steps_count += 1

        def nonlocal_set_complete():
            nonlocal completed
            completed = True

        engine.start()
        engine.tick(7500)

        assert ticks_count > 0
        assert steps_count == 4
        assert completed is True
        assert (PlaybackState.IDLE, PlaybackState.PLAYING) in state_transitions
        assert (PlaybackState.PLAYING, PlaybackState.COMPLETED) in state_transitions

    def test_emergency_stop_halts_simulation_during_tick(self, clean_safety, sample_steps) -> None:
        barrier, guard = clean_safety
        engine = SimulationEngine(steps=sample_steps, barrier=barrier, guard=guard)
        engine.start()

        # Step 1 runs
        engine.tick(0)
        assert engine.step_index == 1

        # Emergency stop trips
        barrier.trigger_emergency_stop(reason="Midway panic button")

        # Subsequent tick must halt and raise EmergencyStopTriggered
        with pytest.raises(EmergencyStopTriggered):
            engine.tick(500)

        assert engine.state == PlaybackState.STOPPED
        assert engine.step_index == 1  # Never advanced to step 2

    def test_controller_integration_dispatches_actions(self, clean_safety, sample_steps) -> None:
        barrier, guard = clean_safety
        sim_hwnd = 7001
        guard.register_simulator_window(hwnd=sim_hwnd)
        controller = SandboxedController(canvas_hwnd=sim_hwnd, guard=guard, barrier=barrier)

        engine = SimulationEngine(
            steps=sample_steps,
            controller=controller,
            barrier=barrier,
            guard=guard,
        )
        engine.start()
        engine.tick(600)  # Executes steps 1 & 2

        assert len(controller.action_history) == 2
        assert controller.action_history[0].x == sample_steps[0].target_x
        assert controller.action_history[0].y == sample_steps[0].target_y
        assert controller.action_history[1].x == sample_steps[1].target_x
        assert controller.action_history[1].y == sample_steps[1].target_y


# ==============================================================================
# Test Suite 4: ReplayRecorder (Features 22, 24)
# ==============================================================================
class TestReplayRecorder:
    """Verifies timeline event recording, JSON serialization, and scrubbing."""

    def test_replay_event_model(self) -> None:
        ev = ReplayEvent(
            timestamp_ms=1250,
            event_type="STEP",
            step_index=2,
            data={"troop": "Dragon", "x": 400, "y": 600},
        )
        data = ev.to_dict()
        assert data["timestamp_ms"] == 1250
        assert data["event_type"] == "STEP"
        assert data["step_index"] == 2
        assert data["data"]["troop"] == "Dragon"

        ev2 = ReplayEvent.from_dict(data)
        assert ev2.timestamp_ms == ev.timestamp_ms
        assert ev2.event_type == ev.event_type
        assert ev2.data == ev.data

    def test_record_steps_ticks_and_actions(self) -> None:
        rec = ReplayRecorder(metadata={"strategy": "SNEAKY_GOBLIN"})
        assert rec.event_count == 0

        step = DeploymentStep(step_id=1, troop_type="Sneaky Goblin", target_x=100, target_y=200)
        rec.record_step(step, timestamp_ms=0)
        rec.record_action(action_type="click", x=100, y=200, timestamp_ms=0)
        rec.record_tick(SimulationState(current_time_ms=500, step_index=1, total_duration_ms=1000))
        rec.record_state_change("IDLE", "PLAYING", timestamp_ms=0)
        rec.record_emergency_stop(timestamp_ms=800, reason="Test trigger")

        assert rec.event_count == 5
        assert len(rec.get_events("STEP")) == 1
        assert len(rec.get_events("ACTION")) == 1
        assert len(rec.get_events("TICK")) == 1
        assert len(rec.get_events("STATE_CHANGE")) == 1
        assert len(rec.get_events("EMERGENCY_STOP")) == 1

    def test_scrubbing_seek_to_time(self) -> None:
        rec = ReplayRecorder()
        s1 = DeploymentStep(step_id=1, troop_type="Goblin")
        s2 = DeploymentStep(step_id=2, troop_type="Goblin")
        s3 = DeploymentStep(step_id=3, troop_type="Wall Breaker")

        rec.record_step(s1, timestamp_ms=0)
        rec.record_step(s2, timestamp_ms=500)
        rec.record_step(s3, timestamp_ms=1200)

        # Seek to 400ms: only step 1
        step_idx, events = rec.seek_to_time(400)
        assert step_idx == 1
        assert len(events) == 1

        # Seek to 600ms: steps 1 and 2
        step_idx, events = rec.seek_to_time(600)
        assert step_idx == 2
        assert len(events) == 2

        # Seek to 1500ms: all 3 steps
        step_idx, events = rec.seek_to_time(1500)
        assert step_idx == 3
        assert len(events) == 3

    def test_scrubbing_seek_to_step(self) -> None:
        rec = ReplayRecorder()
        s1 = DeploymentStep(step_id=1, troop_type="Goblin")
        s2 = DeploymentStep(step_id=2, troop_type="Goblin")
        s3 = DeploymentStep(step_id=3, troop_type="Jump Spell")

        rec.record_step(s1, timestamp_ms=100)
        rec.record_step(s2, timestamp_ms=600)
        rec.record_step(s3, timestamp_ms=1400)

        t_ms, events = rec.seek_to_step(2)
        assert t_ms == 600
        assert len(events) == 2

    def test_json_export_and_import(self) -> None:
        rec = ReplayRecorder(metadata={"map": "TH15_Base"})
        s1 = DeploymentStep(step_id=1, troop_type="Miner", target_x=500, target_y=500)
        rec.record_step(s1, timestamp_ms=250)

        exported = rec.export_replay()
        assert exported["version"] == "1.0"
        assert exported["step_count"] == 1
        assert exported["metadata"]["map"] == "TH15_Base"
        assert len(exported["events"]) == 1

        rec2 = ReplayRecorder()
        rec2.import_replay(exported)
        assert rec2.event_count == 1
        assert rec2.metadata["map"] == "TH15_Base"
        assert rec2.events[0].data["troop_type"] == "Miner"

    def test_file_save_and_load(self) -> None:
        rec = ReplayRecorder(metadata={"session_id": "test_sess_01"})
        step = DeploymentStep(step_id=1, troop_type="Dragon", target_x=200, target_y=300)
        rec.record_step(step, timestamp_ms=0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            rec.save_to_file(temp_path)
            assert os.path.exists(temp_path)

            rec2 = ReplayRecorder()
            rec2.load_from_file(temp_path)
            assert rec2.event_count == 1
            assert rec2.metadata["session_id"] == "test_sess_01"
            assert rec2.events[0].data["troop_type"] == "Dragon"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_attachment_to_simulation_engine(self, clean_safety, sample_steps) -> None:
        barrier, guard = clean_safety
        engine = SimulationEngine(steps=sample_steps, barrier=barrier, guard=guard)
        recorder = ReplayRecorder()
        recorder.attach_to_engine(engine)

        engine.start()
        engine.tick(2500)

        # Recorder should have captured steps, ticks, and state changes automatically
        step_events = recorder.get_events("STEP")
        assert len(step_events) == 4
        assert recorder.get_events("STATE_CHANGE")
        assert recorder.get_events("TICK")


# ==============================================================================
# Test Suite 5: End-to-End Simulation Workflows (Features 22–27)
# ==============================================================================
class TestSimulatorEndToEndScenarios:
    """Verifies full end-to-end simulation pipelines with safety barriers."""

    def test_sneaky_goblin_farming_full_replay(self, clean_safety, sample_recommendation) -> None:
        barrier, guard = clean_safety
        sim_hwnd = 8801
        guard.register_simulator_window(hwnd=sim_hwnd)
        controller = SandboxedController(canvas_hwnd=sim_hwnd, guard=guard, barrier=barrier)

        engine = SimulationEngine(
            recommendation=sample_recommendation,
            controller=controller,
            barrier=barrier,
            guard=guard,
            speed=2.0,
        )
        recorder = ReplayRecorder(metadata={"strategy": sample_recommendation.primary_strategy.value})
        recorder.attach_to_engine(engine)

        barrier.set_simulator_mode()
        assert barrier.is_simulator_mode is True

        engine.start()
        # Advance simulation at 2.0x in increments of 200 ms
        while not engine.is_completed:
            engine.tick(200)

        assert engine.is_completed is True
        assert engine.step_index == len(sample_recommendation.deployment_plan)
        assert len(controller.action_history) == 4
        assert len(recorder.get_events("STEP")) == 4

    def test_emergency_stop_interruption_sub_50ms_guarantee(self, clean_safety, sample_steps) -> None:
        barrier, guard = clean_safety
        sim_hwnd = 8802
        guard.register_simulator_window(hwnd=sim_hwnd)
        controller = SandboxedController(canvas_hwnd=sim_hwnd, guard=guard, barrier=barrier)

        engine = SimulationEngine(
            steps=sample_steps,
            controller=controller,
            barrier=barrier,
            guard=guard,
        )

        engine.start()
        engine.tick(0)    # Step 1 deploys
        engine.tick(500)  # Step 2 deploys
        assert engine.step_index == 2

        # User presses Emergency Stop (measured latency must be <= 50ms)
        t0 = time.perf_counter()
        barrier.trigger_emergency_stop(reason="Abnormal troop path detected")
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        assert elapsed_ms < EMERGENCY_STOP_LATENCY_MAX_MS

        # Next tick halts simulation
        with pytest.raises(EmergencyStopTriggered):
            engine.tick(1000)

        assert engine.state == PlaybackState.STOPPED
        assert engine.step_index == 2
        # No more actions reached controller
        assert len(controller.action_history) == 2

        # Controller directly rejects any further action
        with pytest.raises(EmergencyStopTriggered):
            controller.click(600, 600)
