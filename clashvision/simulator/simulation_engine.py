"""
ClashVision Local Attack Simulator Engine.
Offline battle simulation engine and playback state machine for attack replay.
Derived directly from Requirements R3, R4 and Features 22, 23, 24, 25, 26.
"""

from __future__ import annotations
import logging
import math
import threading
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from clashvision.core.models import (
    DeploymentStep,
    FarmingRecommendation,
    SimulationState,
)
from clashvision.core.exceptions import (
    EmergencyStopTriggered,
    LiveInputSafetyViolation,
)
from clashvision.capture.safety_guard import SafetyGuard, get_safety_guard
from clashvision.simulator.safety_barrier import SafetyBarrier, get_safety_barrier
from clashvision.simulator.sandboxed_controller import SandboxedController

logger = logging.getLogger("clashvision.simulator.engine")


@unique
class PlaybackState(str, Enum):
    """
    Offline playback state machine states.
    """
    IDLE = "IDLE"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"


@dataclass
class ActiveTroop:
    """
    Simulated active troop marker on the battlefield timeline.
    Tracks troop type, coordinates, spawn timestamp, remaining lifetime, and status.
    """
    troop_id: str
    step_id: int
    troop_type: str
    target_x: int
    target_y: int
    normalized_x: float = 0.0
    normalized_y: float = 0.0
    spawn_time_ms: int = 0
    remaining_lifetime_ms: int = 5000
    status: str = "ACTIVE"  # "ACTIVE" or "EXPIRED"
    unit_count: int = 1
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "troop_id": self.troop_id,
            "step_id": self.step_id,
            "troop_type": self.troop_type,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "normalized_x": self.normalized_x,
            "normalized_y": self.normalized_y,
            "spawn_time_ms": self.spawn_time_ms,
            "remaining_lifetime_ms": self.remaining_lifetime_ms,
            "status": self.status,
            "unit_count": self.unit_count,
            "note": self.note,
        }


# Alias for troop marker representation
TroopMarker = ActiveTroop


class SimulationEngine:
    """
    Local Attack Simulator playback engine.
    Features:
    - Full offline playback state machine (IDLE, PLAYING, PAUSED, COMPLETED, STOPPED).
    - Timeline controls: start, pause, resume, stop, reset, step_forward, step_backward.
    - Speed multiplier control: 0.5x, 1.0x, 2.0x, 4.0x (range 0.1x - 10.0x).
    - Accepts list of DeploymentStep or FarmingRecommendation.
    - Tracks timeline progression, progress_fraction, and active troop lifetime.
    - Strict safety integration: halts immediately on emergency stop (<= 50ms).
    - Event callbacks for on_tick, on_step, on_complete, on_state_change.
    """

    def __init__(
        self,
        steps: Optional[List[DeploymentStep]] = None,
        recommendation: Optional[FarmingRecommendation] = None,
        controller: Optional[SandboxedController] = None,
        barrier: Optional[SafetyBarrier] = None,
        guard: Optional[SafetyGuard] = None,
        speed: float = 1.0,
        default_troop_ttl_ms: int = 5000,
    ) -> None:
        self.barrier = barrier if barrier is not None else get_safety_barrier()
        self.guard = guard if guard is not None else get_safety_guard()
        self.controller = controller
        self.default_troop_ttl_ms = default_troop_ttl_ms

        self._lock = threading.RLock()

        self._state: PlaybackState = PlaybackState.IDLE
        self._speed: float = 1.0
        self.set_speed(speed)

        self._steps: List[DeploymentStep] = []
        self._step_times: List[int] = []
        self._total_duration_ms: int = 0
        self._current_time_ms: int = 0
        self._step_index: int = 0
        self._deployed_step_indices: Set[int] = set()
        self._active_troops: List[ActiveTroop] = []

        # Event callbacks
        self._callbacks_tick: List[Callable[[SimulationState], None]] = []
        self._callbacks_step: List[Callable[[DeploymentStep, Dict[str, Any]], None]] = []
        self._callbacks_complete: List[Callable[[SimulationState], None]] = []
        self._callbacks_state_change: List[Callable[[PlaybackState, PlaybackState], None]] = []

        # Load initial plan if provided
        if recommendation is not None:
            self.load_recommendation(recommendation)
        elif steps is not None:
            self.load_steps(steps)

    # --------------------------------------------------------------------------
    # Speed & Timeline Configuration
    # --------------------------------------------------------------------------
    @property
    def speed(self) -> float:
        with self._lock:
            return self._speed

    def set_speed(self, speed: float) -> None:
        """
        Sets the playback speed multiplier.
        Validates 0.1 <= speed <= 10.0 and rejects NaN / non-finite values.
        """
        with self._lock:
            if (
                speed is None
                or isinstance(speed, bool)
                or not isinstance(speed, (int, float))
                or not math.isfinite(speed)
                or speed < 0.1
                or speed > 10.0
            ):
                raise ValueError(
                    f"Playback speed must be a finite number between 0.1 and 10.0, got {speed}"
                )
            self._speed = float(speed)
            logger.debug("SimulationEngine speed set to %.2fx", self._speed)

    # --------------------------------------------------------------------------
    # Step & Recommendation Loading
    # --------------------------------------------------------------------------
    def load_steps(
        self,
        steps: List[DeploymentStep],
        total_duration_ms: Optional[int] = None,
    ) -> None:
        """
        Loads a sequential list of deployment steps into the simulation engine.
        Computes cumulative step deployment timestamps and total timeline duration.
        """
        with self._lock:
            self.reset()
            self._steps = list(steps)
            self._step_times = []

            cumulative_ms = 0
            for step in self._steps:
                cumulative_ms += max(0, step.delay_ms)
                self._step_times.append(cumulative_ms)

            if total_duration_ms is not None:
                self._total_duration_ms = max(0, total_duration_ms)
            elif self._steps:
                self._total_duration_ms = cumulative_ms + self.default_troop_ttl_ms
            else:
                self._total_duration_ms = 0

            logger.info(
                "SimulationEngine loaded %d steps; total duration: %d ms",
                len(self._steps),
                self._total_duration_ms,
            )

    def load_recommendation(self, recommendation: FarmingRecommendation) -> None:
        """
        Loads deployment plan from a FarmingRecommendation contract.
        """
        with self._lock:
            self.load_steps(recommendation.deployment_plan)

    # --------------------------------------------------------------------------
    # State Inspection Properties
    # --------------------------------------------------------------------------
    @property
    def state(self) -> PlaybackState:
        with self._lock:
            return self._state

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._state == PlaybackState.PLAYING

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._state == PlaybackState.PAUSED

    @property
    def is_stopped(self) -> bool:
        with self._lock:
            return self._state in (PlaybackState.STOPPED, PlaybackState.IDLE, PlaybackState.COMPLETED)

    @property
    def is_completed(self) -> bool:
        with self._lock:
            return self._state == PlaybackState.COMPLETED

    @property
    def current_time_ms(self) -> int:
        with self._lock:
            return self._current_time_ms

    @property
    def total_duration_ms(self) -> int:
        with self._lock:
            return self._total_duration_ms

    @property
    def step_index(self) -> int:
        with self._lock:
            return self._step_index

    @property
    def total_steps(self) -> int:
        with self._lock:
            return len(self._steps)

    @property
    def progress_fraction(self) -> float:
        with self._lock:
            if self._total_duration_ms <= 0:
                return 0.0
            return min(1.0, max(0.0, self._current_time_ms / self._total_duration_ms))

    @property
    def active_troops(self) -> List[Dict[str, Any]]:
        """List of active (non-expired) troop dictionaries."""
        with self._lock:
            return [t.to_dict() for t in self._active_troops if t.status == "ACTIVE"]

    @property
    def all_troops(self) -> List[Dict[str, Any]]:
        """List of all troop dictionaries including expired."""
        with self._lock:
            return [t.to_dict() for t in self._active_troops]

    @property
    def simulation_state(self) -> SimulationState:
        """Constructs and returns current SimulationState contract snapshot."""
        with self._lock:
            return SimulationState(
                step_index=self._step_index,
                total_steps=len(self._steps),
                is_playing=self._state == PlaybackState.PLAYING,
                is_paused=self._state == PlaybackState.PAUSED,
                is_stopped=self._state in (PlaybackState.STOPPED, PlaybackState.IDLE, PlaybackState.COMPLETED),
                current_time_ms=self._current_time_ms,
                total_duration_ms=self._total_duration_ms,
                active_troops=[t.to_dict() for t in self._active_troops if t.status == "ACTIVE"],
            )

    def get_state(self) -> SimulationState:
        """Alias for simulation_state property."""
        return self.simulation_state

    # --------------------------------------------------------------------------
    # Playback Controls
    # --------------------------------------------------------------------------
    def _set_state(self, new_state: PlaybackState) -> None:
        old_state = self._state
        if old_state != new_state:
            self._state = new_state
            logger.info("SimulationEngine state transition: %s -> %s", old_state.name, new_state.name)
            callbacks = list(self._callbacks_state_change)
            for cb in callbacks:
                try:
                    cb(old_state, new_state)
                except Exception as e:
                    logger.error("Error in on_state_change callback: %s", e)

    def start(self) -> None:
        """
        Starts playback from the current position or begins new replay.
        """
        with self._lock:
            self._check_emergency_stop()
            if self._state == PlaybackState.PAUSED:
                self.resume()
                return
            if self._state == PlaybackState.PLAYING:
                return

            if self._state == PlaybackState.COMPLETED:
                self.reset()

            self._set_state(PlaybackState.PLAYING)

    def pause(self) -> None:
        """
        Pauses current simulation playback.
        """
        with self._lock:
            if self._state == PlaybackState.PLAYING:
                self._set_state(PlaybackState.PAUSED)

    def resume(self) -> None:
        """
        Resumes playback from a paused state.
        """
        with self._lock:
            self._check_emergency_stop()
            if self._state == PlaybackState.PAUSED:
                self._set_state(PlaybackState.PLAYING)

    def stop(self) -> None:
        """
        Halts simulation playback.
        """
        with self._lock:
            if self._state != PlaybackState.STOPPED:
                self._set_state(PlaybackState.STOPPED)

    def reset(self) -> None:
        """
        Resets the simulation to the initial position (time 0, step 0, no active troops).
        """
        with self._lock:
            self._set_state(PlaybackState.IDLE)
            self._current_time_ms = 0
            self._step_index = 0
            self._deployed_step_indices.clear()
            self._active_troops.clear()

    def step_forward(self) -> Optional[DeploymentStep]:
        """
        Executes the next single deployment step immediately.
        Advances timeline and executes action via sandboxed controller.
        Returns the executed DeploymentStep or None if all steps are completed.
        """
        with self._lock:
            self._check_emergency_stop()
            if self._step_index >= len(self._steps):
                if self._state != PlaybackState.COMPLETED:
                    self._set_state(PlaybackState.COMPLETED)
                    self._notify_complete()
                return None

            next_idx = self._step_index
            scheduled_time = self._step_times[next_idx]
            self._current_time_ms = max(self._current_time_ms, scheduled_time)
            step = self._deploy_step(next_idx)

            if len(self._deployed_step_indices) >= len(self._steps) and self._current_time_ms >= self._total_duration_ms:
                self._set_state(PlaybackState.COMPLETED)
                self._notify_complete()

            self._notify_tick()
            return step

    def step_backward(self) -> None:
        """
        Rewinds simulation by one step index.
        """
        with self._lock:
            if self._step_index <= 0:
                return

            target_index = self._step_index - 1
            target_time = self._step_times[target_index - 1] if target_index > 0 else 0
            self._current_time_ms = target_time

            # Reconstruct deployed steps up to target_index
            self._deployed_step_indices.clear()
            self._active_troops.clear()
            for i in range(target_index):
                self._deploy_step(i, execute_controller=False)

            self._step_index = target_index
            if self._state == PlaybackState.COMPLETED:
                self._set_state(PlaybackState.PAUSED if self.is_paused else PlaybackState.STOPPED)

            self._notify_tick()

    # --------------------------------------------------------------------------
    # Timeline Progression (tick)
    # --------------------------------------------------------------------------
    def tick(self, delta_ms: int) -> SimulationState:
        """
        Advances the timeline by delta_ms scaled by the playback speed.
        Executes pending deployment steps, updates troop lifetimes, and checks completion.
        Raises EmergencyStopTriggered immediately if emergency stop is triggered.
        """
        with self._lock:
            if self._state != PlaybackState.PLAYING:
                return self.simulation_state

            self._check_emergency_stop()

            effective_delta = max(0, int(delta_ms * self._speed))
            self._current_time_ms += effective_delta
            if self._total_duration_ms > 0 and self._current_time_ms > self._total_duration_ms:
                self._current_time_ms = self._total_duration_ms

            # Deploy any steps whose scheduled time has arrived
            for i in range(len(self._steps)):
                if i not in self._deployed_step_indices:
                    if self._current_time_ms >= self._step_times[i]:
                        self._deploy_step(i)

            # Update lifetime of existing active troops
            for troop in self._active_troops:
                if troop.status == "ACTIVE":
                    troop.remaining_lifetime_ms -= effective_delta
                    if troop.remaining_lifetime_ms <= 0:
                        troop.status = "EXPIRED"
                        troop.remaining_lifetime_ms = 0

            # Check for playback completion
            all_deployed = len(self._deployed_step_indices) >= len(self._steps)
            time_elapsed = self._current_time_ms >= self._total_duration_ms
            if all_deployed and time_elapsed:
                self._set_state(PlaybackState.COMPLETED)
                self._notify_complete()

            self._notify_tick()
            return self.simulation_state

    # --------------------------------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------------------------------
    def _check_emergency_stop(self) -> None:
        """Checks emergency barrier status; halts playback immediately if tripped."""
        if self.barrier.is_emergency_stopped or self.guard.is_emergency_stopped:
            self.stop()
            raise EmergencyStopTriggered(
                "Emergency stop triggered; local simulation playback halted immediately."
            )

    def _deploy_step(self, step_idx: int, execute_controller: bool = True) -> DeploymentStep:
        """Deploys a step at step_idx and records its active troop marker."""
        step = self._steps[step_idx]
        self._deployed_step_indices.add(step_idx)
        self._step_index = max(self._step_index, step_idx + 1)

        marker = ActiveTroop(
            troop_id=f"troop_{step.step_id or (step_idx + 1)}_{step_idx + 1}",
            step_id=step.step_id or (step_idx + 1),
            troop_type=step.troop_type or "Troop",
            target_x=step.target_x,
            target_y=step.target_y,
            normalized_x=step.normalized_x,
            normalized_y=step.normalized_y,
            spawn_time_ms=self._step_times[step_idx],
            remaining_lifetime_ms=self.default_troop_ttl_ms,
            status="ACTIVE",
            unit_count=step.unit_count or 1,
            note=step.note,
        )
        self._active_troops.append(marker)

        # Execute simulated click via quarantined controller if attached
        if execute_controller and self.controller is not None:
            self._check_emergency_stop()
            try:
                self.controller.click(step.target_x, step.target_y)
            except EmergencyStopTriggered:
                self.stop()
                raise

        # Notify on_step callbacks
        callbacks = list(self._callbacks_step)
        for cb in callbacks:
            try:
                cb(step, marker.to_dict())
            except Exception as e:
                logger.error("Error in on_step callback: %s", e)

        return step

    def _notify_tick(self) -> None:
        state_snap = self.simulation_state
        for cb in list(self._callbacks_tick):
            try:
                cb(state_snap)
            except Exception as e:
                logger.error("Error in on_tick callback: %s", e)

    def _notify_complete(self) -> None:
        state_snap = self.simulation_state
        for cb in list(self._callbacks_complete):
            try:
                cb(state_snap)
            except Exception as e:
                logger.error("Error in on_complete callback: %s", e)

    # --------------------------------------------------------------------------
    # Callback Registration
    # --------------------------------------------------------------------------
    def register_on_tick(self, callback: Callable[[SimulationState], None]) -> None:
        if callback not in self._callbacks_tick:
            self._callbacks_tick.append(callback)

    def unregister_on_tick(self, callback: Callable[[SimulationState], None]) -> None:
        if callback in self._callbacks_tick:
            self._callbacks_tick.remove(callback)

    def register_on_step(self, callback: Callable[[DeploymentStep, Dict[str, Any]], None]) -> None:
        if callback not in self._callbacks_step:
            self._callbacks_step.append(callback)

    def unregister_on_step(self, callback: Callable[[DeploymentStep, Dict[str, Any]], None]) -> None:
        if callback in self._callbacks_step:
            self._callbacks_step.remove(callback)

    def register_on_complete(self, callback: Callable[[SimulationState], None]) -> None:
        if callback not in self._callbacks_complete:
            self._callbacks_complete.append(callback)

    def unregister_on_complete(self, callback: Callable[[SimulationState], None]) -> None:
        if callback in self._callbacks_complete:
            self._callbacks_complete.remove(callback)

    def register_on_state_change(self, callback: Callable[[PlaybackState, PlaybackState], None]) -> None:
        if callback not in self._callbacks_state_change:
            self._callbacks_state_change.append(callback)

    def unregister_on_state_change(self, callback: Callable[[PlaybackState, PlaybackState], None]) -> None:
        if callback in self._callbacks_state_change:
            self._callbacks_state_change.remove(callback)
