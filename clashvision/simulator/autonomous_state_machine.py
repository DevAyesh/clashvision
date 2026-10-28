"""
ClashVision Autonomous Farming Engine — Event-Driven State Machine.
Defines the 15 formal states for offline simulation, transition validation matrices,
event deduplication, error tracking, and emergency-stop safety integration.
Derived from Master Prompt Section 3.
"""

from __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from clashvision.core.exceptions import EmergencyStopTriggered
from clashvision.simulator.safety_barrier import SafetyBarrier, get_safety_barrier

logger = logging.getLogger("clashvision.simulator.state_machine")


@unique
class FarmingState(str, Enum):
    """
    15 Formal States for the Autonomous Farming Simulator.
    Strictly validated transitions enforce predictable execution cycles.
    """
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    SEARCHING = "SEARCHING"
    ANALYZING_BASE = "ANALYZING_BASE"
    EVALUATING_LOOT = "EVALUATING_LOOT"
    SKIPPING_BASE = "SKIPPING_BASE"
    SELECTING_STRATEGY = "SELECTING_STRATEGY"
    PREPARING_ATTACK = "PREPARING_ATTACK"
    EXECUTING_ATTACK = "EXECUTING_ATTACK"
    WAITING_FOR_RESULT = "WAITING_FOR_RESULT"
    RECORDING_RESULTS = "RECORDING_RESULTS"
    RECOVERING = "RECOVERING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"

    @property
    def is_terminal(self) -> bool:
        """Whether this state represents an end of cycle or session."""
        return self in (FarmingState.STOPPED, FarmingState.ERROR)

    @property
    def is_active(self) -> bool:
        """Whether this state represents actively executing work."""
        return self in (
            FarmingState.SEARCHING,
            FarmingState.ANALYZING_BASE,
            FarmingState.EVALUATING_LOOT,
            FarmingState.SELECTING_STRATEGY,
            FarmingState.PREPARING_ATTACK,
            FarmingState.EXECUTING_ATTACK,
            FarmingState.WAITING_FOR_RESULT,
            FarmingState.RECORDING_RESULTS,
        )


class InvalidStateTransitionError(ValueError):
    """Raised when an unauthorized or invalid state transition is attempted."""
    pass


@dataclass
class TransitionRecord:
    """Audit log entry for state transitions."""
    from_state: FarmingState
    to_state: FarmingState
    timestamp: float = field(default_factory=time.time)
    trigger: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


# Authoritative State Transition Graph
VALID_TRANSITIONS: Dict[FarmingState, Set[FarmingState]] = {
    FarmingState.INITIALIZING: {
        FarmingState.READY,
        FarmingState.STOPPED,
        FarmingState.ERROR,
    },
    FarmingState.READY: {
        FarmingState.SEARCHING,
        FarmingState.PAUSED,
        FarmingState.STOPPED,
        FarmingState.ERROR,
    },
    FarmingState.SEARCHING: {
        FarmingState.ANALYZING_BASE,
        FarmingState.SKIPPING_BASE,
        FarmingState.RECOVERING,
        FarmingState.PAUSED,
        FarmingState.STOPPED,
        FarmingState.ERROR,
    },
    FarmingState.ANALYZING_BASE: {
        FarmingState.EVALUATING_LOOT,
        FarmingState.SKIPPING_BASE,
        FarmingState.RECOVERING,
        FarmingState.STOPPED,
        FarmingState.ERROR,
    },
    FarmingState.EVALUATING_LOOT: {
        FarmingState.SELECTING_STRATEGY,
        FarmingState.SKIPPING_BASE,
        FarmingState.RECOVERING,
        FarmingState.STOPPED,
        FarmingState.ERROR,
    },
    FarmingState.SKIPPING_BASE: {
        FarmingState.SEARCHING,
        FarmingState.RECOVERING,
        FarmingState.PAUSED,
        FarmingState.STOPPED,
        FarmingState.ERROR,
    },
    FarmingState.SELECTING_STRATEGY: {
        FarmingState.PREPARING_ATTACK,
        FarmingState.SKIPPING_BASE,
        FarmingState.RECOVERING,
        FarmingState.STOPPED,
        FarmingState.ERROR,
    },
    FarmingState.PREPARING_ATTACK: {
        FarmingState.EXECUTING_ATTACK,
        FarmingState.RECOVERING,
        FarmingState.STOPPED,
        FarmingState.ERROR,
    },
    FarmingState.EXECUTING_ATTACK: {
        FarmingState.WAITING_FOR_RESULT,
        FarmingState.RECOVERING,
        FarmingState.PAUSED,
        FarmingState.STOPPED,
        FarmingState.ERROR,
    },
    FarmingState.WAITING_FOR_RESULT: {
        FarmingState.RECORDING_RESULTS,
        FarmingState.RECOVERING,
        FarmingState.STOPPED,
        FarmingState.ERROR,
    },
    FarmingState.RECORDING_RESULTS: {
        FarmingState.READY,
        FarmingState.SEARCHING,
        FarmingState.STOPPED,
        FarmingState.ERROR,
    },
    FarmingState.RECOVERING: {
        FarmingState.READY,
        FarmingState.SEARCHING,
        FarmingState.STOPPED,
        FarmingState.ERROR,
    },
    FarmingState.PAUSED: {
        FarmingState.READY,
        FarmingState.SEARCHING,
        FarmingState.EXECUTING_ATTACK,
        FarmingState.STOPPED,
        FarmingState.ERROR,
    },
    FarmingState.STOPPED: {
        FarmingState.INITIALIZING,
        FarmingState.READY,
    },
    FarmingState.ERROR: {
        FarmingState.INITIALIZING,
        FarmingState.READY,
        FarmingState.STOPPED,
    },
}


class AutonomousStateMachine:
    """
    Thread-safe, event-driven state machine managing autonomous farming lifecycles.
    Enforces transition matrix, maintains audit history, eliminates duplicate transitions,
    and checks the hardware safety barrier on every step.
    """

    def __init__(
        self,
        barrier: Optional[SafetyBarrier] = None,
        initial_state: FarmingState = FarmingState.STOPPED,
    ) -> None:
        self.barrier = barrier or get_safety_barrier()
        self._lock = threading.RLock()
        self._state: FarmingState = initial_state
        self._previous_state: Optional[FarmingState] = None
        self._paused_return_state: Optional[FarmingState] = None
        self._history: List[TransitionRecord] = []
        self._listeners: List[Callable[[FarmingState, FarmingState, Dict[str, Any]], None]] = []
        self._error_count: int = 0
        self._last_transition_time: float = time.time()

    @property
    def current_state(self) -> FarmingState:
        with self._lock:
            return self._state

    @property
    def previous_state(self) -> Optional[FarmingState]:
        with self._lock:
            return self._previous_state

    @property
    def history(self) -> List[TransitionRecord]:
        with self._lock:
            return list(self._history)

    @property
    def error_count(self) -> int:
        with self._lock:
            return self._error_count

    def add_listener(
        self, listener: Callable[[FarmingState, FarmingState, Dict[str, Any]], None]
    ) -> None:
        """Register a transition callback."""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(
        self, listener: Callable[[FarmingState, FarmingState, Dict[str, Any]], None]
    ) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def can_transition_to(self, target: FarmingState) -> bool:
        """Evaluates whether transitioning to target state is legally permitted."""
        with self._lock:
            if target == self._state:
                return False  # Deduplicate identical transitions
            allowed = VALID_TRANSITIONS.get(self._state, set())
            return target in allowed

    def transition_to(
        self,
        target: FarmingState,
        trigger: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Executes a validated state transition.
        Raises:
            EmergencyStopTriggered: If hardware/software emergency stop is tripped.
            InvalidStateTransitionError: If the transition violates the state graph.
        """
        payload = payload or {}

        with self._lock:
            # Check safety barrier
            if self.barrier.is_emergency_stopped and target not in (FarmingState.STOPPED, FarmingState.ERROR):
                logger.error("Emergency stop is active; forcing transition to ERROR.")
                target = FarmingState.ERROR
                trigger = "emergency_stop"

            if target == self._state:
                logger.debug("Duplicate transition to %s suppressed.", target.value)
                return

            allowed = VALID_TRANSITIONS.get(self._state, set())
            if target not in allowed and target not in (FarmingState.ERROR, FarmingState.STOPPED):
                raise InvalidStateTransitionError(
                    f"Illegal transition from {self._state.value} to {target.value}. "
                    f"Permitted targets: {[s.value for s in allowed]}"
                )

            old_state = self._state

            if target == FarmingState.ERROR:
                self._error_count += 1

            if target == FarmingState.PAUSED:
                self._paused_return_state = old_state

            self._previous_state = old_state
            self._state = target
            now = time.time()
            self._last_transition_time = now

            record = TransitionRecord(
                from_state=old_state,
                to_state=target,
                timestamp=now,
                trigger=trigger,
                payload=payload,
            )
            self._history.append(record)
            logger.info("FarmingState: %s -> %s [trigger=%s]", old_state.value, target.value, trigger)

            # Fire listeners
            callbacks = list(self._listeners)

        for cb in callbacks:
            try:
                cb(old_state, target, payload)
            except Exception as ex:
                logger.error("Error in state transition listener callback: %s", ex)

    def pause(self, reason: str = "user_pause") -> None:
        """Pauses the engine, remembering pre-pause state."""
        self.transition_to(FarmingState.PAUSED, trigger=reason)

    def resume(self, trigger: str = "user_resume") -> None:
        """Resumes from PAUSED to the pre-pause state or READY."""
        with self._lock:
            target = self._paused_return_state or FarmingState.READY
            if target not in VALID_TRANSITIONS.get(FarmingState.PAUSED, set()):
                target = FarmingState.READY
        self.transition_to(target, trigger=trigger)

    def stop(self, reason: str = "user_stop") -> None:
        """Stops the engine safely."""
        self.transition_to(FarmingState.STOPPED, trigger=reason)

    def trigger_error(self, error_msg: str) -> None:
        """Transitions into ERROR state with error details."""
        self.transition_to(FarmingState.ERROR, trigger="error", payload={"error": error_msg})

    def reset(self) -> None:
        """Resets the state machine back to STOPPED."""
        with self._lock:
            self._state = FarmingState.STOPPED
            self._previous_state = None
            self._paused_return_state = None
            self._error_count = 0
            self._history.clear()
            self._last_transition_time = time.time()

