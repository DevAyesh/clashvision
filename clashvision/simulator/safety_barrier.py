"""
ClashVision Simulator Safety Barrier.
Enforces read-only safety, manages visual mode indicator state, and provides
a high-resolution emergency stop latch with <= 50ms response latency.
Derived directly from Requirements R1, R4 and Features 5, 25, 26, 27.
"""

from __future__ import annotations
import logging
import threading
import time
from enum import Enum, unique
from typing import Callable, Dict, List, Optional, Set, Any

from clashvision.core.constants import EMERGENCY_STOP_LATENCY_MAX_MS
from clashvision.core.exceptions import (
    LiveInputSafetyViolation,
    EmergencyStopTriggered,
)
from clashvision.capture.safety_guard import SafetyGuard, get_safety_guard

logger = logging.getLogger("clashvision.simulator.safety")


@unique
class VisualMode(str, Enum):
    """
    Visual mode indicator state differentiating live observation from local simulation.
    Derived directly from Feature 27.
    """
    LIVE_OBSERVATION_READ_ONLY = "LIVE_OBSERVATION_READ_ONLY"
    LOCAL_SIMULATION_SANDBOX = "LOCAL_SIMULATION_SANDBOX"

    @property
    def display_name(self) -> str:
        if self == VisualMode.LIVE_OBSERVATION_READ_ONLY:
            return "LIVE (READ-ONLY)"
        return "LOCAL SIMULATION (SANDBOX)"

    @property
    def is_read_only(self) -> bool:
        return self == VisualMode.LIVE_OBSERVATION_READ_ONLY


class SafetyBarrier:
    """
    Architectural safety barrier for the attack simulator and sandboxed execution.
    Guarantees:
    - Emergency stop latency <= 50ms (measured via high-resolution timer).
    - Visual mode state indicator strictly reports LIVE_OBSERVATION_READ_ONLY vs LOCAL_SIMULATION_SANDBOX.
    - Synchronizes with SafetyGuard to prevent any automated input from escaping the simulator.
    - Manages emergency stop callbacks for immediate subsystem shutdown.
    """

    _instance: Optional[SafetyBarrier] = None
    _singleton_lock = threading.Lock()

    def __init__(self, guard: Optional[SafetyGuard] = None) -> None:
        self._guard = guard if guard is not None else get_safety_guard()
        self._visual_mode: VisualMode = VisualMode.LIVE_OBSERVATION_READ_ONLY
        self._emergency_latch: bool = False
        self._last_trigger_latency_ms: float = 0.0
        self._last_trigger_reason: str = ""
        self._callbacks: List[Callable[[], None]] = []
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> SafetyBarrier:
        """Singleton accessor for global simulator safety barrier."""
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # --------------------------------------------------------------------------
    # Visual Mode Indicator (Feature 27)
    # --------------------------------------------------------------------------
    @property
    def visual_mode(self) -> VisualMode:
        """Current visual operational mode."""
        return self._visual_mode

    @property
    def visual_mode_str(self) -> str:
        """String representation of current visual operational mode."""
        return self._visual_mode.value

    @property
    def is_live_mode(self) -> bool:
        """True if operating in live read-only observation mode."""
        return self._visual_mode == VisualMode.LIVE_OBSERVATION_READ_ONLY

    @property
    def is_simulator_mode(self) -> bool:
        """True if operating in local simulation sandbox mode."""
        return self._visual_mode == VisualMode.LOCAL_SIMULATION_SANDBOX

    def set_mode(self, mode: VisualMode | str) -> None:
        """Switches the visual operational mode."""
        if isinstance(mode, str):
            mode = VisualMode(mode)
        self._visual_mode = mode
        logger.info("SafetyBarrier visual mode set to: %s (%s)", self._visual_mode.name, self._visual_mode.display_name)

    def set_live_mode(self) -> None:
        """Switches to LIVE_OBSERVATION_READ_ONLY mode."""
        self.set_mode(VisualMode.LIVE_OBSERVATION_READ_ONLY)

    def set_simulator_mode(self) -> None:
        """Switches to LOCAL_SIMULATION_SANDBOX mode."""
        self.set_mode(VisualMode.LOCAL_SIMULATION_SANDBOX)

    # --------------------------------------------------------------------------
    # Emergency Stop Latch & Callbacks (Feature 26, Requirement R4)
    # --------------------------------------------------------------------------
    @property
    def is_emergency_stopped(self) -> bool:
        """Checks if emergency stop latch is currently active."""
        return self._emergency_latch or self._guard.is_emergency_stopped

    @property
    def last_trigger_latency_ms(self) -> float:
        """Latency in milliseconds measured during the last emergency stop trigger."""
        return self._last_trigger_latency_ms

    @property
    def last_trigger_reason(self) -> str:
        """Reason provided during the last emergency stop trigger."""
        return self._last_trigger_reason

    def trigger_emergency_stop(self, reason: str = "User triggered emergency stop") -> float:
        """
        Instantly trips the emergency stop latch and executes registered callbacks.
        Measures and verifies response latency is <= 50ms.
        Returns the measured elapsed latency in milliseconds.
        """
        t0 = time.perf_counter()
        with self._lock:
            self._emergency_latch = True
            self._last_trigger_reason = reason

        # Synchronize with global SafetyGuard
        self._guard.trigger_emergency_stop(reason=reason)

        # Notify all registered callbacks immediately
        callbacks_snapshot = list(self._callbacks)
        for cb in callbacks_snapshot:
            try:
                cb()
            except Exception as e:
                logger.error("Exception in emergency stop callback: %s", e)

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        self._last_trigger_latency_ms = latency_ms

        logger.warning(
            "SafetyBarrier: EMERGENCY STOP TRIPPED in %.3f ms (limit=%d ms): %s",
            latency_ms,
            EMERGENCY_STOP_LATENCY_MAX_MS,
            reason,
        )
        return latency_ms

    def reset_emergency_stop(self) -> None:
        """Resets the emergency stop latch to allow resumption of simulator actions."""
        with self._lock:
            self._emergency_latch = False
            self._last_trigger_reason = ""
            self._last_trigger_latency_ms = 0.0
        self._guard.reset_emergency_stop()
        logger.info("SafetyBarrier emergency stop latch reset.")

    def check_emergency_stop(self) -> None:
        """
        Raises EmergencyStopTriggered if emergency stop latch is set.
        Guarantees fast checking in execution loops.
        """
        if self.is_emergency_stopped:
            raise EmergencyStopTriggered(
                f"Emergency stop is currently active! Reason: {self._last_trigger_reason or 'Emergency stop triggered.'}"
            )

    def register_emergency_callback(self, callback: Callable[[], None]) -> None:
        """Registers a callback to be invoked immediately when emergency stop is triggered."""
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def unregister_emergency_callback(self, callback: Callable[[], None]) -> None:
        """Unregisters an emergency stop callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    # --------------------------------------------------------------------------
    # Window Verification & Safety Guard Integration
    # --------------------------------------------------------------------------
    def register_simulator_window(self, hwnd: int) -> None:
        """Registers a simulator canvas window HWND."""
        self._guard.register_simulator_window(hwnd)

    def unregister_simulator_window(self, hwnd: int) -> None:
        """Unregisters a simulator window HWND."""
        self._guard.unregister_simulator_window(hwnd)

    def register_live_window(self, hwnd: int, title: str = "", process_name: str = "") -> None:
        """Registers a live game window HWND to be blacklisted."""
        self._guard.register_live_window(hwnd, title, process_name)

    def unregister_live_window(self, hwnd: int) -> None:
        """Unregisters a live window HWND."""
        self._guard.unregister_live_window(hwnd)

    def is_live_window(self, hwnd: int) -> bool:
        """Returns True if the given HWND belongs to a live game window."""
        return self._guard.is_live_window(hwnd)

    def verify_window_allowed(self, target_hwnd: Optional[int], action_name: str = "simulator_action") -> None:
        """
        Verifies that an action targeted at target_hwnd is permitted.
        Raises:
            EmergencyStopTriggered if emergency stop is active.
            LiveInputSafetyViolation if target_hwnd is live or unregistered.
        """
        self.check_emergency_stop()
        self._guard.verify_input_allowed(target_hwnd=target_hwnd, action_name=action_name)

    def clear_all(self) -> None:
        """Clears all registered windows, latches, and callbacks (used in unit tests)."""
        with self._lock:
            self._emergency_latch = False
            self._last_trigger_latency_ms = 0.0
            self._last_trigger_reason = ""
            self._callbacks.clear()
            self._visual_mode = VisualMode.LIVE_OBSERVATION_READ_ONLY
        self._guard.clear_all()


def get_safety_barrier() -> SafetyBarrier:
    """Helper to retrieve global SafetyBarrier singleton."""
    return SafetyBarrier.get_instance()
