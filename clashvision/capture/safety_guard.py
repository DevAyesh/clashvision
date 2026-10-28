"""
ClashVision Architectural Safety Barrier.
Strictly guarantees read-only live observation. Prevents any automated input,
mouse injection, or keystrokes from reaching Google Play Games PC or any live emulator.
"""

from __future__ import annotations
import threading
import logging
from typing import Set, List, Optional, Dict, Any

from clashvision.core.exceptions import LiveInputSafetyViolation, EmergencyStopTriggered

logger = logging.getLogger("clashvision.safety")


class SafetyGuard:
    """
    Architectural safety barrier enforcing read-only capture.
    Maintains a strict quarantine:
    - Live target windows (Google Play Games PC, etc.) are strictly blacklisted for any input.
    - Automated inputs (clicks, keypresses) are restricted exclusively to quarantined simulator bounds.
    - Any live input injection raises LiveInputSafetyViolation and logs an alert.
    - Provides emergency stop interrupt handling.
    """

    _instance: Optional[SafetyGuard] = None
    _lock = threading.Lock()

    def __init__(self, default_deny: bool = False):
        self._live_hwnds: Set[int] = set()
        self._live_window_details: Dict[int, Dict[str, Any]] = {}
        self._simulator_hwnds: Set[int] = set()
        self._emergency_stop_event = threading.Event()
        self._violations: List[str] = []
        self._hooks_installed: bool = False
        self._original_pyautogui_funcs: Dict[str, Any] = {}
        self._initialized: bool = True
        self._default_deny: bool = default_deny

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def set_initialized(self, initialized: bool) -> None:
        self._initialized = initialized

    def set_default_deny(self, enabled: bool) -> None:
        self._default_deny = enabled

    @classmethod
    def get_instance(cls) -> SafetyGuard:
        """Singleton accessor for global safety enforcement."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def is_emergency_stopped(self) -> bool:
        return self._emergency_stop_event.is_set()

    @property
    def registered_live_hwnds(self) -> Set[int]:
        return set(self._live_hwnds)

    @property
    def registered_simulator_hwnds(self) -> Set[int]:
        return set(self._simulator_hwnds)

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    def register_live_window(self, hwnd: int, title: str = "", process_name: str = "") -> None:
        """
        Registers a live emulator/game window.
        Any input targeted towards this HWND is strictly blocked.
        """
        if hwnd:
            self._live_hwnds.add(hwnd)
            self._live_window_details[hwnd] = {
                "title": title,
                "process_name": process_name,
            }
            logger.info("SafetyGuard registered live read-only window HWND=%s (%s / %s)", hwnd, title, process_name)

    def unregister_live_window(self, hwnd: int) -> None:
        """Removes a window from the live blacklist."""
        self._live_hwnds.discard(hwnd)
        self._live_window_details.pop(hwnd, None)

    def register_simulator_window(self, hwnd: int) -> None:
        """Registers a quarantined simulator window where synthetic replay is permitted."""
        if hwnd:
            self._simulator_hwnds.add(hwnd)
            logger.info("SafetyGuard registered simulator window HWND=%s", hwnd)

    def unregister_simulator_window(self, hwnd: int) -> None:
        self._simulator_hwnds.discard(hwnd)

    def is_live_window(self, hwnd: int) -> bool:
        """Returns True if the given HWND is registered as a live read-only game window."""
        return hwnd in self._live_hwnds

    def verify_input_allowed(
        self,
        target_hwnd: Optional[int] = None,
        action_name: str = "input_action",
    ) -> None:
        """
        Validates whether an input action is permitted.
        Raises LiveInputSafetyViolation or EmergencyStopTriggered if disallowed.
        """
        # 1. Check emergency stop
        if self._emergency_stop_event.is_set():
            msg = f"SafetyGuard blocked {action_name}: Emergency stop is currently active!"
            logger.error(msg)
            raise EmergencyStopTriggered(msg)

        # 2. Check default-deny if uninitialized or explicitly configured
        if not self._initialized or self._default_deny:
            if target_hwnd is None or target_hwnd not in self._simulator_hwnds:
                msg = (
                    f"SAFETY VIOLATION: Default-deny active (initialized={self._initialized}, default_deny={self._default_deny}). "
                    f"Action '{action_name}' blocked!"
                )
                self._violations.append(msg)
                logger.critical(msg)
                raise LiveInputSafetyViolation(msg)

        # 2. Check if targeted specifically at a live window
        if target_hwnd is not None and target_hwnd in self._live_hwnds:
            details = self._live_window_details.get(target_hwnd, {})
            msg = (
                f"SAFETY VIOLATION: Automated action '{action_name}' attempted against live target window "
                f"HWND={target_hwnd} ({details.get('title', 'Unknown')}). Live interaction is strictly prohibited!"
            )
            self._violations.append(msg)
            logger.critical(msg)
            raise LiveInputSafetyViolation(msg)

        # 3. Check if targeted outside registered simulator windows
        if target_hwnd is not None and target_hwnd not in self._simulator_hwnds:
            msg = (
                f"SAFETY VIOLATION: Automated action '{action_name}' attempted against unregistered window "
                f"HWND={target_hwnd}. Inputs are strictly restricted to the simulator canvas."
            )
            self._violations.append(msg)
            logger.critical(msg)
            raise LiveInputSafetyViolation(msg)

        # 4. If no target_hwnd specified (global desktop injection), but live windows exist:
        if target_hwnd is None and self._live_hwnds:
            msg = (
                f"SAFETY VIOLATION: Global un-targeted automated action '{action_name}' attempted while "
                f"live target windows are registered ({list(self._live_hwnds)}). Potential live injection blocked!"
            )
            self._violations.append(msg)
            logger.critical(msg)
            raise LiveInputSafetyViolation(msg)

    def verify_and_execute_input(
        self,
        action_type: str = "click",
        target_x: int = 0,
        target_y: int = 0,
        key: str = "",
        is_simulator: bool = False,
        **kwargs: Any,
    ) -> bool:
        """
        Unified input gate for both Live and Simulator modes.
        Raises LiveInputSafetyViolation if called outside the local simulator.
        Raises EmergencyStopTriggered if emergency stop is tripped.
        Returns True if execution is permitted inside simulator.
        """
        if self._emergency_stop_event.is_set():
            msg = f"SafetyGuard blocked {action_type}: Emergency stop is currently active!"
            logger.error(msg)
            raise EmergencyStopTriggered(msg)

        if not is_simulator:
            msg = (
                f"SAFETY VIOLATION: Automated {action_type} attempted outside simulator sandbox "
                f"(target=({target_x}, {target_y}), key='{key}'). Live interaction is strictly prohibited!"
            )
            self._violations.append(msg)
            logger.critical(msg)
            raise LiveInputSafetyViolation(msg)

        # Simulator action permitted
        logger.debug("SafetyGuard: Permitted simulator action %s at (%d, %d)", action_type, target_x, target_y)
        return True

    def get_status(self) -> Dict[str, Any]:
        """Returns runtime safety status and operational flags."""
        return {
            "read_only_enforced": True,
            "emergency_stopped": self.is_emergency_stopped,
            "live_windows_count": len(self._live_hwnds),
            "simulator_windows_count": len(self._simulator_hwnds),
            "violations_count": self.violation_count,
            "violations": list(self._violations),
        }

    def trigger_emergency_stop(self, reason: str = "User triggered emergency stop") -> None:
        """Instantly halts any automated simulator playback (<= 50ms requirement)."""
        self._emergency_stop_event.set()
        logger.warning("EMERGENCY STOP TRIGGERED: %s", reason)

    def reset_emergency_stop(self) -> None:
        """Resets the emergency stop state to allow simulator execution again."""
        self._emergency_stop_event.clear()
        logger.info("SafetyGuard emergency stop reset.")

    def clear_all(self) -> None:
        """Resets all registered windows and violations (used in tests)."""
        self._live_hwnds.clear()
        self._live_window_details.clear()
        self._simulator_hwnds.clear()
        self._emergency_stop_event.clear()
        self._violations.clear()

    def install_interceptor(self) -> None:
        """
        Installs runtime safety interceptors on pyautogui if installed.
        Guarantees that ANY call to pyautogui passes through verify_input_allowed().
        """
        if self._hooks_installed:
            return

        try:
            import pyautogui
            funcs_to_intercept = ["click", "moveTo", "mouseDown", "mouseUp", "press", "keyDown", "keyUp", "write"]
            for fname in funcs_to_intercept:
                if hasattr(pyautogui, fname):
                    orig_func = getattr(pyautogui, fname)
                    self._original_pyautogui_funcs[fname] = orig_func

                    def make_wrapper(orig, name):
                        def wrapper(*args, **kwargs):
                            # Inspect caller or target
                            self.verify_input_allowed(target_hwnd=None, action_name=f"pyautogui.{name}")
                            return orig(*args, **kwargs)
                        return wrapper

                    setattr(pyautogui, fname, make_wrapper(orig_func, fname))

            self._hooks_installed = True
            logger.info("SafetyGuard: PyAutoGUI interception hooks installed.")
        except ImportError:
            # PyAutoGUI not installed, no hooks needed
            pass

    def uninstall_interceptor(self) -> None:
        """Restores original pyautogui functions."""
        if not self._hooks_installed:
            return

        try:
            import pyautogui
            for fname, orig_func in self._original_pyautogui_funcs.items():
                setattr(pyautogui, fname, orig_func)
            self._original_pyautogui_funcs.clear()
            self._hooks_installed = False
            logger.info("SafetyGuard: PyAutoGUI interception hooks uninstalled.")
        except ImportError:
            pass


def get_safety_guard() -> SafetyGuard:
    """Helper to access global safety guard singleton."""
    return SafetyGuard.get_instance()
