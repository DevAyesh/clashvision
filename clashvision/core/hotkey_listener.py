"""clashvision/core/hotkey_listener.py
Global keyboard input listener for Emergency Stop.
Provides Windows Win32 asynchronous key-state detection and thread-safe callback invocation.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# Virtual-Key codes on Windows
VK_ESCAPE = 0x1B
VK_F12 = 0x7B
VK_PAUSE = 0x13


class GlobalHotkeyListener:
    """Monitors global keyboard shortcuts (e.g., ESC, F12) to trigger an immediate Emergency Stop.

    Works on Windows via ctypes.windll.user32.GetAsyncKeyState.
    Thread-safe and independent of Qt to avoid circular dependencies.
    """

    def __init__(
        self,
        poll_interval_s: float = 0.03,
        key_codes: Optional[List[int]] = None,
    ) -> None:
        self.poll_interval_s = poll_interval_s
        self.key_codes = key_codes or [VK_ESCAPE, VK_F12]
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[str], None]] = []
        self._is_windows = sys.platform == "win32"
        self._key_down_states = {k: False for k in self.key_codes}

    def register_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback to be invoked upon emergency stop trigger."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def start(self) -> None:
        """Starts the background monitoring thread."""
        if self._is_running or not self._is_windows:
            return

        self._is_running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="GlobalHotkeyListenerThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("GlobalHotkeyListener started (monitoring keys: %s).", self.key_codes)

    def stop(self) -> None:
        """Stops the background monitoring thread."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.2)
        self._thread = None
        logger.info("GlobalHotkeyListener stopped.")

    def _poll_loop(self) -> None:
        """Polling loop inspecting key states via GetAsyncKeyState on Windows."""
        import ctypes

        get_async_key = ctypes.windll.user32.GetAsyncKeyState

        while self._is_running:
            try:
                for vk in self.key_codes:
                    state = get_async_key(vk)
                    is_down = bool(state & 0x8000)

                    # Trigger on key-down edge transition (prevents repeated triggers while held)
                    if is_down and not self._key_down_states.get(vk, False):
                        self._key_down_states[vk] = True
                        key_name = "ESCAPE" if vk == VK_ESCAPE else ("F12" if vk == VK_F12 else f"VK_{vk:#x}")
                        logger.warning("Global emergency stop hotkey [%s] pressed!", key_name)

                        # Invoke registered callbacks
                        for cb in self._callbacks:
                            try:
                                cb(key_name)
                            except Exception as ex:
                                logger.error("Error executing emergency stop callback: %s", ex)
                    elif not is_down and self._key_down_states.get(vk, False):
                        self._key_down_states[vk] = False

                time.sleep(self.poll_interval_s)
            except Exception as ex:
                logger.error("Error in GlobalHotkeyListener loop: %s", ex)
                time.sleep(0.1)
