"""
ClashVision Windows process & window detector with Win32 DPI awareness,
process enumeration, target matching for Google Play Games PC, and offline mock support.
"""

from __future__ import annotations
import sys
import os
import re
import ctypes
from ctypes import wintypes
import logging
from typing import List, Optional, Tuple, Pattern

from clashvision.core.constants import (
    DEFAULT_TARGET_PROCESS_NAMES,
    DEFAULT_TARGET_WINDOW_TITLES,
    VIRTUAL_REFERENCE_WIDTH,
    VIRTUAL_REFERENCE_HEIGHT,
)
from clashvision.core.models import Rect, WindowInfo, Point
from clashvision.capture.safety_guard import get_safety_guard

logger = logging.getLogger("clashvision.capture.detector")

# Try importing psutil for high-fidelity process name resolution
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# Win32 Structures
class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class POINT(ctypes.Structure):
    _fields_ = [
        ("x", wintypes.LONG),
        ("y", wintypes.LONG),
    ]


# DPI Context Constants
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)
DWMWA_EXTENDED_FRAME_BOUNDS = 9


class WindowDetector:
    """
    Win32 and mock window detection engine.
    Finds and tracks Google Play Games PC or emulator windows.
    Automatically registers detected live game windows with SafetyGuard.
    """

    def __init__(
        self,
        target_process_names: Optional[List[str]] = None,
        target_window_titles: Optional[List[str]] = None,
        mock_mode: bool = False,
    ):
        self.target_process_names = target_process_names or list(DEFAULT_TARGET_PROCESS_NAMES)
        self.target_window_titles = target_window_titles or list(DEFAULT_TARGET_WINDOW_TITLES)
        self.compiled_title_regexes: List[Pattern] = [
            re.compile(t, re.IGNORECASE) for t in self.target_window_titles
        ]
        self.mock_mode = mock_mode
        self._mock_windows: List[WindowInfo] = []
        self._dpi_initialized = False

        if sys.platform == "win32" and not self.mock_mode:
            self.init_dpi_awareness()

    def init_dpi_awareness(self) -> bool:
        """
        Enables Per-Monitor V2 DPI awareness on Windows to prevent blurry captures
        and incorrect coordinate scaling.
        """
        if self._dpi_initialized or sys.platform != "win32":
            return self._dpi_initialized

        try:
            # Try SetProcessDpiAwarenessContext (Windows 10 1703+)
            user32 = ctypes.windll.user32
            if hasattr(user32, "SetProcessDpiAwarenessContext"):
                res = user32.SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
                if res:
                    self._dpi_initialized = True
                    logger.debug("Win32 Per-Monitor V2 DPI awareness enabled.")
                    return True

            # Fallback to SetProcessDpiAwareness (Windows 8.1+)
            shcore = ctypes.windll.shcore
            if hasattr(shcore, "SetProcessDpiAwareness"):
                shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
                self._dpi_initialized = True
                logger.debug("Win32 Per-Monitor DPI awareness enabled via SHCore.")
                return True

            # Fallback to SetProcessDPIAware (Vista+)
            if hasattr(user32, "SetProcessDPIAware"):
                user32.SetProcessDPIAware()
                self._dpi_initialized = True
                logger.debug("Win32 System DPI awareness enabled.")
                return True

        except Exception as e:
            logger.warning("Failed to initialize DPI awareness: %s", e)

        return False

    def register_mock_window(self, window_info: WindowInfo) -> None:
        """Registers a mock window for testing in offline/mock environments."""
        self._mock_windows.append(window_info)

    def clear_mock_windows(self) -> None:
        self._mock_windows.clear()

    def set_mock_mode(self, enabled: bool) -> None:
        self.mock_mode = enabled

    def enumerate_windows(self) -> List[WindowInfo]:
        """
        Enumerates all visible top-level windows.
        In mock_mode or non-windows, returns registered mock windows.
        """
        if self.mock_mode or sys.platform != "win32":
            return list(self._mock_windows)

        windows: List[WindowInfo] = []
        user32 = ctypes.windll.user32

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def enum_windows_callback(hwnd: int, lparam: int) -> bool:
            if not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd):
                return True

            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True

            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value.strip()

            if not title:
                return True

            info = self.get_window_info(hwnd, title)
            if info:
                windows.append(info)

            return True

        try:
            callback = WNDENUMPROC(enum_windows_callback)
            user32.EnumWindows(callback, 0)
        except Exception as e:
            logger.error("Error during window enumeration: %s", e)

        return windows

    def get_window_info(self, hwnd: int, known_title: Optional[str] = None) -> Optional[WindowInfo]:
        """Queries Win32 API for full geometry, DPI, process name, and visibility of a window."""
        if self.mock_mode or sys.platform != "win32":
            for m in self._mock_windows:
                if m.hwnd == hwnd:
                    return m
            return None

        user32 = ctypes.windll.user32
        if not user32.IsWindow(hwnd):
            return None

        # 1. Title
        title = known_title
        if title is None:
            length = user32.GetWindowTextLengthW(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value.strip()

        # 2. Process Name
        process_name = ""
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if HAS_PSUTIL and pid.value > 0:
            try:
                proc = psutil.Process(pid.value)
                process_name = proc.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # 3. Visibility and Minimized State
        is_visible = bool(user32.IsWindowVisible(hwnd))
        is_minimized = bool(user32.IsIconic(hwnd))
        is_focused = (hwnd == user32.GetForegroundWindow())

        # 4. Client Rect in Desktop Coordinates
        client_rect_win = RECT()
        user32.GetClientRect(hwnd, ctypes.byref(client_rect_win))
        client_origin = POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(client_origin))

        client_w = client_rect_win.right - client_rect_win.left
        client_h = client_rect_win.bottom - client_rect_win.top
        client_rect = Rect(
            x=client_origin.x,
            y=client_origin.y,
            width=max(0, client_w),
            height=max(0, client_h),
        )

        # 5. Window Outer Rect (via DWM if available for accurate window borders)
        window_rect_win = RECT()
        dwm_success = False
        try:
            dwmapi = ctypes.windll.dwmapi
            if hasattr(dwmapi, "DwmGetWindowAttribute"):
                hr = dwmapi.DwmGetWindowAttribute(
                    hwnd,
                    DWMWA_EXTENDED_FRAME_BOUNDS,
                    ctypes.byref(window_rect_win),
                    ctypes.sizeof(window_rect_win),
                )
                dwm_success = (hr == 0)
        except Exception:
            dwm_success = False

        if not dwm_success:
            user32.GetWindowRect(hwnd, ctypes.byref(window_rect_win))

        window_rect = Rect(
            x=window_rect_win.left,
            y=window_rect_win.top,
            width=max(0, window_rect_win.right - window_rect_win.left),
            height=max(0, window_rect_win.bottom - window_rect_win.top),
        )

        # 6. DPI and Scale Factor
        dpi = 96
        try:
            if hasattr(user32, "GetDpiForWindow"):
                dpi = user32.GetDpiForWindow(hwnd) or 96
        except Exception:
            dpi = 96

        scale_factor = round(dpi / 96.0, 2)

        return WindowInfo(
            hwnd=hwnd,
            title=title,
            process_name=process_name,
            client_rect=client_rect,
            window_rect=window_rect,
            is_visible=is_visible,
            is_minimized=is_minimized,
            is_focused=is_focused,
            scale_factor=scale_factor,
            dpi=dpi,
        )

    def find_target_window(self) -> Optional[WindowInfo]:
        """
        Searches for the target Google Play Games PC / emulator window.
        Uses process name matching, title substring/regex matching,
        and fallback heuristics.
        Automatically registers the matched window with SafetyGuard.
        """
        # If in mock mode, return registered mock window
        if self.mock_mode:
            if self._mock_windows:
                w = self._mock_windows[0]
                self._register_with_safety_guard(w)
                return w
            return None

        windows = self.enumerate_windows()

        # Filter windows with visible geometry
        capturable = [
            w for w in windows
            if w.is_visible and not w.is_minimized and w.client_rect.width >= 200 and w.client_rect.height >= 200
        ]

        # Priority 1: Title contains "Clash of Clans"
        coc_pattern = re.compile(r"Clash\s+of\s+Clans", re.IGNORECASE)
        for w in capturable:
            if w.title and coc_pattern.search(w.title):
                self._register_with_safety_guard(w)
                return w

        # Priority 2: Other emulator title matches (Google Play Games, BlueStacks, LDPlayer, Nox)
        for w in capturable:
            if w.title and any(pattern.search(w.title) for pattern in self.compiled_title_regexes):
                self._register_with_safety_guard(w)
                return w

        # Priority 3: Process name match
        for w in capturable:
            if w.process_name and any(
                p.lower() in (w.process_name.lower(), f"{w.process_name.lower()}.exe")
                for p in self.target_process_names
            ):
                self._register_with_safety_guard(w)
                return w

        # Priority 4: Broad check across any window with Clash of Clans title
        for w in windows:
            if w.title and coc_pattern.search(w.title):
                self._register_with_safety_guard(w)
                return w

        return None

    def create_mock_target(
        self,
        title: str = "Clash of Clans (Mock)",
        process_name: str = "PlayGamesPC.exe",
        width: int = VIRTUAL_REFERENCE_WIDTH,
        height: int = VIRTUAL_REFERENCE_HEIGHT,
        is_minimized: bool = False,
    ) -> WindowInfo:
        """Helper to create and register a synthetic mock target window."""
        mock_win = WindowInfo(
            hwnd=99991,
            title=title,
            process_name=process_name,
            client_rect=Rect(100, 100, width, height),
            window_rect=Rect(92, 68, width + 16, height + 40),
            is_visible=True,
            is_minimized=is_minimized,
            is_focused=True,
            scale_factor=1.0,
            dpi=96,
        )
        self.register_mock_window(mock_win)
        self._register_with_safety_guard(mock_win)
        return mock_win

    def _register_with_safety_guard(self, win: WindowInfo) -> None:
        """Ensures every detected live target is immediately blacklisted for input injection."""
        guard = get_safety_guard()
        guard.register_live_window(
            hwnd=win.hwnd,
            title=win.title,
            process_name=win.process_name,
        )
