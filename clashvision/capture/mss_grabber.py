"""
ClashVision high-performance non-blocking screen capture engine.
Uses MSS for client-window frame grabbing with target FPS rate limiting,
minimized window auto-pausing, and synthetic frame fallback for offline tests.
"""

from __future__ import annotations
import time
import threading
import logging
from typing import Optional, Generator, Callable, Dict, Any
import numpy as np
import mss

from clashvision.core.constants import (
    DEFAULT_CAPTURE_FPS,
    VIRTUAL_REFERENCE_WIDTH,
    VIRTUAL_REFERENCE_HEIGHT,
)
from clashvision.core.models import (
    Rect,
    WindowInfo,
    FramePacket,
)
from clashvision.core.exceptions import (
    CaptureError,
    WindowNotFoundError,
    WindowMinimizedError,
)

logger = logging.getLogger("clashvision.capture.grabber")


class MSSGrabber:
    """
    High-performance screen grabber wrapping MSS.
    Captures visible client area of target window, converts to BGR ndarray,
    enforces target FPS, auto-pauses on minimized state, and supports offline synthetic frames.
    """

    def __init__(
        self,
        target_fps: int = DEFAULT_CAPTURE_FPS,
        auto_pause_on_minimized: bool = True,
        mock_mode: bool = False,
        backend: Optional[Any] = None,
        synthetic_frame_generator: Optional[Callable[[int, int, int], np.ndarray]] = None,
    ):
        self.target_fps = max(1, min(60, target_fps))
        self.auto_pause_on_minimized = auto_pause_on_minimized
        if backend is not None:
            backend_str = str(backend).upper()
            self.mock_mode = ("MOCK" in backend_str) or mock_mode
        else:
            self.mock_mode = mock_mode

        self._synthetic_generator = synthetic_frame_generator or self._default_synthetic_frame

        self._frame_counter: int = 0
        self._is_running: bool = False
        self._is_paused: bool = False
        self._last_grab_time: float = 0.0
        self._target_interval: float = 1.0 / self.target_fps
        self._lock = threading.Lock()
        self._sct: Optional[mss.mss] = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def frame_count(self) -> int:
        return self._frame_counter

    def set_target_fps(self, fps: int) -> None:
        """Dynamically adjusts target frame rate."""
        self.target_fps = max(1, min(60, fps))
        self._target_interval = 1.0 / self.target_fps

    def set_fps(self, fps: int) -> None:
        """Alias for set_target_fps."""
        self.set_target_fps(fps)

    def start(self) -> None:
        """Initializes the MSS context."""
        with self._lock:
            if not self._is_running:
                if not self.mock_mode:
                    try:
                        self._sct = mss.mss()
                    except Exception as e:
                        logger.warning("Could not initialize MSS screen grabber: %s. Falling back to mock mode.", e)
                        self.mock_mode = True
                self._is_running = True
                self._is_paused = False
                logger.info("MSSGrabber started (target_fps=%d, mock_mode=%s)", self.target_fps, self.mock_mode)

    def stop(self) -> None:
        """Releases MSS resources."""
        with self._lock:
            if self._is_running:
                self._is_running = False
                if self._sct is not None:
                    try:
                        self._sct.close()
                    except Exception:
                        pass
                    self._sct = None
                logger.info("MSSGrabber stopped. Total frames captured: %d", self._frame_counter)

    def pause(self) -> None:
        self._is_paused = True

    def resume(self) -> None:
        self._is_paused = False

    def grab_frame(self, window_info: Optional[WindowInfo]) -> Optional[FramePacket]:
        """
        Grabs a single frame from the target window.
        Returns a typed FramePacket, or None if paused/minimized.
        """
        if window_info is None:
            raise WindowNotFoundError("Cannot grab frame: window_info is None.")

        # Check minimized or invisible state
        if window_info.is_minimized or not window_info.is_capturable:
            if self.auto_pause_on_minimized:
                self._is_paused = True
                logger.debug("Window HWND=%d is minimized or not capturable; auto-paused capture.", window_info.hwnd)
                return None
            else:
                raise WindowMinimizedError(f"Target window HWND={window_info.hwnd} is minimized.")

        # Rate limiter / FPS pacer
        now = time.time()
        elapsed = now - self._last_grab_time
        if elapsed < self._target_interval:
            time.sleep(max(0.0, self._target_interval - elapsed))

        # Check mock mode or synthetic fallback
        if self.mock_mode:
            frame_bgr = self._synthetic_generator(
                window_info.client_rect.width or VIRTUAL_REFERENCE_WIDTH,
                window_info.client_rect.height or VIRTUAL_REFERENCE_HEIGHT,
                self._frame_counter + 1,
            )
        else:
            frame_bgr = self._capture_client_rect(window_info.client_rect)

        self._frame_counter += 1
        self._last_grab_time = time.time()

        return FramePacket(
            frame=frame_bgr,
            timestamp=self._last_grab_time,
            window_info=window_info,
            frame_id=self._frame_counter,
        )

    def _capture_client_rect(self, client_rect: Rect) -> np.ndarray:
        """Captures bounding box using MSS and converts BGRA to BGR ndarray."""
        if self._sct is None:
            self._sct = mss.mss()

        bbox = {
            "left": client_rect.x,
            "top": client_rect.y,
            "width": max(1, client_rect.width),
            "height": max(1, client_rect.height),
        }

        try:
            sct_img = self._sct.grab(bbox)
            raw = np.array(sct_img, dtype=np.uint8)
            # MSS returns BGRA; slice out the alpha channel for pure BGR
            return raw[:, :, :3]
        except Exception as e:
            raise CaptureError(f"MSS capture failed on bbox {bbox}: {e}") from e

    @staticmethod
    def _default_synthetic_frame(width: int, height: int, frame_id: int) -> np.ndarray:
        """
        Generates a valid BGR synthetic frame for offline unit and integration tests.
        Includes grass green field, HUD header, and simulated pixel landmarks.
        """
        w = max(32, width)
        h = max(32, height)
        # BGR: CoC grass base tone is roughly (45, 130, 60)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (45, 130, 60)

        # Draw a simulated darker border
        img[0:10, :, :] = (20, 80, 30)
        img[-10:, :, :] = (20, 80, 30)
        img[:, 0:10, :] = (20, 80, 30)
        img[:, -10:, :] = (20, 80, 30)

        # Draw simulated HUD bar in top right (resource area: yellow gold, pink elixir, dark purple DE)
        hud_h = max(20, h // 10)
        hud_w = max(100, w // 4)
        hud_x = w - hud_w - 20
        hud_y = 20
        if hud_x > 0 and hud_y + hud_h < h:
            img[hud_y : hud_y + hud_h, hud_x : hud_x + hud_w] = (30, 30, 30)
            # Gold indicator swatch (BGR: 0, 215, 255)
            img[hud_y + 4 : hud_y + 14, hud_x + 5 : hud_x + 15] = (0, 215, 255)
            # Elixir indicator swatch (BGR: 255, 0, 255)
            img[hud_y + 16 : hud_y + 26, hud_x + 5 : hud_x + 15] = (255, 0, 255)

        return img

    def stream(self, get_window_fn: Callable[[], Optional[WindowInfo]]) -> Generator[FramePacket, None, None]:
        """
        Generator producing continuous FramePackets until stopped.
        Non-blocking pacer throttles to target_fps.
        """
        self.start()
        try:
            while self._is_running:
                if self._is_paused:
                    time.sleep(0.1)
                    continue

                win = get_window_fn()
                if win is None:
                    time.sleep(0.2)
                    continue

                packet = self.grab_frame(win)
                if packet is not None:
                    yield packet
        finally:
            self.stop()
