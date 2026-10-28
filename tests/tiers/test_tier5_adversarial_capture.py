"""
Tier 5: Adversarial Stress Tests & Empirical Verification for Capture Subsystem.
Challenger 2 Verification Suite for Milestone 1.

Tests:
1. High-frequency start/stop cycles & multithreaded termination.
2. Frame rate pacing benchmark & jitter analysis across multiple FPS targets.
3. Boundary & extreme coordinate handling (zero/negative sizes, synthetic generator).
4. Window state transitions: Normal -> Minimized -> Restored -> Invisible -> Restored.
5. Stream generator behavior and auto-pause recovery.
6. WindowDetector non-existent HWNDs, invalid titles, DPI scaling, and concurrent enumeration.
"""

from __future__ import annotations

import time
import threading
from typing import List, Optional
import numpy as np
import pytest

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
from clashvision.capture.mss_grabber import MSSGrabber
from clashvision.capture.window_detector import WindowDetector
from clashvision.capture.safety_guard import get_safety_guard


class TestMSSGrabberAdversarialCycles:
    """Stress testing rapid start/stop cycles and thread termination."""

    def test_rapid_sequential_start_stop_cycles(self) -> None:
        """Rapidly start and stop grabber 50 times in sequence."""
        grabber = MSSGrabber(target_fps=30, mock_mode=True)
        for i in range(50):
            grabber.start()
            assert grabber.is_running is True
            grabber.stop()
            assert grabber.is_running is False
        assert grabber.frame_count == 0

    def test_concurrent_start_stop_stress(self) -> None:
        """Multiple threads concurrently calling start, stop, pause, resume."""
        grabber = MSSGrabber(target_fps=30, mock_mode=True)
        errors: List[Exception] = []

        def worker(action: str) -> None:
            try:
                for _ in range(30):
                    if action == "start":
                        grabber.start()
                    elif action == "stop":
                        grabber.stop()
                    elif action == "pause":
                        grabber.pause()
                    elif action == "resume":
                        grabber.resume()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(act,))
            for act in ["start", "stop", "pause", "resume", "start", "stop"]
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)
            assert not t.is_alive(), "Worker thread hung during concurrent start/stop"

        assert len(errors) == 0, f"Errors encountered during concurrent start/stop: {errors}"
        grabber.stop()

    def test_stream_clean_thread_termination(self) -> None:
        """Stream running in worker thread must terminate within 500ms when stop() is called."""
        grabber = MSSGrabber(target_fps=20, mock_mode=True)
        win = WindowInfo(
            hwnd=1001,
            client_rect=Rect(0, 0, 800, 600),
            is_visible=True,
            is_minimized=False,
        )

        frames_collected: List[FramePacket] = []
        stopped_flag = threading.Event()

        def stream_consumer() -> None:
            for packet in grabber.stream(lambda: win):
                frames_collected.append(packet)
                if len(frames_collected) >= 3:
                    stopped_flag.set()

        t = threading.Thread(target=stream_consumer)
        t.start()

        # Wait for at least 3 frames
        assert stopped_flag.wait(timeout=2.0), "Stream did not produce initial frames"

        # Signal stop
        t_before = time.perf_counter()
        grabber.stop()
        t.join(timeout=0.5)
        t_after = time.perf_counter()

        assert not t.is_alive(), f"Stream thread failed to terminate within 500ms (took {t_after - t_before:.3f}s)"
        assert len(frames_collected) >= 3

    def test_stream_termination_when_paused(self) -> None:
        """Stream running while paused must terminate within 500ms upon stop()."""
        grabber = MSSGrabber(target_fps=20, mock_mode=True)
        win = WindowInfo(
            hwnd=1002,
            client_rect=Rect(0, 0, 800, 600),
            is_visible=True,
            is_minimized=False,
        )

        started = threading.Event()

        def run_paused_stream() -> None:
            grabber.pause()
            started.set()
            for _ in grabber.stream(lambda: win):
                pass

        t = threading.Thread(target=run_paused_stream)
        t.start()
        assert started.wait(timeout=1.0)
        time.sleep(0.1)

        t_before = time.perf_counter()
        grabber.stop()
        t.join(timeout=0.5)
        t_after = time.perf_counter()

        assert not t.is_alive(), f"Paused stream failed to terminate within 500ms (took {t_after - t_before:.3f}s)"

    def test_stream_termination_when_window_none(self) -> None:
        """Stream running when window is repeatedly None terminates cleanly."""
        grabber = MSSGrabber(target_fps=20, mock_mode=True)

        def run_none_stream() -> None:
            for _ in grabber.stream(lambda: None):
                pass

        t = threading.Thread(target=run_none_stream)
        t.start()
        time.sleep(0.3)

        t_before = time.perf_counter()
        grabber.stop()
        t.join(timeout=0.5)
        t_after = time.perf_counter()

        assert not t.is_alive(), f"None-window stream failed to terminate within 500ms (took {t_after - t_before:.3f}s)"


class TestMSSGrabberPacingBenchmark:
    """Rigorous empirical benchmarking of frame rate pacing and jitter."""

    @pytest.mark.parametrize("fps,expected_interval,tolerance", [
        (10, 0.100, 0.035),
        (20, 0.050, 0.025),
        (30, 0.033, 0.020),
    ])
    def test_frame_pacing_rate_and_jitter(self, fps: int, expected_interval: float, tolerance: float) -> None:
        """Benchmark pacing accuracy and inter-frame jitter."""
        grabber = MSSGrabber(target_fps=fps, mock_mode=True)
        grabber.start()
        win = WindowInfo(
            hwnd=2001,
            client_rect=Rect(0, 0, 640, 480),
            is_visible=True,
            is_minimized=False,
        )

        # Warm up first frame
        grabber.grab_frame(win)

        frame_count = 10
        timestamps: List[float] = []
        for _ in range(frame_count):
            pkt = grabber.grab_frame(win)
            assert pkt is not None
            timestamps.append(pkt.timestamp)
        grabber.stop()

        deltas = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        avg_interval = float(np.mean(deltas))
        jitter_std = float(np.std(deltas))

        # Check average interval is close to expected
        assert abs(avg_interval - expected_interval) < tolerance, (
            f"FPS={fps}: avg interval {avg_interval:.4f}s differed from expected {expected_interval:.4f}s "
            f"by more than {tolerance:.4f}s"
        )
        # Jitter standard deviation should be reasonably low (< 35ms)
        assert jitter_std < 0.035, f"FPS={fps}: Excessive jitter std dev {jitter_std:.4f}s"

    def test_dynamic_fps_adjustment_on_the_fly(self) -> None:
        """Dynamically switching target FPS mid-capture adjusts pacing interval immediately."""
        grabber = MSSGrabber(target_fps=10, mock_mode=True)
        grabber.start()
        win = WindowInfo(
            hwnd=2002,
            client_rect=Rect(0, 0, 640, 480),
            is_visible=True,
            is_minimized=False,
        )

        # Step 1: at 10 FPS (~100ms)
        grabber.grab_frame(win)
        t0 = time.perf_counter()
        grabber.grab_frame(win)
        d1 = time.perf_counter() - t0
        assert d1 >= 0.07, f"10 FPS frame interval too fast: {d1:.4f}s"

        # Step 2: change to 50 FPS (~20ms)
        grabber.set_target_fps(50)
        assert grabber.target_fps == 50
        assert abs(grabber._target_interval - 0.02) < 1e-4

        t0 = time.perf_counter()
        for _ in range(3):
            grabber.grab_frame(win)
        d2 = time.perf_counter() - t0
        # 3 frames at 50 FPS should take ~40-80ms, definitely much faster than at 10 FPS
        assert d2 < 0.15, f"50 FPS frame burst took too long: {d2:.4f}s"
        grabber.stop()

    def test_extreme_and_invalid_fps_inputs(self) -> None:
        """Target FPS outside [1, 60] must be clamped cleanly without exception."""
        grabber = MSSGrabber(target_fps=-10, mock_mode=True)
        assert grabber.target_fps == 1
        assert grabber._target_interval == 1.0

        grabber.set_target_fps(1000)
        assert grabber.target_fps == 60
        assert abs(grabber._target_interval - (1.0 / 60.0)) < 1e-4

        grabber.set_fps(0)
        assert grabber.target_fps == 1


class TestMSSGrabberSyntheticAndBoundaries:
    """Empirical verification of synthetic frame fallback and extreme bounds."""

    def test_synthetic_frame_generator_custom_callback(self) -> None:
        """Custom synthetic generator callback is invoked with correct parameters."""
        calls = []

        def custom_gen(w: int, h: int, fid: int) -> np.ndarray:
            calls.append((w, h, fid))
            return np.full((h, w, 3), 128, dtype=np.uint8)

        grabber = MSSGrabber(
            target_fps=30,
            mock_mode=True,
            synthetic_frame_generator=custom_gen,
        )
        grabber.start()

        win = WindowInfo(
            hwnd=3001,
            client_rect=Rect(10, 20, 800, 600),
            is_visible=True,
            is_minimized=False,
        )
        pkt = grabber.grab_frame(win)
        assert pkt is not None
        assert len(calls) == 1
        assert calls[0] == (800, 600, 1)
        assert pkt.frame.shape == (600, 800, 3)
        assert pkt.frame[0, 0, 0] == 128
        grabber.stop()

    def test_extreme_and_zero_rect_synthetic_frames(self) -> None:
        """Grabber safely handles zero or negative client rects in mock mode."""
        grabber = MSSGrabber(target_fps=30, mock_mode=True)
        grabber.start()

        # Zero width and height defaults safely to reference resolution or min clamp
        frame = grabber._default_synthetic_frame(0, 0, frame_id=1)
        assert frame.shape[0] >= 32
        assert frame.shape[1] >= 32
        assert frame.dtype == np.uint8

        # Very small 10x10 clamped to 32x32 minimum
        frame_tiny = grabber._default_synthetic_frame(10, 10, frame_id=2)
        assert frame_tiny.shape == (32, 32, 3)

        # 4K resolution synthetic frame
        frame_4k = grabber._default_synthetic_frame(3840, 2160, frame_id=3)
        assert frame_4k.shape == (2160, 3840, 3)
        grabber.stop()

    def test_grab_frame_none_window_raises_window_not_found(self) -> None:
        grabber = MSSGrabber(mock_mode=True)
        grabber.start()
        with pytest.raises(WindowNotFoundError):
            grabber.grab_frame(None)
        grabber.stop()


class TestMSSGrabberMinimizedStateTransitions:
    """Empirical challenge of window state transitions and auto-pause behavior."""

    def test_direct_grab_frame_minimized_transitions(self) -> None:
        """
        Verify transition: Normal -> Minimized -> Restored using direct grab_frame() calls.
        """
        grabber = MSSGrabber(target_fps=30, auto_pause_on_minimized=True, mock_mode=True)
        grabber.start()

        normal_win = WindowInfo(
            hwnd=4001,
            client_rect=Rect(0, 0, 1280, 720),
            is_visible=True,
            is_minimized=False,
        )
        minimized_win = WindowInfo(
            hwnd=4001,
            client_rect=Rect(0, 0, 1280, 720),
            is_visible=True,
            is_minimized=True,
        )

        # Frame 1: Normal -> FramePacket returned
        pkt1 = grabber.grab_frame(normal_win)
        assert pkt1 is not None
        assert grabber.is_paused is False

        # Frame 2: Window Minimized -> None returned, auto-paused
        pkt2 = grabber.grab_frame(minimized_win)
        assert pkt2 is None
        assert grabber.is_paused is True

        # Frame 3: Window Restored -> grab_frame called with normal window
        pkt3 = grabber.grab_frame(normal_win)
        assert pkt3 is not None
        assert pkt3.frame_id == 2
        grabber.stop()

    def test_stream_auto_pause_and_resume_recovery(self) -> None:
        """
        EMPIRICAL CHALLENGE: Verify whether stream() automatically recovers
        when a window transitions Normal -> Minimized -> Restored.
        """
        grabber = MSSGrabber(target_fps=30, auto_pause_on_minimized=True, mock_mode=True)

        states = [
            WindowInfo(hwnd=5001, client_rect=Rect(0, 0, 640, 480), is_visible=True, is_minimized=False),
            WindowInfo(hwnd=5001, client_rect=Rect(0, 0, 640, 480), is_visible=True, is_minimized=True),  # Minimized!
            WindowInfo(hwnd=5001, client_rect=Rect(0, 0, 640, 480), is_visible=True, is_minimized=False), # Restored!
        ]
        state_idx = 0
        state_lock = threading.Lock()

        def dynamic_window_provider() -> Optional[WindowInfo]:
            nonlocal state_idx
            with state_lock:
                idx = min(state_idx, len(states) - 1)
                return states[idx]

        collected_packets: List[FramePacket] = []

        def run_stream() -> None:
            for pkt in grabber.stream(dynamic_window_provider):
                collected_packets.append(pkt)

        t = threading.Thread(target=run_stream)
        t.start()

        # Step 1: Wait for 1st frame (normal state)
        time.sleep(0.1)
        assert len(collected_packets) >= 1

        # Step 2: Minimize window
        with state_lock:
            state_idx = 1
        time.sleep(0.15)
        count_at_minimized = len(collected_packets)

        # Step 3: Restore window
        with state_lock:
            state_idx = 2
        time.sleep(0.3)

        # Does stream recover on its own, or does it stay paused?
        # Note: If stream() checks `if self._is_paused: time.sleep(0.1); continue`
        # without querying the window provider, count_after_restore will NOT increase!
        # If external resume() is needed, we verify that resume() unblocks it.
        count_after_restore = len(collected_packets)

        stream_auto_resumed = (count_after_restore > count_at_minimized)
        if not stream_auto_resumed:
            # If auto-resume didn't happen, call grabber.resume() to test manual recovery
            grabber.resume()
            time.sleep(0.2)
            assert len(collected_packets) > count_at_minimized, "Even explicit resume() failed to unblock stream"

        grabber.stop()
        t.join(timeout=0.5)
        assert not t.is_alive()


class TestWindowDetectorAdversarial:
    """Stress testing WindowDetector against adversarial inputs, Win32 calls, and edge cases."""

    def test_nonexistent_and_invalid_hwnds(self) -> None:
        """Querying invalid HWNDs (0, negative, dead handles) returns None without crashing."""
        detector = WindowDetector(mock_mode=False)

        invalid_hwnds = [0, -1, -9999, 99999999, 0x7FFFFFFF]
        for hwnd in invalid_hwnds:
            info = detector.get_window_info(hwnd)
            assert info is None

    def test_dpi_awareness_idempotency_and_concurrency(self) -> None:
        """Calling init_dpi_awareness repeatedly or concurrently does not fail or deadlock."""
        detector = WindowDetector(mock_mode=False)

        # Repeated sequential calls
        for _ in range(5):
            res = detector.init_dpi_awareness()
            assert isinstance(res, bool)

        # Concurrent calls
        errors: List[Exception] = []

        def call_dpi() -> None:
            try:
                for _ in range(10):
                    detector.init_dpi_awareness()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call_dpi) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)
            assert not t.is_alive()
        assert len(errors) == 0

    def test_adversarial_title_patterns(self) -> None:
        """Adversarial title regex patterns compile and evaluate without crashing."""
        tricky_titles = [
            r"Google Play Games.*",
            r"",                      # empty
            r"   ",                   # whitespace
            r"^$",                    # empty match
            r".*",                    # match all
            r"(?=.*Clash)(?=.*CoC)",  # lookahead
        ]
        detector = WindowDetector(
            target_window_titles=tricky_titles,
            mock_mode=True,
        )
        assert len(detector.compiled_title_regexes) == len(tricky_titles)

    def test_window_detector_live_enumeration_stability(self) -> None:
        """Live window enumeration on current host returns valid WindowInfo list without exceptions."""
        detector = WindowDetector(mock_mode=False)
        windows = detector.enumerate_windows()
        assert isinstance(windows, list)
        for w in windows:
            assert isinstance(w, WindowInfo)
            assert w.hwnd > 0
            assert isinstance(w.title, str)
            assert w.client_rect.width >= 0
            assert w.client_rect.height >= 0

    def test_fallback_mock_target_safety_guard_registration(self) -> None:
        """Fallback mock target is registered with SafetyGuard."""
        detector = WindowDetector(
            target_process_names=["definitely_nonexistent_proc_123.exe"],
            target_window_titles=["definitely_nonexistent_title_123"],
            mock_mode=True,
        )
        detector.clear_mock_windows()

        # In mock mode with clear_mock_windows, returns None
        assert detector.find_target_window() is None

        # Create mock target and verify registration
        guard = get_safety_guard()
        mock_win = detector.create_mock_target()
        assert guard.is_live_window(mock_win.hwnd)
        guard.clear_all()
