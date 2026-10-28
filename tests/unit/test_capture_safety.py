"""
Unit tests for WindowDetector, MSSGrabber, and SafetyGuard.
Verifies:
- Feature 1: Window Enumeration & Target Discovery
- Feature 2: High-Performance MSS Frame Capture & Rate Limiting
- Feature 4: Window State & Visibility Monitor (Minimization / Occlusion handling)
- Feature 5: Read-Only Live Safety Guard (Live input blocking, emergency stop)
"""

import time
import unittest
import pytest
import numpy as np

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
    LiveInputSafetyViolation,
    EmergencyStopTriggered,
    WindowNotFoundError,
    WindowMinimizedError,
)
from clashvision.capture.safety_guard import (
    SafetyGuard,
    get_safety_guard,
)
from clashvision.capture.window_detector import (
    WindowDetector,
)
from clashvision.capture.mss_grabber import (
    MSSGrabber,
)


class TestSafetyGuard(unittest.TestCase):
    """Rigorous verification of the architectural read-only safety guardrail."""

    def setUp(self):
        self.guard = get_safety_guard()
        self.guard.clear_all()

    def tearDown(self):
        self.guard.clear_all()

    def test_singleton_consistency(self):
        g1 = SafetyGuard.get_instance()
        g2 = get_safety_guard()
        self.assertIs(g1, g2)

    def test_live_window_registration_and_blocking(self):
        live_hwnd = 12345
        self.guard.register_live_window(live_hwnd, title="Clash of Clans", process_name="PlayGamesPC.exe")

        self.assertTrue(self.guard.is_live_window(live_hwnd))
        self.assertIn(live_hwnd, self.guard.registered_live_hwnds)

        # Attempting any automated action targeted at the live window MUST raise LiveInputSafetyViolation
        with self.assertRaises(LiveInputSafetyViolation) as ctx:
            self.guard.verify_input_allowed(target_hwnd=live_hwnd, action_name="click")
        self.assertIn("SAFETY VIOLATION", str(ctx.exception))
        self.assertEqual(self.guard.violation_count, 1)

    def test_global_desktop_input_blocked_when_live_window_registered(self):
        self.guard.register_live_window(999, title="Google Play Games", process_name="HD-Player.exe")

        # Global input without specifying a target hwnd (target_hwnd=None) must be blocked to prevent accidental injection
        with self.assertRaises(LiveInputSafetyViolation):
            self.guard.verify_input_allowed(target_hwnd=None, action_name="pyautogui.click")
        self.assertEqual(self.guard.violation_count, 1)

    def test_unregistered_window_input_blocked(self):
        # Only simulator windows are allowed; arbitrary windows are blocked
        unregistered_hwnd = 54321
        with self.assertRaises(LiveInputSafetyViolation):
            self.guard.verify_input_allowed(target_hwnd=unregistered_hwnd, action_name="key_press")
        self.assertEqual(self.guard.violation_count, 1)

    def test_simulator_window_allowed_when_no_emergency_stop(self):
        sim_hwnd = 8888
        self.guard.register_simulator_window(sim_hwnd)

        self.assertIn(sim_hwnd, self.guard.registered_simulator_hwnds)

        # Input targeted directly at the registered simulator window must succeed with zero exceptions
        self.guard.verify_input_allowed(target_hwnd=sim_hwnd, action_name="simulator_click")
        self.assertEqual(self.guard.violation_count, 0)

    def test_emergency_stop_overrides_everything(self):
        sim_hwnd = 8888
        self.guard.register_simulator_window(sim_hwnd)

        self.assertFalse(self.guard.is_emergency_stopped)
        self.guard.trigger_emergency_stop(reason="User clicked Emergency Stop button")
        self.assertTrue(self.guard.is_emergency_stopped)

        # Even simulator window inputs must now be aborted immediately
        with self.assertRaises(EmergencyStopTriggered) as ctx:
            self.guard.verify_input_allowed(target_hwnd=sim_hwnd, action_name="simulator_click")
        self.assertIn("Emergency stop is currently active", str(ctx.exception))

        # Resetting emergency stop restores simulator access
        self.guard.reset_emergency_stop()
        self.assertFalse(self.guard.is_emergency_stopped)
        self.guard.verify_input_allowed(target_hwnd=sim_hwnd, action_name="simulator_click")

    def test_unregister_live_window(self):
        hwnd = 777
        self.guard.register_live_window(hwnd, "Test Live", "Test.exe")
        self.assertTrue(self.guard.is_live_window(hwnd))
        self.guard.unregister_live_window(hwnd)
        self.assertFalse(self.guard.is_live_window(hwnd))


class TestWindowDetector(unittest.TestCase):
    """Tests for WindowDetector matching, enumeration, and mock support."""

    def setUp(self):
        get_safety_guard().clear_all()

    def tearDown(self):
        get_safety_guard().clear_all()

    def test_mock_target_creation_and_discovery(self):
        detector = WindowDetector(mock_mode=True)
        mock_win = detector.create_mock_target(
            title="Clash of Clans (Mock Target)",
            process_name="PlayGamesPC.exe",
            width=1920,
            height=1080,
        )

        self.assertEqual(mock_win.hwnd, 99991)
        self.assertEqual(mock_win.client_rect.width, 1920)
        self.assertEqual(mock_win.client_rect.height, 1080)
        self.assertTrue(mock_win.is_capturable)
        self.assertFalse(mock_win.is_minimized)

        # Target finding should match the registered mock target
        found = detector.find_target_window()
        self.assertIsNotNone(found)
        self.assertEqual(found.hwnd, mock_win.hwnd)
        self.assertEqual(found.process_name, "PlayGamesPC.exe")

        # Verify that finding the target automatically registered it with SafetyGuard
        guard = get_safety_guard()
        self.assertTrue(guard.is_live_window(mock_win.hwnd))

    def test_target_title_regex_matching(self):
        detector = WindowDetector(
            target_process_names=["NonExistentProcess.exe"],
            target_window_titles=[r"Google Play Games.*", r"Clash of Clans.*"],
            mock_mode=True,
        )
        mock_win = WindowInfo(
            hwnd=10101,
            title="Google Play Games on PC - Running Clash of Clans",
            process_name="Unknown.exe",
            client_rect=Rect(0, 0, 1920, 1080),
            window_rect=Rect(0, 0, 1920, 1080),
            is_visible=True,
            is_minimized=False,
            is_focused=True,
        )
        detector.register_mock_window(mock_win)

        found = detector.find_target_window()
        self.assertIsNotNone(found)
        self.assertEqual(found.hwnd, 10101)

    def test_window_info_aspect_ratio_and_capturable(self):
        # 16:9 capturable window
        w1 = WindowInfo(
            client_rect=Rect(0, 0, 1920, 1080),
            is_visible=True,
            is_minimized=False,
        )
        self.assertAlmostEqual(w1.aspect_ratio, 16.0 / 9.0, places=3)
        self.assertTrue(w1.is_capturable)

        # Minimized window
        w2 = WindowInfo(
            client_rect=Rect(0, 0, 1920, 1080),
            is_visible=True,
            is_minimized=True,
        )
        self.assertFalse(w2.is_capturable)

        # Zero area client rect (invisible or collapsed)
        w3 = WindowInfo(
            client_rect=Rect(0, 0, 0, 0),
            is_visible=True,
            is_minimized=False,
        )
        self.assertFalse(w3.is_capturable)
        self.assertEqual(w3.aspect_ratio, 0.0)

    def test_target_not_found(self):
        detector = WindowDetector(
            target_process_names=["definitely_not_running_xyz.exe"],
            target_window_titles=["definitely_not_existing_title_xyz"],
            mock_mode=True,
        )
        detector.clear_mock_windows()
        self.assertIsNone(detector.find_target_window())


class TestMSSGrabber(unittest.TestCase):
    """Tests for frame capture, FPS rate control, and minimized auto-pause."""

    def test_grabber_initialization_and_lifecycle(self):
        grabber = MSSGrabber(target_fps=15, mock_mode=True)
        self.assertEqual(grabber.target_fps, 15)
        self.assertFalse(grabber.is_running)
        self.assertEqual(grabber.frame_count, 0)

        grabber.start()
        self.assertTrue(grabber.is_running)

        grabber.pause()
        self.assertTrue(grabber.is_paused)

        grabber.resume()
        self.assertFalse(grabber.is_paused)

        grabber.stop()
        self.assertFalse(grabber.is_running)

    def test_grab_frame_packet_creation(self):
        grabber = MSSGrabber(target_fps=30, mock_mode=True)
        grabber.start()

        mock_win = WindowInfo(
            hwnd=555,
            title="Clash of Clans",
            process_name="PlayGamesPC.exe",
            client_rect=Rect(100, 100, 1280, 720),
            window_rect=Rect(92, 68, 1296, 760),
            is_visible=True,
            is_minimized=False,
        )

        packet = grabber.grab_frame(mock_win)
        self.assertIsInstance(packet, FramePacket)
        self.assertEqual(packet.frame_id, 1)
        self.assertGreater(packet.timestamp, 0)
        self.assertEqual(packet.window_info.hwnd, 555)
        self.assertIsNotNone(packet.frame)
        self.assertEqual(packet.width, 1280)
        self.assertEqual(packet.height, 720)
        self.assertEqual(packet.channels, 3)
        self.assertIsInstance(packet.frame, np.ndarray)
        self.assertEqual(packet.frame.dtype, np.uint8)

        # Grab a second frame to test monotonic counter
        packet2 = grabber.grab_frame(mock_win)
        self.assertEqual(packet2.frame_id, 2)
        self.assertGreaterEqual(packet2.timestamp, packet.timestamp)

        grabber.stop()

    def test_grab_frame_none_window_raises(self):
        grabber = MSSGrabber(mock_mode=True)
        grabber.start()
        with self.assertRaises(WindowNotFoundError):
            grabber.grab_frame(None)
        grabber.stop()

    def test_auto_pause_on_minimized_window(self):
        grabber = MSSGrabber(auto_pause_on_minimized=True, mock_mode=True)
        grabber.start()

        minimized_win = WindowInfo(
            hwnd=666,
            client_rect=Rect(0, 0, 1920, 1080),
            is_visible=True,
            is_minimized=True,  # Minimized!
        )

        packet = grabber.grab_frame(minimized_win)
        self.assertIsNone(packet)
        self.assertTrue(grabber.is_paused)
        grabber.stop()

    def test_raise_on_minimized_when_auto_pause_false(self):
        grabber = MSSGrabber(auto_pause_on_minimized=False, mock_mode=True)
        grabber.start()

        minimized_win = WindowInfo(
            hwnd=666,
            client_rect=Rect(0, 0, 1920, 1080),
            is_visible=True,
            is_minimized=True,
        )

        with self.assertRaises(WindowMinimizedError):
            grabber.grab_frame(minimized_win)
        grabber.stop()

    def test_fps_rate_limiting(self):
        # Setting 20 FPS -> ~50ms interval
        target_fps = 20
        grabber = MSSGrabber(target_fps=target_fps, mock_mode=True)
        grabber.start()

        mock_win = WindowInfo(
            hwnd=123,
            client_rect=Rect(0, 0, 640, 360),
            is_visible=True,
            is_minimized=False,
        )

        start_time = time.time()
        for _ in range(3):
            grabber.grab_frame(mock_win)
        elapsed = time.time() - start_time

        # 3 frames at 20 FPS should take at least 2 intervals = 2 * 0.05 = 0.10s (allowing small timing delta)
        self.assertGreaterEqual(elapsed, 0.08)
        grabber.stop()

    def test_synthetic_frame_generator_contents(self):
        grabber = MSSGrabber(mock_mode=True)
        frame = grabber._default_synthetic_frame(1920, 1080, frame_id=1)
        self.assertEqual(frame.shape, (1080, 1920, 3))
        self.assertEqual(frame.dtype, np.uint8)

        # Verify base grass tone exists (roughly BGR: 45, 130, 60)
        center_pixel = frame[540, 960]
        self.assertEqual(center_pixel[0], 45)  # B
        self.assertEqual(center_pixel[1], 130) # G
        self.assertEqual(center_pixel[2], 60)  # R
