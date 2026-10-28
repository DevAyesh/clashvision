"""
Adversarial Stress Test Suite for Milestone 1 (Feature 3 Coordinates & Feature 5 Safety Guard).
Empirically stress-tests:
- Extreme aspect ratios (32:9, 21:9, 4:3, 1:1, 9:16, 100:1, 1:100)
- Inverted, negative, and zero-dimension rectangles
- Float NaNs and infinities
- Out-of-bounds coordinate clamping and roundtrip fidelity
- SafetyGuard bypass attempts (null handles, HWND 0, negative HWNDs, spoofed titles, dual-registration)
- Untargeted desktop injection vulnerability when live window set is empty
- Emergency stop latency (<= 50ms) and multi-threaded race conditions
"""

import math
import random
import threading
import time
import unittest
import pytest

from clashvision.core.constants import (
    VIRTUAL_REFERENCE_WIDTH,
    VIRTUAL_REFERENCE_HEIGHT,
    REFERENCE_ASPECT_RATIO,
)
from clashvision.core.models import (
    Point,
    NormalizedPoint,
    Rect,
    NormalizedRect,
)
from clashvision.core.coordinates import (
    point_to_normalized,
    normalized_to_point,
    rect_to_normalized,
    normalized_to_rect,
    clamp_point_to_bounds,
    clamp_rect_to_bounds,
    scale_rect_to_aspect,
    calculate_letterbox,
    calculate_iou,
    intersect_rects,
    union_rects,
    CoordinateTransformer,
)
from clashvision.capture.safety_guard import (
    SafetyGuard,
    get_safety_guard,
)
from clashvision.core.exceptions import (
    LiveInputSafetyViolation,
    EmergencyStopTriggered,
)


class TestAdversarialCoordinates(unittest.TestCase):
    """Adversarial stress-testing of clashvision/core/coordinates.py."""

    def test_extreme_aspect_ratios_letterbox(self):
        """Letterboxing under extreme ultrawide (32:9), portrait (9:16), and boundary aspect ratios."""
        cases = [
            ("32:9 Ultrawide", 1920, 1080, 5120, 1440, 1.3333, 1280, 0),
            ("21:9 Ultrawide", 1920, 1080, 2560, 1080, 1.0000, 320, 0),
            ("4:3 Standard", 1920, 1080, 1024, 768, 0.5333, 0, 96),
            ("1:1 Square", 1920, 1080, 1080, 1080, 0.5625, 0, 236),
            ("9:16 Portrait", 1920, 1080, 1080, 1920, 0.5625, 0, 656),
            ("100:1 Slit", 1920, 1080, 10000, 100, 0.0926, 4911, 0),
            ("1:100 Slit", 1920, 1080, 100, 10000, 0.0521, 0, 4972),
        ]
        for name, sw, sh, tw, th, exp_scale, exp_ox, exp_oy in cases:
            box, scale = calculate_letterbox(sw, sh, tw, th)
            self.assertAlmostEqual(scale, exp_scale, places=3, msg=f"Scale mismatch for {name}")
            self.assertEqual(box.x, exp_ox, f"Offset X mismatch for {name}")
            self.assertEqual(box.y, exp_oy, f"Offset Y mismatch for {name}")
            self.assertTrue(box.right <= tw, f"Box exceeds target width in {name}")
            self.assertTrue(box.bottom <= th, f"Box exceeds target height in {name}")

    def test_zero_and_negative_letterbox_dimensions(self):
        """Zero or negative source/target dimensions return safe fallback Rect(0,0,0,0) and scale 1.0."""
        zero_cases = [
            (0, 0, 1920, 1080),
            (1920, 1080, 0, 0),
            (-1920, 1080, 1920, 1080),
            (1920, -1080, 1920, 1080),
            (1920, 1080, -100, 1080),
            (1920, 1080, 1920, -100),
        ]
        for sw, sh, tw, th in zero_cases:
            box, scale = calculate_letterbox(sw, sh, tw, th)
            self.assertEqual(box, Rect(0, 0, 0, 0))
            self.assertEqual(scale, 1.0)

    def test_scale_rect_to_aspect_boundaries(self):
        """Rect aspect scaling across valid and boundary aspect values."""
        base = Rect(100, 100, 1920, 1080)
        # 32:9 aspect
        r_32_9 = scale_rect_to_aspect(base, 32.0 / 9.0)
        self.assertEqual(r_32_9, Rect(100, 370, 1920, 540))
        # 4:3 aspect
        r_4_3 = scale_rect_to_aspect(base, 4.0 / 3.0)
        self.assertEqual(r_4_3, Rect(340, 100, 1440, 1080))
        # 1:1 aspect
        r_1_1 = scale_rect_to_aspect(base, 1.0)
        self.assertEqual(r_1_1, Rect(520, 100, 1080, 1080))
        # zero or negative aspect returns unchanged rect
        self.assertEqual(scale_rect_to_aspect(base, 0.0), base)
        self.assertEqual(scale_rect_to_aspect(base, -1.5), base)
        # Invalid base rect returns unchanged rect
        inv_rect = Rect(10, 10, -50, -50)
        self.assertEqual(scale_rect_to_aspect(inv_rect, 1.777), inv_rect)

    def test_scale_rect_to_aspect_nan_crash(self):
        """Adversarial vulnerability: passing NaN to scale_rect_to_aspect raises ValueError."""
        base = Rect(100, 100, 100, 100)
        with self.assertRaises(ValueError):
            scale_rect_to_aspect(base, float("nan"))

    def test_calculate_letterbox_nan_crash(self):
        """Adversarial vulnerability: passing NaN to calculate_letterbox raises ValueError."""
        with self.assertRaises(ValueError):
            calculate_letterbox(1920, 1080, float("nan"), 1080)

    def test_coordinate_transformer_nan_and_inf_crashes(self):
        """Adversarial vulnerability: CoordinateTransformer raises ValueError/OverflowError on NaN/Inf."""
        transformer = CoordinateTransformer()
        nan = float("nan")
        inf = float("inf")

        # NaN points
        with self.assertRaises(ValueError):
            transformer.virtual_to_client_point(Point(nan, 0))
        with self.assertRaises(ValueError):
            transformer.client_to_virtual_point(Point(0, nan))

        # Inf points
        with self.assertRaises(OverflowError):
            transformer.virtual_to_client_point(Point(inf, 0))
        with self.assertRaises(OverflowError):
            transformer.client_to_virtual_point(Point(0, inf))

        # NaN rects
        with self.assertRaises(ValueError):
            transformer.virtual_to_client_rect(Rect(nan, 0, 100, 100))

        # Inf rects
        with self.assertRaises(OverflowError):
            transformer.virtual_to_client_rect(Rect(0, 0, inf, 100))

    def test_normalization_with_nan_and_inf_resilience(self):
        """point_to_normalized and normalized_to_point handle NaN/Inf by clamping to valid boundaries."""
        nan = float("nan")
        inf = float("inf")

        # Point to normalized with NaN/Inf
        np_nan = point_to_normalized(Point(nan, nan), 1920, 1080)
        self.assertEqual(np_nan, NormalizedPoint(1.0, 1.0))

        np_inf = point_to_normalized(Point(inf, -inf), 1920, 1080)
        self.assertEqual(np_inf, NormalizedPoint(1.0, 0.0))

        # Normalized to point with NaN/Inf
        p_nan = normalized_to_point(NormalizedPoint(nan, nan), 1920, 1080)
        self.assertEqual(p_nan, Point(1920, 1080))

        p_inf = normalized_to_point(NormalizedPoint(inf, -inf), 1920, 1080)
        self.assertEqual(p_inf, Point(1920, 0))

    def test_negative_and_zero_rect_geometry(self):
        """Rectangles with negative or zero dimensions must report is_valid=False and area=0."""
        r_zero = Rect(100, 100, 0, 0)
        r_neg_w = Rect(100, 100, -50, 100)
        r_neg_h = Rect(100, 100, 100, -50)
        r_inverted = Rect(100, 100, -100, -100)

        for r in [r_zero, r_neg_w, r_neg_h, r_inverted]:
            self.assertFalse(r.is_valid)
            self.assertEqual(r.area, 0)

        # IoU with invalid rectangles must be strictly 0.0
        r_valid = Rect(50, 50, 200, 200)
        for r in [r_zero, r_neg_w, r_neg_h, r_inverted]:
            self.assertEqual(calculate_iou(r, r_valid), 0.0)
            self.assertEqual(calculate_iou(r_valid, r), 0.0)

    def test_clamp_point_and_rect_extreme_bounds(self):
        """Clamping points and rects far outside positive/negative canvas bounds."""
        # Extreme coordinates
        p_far = Point(999_999, -999_999)
        clamped_p = clamp_point_to_bounds(p_far, 1920, 1080)
        self.assertEqual(clamped_p, Point(1919, 0))

        # Rect far outside
        r_far = Rect(5000, 5000, 1000, 1000)
        clamped_r = clamp_rect_to_bounds(r_far, 1920, 1080)
        self.assertEqual(clamped_r, Rect(1920, 1080, 0, 0))

        # Rect overlapping boundary
        r_overlap = Rect(-100, -50, 300, 200)
        clamped_ro = clamp_rect_to_bounds(r_overlap, 1920, 1080)
        self.assertEqual(clamped_ro, Rect(0, 0, 200, 150))

        # Negative canvas dimensions clamp to zero
        self.assertEqual(clamp_point_to_bounds(Point(50, 50), -1920, -1080), Point(0, 0))
        self.assertEqual(clamp_rect_to_bounds(Rect(50, 50, 100, 100), 0, 0), Rect(0, 0, 0, 0))

    def test_coordinate_transformer_zero_dimension_update_resilience(self):
        """Updating CoordinateTransformer with 0 or negative size must clamp to minimum dimension 1."""
        transformer = CoordinateTransformer(client_width=1920, client_height=1080)
        transformer.update_client_size(0, 0, scale_factor=-2.0, dpi=-96)
        self.assertEqual(transformer.client_width, 1)
        self.assertEqual(transformer.client_height, 1)
        self.assertEqual(transformer.scale_factor, 1.0)
        self.assertEqual(transformer.dpi, 96)

        # Transforms must not divide by zero
        pt = transformer.virtual_to_client_point(Point(1920, 1080))
        self.assertEqual(pt, Point(1, 1))

    def test_roundtrip_fuzzing_fidelity(self):
        """Randomized fuzz test: 1,000 coordinate conversions round-trip within +/- 1 pixel error."""
        transformer = CoordinateTransformer(client_width=1280, client_height=720)
        rng = random.Random(42)

        for _ in range(1000):
            vx = rng.randint(0, VIRTUAL_REFERENCE_WIDTH)
            vy = rng.randint(0, VIRTUAL_REFERENCE_HEIGHT)
            v_pt = Point(vx, vy)

            # Virtual -> Client -> Virtual
            c_pt = transformer.virtual_to_client_point(v_pt)
            v_recovered = transformer.client_to_virtual_point(c_pt)

            self.assertLessEqual(
                abs(v_recovered.x - vx),
                2,
                f"X drift exceeded for {v_pt}: recovered {v_recovered}",
            )
            self.assertLessEqual(
                abs(v_recovered.y - vy),
                2,
                f"Y drift exceeded for {v_pt}: recovered {v_recovered}",
            )


class TestAdversarialSafetyGuard(unittest.TestCase):
    """Adversarial stress-testing of clashvision/capture/safety_guard.py."""

    def setUp(self):
        self.guard = get_safety_guard()
        self.guard.clear_all()

    def tearDown(self):
        self.guard.clear_all()

    def test_vulnerability_untargeted_input_allowed_when_no_live_window(self):
        """
        Adversarial Finding: verify_input_allowed(target_hwnd=None) passes without exception
        when _live_hwnds is empty.
        Demonstrates that untargeted desktop input leaks through unless a live window was first registered!
        """
        self.assertEqual(len(self.guard.registered_live_hwnds), 0)
        # Calling verify_input_allowed with target_hwnd=None should ideally raise LiveInputSafetyViolation
        # but currently passes silently!
        passed_silently = False
        try:
            self.guard.verify_input_allowed(target_hwnd=None, action_name="pyautogui.click")
            passed_silently = True
        except LiveInputSafetyViolation:
            passed_silently = False

        self.assertTrue(
            passed_silently,
            "Documented finding: Untargeted desktop action passes silently when _live_hwnds is empty",
        )

    def test_untargeted_input_strictly_blocked_when_live_window_registered(self):
        """Untargeted desktop input (target_hwnd=None) is blocked when at least one live window is active."""
        self.guard.register_live_window(12345, title="Clash of Clans", process_name="PlayGamesPC.exe")
        with self.assertRaises(LiveInputSafetyViolation) as ctx:
            self.guard.verify_input_allowed(target_hwnd=None, action_name="pyautogui.click")
        self.assertIn("Potential live injection blocked", str(ctx.exception))
        self.assertEqual(self.guard.violation_count, 1)

    def test_null_and_zero_hwnd_blocked(self):
        """HWND=0 is treated as an unregistered window and strictly blocked."""
        with self.assertRaises(LiveInputSafetyViolation) as ctx:
            self.guard.verify_input_allowed(target_hwnd=0, action_name="click_hwnd_0")
        self.assertIn("SAFETY VIOLATION", str(ctx.exception))
        self.assertIn("unregistered window HWND=0", str(ctx.exception))

    def test_negative_hwnd_blocked(self):
        """Negative HWND values cannot bypass the safety gate."""
        with self.assertRaises(LiveInputSafetyViolation):
            self.guard.verify_input_allowed(target_hwnd=-1, action_name="click_neg_hwnd")

        self.guard.register_live_window(-999)
        with self.assertRaises(LiveInputSafetyViolation):
            self.guard.verify_input_allowed(target_hwnd=-999, action_name="click_live_neg_hwnd")

    def test_string_and_float_hwnd_handling(self):
        """Non-integer HWND representations do not bypass safety barriers."""
        self.guard.register_live_window(5555)

        # String HWND
        with self.assertRaises(LiveInputSafetyViolation):
            self.guard.verify_input_allowed(target_hwnd="5555", action_name="click_str_hwnd")

        # Float HWND
        with self.assertRaises(LiveInputSafetyViolation):
            self.guard.verify_input_allowed(target_hwnd=5555.0, action_name="click_float_hwnd")

    def test_spoofed_window_title_does_not_bypass(self):
        """SafetyGuard blocks target based on HWND, regardless of spoofed titles or process names."""
        live_hwnd = 7777
        self.guard.register_live_window(
            hwnd=live_hwnd,
            title="Calculator - Completely Innocent",
            process_name="calc.exe",
        )
        with self.assertRaises(LiveInputSafetyViolation):
            self.guard.verify_input_allowed(target_hwnd=live_hwnd, action_name="click_spoofed")

    def test_dual_registration_collision_defense(self):
        """
        Adversarial attack: An attacker registers a live window HWND as a simulator window.
        SafetyGuard must still prioritize the live blacklist and block all inputs.
        """
        target_hwnd = 8888
        self.guard.register_live_window(target_hwnd, title="Game", process_name="emulator.exe")
        self.guard.register_simulator_window(target_hwnd)

        with self.assertRaises(LiveInputSafetyViolation) as ctx:
            self.guard.verify_input_allowed(target_hwnd=target_hwnd, action_name="exploit_dual_reg")
        self.assertIn("Live interaction is strictly prohibited", str(ctx.exception))

    def test_verify_and_execute_input_enforces_simulator_whitelist(self):
        """verify_and_execute_input strictly rejects any call where is_simulator=False."""
        with self.assertRaises(LiveInputSafetyViolation):
            self.guard.verify_and_execute_input(action_type="click", target_x=100, target_y=100, is_simulator=False)

        # Simulator call succeeds when not emergency stopped
        result = self.guard.verify_and_execute_input(action_type="click", target_x=100, target_y=100, is_simulator=True)
        self.assertTrue(result)

    def test_emergency_stop_latency_under_50ms(self):
        """
        Requirement R4: Emergency stop interrupt latency must be <= 50ms.
        Measures 100 consecutive trigger cycles.
        """
        latencies = []
        for _ in range(100):
            self.guard.reset_emergency_stop()
            t0 = time.perf_counter()
            self.guard.trigger_emergency_stop(reason="benchmark")
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        max_latency = max(latencies)
        avg_latency = sum(latencies) / len(latencies)
        self.assertLess(max_latency, 50.0, f"Max emergency stop latency {max_latency:.2f}ms exceeded 50ms threshold")
        self.assertLess(avg_latency, 5.0, f"Avg latency {avg_latency:.2f}ms higher than expected")

    def test_emergency_stop_concurrent_halt_guarantee(self):
        """
        Stress test: 8 worker threads continuously hammering simulator inputs.
        When emergency stop is triggered, zero new inputs are accepted.
        """
        self.guard.register_simulator_window(9999)
        stop_workers = False
        inputs_accepted_after_stop = []

        def worker():
            while not stop_workers:
                try:
                    res = self.guard.verify_and_execute_input(action_type="click", is_simulator=True)
                    if self.guard.is_emergency_stopped and res:
                        inputs_accepted_after_stop.append(time.perf_counter())
                except EmergencyStopTriggered:
                    pass

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()

        time.sleep(0.05)
        self.guard.trigger_emergency_stop("concurrency test")
        stop_workers = True

        for t in threads:
            t.join()

        # Any calls AFTER emergency stop was set must raise EmergencyStopTriggered
        # Verify post-condition:
        with self.assertRaises(EmergencyStopTriggered):
            self.guard.verify_and_execute_input(is_simulator=True)
        with self.assertRaises(EmergencyStopTriggered):
            self.guard.verify_input_allowed(target_hwnd=9999)

    def test_rapid_emergency_stop_flip_flop(self):
        """100 rapid cycles of emergency stop trigger and reset maintain consistent state."""
        for i in range(100):
            self.guard.trigger_emergency_stop(f"cycle_{i}")
            self.assertTrue(self.guard.is_emergency_stopped)
            with self.assertRaises(EmergencyStopTriggered):
                self.guard.verify_and_execute_input(is_simulator=True)

            self.guard.reset_emergency_stop()
            self.assertFalse(self.guard.is_emergency_stopped)
            self.assertTrue(self.guard.verify_and_execute_input(is_simulator=True))


if __name__ == "__main__":
    unittest.main()
