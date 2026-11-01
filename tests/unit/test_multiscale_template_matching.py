"""
Unit Test & Benchmark Suite for ClashVision Multi-Scale Template Matching.
Verifies resolution invariance (720p, 1080p, 1440p, 4K), grayscale vs edge matching,
noise/brightness tolerance, boundary safety, and performance benchmarks.
Derived from Phase 2 Requirements.
"""

import time
import unittest
from typing import Dict, Any, Tuple

import cv2
import numpy as np
import pytest

from clashvision.core.models import Rect
from clashvision.vision.preprocessor import VisionPreprocessor, TemplateMatchResult


def create_synthetic_target(width: int = 60, height: int = 60) -> np.ndarray:
    """Generates a distinctive synthetic UI button/building template with borders and inner shape."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Background fill
    img[:] = (35, 45, 60)
    # Outer rectangle
    cv2.rectangle(img, (4, 4), (width - 5, height - 5), (0, 220, 255), 2)
    # Inner circular icon
    center = (width // 2, height // 2)
    radius = min(width, height) // 4
    cv2.circle(img, center, radius, (0, 255, 120), -1)
    # Distinctive cross
    cv2.line(img, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (255, 255, 255), 2)
    return img


def embed_target_in_scene(
    scene_width: int,
    scene_height: int,
    target: np.ndarray,
    scale: float = 1.0,
    pos: Tuple[int, int] = (100, 100),
    brightness_offset: int = 0,
    noise_sigma: float = 0.0,
) -> Tuple[np.ndarray, Rect]:
    """Embeds scaled target into a textured synthetic scene."""
    scene = np.zeros((scene_height, scene_width, 3), dtype=np.uint8)
    # Add subtle background texture
    grid_size = 50
    for y in range(0, scene_height, grid_size):
        for x in range(0, scene_width, grid_size):
            val = ((x // grid_size + y // grid_size) % 2) * 15 + 20
            scene[y : y + grid_size, x : x + grid_size] = (val, val + 5, val)

    # Scale target
    tw = int(round(target.shape[1] * scale))
    th = int(round(target.shape[0] * scale))
    scaled_target = cv2.resize(target, (tw, th), interpolation=cv2.INTER_CUBIC)

    # Apply brightness
    if brightness_offset != 0:
        scaled_target = np.clip(scaled_target.astype(np.int16) + brightness_offset, 0, 255).astype(np.uint8)

    # Apply noise
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, scaled_target.shape).astype(np.int16)
        scaled_target = np.clip(scaled_target.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Place in scene
    px, py = pos
    px = max(0, min(scene_width - tw, px))
    py = max(0, min(scene_height - th, py))
    scene[py : py + th, px : px + tw] = scaled_target

    ground_truth = Rect(x=px, y=py, width=tw, height=th)
    return scene, ground_truth


class TestMultiScaleTemplateMatchingBasics(unittest.TestCase):
    """Verifies interface contracts, boundary handling, and typed returns."""

    def setUp(self):
        self.template = create_synthetic_target(50, 50)

    def test_empty_and_none_images_safe_return(self):
        """None, empty, or zero-dimension images must return TemplateMatchResult without exception."""
        # None inputs
        res = VisionPreprocessor.match_template_multiscale(None, self.template)
        self.assertFalse(res.found)
        self.assertEqual(res.confidence, 0.0)

        res2 = VisionPreprocessor.match_template_multiscale(self.template, None)
        self.assertFalse(res2.found)

        # Empty array
        empty = np.zeros((0, 0), dtype=np.uint8)
        res3 = VisionPreprocessor.match_template_multiscale(empty, self.template)
        self.assertFalse(res3.found)

    def test_template_larger_than_scene(self):
        """When template is larger than scene, it must not throw OpenCV exception."""
        small_scene = np.zeros((30, 30, 3), dtype=np.uint8)
        large_template = np.zeros((100, 100, 3), dtype=np.uint8)
        res = VisionPreprocessor.match_template_multiscale(small_scene, large_template, scales=[1.0, 1.5])
        self.assertFalse(res.found)
        self.assertEqual(res.confidence, 0.0)

    def test_single_scale_wrapper_convenience(self):
        """match_template() convenience wrapper works identically to 1.0x matching."""
        scene, gt = embed_target_in_scene(640, 480, self.template, scale=1.0, pos=(120, 80))
        res = VisionPreprocessor.match_template(scene, self.template, threshold=0.85)
        self.assertTrue(res.found)
        self.assertGreaterEqual(res.confidence, 0.90)
        self.assertAlmostEqual(res.x, gt.x, delta=3)
        self.assertAlmostEqual(res.y, gt.y, delta=3)
        self.assertEqual(res.scale, 1.0)
        self.assertIsInstance(res.rect, Rect)
        self.assertIsInstance(res.to_dict(), dict)


class TestResolutionInvarianceAndScales(unittest.TestCase):
    """Tests multi-scale matching across 720p, 1080p, 1440p, and 4K resolutions."""

    @classmethod
    def setUpClass(cls):
        cls.template = create_synthetic_target(60, 60)

    def test_720p_matching_at_varying_scales(self):
        """720p (1280x720) with target scaled at 0.8x, 1.0x, and 1.25x."""
        for target_scale in [0.8, 1.0, 1.25]:
            scene, gt = embed_target_in_scene(
                1280, 720, self.template, scale=target_scale, pos=(300, 200)
            )
            res = VisionPreprocessor.match_template_multiscale(
                scene, self.template, scales=[0.75, 0.8, 0.9, 1.0, 1.1, 1.25], threshold=0.75
            )
            self.assertTrue(res.found, f"Failed at 720p with scale={target_scale}")
            self.assertGreaterEqual(res.confidence, 0.80)
            self.assertAlmostEqual(res.scale, target_scale, delta=0.06)
            self.assertAlmostEqual(res.x, gt.x, delta=5)
            self.assertAlmostEqual(res.y, gt.y, delta=5)

    def test_1080p_standard_fullhd(self):
        """1080p (1920x1080) with target scaled at 1.15x."""
        scene, gt = embed_target_in_scene(
            1920, 1080, self.template, scale=1.15, pos=(850, 420)
        )
        res = VisionPreprocessor.match_template_multiscale(
            scene, self.template, scales=[0.85, 1.0, 1.15, 1.3], threshold=0.80
        )
        self.assertTrue(res.found)
        self.assertAlmostEqual(res.scale, 1.15, delta=0.05)
        self.assertAlmostEqual(res.x, gt.x, delta=5)
        self.assertAlmostEqual(res.y, gt.y, delta=5)

    def test_1440p_quadhd(self):
        """1440p (2560x1440) high resolution match."""
        scene, gt = embed_target_in_scene(
            2560, 1440, self.template, scale=1.3, pos=(1200, 700)
        )
        res = VisionPreprocessor.match_template_multiscale(
            scene, self.template, scales=[1.0, 1.15, 1.3], threshold=0.80
        )
        self.assertTrue(res.found)
        self.assertAlmostEqual(res.x, gt.x, delta=6)
        self.assertAlmostEqual(res.y, gt.y, delta=6)

    def test_4k_ultra_high_definition(self):
        """4K (3840x2160) UHD with downscaled template search space."""
        scene, gt = embed_target_in_scene(
            3840, 2160, self.template, scale=1.2, pos=(2100, 1100)
        )
        # Search using specific scale candidates
        res = VisionPreprocessor.match_template_multiscale(
            scene, self.template, scales=[1.0, 1.2], threshold=0.80
        )
        self.assertTrue(res.found)
        self.assertAlmostEqual(res.x, gt.x, delta=6)
        self.assertAlmostEqual(res.y, gt.y, delta=6)


class TestRobustnessNoiseAndLighting(unittest.TestCase):
    """Evaluates resilience to visual noise, brightness shift, and false-positive rejections."""

    def setUp(self):
        self.template = create_synthetic_target(60, 60)

    def test_brightness_variance_tolerance(self):
        """Target with +35 brightness shift maintains high correlation."""
        scene, gt = embed_target_in_scene(
            1280, 720, self.template, scale=1.0, pos=(400, 300), brightness_offset=35
        )
        res = VisionPreprocessor.match_template_multiscale(
            scene, self.template, scales=[0.9, 1.0, 1.1], threshold=0.75
        )
        self.assertTrue(res.found)
        self.assertGreaterEqual(res.confidence, 0.85)

    def test_gaussian_noise_tolerance(self):
        """Target with Gaussian pixel noise (sigma=15) correctly detected."""
        scene, gt = embed_target_in_scene(
            1280, 720, self.template, scale=1.0, pos=(400, 300), noise_sigma=15.0
        )
        res = VisionPreprocessor.match_template_multiscale(
            scene, self.template, scales=[0.95, 1.0, 1.05], threshold=0.70
        )
        self.assertTrue(res.found)
        self.assertAlmostEqual(res.x, gt.x, delta=6)

    def test_edge_based_matching_method(self):
        """Verifies method='edge' Canny edge detection matching."""
        scene, gt = embed_target_in_scene(
            1280, 720, self.template, scale=1.0, pos=(250, 180)
        )
        res = VisionPreprocessor.match_template_multiscale(
            scene, self.template, scales=[1.0], method="edge", threshold=0.60
        )
        self.assertTrue(res.found)
        self.assertEqual(res.method, "edge")
        self.assertAlmostEqual(res.x, gt.x, delta=4)

    def test_false_positive_rejection_on_unrelated_scene(self):
        """Unrelated scene without target must return found=False with low confidence."""
        empty_scene = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Fill with random textured noise
        np.random.seed(42)
        empty_scene[:] = np.random.randint(10, 50, (720, 1280, 3), dtype=np.uint8)

        res = VisionPreprocessor.match_template_multiscale(
            empty_scene, self.template, scales=[0.8, 1.0, 1.2], threshold=0.75
        )
        self.assertFalse(res.found)
        self.assertLess(res.confidence, 0.70)


class TestMultiScalePerformanceBenchmark(unittest.TestCase):
    """Repeatable benchmark testing latency across resolution and scale search iterations."""

    def test_benchmark_multi_scale_matching_latency(self):
        template = create_synthetic_target(50, 50)
        scene, _ = embed_target_in_scene(1920, 1080, template, scale=1.05, pos=(800, 450))

        # 1. Benchmark 5 scale evaluations across 5 trials (exhaustive scan)
        candidate_scales = [0.85, 0.95, 1.05, 1.15, 1.25]
        latencies_exhaustive = []

        for _ in range(5):
            t0 = time.perf_counter()
            res = VisionPreprocessor.match_template_multiscale(
                scene, template, scales=candidate_scales, threshold=0.75
            )
            latencies_exhaustive.append((time.perf_counter() - t0) * 1000.0)
            self.assertTrue(res.found)

        avg_lat_exh = sum(latencies_exhaustive) / len(latencies_exhaustive)
        max_lat_exh = max(latencies_exhaustive)

        print(f"\n[BENCHMARK] Multi-Scale Template Matching Exhaustive (1080p, 5 scales):")
        print(f"  Average Latency: {avg_lat_exh:.2f} ms (~{avg_lat_exh/5:.2f} ms/scale)")
        print(f"  Max Latency:     {max_lat_exh:.2f} ms")
        print(f"  Confidence:      {res.confidence:.4f}")
        print(f"  Detected Scale:  {res.scale:.2f}")

        # Full 1080p 5-scale exhaustive scan on CPU takes < 1000 ms (~80-180 ms/scale depending on CPU load)
        self.assertLess(avg_lat_exh, 1000.0, f"Exhaustive latency too high: {avg_lat_exh:.2f}ms")

        # 2. Benchmark accelerated search with early exit on high confidence
        latencies_early_exit = []
        for _ in range(5):
            t0 = time.perf_counter()
            res_fast = VisionPreprocessor.match_template_multiscale(
                scene, template, scales=candidate_scales, threshold=0.75, early_exit_threshold=0.95
            )
            latencies_early_exit.append((time.perf_counter() - t0) * 1000.0)
            self.assertTrue(res_fast.found)

        avg_lat_fast = sum(latencies_early_exit) / len(latencies_early_exit)
        max_lat_fast = max(latencies_early_exit)

        print(f"[BENCHMARK] Multi-Scale Template Matching Early-Exit (1080p, threshold=0.95):")
        print(f"  Average Latency: {avg_lat_fast:.2f} ms")
        print(f"  Max Latency:     {max_lat_fast:.2f} ms")
        print(f"  Confidence:      {res_fast.confidence:.4f}")

        # Early exit should be significantly faster than exhaustive search
        self.assertLessEqual(avg_lat_fast, avg_lat_exh)


if __name__ == "__main__":
    unittest.main()
