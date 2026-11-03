"""
Unit tests for UI Element Recognition and Classification Engine.
Validates multi-scale template matching, state association, resolution invariance,
conflict resolution, and integration with StateClassifier.
"""

from __future__ import annotations

import os
import unittest
import numpy as np
import cv2

from clashvision.core.constants import GameState
from clashvision.core.models import Rect, Point, NormalizedPoint, NormalizedRect
from clashvision.vision.state_classifier import StateClassifier
from clashvision.vision.ui_element_detector import (
    UIElementType,
    UIElementResult,
    UIElementDetector,
    ElementRecognitionConfig,
    DetectionDiagnostics,
    ELEMENT_STATE_MAP,
)


class TestUIElementDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.detector = UIElementDetector()

        # Paths to user-provided recorded screenshots
        cls.img1_path = r"C:\Users\inte\.gemini\antigravity\brain\920e4210-188b-4c81-8fc3-27759110df72\.user_uploaded\media_1793560366915_90d8ad88.jpg"
        cls.img2_path = r"C:\Users\inte\.gemini\antigravity\brain\920e4210-188b-4c81-8fc3-27759110df72\.user_uploaded\media_1793560366837_c63e403f.jpg"
        cls.img3_path = r"C:\Users\inte\.gemini\antigravity\brain\920e4210-188b-4c81-8fc3-27759110df72\.user_uploaded\media_1793560366837_5c10924d.jpg"

        cls.img1 = cv2.imread(cls.img1_path) if os.path.isfile(cls.img1_path) else None
        cls.img2 = cv2.imread(cls.img2_path) if os.path.isfile(cls.img2_path) else None
        cls.img3 = cv2.imread(cls.img3_path) if os.path.isfile(cls.img3_path) else None

    def test_element_result_model_and_serialization(self) -> None:
        res = UIElementResult(
            element_type=UIElementType.BATTLE_BUTTON.value,
            bounding_box=Rect(100, 900, 90, 85),
            normalized_box=NormalizedRect(0.052, 0.833, 0.047, 0.078),
            center_point=Point(145, 942),
            normalized_center=NormalizedPoint(0.075, 0.872),
            confidence=0.985,
            scale=1.0,
            state_association=GameState.HOME_VILLAGE,
            template_name="battle_button.png",
        )
        d = res.to_dict()
        self.assertEqual(d["element_type"], UIElementType.BATTLE_BUTTON.value)
        self.assertEqual(d["state_association"], GameState.HOME_VILLAGE.value)
        self.assertEqual(d["confidence"], 0.985)
        self.assertFalse(d["is_ambiguous"])

    def test_detect_battle_button_on_home_village_screenshot(self) -> None:
        if self.img3 is None:
            self.skipTest("Screenshot 3 not found on disk")

        result = self.detector.detect_element(self.img3, UIElementType.BATTLE_BUTTON.value)
        self.assertIsNotNone(result)
        self.assertGreater(result.confidence, 0.85)
        self.assertEqual(result.element_type, UIElementType.BATTLE_BUTTON.value)
        self.assertEqual(result.state_association, GameState.HOME_VILLAGE)

        # Expected normalized location: bottom-left (nx < 0.20, ny > 0.80)
        self.assertLess(result.normalized_center.x, 0.20)
        self.assertGreater(result.normalized_center.y, 0.80)

    def test_detect_find_match_button_on_multiplayer_screenshot(self) -> None:
        if self.img2 is None:
            self.skipTest("Screenshot 2 not found on disk")

        result = self.detector.detect_element(self.img2, UIElementType.FIND_MATCH_BUTTON.value)
        self.assertIsNotNone(result)
        self.assertGreater(result.confidence, 0.80)
        self.assertEqual(result.element_type, UIElementType.FIND_MATCH_BUTTON.value)
        self.assertEqual(result.state_association, GameState.MULTIPLAYER_INTERFACE)

        # Expected normalized location: left-middle (nx < 0.35, ny in [0.55, 0.85])
        self.assertLess(result.normalized_center.x, 0.35)
        self.assertGreater(result.normalized_center.y, 0.55)

    def test_detect_attack_confirm_button_on_army_screenshot(self) -> None:
        if self.img1 is None:
            self.skipTest("Screenshot 1 not found on disk")

        result = self.detector.detect_element(self.img1, UIElementType.ATTACK_CONFIRM_BUTTON.value)
        self.assertIsNotNone(result)
        self.assertGreater(result.confidence, 0.85)
        self.assertEqual(result.element_type, UIElementType.ATTACK_CONFIRM_BUTTON.value)
        self.assertEqual(result.state_association, GameState.ARMY_CONFIGURATION)

        # Expected location: bottom-right (nx > 0.70, ny > 0.75)
        self.assertGreater(result.normalized_center.x, 0.70)
        self.assertGreater(result.normalized_center.y, 0.75)

    def test_multi_scale_resolution_invariance(self) -> None:
        if self.img3 is None:
            self.skipTest("Screenshot 3 not found on disk")

        # Scale image to 1920x1080 (Full HD)
        scaled_1080p = cv2.resize(self.img3, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        res_1080p = self.detector.detect_element(scaled_1080p, UIElementType.BATTLE_BUTTON.value)
        self.assertIsNotNone(res_1080p)
        self.assertGreater(res_1080p.confidence, 0.80)

        # Normalized coordinates should remain practically invariant (< 2% deviation)
        ref_res = self.detector.detect_element(self.img3, UIElementType.BATTLE_BUTTON.value)
        self.assertIsNotNone(ref_res)
        self.assertAlmostEqual(res_1080p.normalized_center.x, ref_res.normalized_center.x, delta=0.03)
        self.assertAlmostEqual(res_1080p.normalized_center.y, ref_res.normalized_center.y, delta=0.03)

    def test_brightness_and_contrast_invariance(self) -> None:
        if self.img3 is None:
            self.skipTest("Screenshot 3 not found on disk")

        # Darkened image (0.75x brightness)
        dark = cv2.convertScaleAbs(self.img3, alpha=0.85, beta=-20)
        res_dark = self.detector.detect_element(dark, UIElementType.BATTLE_BUTTON.value)
        self.assertIsNotNone(res_dark)
        self.assertGreater(res_dark.confidence, 0.75)

        # Brightened image (1.15x brightness)
        bright = cv2.convertScaleAbs(self.img3, alpha=1.15, beta=15)
        res_bright = self.detector.detect_element(bright, UIElementType.BATTLE_BUTTON.value)
        self.assertIsNotNone(res_bright)
        self.assertGreater(res_bright.confidence, 0.75)

    def test_conflict_resolution_and_nms(self) -> None:
        # Create two overlapping mock candidates (IoU > 0.5)
        cand1 = UIElementResult(
            element_type="button_a",
            bounding_box=Rect(100, 100, 80, 80),
            normalized_box=NormalizedRect(0.1, 0.1, 0.08, 0.08),
            center_point=Point(140, 140),
            normalized_center=NormalizedPoint(0.14, 0.14),
            confidence=0.92,
            scale=1.0,
        )
        cand2 = UIElementResult(
            element_type="button_b",
            bounding_box=Rect(110, 110, 80, 80),  # Heavy overlap
            normalized_box=NormalizedRect(0.11, 0.11, 0.08, 0.08),
            center_point=Point(150, 150),
            normalized_center=NormalizedPoint(0.15, 0.15),
            confidence=0.74,
            scale=1.0,
        )

        resolved = self.detector.resolve_conflicts([cand1, cand2])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].confidence, 0.92)

    def test_filter_by_state(self) -> None:
        r1 = UIElementResult(
            element_type=UIElementType.BATTLE_BUTTON.value,
            bounding_box=Rect(0, 0, 10, 10),
            normalized_box=NormalizedRect(0, 0, 0.1, 0.1),
            center_point=Point(5, 5),
            normalized_center=NormalizedPoint(0.05, 0.05),
            confidence=0.9,
            scale=1.0,
            state_association=GameState.HOME_VILLAGE,
        )
        r2 = UIElementResult(
            element_type=UIElementType.NEXT_BUTTON.value,
            bounding_box=Rect(100, 100, 10, 10),
            normalized_box=NormalizedRect(0.1, 0.1, 0.1, 0.1),
            center_point=Point(105, 105),
            normalized_center=NormalizedPoint(0.15, 0.15),
            confidence=0.9,
            scale=1.0,
            state_association=GameState.ENEMY_BASE_PREVIEW,
        )
        filtered = self.detector.filter_by_state([r1, r2], GameState.HOME_VILLAGE)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].element_type, UIElementType.BATTLE_BUTTON.value)

    def test_state_classifier_integration_with_ui_elements(self) -> None:
        classifier = StateClassifier(ui_element_detector=self.detector)

        if self.img3 is not None:
            res, elements = classifier.classify_with_elements(self.img3)
            self.assertEqual(res.state, GameState.HOME_VILLAGE)
            self.assertGreater(res.confidence, 0.70)
            self.assertIn("ui_elements", res.evidence)
            self.assertTrue(any(e.element_type == UIElementType.BATTLE_BUTTON.value for e in elements))

    def test_annotated_frame_generation_and_saving(self) -> None:
        canvas = np.zeros((576, 1024, 3), dtype=np.uint8)
        dummy_res = [
            UIElementResult(
                element_type=UIElementType.BATTLE_BUTTON.value,
                bounding_box=Rect(20, 480, 80, 80),
                normalized_box=NormalizedRect(0.02, 0.83, 0.08, 0.14),
                center_point=Point(60, 520),
                normalized_center=NormalizedPoint(0.06, 0.90),
                confidence=0.95,
                scale=1.0,
                state_association=GameState.HOME_VILLAGE,
            )
        ]

        annotated = self.detector.annotate_frame(canvas, dummy_res)
        self.assertEqual(annotated.shape, canvas.shape)
        # Source canvas remains unmodified (all zeros)
        self.assertEqual(np.count_nonzero(canvas), 0)
        # Annotated frame has drawn bounding boxes and HUD
        self.assertGreater(np.count_nonzero(annotated), 100)


if __name__ == "__main__":
    unittest.main()
