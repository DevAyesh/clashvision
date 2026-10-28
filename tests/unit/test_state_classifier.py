"""
Unit tests for ClashVision 12-Game-State Classifier and Debouncer.
Verifies Requirement R2 and Features 6, 7, 8, 15:
- All 12 game states classified correctly on 1080p and 720p frames
- Confidence score strictly bounded in [0.0, 1.0]
- Structured explanatory evidence dictionary
- 2-frame debouncing filter eliminating transient noise
- Noise resistance and empty frame resilience
- BaseLayoutAnalyzer integration
"""

import unittest
from typing import Dict, Any
import cv2
import numpy as np
import pytest

from clashvision.core.constants import GameState
from clashvision.core.models import VisionAnalysisResult, BaseLayoutFeatures
from clashvision.vision.state_classifier import StateClassifier, StateDebouncer
from clashvision.vision.layout_analyzer import BaseLayoutAnalyzer
from tests.fixtures.synthetic_generator import SyntheticGenerator, GameStateType
from tests.fixtures.layout_generator import LayoutGenerator, BaseLayoutType


class TestGameStateClassifierAll12States(unittest.TestCase):
    """Verifies classification accuracy for all 12 game states on 1080p and 720p."""

    @classmethod
    def setUpClass(cls):
        cls.gen = SyntheticGenerator(default_seed=42)
        cls.clf = StateClassifier(enable_debouncing=False)

    def test_classify_all_12_states_1080p(self):
        for state in GameState:
            frame = self.gen.generate_frame(
                state=GameStateType(state.value), width=1920, height=1080
            )
            result = self.clf.classify(frame)
            self.assertIsInstance(result, VisionAnalysisResult)
            self.assertEqual(
                result.state,
                state,
                f"1080p State mismatch: expected {state.value}, got {result.state.value} (evidence={result.evidence})",
            )
            self.assertTrue(
                0.0 <= result.confidence <= 1.0,
                f"Confidence out of range: {result.confidence}",
            )
            if state != GameState.UNKNOWN:
                self.assertGreaterEqual(
                    result.confidence,
                    0.65,
                    f"Confidence too low for {state.value}: {result.confidence}",
                )
            else:
                self.assertLessEqual(
                    result.confidence,
                    0.35,
                    f"Confidence too high for UNKNOWN: {result.confidence}",
                )

    def test_classify_all_12_states_720p(self):
        for state in GameState:
            frame = self.gen.generate_frame(
                state=GameStateType(state.value), width=1280, height=720
            )
            result = self.clf.classify(frame)
            self.assertEqual(
                result.state,
                state,
                f"720p State mismatch: expected {state.value}, got {result.state.value}",
            )


class TestStateClassifierEvidenceAndConfidence(unittest.TestCase):
    """Verifies confidence score bounds and supporting evidence dictionary."""

    def setUp(self):
        self.gen = SyntheticGenerator()
        self.clf = StateClassifier(enable_debouncing=False)

    def test_home_village_evidence_contains_landmarks(self):
        frame = self.gen.generate_frame(state=GameStateType.HOME_VILLAGE)
        result = self.clf.classify(frame)
        self.assertEqual(result.state, GameState.HOME_VILLAGE)
        self.assertIn("grass_ratio", result.evidence)
        self.assertIn("attack_button_detected", result.evidence)
        self.assertTrue(result.evidence["attack_button_detected"])

    def test_enemy_preview_evidence_contains_loot_box(self):
        frame = self.gen.generate_frame(state=GameStateType.ENEMY_BASE_PREVIEW)
        result = self.clf.classify(frame)
        self.assertEqual(result.state, GameState.ENEMY_BASE_PREVIEW)
        self.assertIn("loot_box_top_left", result.evidence)
        self.assertTrue(result.evidence["loot_box_top_left"])

    def test_battle_in_progress_evidence_contains_timer(self):
        frame = self.gen.generate_frame(state=GameStateType.BATTLE_IN_PROGRESS)
        result = self.clf.classify(frame)
        self.assertEqual(result.state, GameState.BATTLE_IN_PROGRESS)
        self.assertIn("end_battle_button", result.evidence)
        self.assertTrue(result.evidence["end_battle_button"])

    def test_matchmaking_evidence_contains_swirl(self):
        frame = self.gen.generate_frame(state=GameStateType.MATCHMAKING_INTERFACE)
        result = self.clf.classify(frame)
        self.assertEqual(result.state, GameState.MATCHMAKING_INTERFACE)
        self.assertGreater(result.evidence.get("white_ratio", 0), 0.5)

    def test_evidence_dictionary_serializability(self):
        frame = self.gen.generate_frame(state=GameStateType.LOADING_SCREEN)
        result = self.clf.classify(frame)
        d = result.to_dict()
        self.assertEqual(d["state"], "LOADING_SCREEN")
        self.assertIsInstance(d["evidence"], dict)


class TestStateDebouncing(unittest.TestCase):
    """Verifies 2-frame temporal consistency filter eliminating transient flickers (Feature 8)."""

    def test_debouncer_initial_state_unknown(self):
        deb = StateDebouncer(required_consecutive=2)
        self.assertEqual(deb.current_state, GameState.UNKNOWN)

    def test_debouncer_single_frame_flicker_ignored(self):
        deb = StateDebouncer(required_consecutive=2)
        # Establish HOME_VILLAGE
        deb.update(GameState.HOME_VILLAGE)
        deb.update(GameState.HOME_VILLAGE)
        self.assertEqual(deb.current_state, GameState.HOME_VILLAGE)

        # 1-frame glitch to UNEXPECTED_POPUP
        res = deb.update(GameState.UNEXPECTED_POPUP)
        self.assertEqual(res, GameState.HOME_VILLAGE)
        self.assertEqual(deb.current_state, GameState.HOME_VILLAGE)

    def test_debouncer_transitions_on_two_consecutive_frames(self):
        deb = StateDebouncer(required_consecutive=2)
        deb.update(GameState.HOME_VILLAGE)
        deb.update(GameState.HOME_VILLAGE)
        self.assertEqual(deb.current_state, GameState.HOME_VILLAGE)

        # First frame of new state -> unchanged
        deb.update(GameState.BATTLE_IN_PROGRESS)
        self.assertEqual(deb.current_state, GameState.HOME_VILLAGE)

        # Second consecutive frame -> transitions
        deb.update(GameState.BATTLE_IN_PROGRESS)
        self.assertEqual(deb.current_state, GameState.BATTLE_IN_PROGRESS)

    def test_debouncer_rapid_flicker_stability(self):
        deb = StateDebouncer(required_consecutive=2)
        deb.update(GameState.HOME_VILLAGE)
        deb.update(GameState.HOME_VILLAGE)

        for _ in range(5):
            deb.update(GameState.LOADING_SCREEN)
            deb.update(GameState.HOME_VILLAGE)

        self.assertEqual(deb.current_state, GameState.HOME_VILLAGE)

    def test_debouncer_reset(self):
        deb = StateDebouncer(required_consecutive=2)
        deb.update(GameState.ENEMY_BASE_PREVIEW)
        deb.update(GameState.ENEMY_BASE_PREVIEW)
        self.assertEqual(deb.current_state, GameState.ENEMY_BASE_PREVIEW)

        deb.reset()
        self.assertEqual(deb.current_state, GameState.UNKNOWN)

    def test_classifier_with_debouncing_integration(self):
        clf = StateClassifier(enable_debouncing=True, debounce_frames=2)
        gen = SyntheticGenerator()

        frame_home = gen.generate_frame(state=GameStateType.HOME_VILLAGE)
        frame_enemy = gen.generate_frame(state=GameStateType.ENEMY_BASE_PREVIEW)

        # Frame 1: Home
        r1 = clf.classify(frame_home)
        self.assertEqual(r1.state, GameState.UNKNOWN)  # Needs 2 frames

        # Frame 2: Home
        r2 = clf.classify(frame_home)
        self.assertEqual(r2.state, GameState.HOME_VILLAGE)

        # Frame 3: 1-frame enemy flicker
        r3 = clf.classify(frame_enemy)
        self.assertEqual(r3.state, GameState.HOME_VILLAGE)  # Ignored!

        # Frame 4: Second enemy frame
        r4 = clf.classify(frame_enemy)
        self.assertEqual(r4.state, GameState.ENEMY_BASE_PREVIEW)  # Transition!


class TestStateClassifierResilience(unittest.TestCase):
    """Verifies handling of noisy, corrupted, or empty frames."""

    def setUp(self):
        self.gen = SyntheticGenerator()
        self.clf = StateClassifier(enable_debouncing=False)

    def test_noisy_home_village_frame(self):
        frame = self.gen.generate_frame(
            state=GameStateType.HOME_VILLAGE,
            add_noise=True,
            noise_sigma=15.0,
        )
        result = self.clf.classify(frame)
        self.assertEqual(result.state, GameState.HOME_VILLAGE)

    def test_noisy_enemy_preview_frame(self):
        frame = self.gen.generate_frame(
            state=GameStateType.ENEMY_BASE_PREVIEW,
            add_noise=True,
            noise_sigma=15.0,
        )
        result = self.clf.classify(frame)
        self.assertEqual(result.state, GameState.ENEMY_BASE_PREVIEW)

    def test_empty_frame_returns_unknown(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        result = self.clf.classify(empty)
        self.assertEqual(result.state, GameState.UNKNOWN)
        self.assertEqual(result.confidence, 0.0)

    def test_none_frame_returns_unknown(self):
        result = self.clf.classify(None)  # type: ignore
        self.assertEqual(result.state, GameState.UNKNOWN)
        self.assertEqual(result.confidence, 0.0)

    def test_tiny_frame_returns_unknown(self):
        tiny = np.zeros((10, 10, 3), dtype=np.uint8)
        result = self.clf.classify(tiny)
        self.assertEqual(result.state, GameState.UNKNOWN)


class TestBaseLayoutAnalyzerIntegration(unittest.TestCase):
    """Verifies BaseLayoutAnalyzer extraction of collector exposure, TH accessibility, and ADs."""

    def setUp(self):
        self.analyzer = BaseLayoutAnalyzer()

    def test_exposed_collectors_layout(self):
        model = LayoutGenerator.generate_layout(BaseLayoutType.EXPOSED_COLLECTORS)
        features = self.analyzer.analyze_layout(model)
        self.assertIsInstance(features, BaseLayoutFeatures)
        self.assertAlmostEqual(features.exposed_collector_ratio, 1.0, places=2)
        self.assertEqual(len(features.air_defense_locations), 4)

    def test_core_storages_layout(self):
        model = LayoutGenerator.generate_layout(BaseLayoutType.CORE_STORAGES)
        features = self.analyzer.analyze_layout(model)
        self.assertAlmostEqual(features.exposed_collector_ratio, 0.0, places=2)

    def test_town_hall_exposed_layout(self):
        model = LayoutGenerator.generate_layout(BaseLayoutType.TOWN_HALL_EXPOSED)
        features = self.analyzer.analyze_layout(model)
        self.assertTrue(features.th_accessible)

    def test_rendered_map_image_analysis(self):
        model = LayoutGenerator.generate_layout(BaseLayoutType.AIR_DEFENSE_CORRIDOR)
        img = model.render_map(size=600)
        features = self.analyzer.analyze_layout(img)
        self.assertIsInstance(features, BaseLayoutFeatures)
        self.assertGreaterEqual(len(features.air_defense_locations), 2)

    def test_image_zero_collectors_defaults_to_zero(self):
        blank = np.zeros((300, 300, 3), dtype=np.uint8)
        features = self.analyzer.analyze_layout(blank)
        self.assertEqual(features.exposed_collector_ratio, 0.0)

    def test_town_hall_max_contour_selection(self):
        img = np.zeros((600, 600, 3), dtype=np.uint8)
        # Small red patch at (100, 100) size 11x11 (area 121)
        cv2.rectangle(img, (95, 95), (106, 106), (0, 0, 200), -1)
        # Large red TH at outer area (500, 500) size 30x30 (area 900)
        cv2.rectangle(img, (485, 485), (515, 515), (0, 0, 200), -1)
        features = self.analyzer.analyze_layout(img)
        self.assertIsNotNone(features.metadata.get("th_coordinates"))
        th_cx, th_cy = features.metadata["th_coordinates"]
        self.assertAlmostEqual(th_cx, 500, delta=15)
        self.assertAlmostEqual(th_cy, 500, delta=15)
        self.assertTrue(features.th_accessible)


if __name__ == "__main__":
    unittest.main()
