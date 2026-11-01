"""
Unit tests for Advanced Base Layout Analysis, Topology, Symmetry, and Difficulty Scoring.
Verifies Phase 4 improvements to BaseLayoutAnalyzer and BaseLayoutFeatures.
"""

import unittest
from types import SimpleNamespace
import numpy as np
import cv2

from clashvision.core.models import BaseLayoutFeatures
from clashvision.vision.layout_analyzer import BaseLayoutAnalyzer


class TestBaseTopologyAndLayoutAnalysis(unittest.TestCase):
    """Test suite covering base symmetry, resource distribution, entry points, and difficulty."""

    def setUp(self):
        self.analyzer = BaseLayoutAnalyzer(reference_width=1920, reference_height=1080)

    def test_empty_and_none_source_handling(self):
        """Verifies graceful defaults when input is None or empty."""
        features_none = self.analyzer.analyze_layout(None)
        self.assertIsInstance(features_none, BaseLayoutFeatures)
        self.assertEqual(features_none.difficulty_score, 0.0)

        symm_none = self.analyzer.compute_base_symmetry(None)
        self.assertEqual(symm_none, 0.0)

        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        features_empty = self.analyzer.analyze_layout(empty_img)
        self.assertEqual(features_empty.compartment_count, 0)

    def test_synthetic_symmetric_and_asymmetric_models(self):
        """Verifies symmetry calculation on structured BaseLayoutModel-like objects."""
        # Perfectly 4-way symmetric layout
        symm_buildings = [
            SimpleNamespace(x=0.3, y=0.3, name="Cannon", is_defense=True),
            SimpleNamespace(x=0.7, y=0.3, name="Cannon", is_defense=True),
            SimpleNamespace(x=0.3, y=0.7, name="Cannon", is_defense=True),
            SimpleNamespace(x=0.7, y=0.7, name="Cannon", is_defense=True),
        ]
        symm_model = SimpleNamespace(buildings=symm_buildings, wall_coordinates=[])
        symm_score = self.analyzer.compute_base_symmetry(symm_model)
        self.assertGreaterEqual(symm_score, 0.90)

        # Strongly asymmetric layout
        asymm_buildings = [
            SimpleNamespace(x=0.1, y=0.2, name="Cannon", is_defense=True),
            SimpleNamespace(x=0.15, y=0.8, name="Mortar", is_defense=True),
            SimpleNamespace(x=0.4, y=0.3, name="Storage", is_storage=True),
            SimpleNamespace(x=0.9, y=0.9, name="TH", is_town_hall=True),
        ]
        asymm_model = SimpleNamespace(buildings=asymm_buildings, wall_coordinates=[])
        asymm_score = self.analyzer.compute_base_symmetry(asymm_model)
        self.assertLess(asymm_score, 0.60)

    def test_image_based_symmetry_scoring(self):
        """Verifies CV2 template-based symmetry calculation on rendered image matrices."""
        # Horizontally symmetric canvas
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (60, 100), 25, (0, 255, 0), -1)
        cv2.circle(img, (140, 100), 25, (0, 255, 0), -1)

        score = self.analyzer.compute_base_symmetry(img)
        self.assertGreaterEqual(score, 0.85)

    def test_resource_distribution_core_vs_perimeter(self):
        """Verifies core storages vs perimeter collectors classification."""
        buildings = [
            # Core storages (centered at 0.5, 0.5)
            SimpleNamespace(x=0.48, y=0.48, name="Gold Storage", is_storage=True, is_exposed=False),
            SimpleNamespace(x=0.52, y=0.52, name="Elixir Storage", is_storage=True, is_exposed=False),
            # Perimeter collectors (outer perimeter)
            SimpleNamespace(x=0.1, y=0.1, name="Gold Mine", is_collector=True, is_exposed=True),
            SimpleNamespace(x=0.9, y=0.9, name="Elixir Collector", is_collector=True, is_exposed=True),
        ]
        walls = [(int(0.4 * 1920) + i * 10, int(0.4 * 1080)) for i in range(50)]
        model = SimpleNamespace(
            buildings=buildings,
            wall_coordinates=walls,
            th_accessible=False,
            layout_type="FARMING_CORE",
        )

        features = self.analyzer.analyze_layout(model)
        self.assertEqual(features.metadata["core_resources"], 2)
        self.assertEqual(features.metadata["perimeter_resources"], 2)
        self.assertAlmostEqual(features.core_resource_ratio, 0.5, delta=0.05)
        self.assertAlmostEqual(features.perimeter_resource_ratio, 0.5, delta=0.05)

        dist = self.analyzer.analyze_resource_distribution(model)
        self.assertEqual(dist["core_resource_ratio"], features.core_resource_ratio)
        self.assertEqual(dist["perimeter_resource_ratio"], features.perimeter_resource_ratio)

    def test_entry_point_and_hazard_evaluation(self):
        """Verifies directional hazard rating and optimal entry selection."""
        buildings = [
            # Heavy defenses concentrated on North side
            SimpleNamespace(x=0.5, y=0.2, name="Inferno Tower", is_defense=True),
            SimpleNamespace(x=0.48, y=0.22, name="Eagle Artillery", is_defense=True),
            SimpleNamespace(x=0.52, y=0.18, name="Air Defense", is_defense=True, is_air_defense=True),
            # South side has collectors with zero defenses
            SimpleNamespace(x=0.5, y=0.85, name="Gold Mine", is_collector=True, is_exposed=True),
            SimpleNamespace(x=0.52, y=0.88, name="Elixir Collector", is_collector=True, is_exposed=True),
        ]
        model = SimpleNamespace(buildings=buildings, wall_coordinates=[], th_accessible=False)

        entry_points = self.analyzer.evaluate_entry_points(model)
        self.assertEqual(len(entry_points), 4)

        north = next(e for e in entry_points if e["sector"] == "North")
        south = next(e for e in entry_points if e["sector"] == "South")

        self.assertGreater(north["hazard_score"], south["hazard_score"])
        self.assertGreater(north["defenses_count"], 0)
        self.assertEqual(south["defenses_count"], 0)
        self.assertTrue(south["recommended"], "South should be recommended due to zero hazard and exposed loot")

    def test_difficulty_scoring_dead_vs_fortified_base(self):
        """Verifies base difficulty scoring produces expected contrasts between dead and fortified bases."""
        # Dead base: exposed collectors, accessible TH, no walls
        dead_base = SimpleNamespace(
            buildings=[SimpleNamespace(x=0.1, y=0.1, name="Mine", is_collector=True, is_exposed=True)],
            wall_coordinates=[],
            th_accessible=True,
            dead_base=True,
            exposed_collector_ratio=1.0,
        )
        dead_features = self.analyzer.analyze_layout(dead_base)
        dead_diff = self.analyzer.compute_difficulty_score(dead_base, dead_features)

        # Heavily fortified base: 100 walls, inaccessible TH, zero exposed collectors
        fortified_base = SimpleNamespace(
            buildings=[
                SimpleNamespace(x=0.5, y=0.5, name="TH", is_town_hall=True),
                SimpleNamespace(x=0.48, y=0.48, name="Eagle", is_defense=True),
            ],
            wall_coordinates=[(i, i) for i in range(100)],
            th_accessible=False,
            dead_base=False,
            exposed_collector_ratio=0.0,
        )
        fortified_features = self.analyzer.analyze_layout(fortified_base)
        fortified_diff = self.analyzer.compute_difficulty_score(fortified_base, fortified_features)

        self.assertLess(dead_diff, 0.40)
        self.assertGreater(fortified_diff, 0.65)
        self.assertGreater(fortified_diff, dead_diff)


if __name__ == "__main__":
    unittest.main()

