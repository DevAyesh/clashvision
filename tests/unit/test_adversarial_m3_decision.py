"""
Milestone 3 Adversarial Stress & Vulnerability Test Suite for ClashVision Decision Subsystem.
Covers:
- TestDecisionRobustnessPassing: Boundary stability, extreme resource scaling, deterministic rankings,
  custom configuration parsing, layout geometry heuristics.
- TestDecisionVulnerabilities: Empirical reproduction of discovered bugs, crashes, and contract violations:
  1. ZeroDivisionError crash on zero canvas width/height (minimized/collapsed window).
  2. Viable primary CustomStrategy emitting empty deployment plan ([]) when army_composition is empty.
  3. Negative unit count leakage (unit_count=-2) and zero-count hallucination in CustomStrategy.
  4. Zero-resource base receiving non-zero suitability score and full attack deployment plan.
  5. Negative score leakage (score < 0.0) in Sneaky Goblin / BARCH on negative OCR values.
  6. NaN score leakage into StrategyEvaluation rankings corrupting sorting.

Derived from Requirement R3, Features 14-21, and Dispatch Instructions.
"""

from __future__ import annotations
import math
import unittest
import pytest
from typing import Any, Dict, List, Optional, Tuple

from clashvision.core.constants import (
    ResourcePriority,
    RiskLevel,
    StrategyType,
    VIRTUAL_REFERENCE_WIDTH,
    VIRTUAL_REFERENCE_HEIGHT,
)
from clashvision.core.models import (
    BaseLayoutFeatures,
    DeploymentStep,
    FarmingRecommendation,
    ResourceReading,
    StrategyEvaluation,
)
from clashvision.core.config import ResourceThresholds
from clashvision.decision.evaluator import BaseEvaluator, BaseEvaluationResult
from clashvision.decision.base_strategy import BaseStrategy
from clashvision.decision.sneaky_goblin import SneakyGoblinStrategy
from clashvision.decision.barch import BarchStrategy
from clashvision.decision.mass_miner import MassMinerStrategy
from clashvision.decision.mass_dragon import MassDragonStrategy
from clashvision.decision.custom_strategy import CustomStrategy, CustomStrategyConfig
from clashvision.decision.recommendation_engine import RecommendationEngine
from tests.fixtures.layout_generator import (
    BaseLayoutType,
    LayoutGenerator,
    generate_canonical_layout,
)
from clashvision.vision.layout_analyzer import BaseLayoutAnalyzer


class TestDecisionRobustnessPassing(unittest.TestCase):
    """Verifies that the decision subsystem holds its invariant contracts under boundary inputs."""

    def setUp(self) -> None:
        self.evaluator = BaseEvaluator()
        self.engine = RecommendationEngine(self.evaluator)
        self.analyzer = BaseLayoutAnalyzer()

    def test_none_resource_reading_evaluates_to_zero(self) -> None:
        """ResourceReading with all None fields evaluates to 0.0 with empty deployment plan."""
        reading = ResourceReading(gold=None, elixir=None, dark_elixir=None)
        res = self.evaluator.evaluate(reading)
        self.assertEqual(res.normalized_score, 0.0)
        self.assertFalse(res.meets_thresholds)

        rec = self.engine.evaluate_base(reading)
        self.assertEqual(rec.suitability_score, 0.0)
        self.assertEqual(len(rec.deployment_plan), 0)
        self.assertIn("No harvestable resources detected", rec.explanation)

    def test_completely_none_inputs_handled_safely(self) -> None:
        """Calling evaluate_base(None, None) executes without exception and returns 0 suitability."""
        rec = self.engine.evaluate_base(None, None)
        self.assertIsInstance(rec, FarmingRecommendation)
        self.assertEqual(rec.suitability_score, 0.0)
        self.assertEqual(len(rec.deployment_plan), 0)

    def test_multi_billion_resources_bounded_in_score_range(self) -> None:
        """Multi-billion resource values produce scores in [95.0, 100.0] without overflow."""
        huge_reading = ResourceReading(
            gold=99_999_999_999,
            elixir=88_888_888_888,
            dark_elixir=50_000_000,
        )
        eval_res = self.evaluator.evaluate(huge_reading)
        self.assertEqual(eval_res.normalized_score, 100.0)
        self.assertTrue(eval_res.meets_thresholds)

        rec = self.engine.evaluate_base(huge_reading)
        self.assertGreaterEqual(rec.suitability_score, 95.0)
        self.assertLessEqual(rec.suitability_score, 100.0)
        self.assertGreater(len(rec.deployment_plan), 0)
        self.assertIn("99,999,999,999", rec.explanation)

    def test_asymmetric_negative_and_positive_resources(self) -> None:
        """One negative resource and one positive resource clamps negative to 0."""
        reading = ResourceReading(gold=-500_000, elixir=1_000_000, dark_elixir=None)
        eval_res = self.evaluator.evaluate(reading)
        self.assertGreater(eval_res.normalized_score, 0.0)
        self.assertLessEqual(eval_res.normalized_score, 100.0)

    def test_zero_thresholds_allows_any_positive_loot(self) -> None:
        """Thresholds set to 0 treat any positive resource total as satisfying threshold."""
        thresh_zero = ResourceThresholds(min_gold=0, min_elixir=0, min_dark_elixir=0)
        reading = ResourceReading(gold=100, elixir=100, dark_elixir=1)
        eval_res = self.evaluator.evaluate(reading, thresholds=thresh_zero)
        self.assertTrue(eval_res.meets_thresholds)
        self.assertGreater(eval_res.normalized_score, 0.0)

    def test_negative_thresholds_clamped_to_zero(self) -> None:
        """Negative thresholds clamp to 0 and do not divide by negative numbers."""
        thresh_neg = ResourceThresholds(min_gold=-500_000, min_elixir=-200_000, min_dark_elixir=-10)
        reading = ResourceReading(gold=50_000, elixir=50_000, dark_elixir=100)
        eval_res = self.evaluator.evaluate(reading, thresholds=thresh_neg)
        self.assertTrue(eval_res.meets_thresholds)

    def test_zero_resource_weights_no_division_by_zero(self) -> None:
        """All weights set to 0.0 does not cause ZeroDivisionError in BaseEvaluator."""
        thresh_zero_w = ResourceThresholds(gold_weight=0.0, elixir_weight=0.0, dark_elixir_weight=0.0)
        reading = ResourceReading(gold=500_000, elixir=500_000, dark_elixir=5_000)
        eval_res = self.evaluator.evaluate(reading, thresholds=thresh_zero_w)
        self.assertGreaterEqual(eval_res.normalized_score, 0.0)
        self.assertLessEqual(eval_res.normalized_score, 100.0)

    def test_extreme_priority_switches(self) -> None:
        """Priority switches correctly gate meets_thresholds under single-resource peaks."""
        reading = ResourceReading(gold=1_000_000, elixir=0, dark_elixir=0)

        t_gold = ResourceThresholds(priority=ResourcePriority.GOLD_FIRST, min_gold=400_000)
        self.assertTrue(self.evaluator.evaluate(reading, thresholds=t_gold).meets_thresholds)

        t_elixir = ResourceThresholds(priority=ResourcePriority.ELIXIR_FIRST, min_elixir=400_000)
        self.assertFalse(self.evaluator.evaluate(reading, thresholds=t_elixir).meets_thresholds)

        t_de = ResourceThresholds(priority=ResourcePriority.DARK_ELIXIR_FIRST, min_dark_elixir=3_000)
        self.assertFalse(self.evaluator.evaluate(reading, thresholds=t_de).meets_thresholds)

    def test_fuzzed_troop_names_deployment_generation(self) -> None:
        """Adversarial troop names (Unicode, SQLi, control chars) generate valid steps."""
        fuzz_troops = {
            "⚔️ Super Barbarian 👑": 20,
            "'; DROP TABLE recommendations; --": 10,
            "\x00\r\n\t": 5,
            "A" * 5000: 2,
            "Normal Spell": 2,
        }
        config = CustomStrategyConfig(name="Fuzzed Troop Strategy", army_composition=fuzz_troops)
        strategy = CustomStrategy(config)
        layout = BaseLayoutFeatures(exposed_collector_ratio=0.5)

        plan = strategy.generate_deployment_plan(layout, canvas_width=1920, canvas_height=1080)
        self.assertGreater(len(plan), 0)
        for step in plan:
            self.assertIsInstance(step.troop_type, str)
            self.assertGreater(step.unit_count, 0)
            self.assertGreaterEqual(step.delay_ms, 0)

    def test_conflicting_custom_prerequisites(self) -> None:
        """Conflicting prerequisites mark viable=False and explain reasons."""
        config = CustomStrategyConfig(
            requires_accessible_th=True,
            max_acceptable_risk=RiskLevel.LOW,
            preferred_layout_types=["NON_EXISTENT_LAYOUT"],
        )
        strategy = CustomStrategy(config)
        layout = BaseLayoutFeatures(
            exposed_collector_ratio=0.1,
            th_accessible=False,
            metadata={"layout_type": "CORE_STORAGES"},
        )
        reading = ResourceReading(gold=1_000_000, elixir=1_000_000, dark_elixir=5_000)
        ev = strategy.evaluate(reading, layout)
        self.assertFalse(ev.viable)
        self.assertFalse(ev.prerequisites_met)

    def test_malformed_custom_steps_parsed_safely(self) -> None:
        """Malformed custom steps with missing fields or string numbers parse safely."""
        malformed_steps = [
            {},
            {"troop_type": "Custom Unit", "normalized_x": -5.0, "normalized_y": 10.0},
            {"unit_count": "10", "delay_ms": "500"},
            {"note": None},
        ]
        config = CustomStrategyConfig(custom_steps=malformed_steps)
        strategy = CustomStrategy(config)
        plan = strategy.generate_deployment_plan(None, canvas_width=1920, canvas_height=1080)
        self.assertEqual(len(plan), len(malformed_steps))

    def test_deterministic_rankings_across_repeats(self) -> None:
        """Calling evaluate_base 10 times consecutively produces identical scores and rankings."""
        reading = ResourceReading(gold=750_000, elixir=650_000, dark_elixir=5_000)
        layout = BaseLayoutFeatures(exposed_collector_ratio=0.6, compartment_count=4)

        base_rec = self.engine.evaluate_base(reading, layout)
        base_scores = [(e.strategy_type, e.score, e.viable) for e in base_rec.rankings]

        for _ in range(10):
            rec = self.engine.evaluate_base(reading, layout)
            scores = [(e.strategy_type, e.score, e.viable) for e in rec.rankings]
            self.assertEqual(scores, base_scores)

    def test_viable_canonical_bases_output_non_empty_deployment_plans(self) -> None:
        """All 6 canonical layout models produce viable primary strategies with non-empty deployment plans."""
        for layout_type in BaseLayoutType:
            model = generate_canonical_layout(layout_type)
            layout = self.analyzer.analyze_layout(model)
            reading = ResourceReading(
                gold=model.total_gold, elixir=model.total_elixir, dark_elixir=model.total_dark_elixir
            )
            rec = self.engine.evaluate_base(reading, layout)
            self.assertTrue(rec.rankings[0].viable)
            self.assertGreater(len(rec.deployment_plan), 0)

    def test_ranking_sort_order_strictly_descending(self) -> None:
        """Rankings must always be sorted by (viable, score) descending."""
        reading = ResourceReading(gold=500_000, elixir=500_000, dark_elixir=3_500)
        layout = BaseLayoutFeatures(exposed_collector_ratio=0.5, compartment_count=3)
        rec = self.engine.evaluate_base(reading, layout)
        for i in range(len(rec.rankings) - 1):
            curr = rec.rankings[i]
            nxt = rec.rankings[i + 1]
            self.assertGreaterEqual((curr.viable, curr.score), (nxt.viable, nxt.score))

    def test_high_resolution_canvas_scaling(self) -> None:
        """4K and 8K resolutions scale deployment targets within canvas bounds."""
        reading = ResourceReading(gold=800_000, elixir=800_000, dark_elixir=5_000)
        layout = BaseLayoutFeatures(exposed_collector_ratio=0.7)

        rec_4k = self.engine.evaluate_base(reading, layout, canvas_width=3840, canvas_height=2160)
        self.assertGreater(len(rec_4k.deployment_plan), 0)
        for step in rec_4k.deployment_plan:
            self.assertLessEqual(step.target_x, 3840)
            self.assertLessEqual(step.target_y, 2160)

    def test_calculate_perimeter_point_extremes(self) -> None:
        """Perimeter point calculation with extreme angles and large radius fractions."""
        px, py, nx, ny = BaseStrategy.calculate_perimeter_point(-100.0 * math.pi)
        self.assertTrue(0.0 <= nx <= 1.0)
        self.assertTrue(0.0 <= ny <= 1.0)

        px, py, nx, ny = BaseStrategy.calculate_perimeter_point(0.0, radius_fraction=10.0)
        self.assertLessEqual(nx, 0.98)
        self.assertGreaterEqual(nx, 0.02)

    def test_find_safest_corridor_angle_extremes(self) -> None:
        """Safest corridor angle with hazard at center (0.5, 0.5) and 1000 dense hazards."""
        # Hazard at exact center (0.5, 0.5)
        angle_center = BaseStrategy.find_safest_corridor_angle([(0.5, 0.5)])
        self.assertTrue(0.0 <= angle_center <= 2.0 * math.pi)

        # 1,000 hazards in East sector
        dense_hazards = [(0.75, 0.5) for _ in range(1000)]
        safest_angle = BaseStrategy.find_safest_corridor_angle(dense_hazards)
        self.assertTrue(0.0 <= safest_angle <= 2.0 * math.pi)


class TestDecisionVulnerabilities(unittest.TestCase):
    """
    Verifies remediation and hardening against the 6 vulnerabilities discovered in the decision subsystem.
    Asserts that all invalidation conditions hold and contracts are strictly maintained.
    """

    def setUp(self) -> None:
        self.engine = RecommendationEngine()

    def test_vuln_canvas_zero_division_crash(self) -> None:
        """
        REMEDIATION 1: ZeroDivisionError prevented when canvas_width=0 or canvas_height=0.
        When an emulator window is minimized/collapsed, client_rect is (0, 0, 0, 0).
        Engine handles zero canvas dimensions gracefully without crashing.
        """
        reading = ResourceReading(gold=800_000, elixir=800_000, dark_elixir=5_000)
        layout = BaseLayoutFeatures(exposed_collector_ratio=0.8)

        # Verified: Engine handles zero canvas dimensions gracefully without crashing
        rec = self.engine.evaluate_base(reading, layout, canvas_width=0, canvas_height=0)
        self.assertIsInstance(rec, FarmingRecommendation)
        self.assertGreater(len(rec.deployment_plan), 0)

    def test_vuln_empty_army_custom_strategy_emits_empty_deployment_plan(self) -> None:
        """
        REMEDIATION 2: CustomStrategy with army_composition={} and custom_steps=[]
        is evaluated as viable=False and prerequisites_met=False.
        RecommendationEngine falls back to viable strategies and never emits an empty plan.
        """
        cfg = CustomStrategyConfig(
            name="Empty Army Strategy",
            army_composition={},
            custom_steps=[],
            min_gold=100_000,
            min_elixir=100_000,
            min_dark_elixir=1_000,
            preferred_layout_types=["CUSTOM_PARADISE"],
        )
        self.engine.set_custom_config(cfg)
        reading = ResourceReading(gold=1_000_000, elixir=1_000_000, dark_elixir=10_000)
        layout = BaseLayoutFeatures(metadata={"layout_type": "CUSTOM_PARADISE"})

        rec = self.engine.evaluate_base(reading, layout)
        custom_eval = next(e for e in rec.rankings if e.strategy_type == StrategyType.CUSTOM)
        self.assertFalse(custom_eval.viable)
        self.assertFalse(custom_eval.prerequisites_met)
        self.assertEqual(custom_eval.layout_suitability, 0.0)
        self.assertNotEqual(rec.primary_strategy, StrategyType.CUSTOM)
        self.assertTrue(rec.rankings[0].viable)
        self.assertGreater(len(rec.deployment_plan), 0)

    def test_vuln_negative_and_zero_unit_count_corruption(self) -> None:
        """
        REMEDIATION 3: Configured troop counts <= 0 are skipped entirely.
        Never emits a DeploymentStep with unit_count <= 0 or phantom units.
        """
        cfg = CustomStrategyConfig(
            army_composition={"Barbarian": 0, "Rage Spell": -2},
        )
        strategy = CustomStrategy(cfg)
        plan = strategy.generate_deployment_plan(None)

        # Neither negative spell nor zero barbarian produces deployment steps
        self.assertEqual(len(plan), 0)

    def test_vuln_zero_resource_phantom_attack_plan_and_non_zero_score(self) -> None:
        """
        REMEDIATION 4: When gold=0, elixir=0, dark_elixir=0 on a base with layout features,
        BaseEvaluator returns normalized_score=0.0 and RecommendationEngine returns
        suitability_score=0.0 with deployment_plan=[] (no phantom attacks on empty bases).
        """
        zero_reading = ResourceReading(gold=0, elixir=0, dark_elixir=0)
        layout = BaseLayoutFeatures(exposed_collector_ratio=0.8, compartment_count=5)

        rec = self.engine.evaluate_base(zero_reading, layout)
        self.assertEqual(rec.suitability_score, 0.0)
        self.assertEqual(len(rec.deployment_plan), 0)
        self.assertIn("No harvestable resources detected", rec.explanation)

    def test_vuln_negative_score_leakage_in_sneaky_goblin(self) -> None:
        """
        REMEDIATION 5: Negative resource values are clamped to 0. All strategy scores
        and subscores are strictly bounded in [0.0, 100.0].
        """
        neg_reading = ResourceReading(gold=-500_000, elixir=-500_000, dark_elixir=-50_000)
        layout = BaseLayoutFeatures(exposed_collector_ratio=0.8, compartment_count=5)

        rec = self.engine.evaluate_base(neg_reading, layout)
        for e in rec.rankings:
            self.assertGreaterEqual(e.score, 0.0)
            self.assertLessEqual(e.score, 100.0)
            self.assertGreaterEqual(e.resource_suitability, 0.0)
            self.assertGreaterEqual(e.layout_suitability, 0.0)

    def test_vuln_nan_score_leakage_and_sort_corruption(self) -> None:
        """
        REMEDIATION 6: Float NaN in ResourceReading is sanitized. StrategyEvaluation.score
        contains no NaN values, maintaining deterministic sorting.
        """
        nan_reading = ResourceReading(gold=float("nan"), elixir=float("nan"), dark_elixir=float("nan"))
        rec = self.engine.evaluate_base(nan_reading)
        scores = [e.score for e in rec.rankings]
        self.assertFalse(any(math.isnan(s) for s in scores))
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 100.0)


if __name__ == "__main__":
    unittest.main()
