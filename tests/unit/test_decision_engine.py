"""
Unit tests for ClashVision Farming Decision Subsystem:
- BaseEvaluator scoring, priority weighting, threshold checks, layout modifiers
- RecommendationEngine multi-strategy ranking, risk vs reward, score breakdown, explainability
Derived from Requirement R3 and Features 14, 15, 21.
"""

from __future__ import annotations
import pytest

from clashvision.core.constants import (
    ResourcePriority,
    RiskLevel,
    StrategyType,
    DEFAULT_MIN_GOLD,
    DEFAULT_MIN_ELIXIR,
    DEFAULT_MIN_DARK_ELIXIR,
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
from clashvision.decision.recommendation_engine import RecommendationEngine
from clashvision.decision.custom_strategy import CustomStrategyConfig
from tests.fixtures.layout_generator import (
    BaseLayoutType,
    LayoutGenerator,
    generate_canonical_layout,
)
from clashvision.vision.layout_analyzer import BaseLayoutAnalyzer


# ==============================================================================
# BaseEvaluator Tests
# ==============================================================================
class TestBaseEvaluator:
    """Tests resource evaluation, priority modes, thresholds, and layout modifiers."""

    def test_evaluator_initialization_defaults(self) -> None:
        evaluator = BaseEvaluator()
        assert evaluator.default_thresholds.min_gold == DEFAULT_MIN_GOLD
        assert evaluator.default_thresholds.min_elixir == DEFAULT_MIN_ELIXIR
        assert evaluator.default_thresholds.min_dark_elixir == DEFAULT_MIN_DARK_ELIXIR

    def test_evaluate_none_resources_returns_zero(self) -> None:
        evaluator = BaseEvaluator()
        res = evaluator.evaluate(None)
        assert res.normalized_score == 0.0
        assert not res.meets_thresholds
        assert res.resource_score == 0.0
        assert "No resources detected" in res.summary

    def test_evaluate_empty_resource_reading_returns_zero(self) -> None:
        evaluator = BaseEvaluator()
        reading = ResourceReading(gold=None, elixir=None, dark_elixir=None)
        res = evaluator.evaluate(reading)
        assert res.normalized_score == 0.0
        assert not res.meets_thresholds

    def test_evaluate_zero_resources_returns_zero(self) -> None:
        evaluator = BaseEvaluator()
        reading = ResourceReading(gold=0, elixir=0, dark_elixir=0)
        res = evaluator.evaluate(reading)
        assert res.normalized_score == 0.0
        assert not res.meets_thresholds

    def test_evaluate_exact_thresholds_balanced(self) -> None:
        evaluator = BaseEvaluator()
        reading = ResourceReading(gold=400_000, elixir=400_000, dark_elixir=3_000)
        res = evaluator.evaluate(reading)
        assert res.meets_thresholds
        # Exact threshold maps to ratio 1.0 -> subscore 60.0
        assert 58.0 <= res.resource_score <= 62.0
        assert 58.0 <= res.normalized_score <= 62.0

    def test_evaluate_double_thresholds_high_score(self) -> None:
        evaluator = BaseEvaluator()
        reading = ResourceReading(gold=800_000, elixir=800_000, dark_elixir=6_000)
        res = evaluator.evaluate(reading)
        assert res.meets_thresholds
        # 2x threshold maps to ~92.0
        assert 90.0 <= res.resource_score <= 95.0

    def test_evaluate_huge_resources_caps_at_100(self) -> None:
        evaluator = BaseEvaluator()
        reading = ResourceReading(gold=2_500_000, elixir=2_500_000, dark_elixir=20_000)
        res = evaluator.evaluate(reading)
        assert res.normalized_score == 100.0

    def test_priority_gold_first_weights_gold_higher(self) -> None:
        evaluator = BaseEvaluator()
        thresh_gold_first = ResourceThresholds(priority=ResourcePriority.GOLD_FIRST)
        thresh_elixir_first = ResourceThresholds(priority=ResourcePriority.ELIXIR_FIRST)

        # Base with high gold, low elixir
        reading = ResourceReading(gold=900_000, elixir=100_000, dark_elixir=1_000)

        res_gold_first = evaluator.evaluate(reading, thresholds=thresh_gold_first)
        res_elixir_first = evaluator.evaluate(reading, thresholds=thresh_elixir_first)

        assert res_gold_first.normalized_score > res_elixir_first.normalized_score
        assert res_gold_first.meets_thresholds

    def test_priority_dark_elixir_first(self) -> None:
        evaluator = BaseEvaluator()
        thresh_de = ResourceThresholds(priority=ResourcePriority.DARK_ELIXIR_FIRST)

        # Base with massive dark elixir
        reading = ResourceReading(gold=200_000, elixir=200_000, dark_elixir=9_000)
        res = evaluator.evaluate(reading, thresholds=thresh_de)
        assert res.meets_thresholds
        assert res.normalized_score >= 70.0

    def test_layout_modifier_exposed_collectors_bonus(self) -> None:
        evaluator = BaseEvaluator()
        reading = ResourceReading(gold=500_000, elixir=500_000, dark_elixir=3_500)

        layout_exposed = BaseLayoutFeatures(exposed_collector_ratio=1.0)
        layout_protected = BaseLayoutFeatures(exposed_collector_ratio=0.0)

        res_exposed = evaluator.evaluate(reading, layout=layout_exposed)
        res_protected = evaluator.evaluate(reading, layout=layout_protected)

        assert res_exposed.normalized_score > res_protected.normalized_score
        assert res_exposed.layout_modifier > 0.0

    def test_layout_modifier_dead_base_bonus(self) -> None:
        evaluator = BaseEvaluator()
        reading = ResourceReading(gold=800_000, elixir=800_000, dark_elixir=6_000)

        layout_dead = BaseLayoutFeatures(
            exposed_collector_ratio=1.0, metadata={"dead_base": True}
        )
        res = evaluator.evaluate(reading, layout=layout_dead)
        assert res.breakdown["dead_base_bonus"] == 10.0
        assert "DEAD BASE" in res.summary

    def test_layout_modifier_th_accessible_bonus(self) -> None:
        evaluator = BaseEvaluator()
        reading = ResourceReading(gold=500_000, elixir=500_000, dark_elixir=3_500)

        layout_th = BaseLayoutFeatures(exposed_collector_ratio=0.5, th_accessible=True)
        res = evaluator.evaluate(reading, layout=layout_th)
        assert res.breakdown["th_accessibility_bonus"] == 5.0
        assert "Town Hall" in res.summary

    def test_layout_modifier_core_storages_penalty(self) -> None:
        evaluator = BaseEvaluator()
        reading = ResourceReading(gold=500_000, elixir=500_000, dark_elixir=3_500)

        layout_core = BaseLayoutFeatures(
            exposed_collector_ratio=0.0,
            metadata={"storages_in_core": True, "dead_base": False},
        )
        res = evaluator.evaluate(reading, layout=layout_core)
        assert res.breakdown["core_storage_penalty"] < 0.0

    def test_evaluation_result_to_dict_structure(self) -> None:
        evaluator = BaseEvaluator()
        reading = ResourceReading(gold=600_000, elixir=600_000, dark_elixir=4_000)
        res = evaluator.evaluate(reading)
        d = res.to_dict()
        assert "normalized_score" in d
        assert "meets_thresholds" in d
        assert "resource_score" in d
        assert "layout_modifier" in d
        assert "breakdown" in d
        assert "summary" in d
        assert isinstance(d["breakdown"], dict)


# ==============================================================================
# RecommendationEngine Tests
# ==============================================================================
class TestRecommendationEngine:
    """Tests overall multi-strategy recommendations, rankings, breakdowns, and plans."""

    @pytest.fixture
    def engine(self) -> RecommendationEngine:
        return RecommendationEngine()

    @pytest.fixture
    def analyzer(self) -> BaseLayoutAnalyzer:
        return BaseLayoutAnalyzer()

    def test_engine_initializes_all_5_strategies(self, engine: RecommendationEngine) -> None:
        assert engine.get_strategy(StrategyType.SNEAKY_GOBLIN) is not None
        assert engine.get_strategy(StrategyType.BARCH) is not None
        assert engine.get_strategy(StrategyType.MASS_MINER) is not None
        assert engine.get_strategy(StrategyType.MASS_DRAGON) is not None
        assert engine.get_strategy(StrategyType.CUSTOM) is not None

    def test_evaluate_base_with_none_resources(self, engine: RecommendationEngine) -> None:
        rec = engine.evaluate_base(None)
        assert rec.suitability_score == 0.0
        assert len(rec.deployment_plan) == 0
        assert "No harvestable resources detected" in rec.explanation
        d = rec.to_dict()
        assert d["suitability_score"] == 0.0

    def test_evaluate_base_exposed_collectors_favors_sneaky_or_barch(
        self, engine: RecommendationEngine, analyzer: BaseLayoutAnalyzer
    ) -> None:
        model = generate_canonical_layout(BaseLayoutType.EXPOSED_COLLECTORS)
        layout = analyzer.analyze_layout(model)
        reading = ResourceReading(
            gold=model.total_gold, elixir=model.total_elixir, dark_elixir=model.total_dark_elixir
        )

        rec = engine.evaluate_base(reading, layout)
        assert rec.suitability_score >= 80.0
        assert rec.primary_strategy in (StrategyType.SNEAKY_GOBLIN, StrategyType.BARCH)
        assert len(rec.deployment_plan) > 0
        assert rec.rankings[0].strategy_type == rec.primary_strategy
        assert rec.rankings[0].viable

    def test_evaluate_base_dead_base_maximum_recommendation(
        self, engine: RecommendationEngine, analyzer: BaseLayoutAnalyzer
    ) -> None:
        model = generate_canonical_layout(BaseLayoutType.DEAD_BASE)
        layout = analyzer.analyze_layout(model)
        reading = ResourceReading(
            gold=model.total_gold, elixir=model.total_elixir, dark_elixir=model.total_dark_elixir
        )

        rec = engine.evaluate_base(reading, layout)
        assert rec.suitability_score >= 85.0
        assert rec.primary_strategy in (StrategyType.SNEAKY_GOBLIN, StrategyType.BARCH)
        assert rec.score_breakdown["risk_discount"] == 0.0  # LOW risk
        assert "RECOMMENDED TARGET" in rec.explanation

    def test_evaluate_base_core_storages_favors_miner_or_sneaky_over_barch(
        self, engine: RecommendationEngine, analyzer: BaseLayoutAnalyzer
    ) -> None:
        model = generate_canonical_layout(BaseLayoutType.CORE_STORAGES)
        layout = analyzer.analyze_layout(model)
        reading = ResourceReading(
            gold=model.total_gold, elixir=model.total_elixir, dark_elixir=model.total_dark_elixir
        )

        rec = engine.evaluate_base(reading, layout)
        # BARCH should have lower score than Mass Miner or Sneaky Goblin for core storages
        barch_eval = next(e for e in rec.rankings if e.strategy_type == StrategyType.BARCH)
        miner_eval = next(e for e in rec.rankings if e.strategy_type == StrategyType.MASS_MINER)
        assert miner_eval.score > barch_eval.score
        assert rec.primary_strategy != StrategyType.BARCH

    def test_evaluate_base_air_corridor_favors_mass_dragon(
        self, engine: RecommendationEngine, analyzer: BaseLayoutAnalyzer
    ) -> None:
        model = generate_canonical_layout(BaseLayoutType.AIR_DEFENSE_CORRIDOR)
        layout = analyzer.analyze_layout(model)
        reading = ResourceReading(
            gold=model.total_gold, elixir=model.total_elixir, dark_elixir=model.total_dark_elixir
        )

        rec = engine.evaluate_base(reading, layout)
        dragon_eval = next(e for e in rec.rankings if e.strategy_type == StrategyType.MASS_DRAGON)
        assert dragon_eval.viable
        assert dragon_eval.layout_suitability >= 85.0
        assert dragon_eval.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)

    def test_evaluate_base_compartment_ring_favors_mass_miner(
        self, engine: RecommendationEngine, analyzer: BaseLayoutAnalyzer
    ) -> None:
        model = generate_canonical_layout(BaseLayoutType.COMPARTMENT_RING)
        layout = analyzer.analyze_layout(model)
        reading = ResourceReading(
            gold=model.total_gold, elixir=model.total_elixir, dark_elixir=model.total_dark_elixir
        )

        rec = engine.evaluate_base(reading, layout)
        miner_eval = next(e for e in rec.rankings if e.strategy_type == StrategyType.MASS_MINER)
        assert miner_eval.viable
        assert miner_eval.layout_suitability >= 85.0

    def test_evaluate_base_town_hall_exposed(
        self, engine: RecommendationEngine, analyzer: BaseLayoutAnalyzer
    ) -> None:
        model = generate_canonical_layout(BaseLayoutType.TOWN_HALL_EXPOSED)
        layout = analyzer.analyze_layout(model)
        reading = ResourceReading(
            gold=model.total_gold, elixir=model.total_elixir, dark_elixir=model.total_dark_elixir
        )

        rec = engine.evaluate_base(reading, layout)
        assert layout.th_accessible
        # Sneaky Goblin plan should feature Town Hall snipe
        assert any("Town Hall" in s.note for s in rec.deployment_plan)

    def test_rankings_sorted_descending(
        self, engine: RecommendationEngine
    ) -> None:
        reading = ResourceReading(gold=650_000, elixir=650_000, dark_elixir=4_500)
        rec = engine.evaluate_base(reading)
        scores = [e.score for e in rec.rankings]
        assert scores == sorted(scores, reverse=True)

    def test_score_breakdown_contains_all_keys(
        self, engine: RecommendationEngine
    ) -> None:
        reading = ResourceReading(gold=700_000, elixir=700_000, dark_elixir=5_000)
        rec = engine.evaluate_base(reading)
        sb = rec.score_breakdown
        assert "resource_score" in sb
        assert "accessibility_score" in sb
        assert "risk_discount" in sb
        assert "synergy_bonus" in sb
        assert "final_score" in sb

    def test_explainable_rationale_structure(
        self, engine: RecommendationEngine
    ) -> None:
        reading = ResourceReading(gold=800_000, elixir=800_000, dark_elixir=6_000)
        layout = BaseLayoutFeatures(exposed_collector_ratio=0.8, compartment_count=3)
        rec = engine.evaluate_base(reading, layout)
        assert len(rec.explanation) > 0
        assert "Overall Suitability Score:" in rec.explanation
        assert "Selected Strategy:" in rec.explanation
        assert "Alternative options considered:" in rec.explanation

    def test_custom_strategy_configuration_update(
        self, engine: RecommendationEngine
    ) -> None:
        custom_cfg = CustomStrategyConfig(
            name="Super Dragon Rush",
            min_gold=600_000,
            min_elixir=600_000,
            army_composition={"Super Dragon": 6, "Balloon": 4, "Rage Spell": 3},
        )
        engine.set_custom_config(custom_cfg)
        reading = ResourceReading(gold=750_000, elixir=750_000, dark_elixir=4_000)
        rec = engine.evaluate_base(reading)
        custom_eval = next(e for e in rec.rankings if e.strategy_type == StrategyType.CUSTOM)
        assert custom_eval.prerequisites_met
        assert custom_eval.viable

    def test_deployment_plan_coordinate_bounds(
        self, engine: RecommendationEngine
    ) -> None:
        reading = ResourceReading(gold=900_000, elixir=900_000, dark_elixir=7_000)
        layout = BaseLayoutFeatures(exposed_collector_ratio=0.8)
        rec = engine.evaluate_base(reading, layout, canvas_width=1920, canvas_height=1080)
        for step in rec.deployment_plan:
            assert 0 <= step.target_x <= 1920
            assert 0 <= step.target_y <= 1080
            assert 0.0 <= step.normalized_x <= 1.0
            assert 0.0 <= step.normalized_y <= 1.0
            assert step.delay_ms >= 0
            assert step.unit_count > 0
            assert len(step.troop_type) > 0
