"""
ClashVision Farming Decision Subsystem — Multi-Strategy Recommendation Engine.
Ranks all available strategies, calculates risk vs. reward trade-offs, generates
comprehensive score breakdowns, constructs explainable rationale, and emits
the authoritative FarmingRecommendation contract.
Derived from Requirement R3 and Feature 21.
"""

from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from clashvision.core.constants import (
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


class RecommendationEngine:
    """
    Tactical decision and multi-strategy recommendation engine.
    Orchestrates:
    - Base resource and layout evaluation
    - Strategy candidate evaluations across all 5 tactical archetypes
    - Multi-criteria ranking and risk vs. reward assessments
    - Explainable decision rationale and step-by-step deployment planning
    """

    def __init__(
        self,
        evaluator: Optional[BaseEvaluator] = None,
        custom_config: Optional[CustomStrategyConfig] = None,
    ) -> None:
        self.evaluator = evaluator or BaseEvaluator()
        self._strategies: Dict[StrategyType, BaseStrategy] = {
            StrategyType.SNEAKY_GOBLIN: SneakyGoblinStrategy(),
            StrategyType.BARCH: BarchStrategy(),
            StrategyType.MASS_MINER: MassMinerStrategy(),
            StrategyType.MASS_DRAGON: MassDragonStrategy(),
            StrategyType.CUSTOM: CustomStrategy(custom_config),
        }

    def register_strategy(self, strategy: BaseStrategy) -> None:
        """Registers or replaces an attack strategy in the engine."""
        self._strategies[strategy.strategy_type] = strategy

    def get_strategy(self, strategy_type: StrategyType) -> Optional[BaseStrategy]:
        """Retrieves a registered strategy by type."""
        return self._strategies.get(strategy_type)

    def set_custom_config(self, config: CustomStrategyConfig) -> None:
        """Updates the configuration for Strategy E (Custom Strategy)."""
        cust = self._strategies.get(StrategyType.CUSTOM)
        if isinstance(cust, CustomStrategy):
            cust.update_config(config)
        else:
            self._strategies[StrategyType.CUSTOM] = CustomStrategy(config)

    def recommend(
        self,
        resources: Optional[ResourceReading],
        layout: Optional[BaseLayoutFeatures] = None,
        game_state: Optional[Any] = None,
        **kwargs: Any,
    ) -> FarmingRecommendation:
        """Alias for evaluate_base for API convenience and backwards-compatibility."""
        return self.evaluate_base(resources=resources, layout=layout, **kwargs)

    def evaluate_base(
        self,
        resources: Optional[ResourceReading],
        layout: Optional[BaseLayoutFeatures] = None,
        thresholds: Optional[ResourceThresholds] = None,
        custom_config: Optional[CustomStrategyConfig] = None,
        canvas_width: int = VIRTUAL_REFERENCE_WIDTH,
        canvas_height: int = VIRTUAL_REFERENCE_HEIGHT,
    ) -> FarmingRecommendation:
        """
        Executes complete tactical evaluation of an enemy base.
        Returns the unified FarmingRecommendation contract for downstream presentation/simulation.
        """
        if custom_config is not None:
            self.set_custom_config(custom_config)

        # 1. Base Resource & Layout Evaluation
        base_result = self.evaluator.evaluate(resources, layout, thresholds)

        # 2. Evaluate All Registered Strategies
        evaluations: List[StrategyEvaluation] = []
        for strat_type, strat in self._strategies.items():
            ev = strat.evaluate(resources, layout, thresholds)
            evaluations.append(ev)

        # 3. Sort Rankings by Viability and Score (descending)
        evaluations.sort(key=lambda e: (e.viable, e.score), reverse=True)

        # Determine Primary Strategy
        viable_evals = [e for e in evaluations if e.viable]
        if viable_evals:
            top_eval = viable_evals[0]
        else:
            top_eval = evaluations[0] if evaluations else StrategyEvaluation(
                strategy_type=StrategyType.SNEAKY_GOBLIN,
                score=0.0,
                viable=False,
                risk_level=RiskLevel.HIGH,
                reasons=["No viable attack strategies identified."],
            )

        primary_strategy_type = top_eval.strategy_type
        primary_strategy = self._strategies.get(primary_strategy_type)

        # 4. Suitability Score & Risk vs. Reward Calculations
        if top_eval.viable and base_result.meets_thresholds:
            # Composite suitability merges base value and primary strategy effectiveness
            suitability_score = 0.45 * base_result.normalized_score + 0.55 * top_eval.score
        elif top_eval.viable:
            # Strategies are viable but resources are below standard thresholds
            suitability_score = min(base_result.normalized_score, top_eval.score)
        else:
            # Neither base nor strategy is viable
            suitability_score = min(base_result.normalized_score, top_eval.score)

        suitability_score = round(max(0.0, min(100.0, suitability_score if math.isfinite(suitability_score) else 0.0)), 1)

        # Risk discount calculation
        risk_discounts = {
            RiskLevel.LOW: 0.0,
            RiskLevel.MEDIUM: -5.0,
            RiskLevel.HIGH: -15.0,
            RiskLevel.EXTREME: -30.0,
        }
        risk_discount = risk_discounts.get(top_eval.risk_level, 0.0)

        # Synergy bonus (e.g. exposed collectors with sneaky goblins/barch, or corridors with dragon/miner)
        synergy_bonus = 0.0
        if layout:
            if layout.exposed_collector_ratio >= 0.7 and primary_strategy_type in (
                StrategyType.SNEAKY_GOBLIN,
                StrategyType.BARCH,
            ):
                synergy_bonus = 6.0
            elif layout.compartment_count >= 4 and primary_strategy_type == StrategyType.MASS_MINER:
                synergy_bonus = 6.0
            elif (
                len(layout.air_defense_locations) <= 2
                or layout.metadata.get("layout_type") == "AIR_DEFENSE_CORRIDOR"
            ) and primary_strategy_type == StrategyType.MASS_DRAGON:
                synergy_bonus = 6.0

        # Formulate structured score breakdown dictionary
        score_breakdown = {
            "resource_score": round(base_result.resource_score, 1),
            "accessibility_score": round(top_eval.layout_suitability, 1),
            "risk_discount": round(risk_discount, 1),
            "synergy_bonus": round(synergy_bonus, 1),
            "final_score": suitability_score,
        }

        # 5. Generate Procedural Deployment Plan
        if primary_strategy and top_eval.viable and suitability_score > 0.0 and top_eval.score > 0:
            deployment_plan = primary_strategy.generate_deployment_plan(
                layout, canvas_width=canvas_width, canvas_height=canvas_height
            )
        else:
            deployment_plan = []

        # 6. Construct Explainable Rationale Text
        explanation = self._build_explanation(
            base_result=base_result,
            top_eval=top_eval,
            all_evals=evaluations,
            resources=resources,
            layout=layout,
            suitability_score=suitability_score,
        )

        return FarmingRecommendation(
            suitability_score=suitability_score,
            primary_strategy=primary_strategy_type,
            rankings=evaluations,
            deployment_plan=deployment_plan,
            score_breakdown=score_breakdown,
            timestamp=time.time(),
            explanation=explanation,
        )

    def _build_explanation(
        self,
        base_result: BaseEvaluationResult,
        top_eval: StrategyEvaluation,
        all_evals: List[StrategyEvaluation],
        resources: Optional[ResourceReading],
        layout: Optional[BaseLayoutFeatures],
        suitability_score: float,
    ) -> str:
        """Builds a rich, explainable tactical rationale for UI and logs."""
        parts: List[str] = []

        # 1. Headline Assessment
        if not resources or not resources.has_any() or suitability_score == 0:
            return "Base unsuitable: No harvestable resources detected or OCR parsing returned empty values."

        gold = resources.gold or 0
        elixir = resources.elixir or 0
        dark_elixir = resources.dark_elixir or 0

        status = "RECOMMENDED TARGET" if top_eval.viable else "SUB-OPTIMAL TARGET"
        parts.append(
            f"[{status}] Overall Suitability Score: {suitability_score:.1f}/100. "
            f"Available Loot: {gold:,} Gold, {elixir:,} Elixir, {dark_elixir:,} Dark Elixir."
        )

        # 2. Layout Context
        if layout:
            exp_pct = int(layout.exposed_collector_ratio * 100)
            layout_name = layout.metadata.get("layout_type", "Standard Layout")
            parts.append(
                f"Layout: {layout_name} with {exp_pct}% exposed collectors, "
                f"{layout.compartment_count} compartments, "
                f"Town Hall {'accessible on perimeter' if layout.th_accessible else 'protected in core'}."
            )

        # 3. Strategy Selection Justification
        strat_name = top_eval.strategy_type.display_name
        parts.append(
            f"Selected Strategy: {strat_name} (Score: {top_eval.score:.1f}, Risk: {top_eval.risk_level.value}). "
            f"Rationale: {' '.join(top_eval.reasons[:2])}"
        )

        # 4. Comparative Alternative Summary
        runners_up = [e for e in all_evals if e.strategy_type != top_eval.strategy_type and e.score > 0]
        if runners_up:
            alt_texts = [
                f"{e.strategy_type.display_name} ({e.score:.1f})" for e in runners_up[:2]
            ]
            parts.append(f"Alternative options considered: {', '.join(alt_texts)}.")

        return "\n".join(parts)
