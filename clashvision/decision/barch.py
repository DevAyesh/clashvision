"""
ClashVision Farming Decision Subsystem — Strategy B: BARCH Farming.
Swarm strategy focusing on outer ring building clearing, perimeter collector harvesting,
and alternating Barbarian meat-shield / Archer firing line waves.
Derived from Requirement R3 and Feature 17.
"""

from __future__ import annotations
import math
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
    ResourceReading,
    StrategyEvaluation,
)
from clashvision.core.config import ResourceThresholds
from clashvision.decision.base_strategy import BaseStrategy


class BarchStrategy(BaseStrategy):
    """
    Strategy B: BARCH Farming (Barbarians + Archers).
    Classic high-tempo, zero dark-elixir cost swarm attack.
    Ideal for dead bases and exposed collector perimeters; ineffective against core storages.
    """

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.BARCH

    @property
    def name(self) -> str:
        return "BARCH Farming"

    @property
    def description(self) -> str:
        return (
            "Low-cost perimeter sweep deploying alternating waves of Barbarians as damage sponges "
            "and Archers to target exposed collectors and outer ring structures outside walls."
        )

    @property
    def army_composition(self) -> Dict[str, int]:
        return {
            "Barbarian": 120,
            "Archer": 120,
            "Lightning Spell": 4,
        }

    @property
    def estimated_training_time_seconds(self) -> int:
        return 480  # 8 minutes

    @property
    def estimated_elixir_cost(self) -> int:
        return 40_000

    @property
    def estimated_dark_elixir_cost(self) -> int:
        return 0

    def evaluate(
        self,
        resources: Optional[ResourceReading],
        layout: Optional[BaseLayoutFeatures],
        thresholds: Optional[ResourceThresholds] = None,
    ) -> StrategyEvaluation:
        reasons: List[str] = []

        if resources is None or not resources.has_any():
            return StrategyEvaluation(
                strategy_type=self.strategy_type,
                score=0.0,
                viable=False,
                risk_level=RiskLevel.LOW,
                resource_suitability=0.0,
                layout_suitability=0.0,
                prerequisites_met=False,
                reasons=["No resource readings available for evaluation."],
            )

        def _sanitize_val(val: Any) -> int:
            if val is None:
                return 0
            try:
                if not math.isfinite(val):
                    return 0
                return max(0, int(val))
            except (TypeError, ValueError):
                return 0

        gold = _sanitize_val(resources.gold)
        elixir = _sanitize_val(resources.elixir)
        dark_elixir = _sanitize_val(resources.dark_elixir)
        total_loot = gold + elixir

        # Zero-loot protection
        if gold <= 0 and elixir <= 0 and dark_elixir <= 0:
            return StrategyEvaluation(
                strategy_type=self.strategy_type,
                score=0.0,
                viable=False,
                risk_level=RiskLevel.LOW,
                resource_suitability=0.0,
                layout_suitability=0.0,
                prerequisites_met=False,
                reasons=["Zero harvestable loot detected on target base."],
            )

        # 1. Resource Suitability [0.0, 100.0]
        # BARCH prioritizes regular resources (Gold & Elixir)
        min_gold = _sanitize_val(thresholds.min_gold) if thresholds else 400_000
        min_elixir = _sanitize_val(thresholds.min_elixir) if thresholds else 400_000
        min_total = min_gold + min_elixir

        res_ratio = total_loot / max(1, min_total)
        if res_ratio >= 1.8:
            res_suitability = 100.0
        elif res_ratio >= 1.0:
            res_suitability = 65.0 + (res_ratio - 1.0) * 43.75
        else:
            res_suitability = res_ratio * 65.0

        if total_loot >= 750_000:
            reasons.append(f"Abundant Gold and Elixir available: {total_loot:,} total.")
        elif total_loot < 250_000:
            reasons.append("Low regular resource quantity makes BARCH troop investment marginal.")

        # 2. Layout Suitability [0.0, 100.0]
        # BARCH heavily depends on exposed collectors; struggles severely against deep core storages
        exp_ratio = layout.exposed_collector_ratio if layout else 0.5
        is_dead = bool(layout.metadata.get("dead_base", False)) if layout else False
        walls_intact = layout.wall_perimeter_intact if layout else True

        if is_dead:
            layout_suitability = 96.0
            reasons.append("Dead base: Inactive defenses and full perimeter collectors maximize BARCH efficiency.")
        elif exp_ratio >= 0.80:
            layout_suitability = 92.0
            reasons.append(f"High collector exposure ({int(exp_ratio * 100)}%): Outer sweep captures nearly all available loot.")
        elif exp_ratio >= 0.50:
            layout_suitability = 75.0
            reasons.append(f"Moderate collector exposure ({int(exp_ratio * 100)}%): Outer loot harvestable; core storages unreachable.")
        elif exp_ratio >= 0.25:
            layout_suitability = 45.0
            reasons.append(f"Low collector exposure ({int(exp_ratio * 100)}%): Most loot locked behind inner walls.")
        else:
            layout_suitability = 10.0
            reasons.append("Zero/minimal exposed collectors: BARCH cannot penetrate multi-layer core compartments.")

        if not walls_intact:
            layout_suitability = min(100.0, layout_suitability + 15.0)
            reasons.append("Weak or incomplete wall perimeter allows Barbarian swarm to enter base.")

        # 3. Risk Assessment
        if is_dead or exp_ratio >= 0.75:
            risk = RiskLevel.LOW
            reasons.append("Low risk: Troops stay outside wall defenses and avoid core splash damage.")
        elif exp_ratio >= 0.40:
            risk = RiskLevel.MEDIUM
            reasons.append("Medium risk: Mortars and Wizard Towers can devastate swarm if clumped.")
        elif exp_ratio >= 0.20:
            risk = RiskLevel.HIGH
            reasons.append("High risk: Unfavorable layout; troops will perish on walls under splash fire.")
        else:
            risk = RiskLevel.EXTREME
            reasons.append("Extreme risk: Total army loss expected with minimal core loot extracted.")

        # 4. Composite Score & Viability
        # If collectors are unexposed and walls are intact, core loot is inaccessible to BARCH
        if exp_ratio < 0.20 and walls_intact and not is_dead:
            effective_res = res_suitability * exp_ratio
            composite_score = round(0.30 * effective_res + 0.70 * layout_suitability, 2)
        else:
            composite_score = round(0.40 * res_suitability + 0.60 * layout_suitability, 2)

        composite_score = max(0.0, min(100.0, composite_score if math.isfinite(composite_score) else 0.0))
        res_suitability = max(0.0, min(100.0, res_suitability if math.isfinite(res_suitability) else 0.0))
        layout_suitability = max(0.0, min(100.0, layout_suitability if math.isfinite(layout_suitability) else 0.0))
        viable = (exp_ratio >= 0.30 or is_dead or not walls_intact) and (total_loot >= 200_000)

        return StrategyEvaluation(
            strategy_type=self.strategy_type,
            score=composite_score,
            viable=viable,
            risk_level=risk,
            resource_suitability=round(res_suitability, 2),
            layout_suitability=round(layout_suitability, 2),
            prerequisites_met=True,
            reasons=reasons,
        )

    def generate_deployment_plan(
        self,
        layout: Optional[BaseLayoutFeatures],
        canvas_width: int = VIRTUAL_REFERENCE_WIDTH,
        canvas_height: int = VIRTUAL_REFERENCE_HEIGHT,
    ) -> List[DeploymentStep]:
        plan: List[DeploymentStep] = []
        step_id = 1
        current_delay = 0

        cw = max(1, canvas_width)
        ch = max(1, canvas_height)

        # Generate 10 equidistant perimeter points for continuous perimeter ring deployment
        perimeter_points = self.get_cardinal_perimeter_points(
            count=10, radius_fraction=0.44, canvas_width=cw, canvas_height=ch
        )

        # Wave 1: Barbarian Meat-Shield Screen (draws mortar, archer tower, and cannon aggro)
        for idx, (px, py, nx, ny) in enumerate(perimeter_points):
            plan.append(
                DeploymentStep(
                    step_id=step_id,
                    troop_type="Barbarian",
                    target_x=px,
                    target_y=py,
                    normalized_x=nx,
                    normalized_y=ny,
                    delay_ms=current_delay,
                    unit_count=6,
                    note=f"Barbarian meat shield sector {idx + 1}: Absorb outer perimeter fire.",
                )
            )
            step_id += 1
            current_delay += 250

        # Wave 2: Archer Firing Line (deployed 1.5 seconds later behind Barbarians)
        current_delay += 800
        for idx, (px, py, nx, ny) in enumerate(perimeter_points):
            # Archers drop slightly behind (radius 0.46)
            plan.append(
                DeploymentStep(
                    step_id=step_id,
                    troop_type="Archer",
                    target_x=px,
                    target_y=py,
                    normalized_x=nx,
                    normalized_y=ny,
                    delay_ms=current_delay,
                    unit_count=6,
                    note=f"Archer firing line sector {idx + 1}: Snipe exposed collectors over walls.",
                )
            )
            step_id += 1
            current_delay += 250

        # Wave 3: Secondary Barbarian Wave (reinforcing active breach areas)
        current_delay += 1200
        for idx in [0, 2, 4, 6, 8]:
            px, py, nx, ny = perimeter_points[idx]
            plan.append(
                DeploymentStep(
                    step_id=step_id,
                    troop_type="Barbarian",
                    target_x=px,
                    target_y=py,
                    normalized_x=nx,
                    normalized_y=ny,
                    delay_ms=current_delay,
                    unit_count=8,
                    note=f"Reinforcement Barbarians sector {idx + 1}: Maintain front-line aggro.",
                )
            )
            step_id += 1
            current_delay += 300

        # Wave 4: Secondary Archer Cleanup Wave
        current_delay += 1000
        for idx in [1, 3, 5, 7, 9]:
            px, py, nx, ny = perimeter_points[idx]
            plan.append(
                DeploymentStep(
                    step_id=step_id,
                    troop_type="Archer",
                    target_x=px,
                    target_y=py,
                    normalized_x=nx,
                    normalized_y=ny,
                    delay_ms=current_delay,
                    unit_count=8,
                    note=f"Cleanup Archers sector {idx + 1}: Finish off depleted resource collectors.",
                )
            )
            step_id += 1
            current_delay += 300

        # Phase 5: Tactical Lightning Spell on perimeter splash defenses
        splash_x = int(cw * 0.35)
        splash_y = int(ch * 0.40)
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Lightning Spell",
                target_x=splash_x,
                target_y=splash_y,
                normalized_x=round(splash_x / cw, 4),
                normalized_y=round(splash_y / ch, 4),
                delay_ms=current_delay + 500,
                unit_count=2,
                note="Lightning Strike on high-threat Wizard Tower threatening archer line.",
            )
        )

        return plan
