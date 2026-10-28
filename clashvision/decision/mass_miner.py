"""
ClashVision Farming Decision Subsystem — Strategy C: Mass Miner.
Subterranean assault analyzing compartment paths, core routing, and heal spell sequencing
over splash damage clusters.
Derived from Requirement R3 and Feature 18.
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


class MassMinerStrategy(BaseStrategy):
    """
    Strategy C: Mass Miner Assault.
    Miners burrow underground while moving, rendering walls completely irrelevant.
    Highly effective on dense multi-compartment bases and core storage layouts.
    Vulnerable to high-density splash clusters (Wizard Towers, Bomb Towers, Scattershots).
    """

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.MASS_MINER

    @property
    def name(self) -> str:
        return "Mass Miner Assault"

    @property
    def description(self) -> str:
        return (
            "Subterranean pathing assault bypassing all walls entirely. Funnels miners into "
            "a tight destructive corridor through core compartments backed by rolling Heal spells."
        )

    @property
    def army_composition(self) -> Dict[str, int]:
        return {
            "Miner": 40,
            "Heal Spell": 5,
            "Poison Spell": 1,
        }

    @property
    def estimated_training_time_seconds(self) -> int:
        return 1440  # 24 minutes

    @property
    def estimated_elixir_cost(self) -> int:
        return 180_000

    @property
    def estimated_dark_elixir_cost(self) -> int:
        return 200

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
        # Mass Miner has higher training cost (~180k Elixir), so requires solid loot pool
        min_gold = _sanitize_val(thresholds.min_gold) if thresholds else 400_000
        min_elixir = _sanitize_val(thresholds.min_elixir) if thresholds else 400_000
        min_dark = _sanitize_val(thresholds.min_dark_elixir) if thresholds else 3_000

        total_target = min_gold + min_elixir
        loot_margin = total_loot - self.estimated_elixir_cost

        if loot_margin >= total_target * 1.5:
            res_suitability = 95.0
            reasons.append(f"Substantial net profit: {total_loot:,} loot offsets army investment.")
        elif loot_margin >= total_target:
            res_suitability = 80.0
            reasons.append(f"Profitable target: {total_loot:,} total Gold/Elixir.")
        elif loot_margin >= 200_000:
            res_suitability = 58.0
            reasons.append("Moderate loot margin; training cost represents notable overhead.")
        else:
            res_suitability = 30.0
            reasons.append("Low loot margin; high elixir training cost makes Mass Miner uneconomical.")

        # 2. Layout Suitability [0.0, 100.0]
        # Compartment analysis: Miners shine when compartments are dense (>= 4) and storages are in core
        compartments = layout.compartment_count if layout else 2
        storages_in_core = bool(layout.metadata.get("storages_in_core", True)) if layout else True
        layout_type = str(layout.metadata.get("layout_type", "")) if layout else ""

        if layout_type == "COMPARTMENT_RING" or compartments >= 4:
            layout_suitability = 94.0
            reasons.append(f"Dense compartment grid ({compartments} compartments): Miners bypass all walls effortlessly.")
        elif storages_in_core:
            layout_suitability = 85.0
            reasons.append("Core storages: Subterranean routing pierces directly to central high-value targets.")
        elif compartments >= 2:
            layout_suitability = 72.0
            reasons.append(f"Standard base topology ({compartments} compartments): Solid pathing.")
        else:
            layout_suitability = 55.0
            reasons.append("Open or sparse base: Miner wall-bypass tactical advantage is diminished.")

        # 3. Risk Assessment
        if layout_type == "COMPARTMENT_RING" or compartments >= 4:
            risk = RiskLevel.MEDIUM
            reasons.append("Medium risk: Predictable corridor pathing; requires synchronized Heal spell drops.")
        elif storages_in_core:
            risk = RiskLevel.MEDIUM
            reasons.append("Medium risk: Core splash damage requires overlapping Heal spell coverage.")
        else:
            risk = RiskLevel.LOW
            reasons.append("Low risk: Wide routing corridor reduces splash damage stacking.")

        # 4. Composite Score & Viability
        composite_score = round(0.50 * res_suitability + 0.50 * layout_suitability, 2)
        composite_score = max(0.0, min(100.0, composite_score if math.isfinite(composite_score) else 0.0))
        res_suitability = max(0.0, min(100.0, res_suitability if math.isfinite(res_suitability) else 0.0))
        layout_suitability = max(0.0, min(100.0, layout_suitability if math.isfinite(layout_suitability) else 0.0))
        viable = (total_loot >= 380_000) and (composite_score >= 45.0)

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

        center_x = cw // 2
        center_y = ch // 2

        # Primary approach angle: South-West flank (225 degrees = 5*pi/4)
        main_angle = 5.0 * math.pi / 4.0
        flank_left = main_angle - (math.pi / 6.0)
        flank_right = main_angle + (math.pi / 6.0)

        # Phase 1: Funnel Trimming
        # Deploy small miner groups on outer left and right corners to channel the main pack inward
        fl_px, fl_py, fl_nx, fl_ny = self.calculate_perimeter_point(
            flank_left, radius_fraction=0.42, canvas_width=cw, canvas_height=ch
        )
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Miner",
                target_x=fl_px,
                target_y=fl_py,
                normalized_x=fl_nx,
                normalized_y=fl_ny,
                delay_ms=current_delay,
                unit_count=4,
                note="Left flank funnel: Carve outer corridor edge.",
            )
        )
        step_id += 1
        current_delay += 400

        fr_px, fr_py, fr_nx, fr_ny = self.calculate_perimeter_point(
            flank_right, radius_fraction=0.42, canvas_width=cw, canvas_height=ch
        )
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Miner",
                target_x=fr_px,
                target_y=fr_py,
                normalized_x=fr_nx,
                normalized_y=fr_ny,
                delay_ms=current_delay,
                unit_count=4,
                note="Right flank funnel: Prevent miners from straying around perimeter.",
            )
        )
        step_id += 1
        current_delay += 800

        # Phase 2: Main Miner Line Deployment (Center of the funnel corridor)
        main_px, main_py, main_nx, main_ny = self.calculate_perimeter_point(
            main_angle, radius_fraction=0.40, canvas_width=cw, canvas_height=ch
        )

        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Miner",
                target_x=main_px,
                target_y=main_py,
                normalized_x=main_nx,
                normalized_y=main_ny,
                delay_ms=current_delay,
                unit_count=16,
                note="Main miner wave 1: Core subterranean assault.",
            )
        )
        step_id += 1
        current_delay += 600

        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Miner",
                target_x=main_px,
                target_y=main_py,
                normalized_x=main_nx,
                normalized_y=main_ny,
                delay_ms=current_delay,
                unit_count=16,
                note="Main miner wave 2: Deep corridor reinforcement.",
            )
        )
        step_id += 1
        current_delay += 1200

        # Phase 3: Rolling Heal Spells along the approach corridor toward the core
        # Heal 1: Entry splash zone (halfway between entry and center)
        h1_x = int(round(main_px * 0.65 + center_x * 0.35))
        h1_y = int(round(main_py * 0.65 + center_y * 0.35))
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Heal Spell",
                target_x=h1_x,
                target_y=h1_y,
                normalized_x=round(h1_x / cw, 4),
                normalized_y=round(h1_y / ch, 4),
                delay_ms=current_delay,
                unit_count=1,
                note="Heal Spell 1: Sustain miner health against outer Wizard/Bomb tower splash.",
            )
        )
        step_id += 1
        current_delay += 2000

        # Poison spell on Clan Castle / defending hero zone near core
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Poison Spell",
                target_x=center_x,
                target_y=center_y,
                normalized_x=round(center_x / cw, 4),
                normalized_y=round(center_y / ch, 4),
                delay_ms=current_delay,
                unit_count=1,
                note="Poison Spell: Neutralize enemy Clan Castle troops and defending heroes in core.",
            )
        )
        step_id += 1
        current_delay += 1500

        # Heal 2: Base Core / Town Hall compartment
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Heal Spell",
                target_x=center_x,
                target_y=center_y,
                normalized_x=round(center_x / cw, 4),
                normalized_y=round(center_y / ch, 4),
                delay_ms=current_delay,
                unit_count=1,
                note="Heal Spell 2: Full core protection over Town Hall and Giga Tesla/Inferno.",
            )
        )
        step_id += 1
        current_delay += 2500

        # Heal 3: Back-end exit quadrant (North-East)
        exit_angle = main_angle - math.pi  # Opposite quadrant
        h3_x, h3_y, h3_nx, h3_ny = self.calculate_perimeter_point(
            exit_angle, radius_fraction=0.20, canvas_width=cw, canvas_height=ch
        )
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Heal Spell",
                target_x=h3_x,
                target_y=h3_y,
                normalized_x=h3_nx,
                normalized_y=h3_ny,
                delay_ms=current_delay,
                unit_count=1,
                note="Heal Spell 3: Back-end healing over remaining storages and defensive ring.",
            )
        )

        return plan
