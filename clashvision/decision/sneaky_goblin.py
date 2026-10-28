"""
ClashVision Farming Decision Subsystem — Strategy A: Sneaky Goblin Farming.
Prioritizes collector and storage routing analysis, perimeter collector harvesting,
and Town Hall accessibility / sniping using Jump, Haste, and Invisibility sequencing.
Derived from Requirement R3 and Feature 16.
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


class SneakyGoblinStrategy(BaseStrategy):
    """
    Strategy A: Sneaky Goblin Farming.
    Stealth infiltration prioritizing:
    - Exposed perimeter gold mines, elixir collectors, and dark elixir drills
    - Core storage sniping using Jump and Invisibility spells
    - Town Hall sniping for trophy preservation and star bonus
    """

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.SNEAKY_GOBLIN

    @property
    def name(self) -> str:
        return "Sneaky Goblin Farming"

    @property
    def description(self) -> str:
        return (
            "Targeted resource raid utilizing 5-second stealth invisibility to clear "
            "perimeter collectors and breach core storages with Jump and Invisibility spells."
        )

    @property
    def army_composition(self) -> Dict[str, int]:
        return {
            "Sneaky Goblin": 80,
            "Super Wall Breaker": 4,
            "Jump Spell": 2,
            "Invisibility Spell": 3,
            "Haste Spell": 2,
        }

    @property
    def estimated_training_time_seconds(self) -> int:
        return 720  # 12 minutes

    @property
    def estimated_elixir_cost(self) -> int:
        return 65_000

    @property
    def estimated_dark_elixir_cost(self) -> int:
        return 25_000  # Super troop activation cost (amortized)

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
        min_gold = _sanitize_val(thresholds.min_gold) if thresholds else 400_000
        min_elixir = _sanitize_val(thresholds.min_elixir) if thresholds else 400_000
        min_dark = _sanitize_val(thresholds.min_dark_elixir) if thresholds else 3_000

        g_ratio = gold / max(1, min_gold)
        e_ratio = elixir / max(1, min_elixir)
        de_ratio = dark_elixir / max(1, min_dark)

        avg_ratio = (g_ratio + e_ratio + de_ratio * 1.2) / 3.2
        if avg_ratio >= 2.0:
            res_suitability = 100.0
        elif avg_ratio >= 1.0:
            res_suitability = 70.0 + (avg_ratio - 1.0) * 30.0
        else:
            res_suitability = avg_ratio * 70.0

        if total_loot >= 800_000 or dark_elixir >= 5_000:
            reasons.append(f"Abundant target loot: {total_loot:,} total Gold/Elixir, {dark_elixir:,} DE.")
        elif total_loot < 300_000 and dark_elixir < 2_000:
            reasons.append("Loot values below standard profitable farming margin.")

        # 2. Layout Suitability [0.0, 100.0]
        layout_suitability = 70.0  # Goblins have a solid baseline because they target resources
        exp_ratio = layout.exposed_collector_ratio if layout else 0.5
        is_dead = bool(layout.metadata.get("dead_base", False)) if layout else False
        th_accessible = layout.th_accessible if layout else False

        if is_dead:
            layout_suitability = 98.0
            reasons.append("Dead base detected: 100% full perimeter collectors outside defenses.")
        elif exp_ratio >= 0.75:
            layout_suitability = 92.0
            reasons.append(f"High collector exposure ({int(exp_ratio * 100)}%): Rapid stealth collection without wall penetration.")
        elif exp_ratio >= 0.40:
            layout_suitability = 80.0
            reasons.append(f"Moderate collector exposure ({int(exp_ratio * 100)}%): Perimeter sweep followed by Jump routing.")
        else:
            layout_suitability = 62.0
            reasons.append("Storages concentrated in deep core: Requires Jump and Invisibility spell commitment.")

        if th_accessible:
            layout_suitability = min(100.0, layout_suitability + 8.0)
            reasons.append("Town Hall is exposed on perimeter: Easy 1-star snipe for trophy maintenance.")

        # 3. Risk Assessment
        if is_dead or exp_ratio >= 0.75:
            risk = RiskLevel.LOW
            reasons.append("Low risk: Troops can harvest loot before stealth expires.")
        elif exp_ratio >= 0.35 or th_accessible:
            risk = RiskLevel.MEDIUM
            reasons.append("Medium risk: Core storage routing requires accurate spell timing.")
        else:
            risk = RiskLevel.HIGH
            reasons.append("High risk: Multi-compartment pathing under splash defense fire.")

        # 4. Composite Score & Viability
        composite_score = round(0.55 * res_suitability + 0.45 * layout_suitability, 2)
        composite_score = max(0.0, min(100.0, composite_score if math.isfinite(composite_score) else 0.0))
        res_suitability = max(0.0, min(100.0, res_suitability if math.isfinite(res_suitability) else 0.0))
        layout_suitability = max(0.0, min(100.0, layout_suitability if math.isfinite(layout_suitability) else 0.0))
        viable = (total_loot >= 250_000 or dark_elixir >= 2_000) and (composite_score >= 35.0)

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

        # Phase 1: Perimeter Collector Sweep (8 outer cardinal angles)
        perimeter_points = self.get_cardinal_perimeter_points(
            count=8, radius_fraction=0.42, canvas_width=cw, canvas_height=ch
        )

        # Deploy 2 sneaky goblins per perimeter sector to sweep exposed collectors
        for idx, (px, py, nx, ny) in enumerate(perimeter_points):
            plan.append(
                DeploymentStep(
                    step_id=step_id,
                    troop_type="Sneaky Goblin",
                    target_x=px,
                    target_y=py,
                    normalized_x=nx,
                    normalized_y=ny,
                    delay_ms=current_delay,
                    unit_count=2,
                    note=f"Perimeter sweep sector {idx + 1}: Clear outer collectors under stealth.",
                )
            )
            step_id += 1
            current_delay += 350

        # Phase 2: Wall Breaker Entry / Jump Spell
        # If Town Hall is accessible on perimeter, target TH angle; else default south-west entry (225 deg)
        entry_angle = (5.0 * math.pi / 4.0)  # South-West
        entry_px, entry_py, entry_nx, entry_ny = self.calculate_perimeter_point(
            entry_angle, radius_fraction=0.35, canvas_width=cw, canvas_height=ch
        )

        current_delay += 500
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Super Wall Breaker",
                target_x=entry_px,
                target_y=entry_py,
                normalized_x=entry_nx,
                normalized_y=entry_ny,
                delay_ms=current_delay,
                unit_count=2,
                note="Breach outer wall layer toward primary storage compartment.",
            )
        )
        step_id += 1
        current_delay += 700

        # Jump spell over inner compartment boundary
        jump_px = int(round(center_x * 0.75 + entry_px * 0.25))
        jump_py = int(round(center_y * 0.75 + entry_py * 0.25))
        jump_nx = round(jump_px / cw, 4)
        jump_ny = round(jump_py / ch, 4)

        current_delay += 400
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Jump Spell",
                target_x=jump_px,
                target_y=jump_py,
                normalized_x=jump_nx,
                normalized_y=jump_ny,
                delay_ms=current_delay,
                unit_count=1,
                note="Jump Spell bridging outer corridor to core storages.",
            )
        )
        step_id += 1
        current_delay += 400

        # Haste spell at entry point
        current_delay += 300
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Haste Spell",
                target_x=entry_px,
                target_y=entry_py,
                normalized_x=entry_nx,
                normalized_y=entry_ny,
                delay_ms=current_delay,
                unit_count=1,
                note="Haste spell accelerating goblins into core before stealth depletion.",
            )
        )
        step_id += 1
        current_delay += 500

        # Phase 3: Core Extraction Waves
        for wave in range(2):
            plan.append(
                DeploymentStep(
                    step_id=step_id,
                    troop_type="Sneaky Goblin",
                    target_x=entry_px,
                    target_y=entry_py,
                    normalized_x=entry_nx,
                    normalized_y=entry_ny,
                    delay_ms=current_delay,
                    unit_count=6,
                    note=f"Core extraction wave {wave + 1}: Rush central dark elixir & storages.",
                )
            )
            step_id += 1
            current_delay += 800

        # Phase 4: Town Hall Snipe Sequence
        th_accessible = layout.th_accessible if layout else False
        th_target_x = int(cw * 0.2) if th_accessible else center_x
        th_target_y = center_y
        th_nx = round(th_target_x / cw, 4)
        th_ny = round(th_target_y / ch, 4)

        # Invisibility spell covering Town Hall
        current_delay += 600
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Invisibility Spell",
                target_x=th_target_x,
                target_y=th_target_y,
                normalized_x=th_nx,
                normalized_y=th_ny,
                delay_ms=current_delay,
                unit_count=1,
                note="Invisibility spell covering goblins while securing Town Hall.",
            )
        )
        step_id += 1
        current_delay += 400

        # Town Hall takedown pack
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Sneaky Goblin",
                target_x=th_target_x,
                target_y=th_target_y,
                normalized_x=th_nx,
                normalized_y=th_ny,
                delay_ms=current_delay,
                unit_count=7,
                note="Town Hall strike: Secure final star and remaining 20% resource share.",
            )
        )

        return plan
