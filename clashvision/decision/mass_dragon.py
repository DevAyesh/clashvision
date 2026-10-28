"""
ClashVision Farming Decision Subsystem — Strategy D: Mass Dragon.
Air attack calculating optimal approach corridors, analyzing Air Defense spatial positioning,
and coordinating dragon funnels, balloon sweeps, rage amplification, and freeze locks.
Derived from Requirement R3 and Feature 19.
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


class MassDragonStrategy(BaseStrategy):
    """
    Strategy D: Mass Dragon Corridor Assault.
    Analyzes Air Defense (AD) coordinates to identify vulnerability corridors.
    Deploys lightning strikes against inaccessible ADs, funnels flanking dragons,
    and propels balloon squads through rage-boosted air corridors into core storages.
    """

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.MASS_DRAGON

    @property
    def name(self) -> str:
        return "Mass Dragon Corridor"

    @property
    def description(self) -> str:
        return (
            "Heavy aerial bombardment bypassing ground defenses and walls. Identifies safest "
            "air attack corridor away from Air Defense clusters, using lightning strikes on "
            "remote ADs and funneling dragons directly into core storages."
        )

    @property
    def army_composition(self) -> Dict[str, int]:
        return {
            "Dragon": 12,
            "Balloon": 8,
            "Lightning Spell": 6,
            "Rage Spell": 2,
            "Freeze Spell": 2,
        }

    @property
    def estimated_training_time_seconds(self) -> int:
        return 1800  # 30 minutes

    @property
    def estimated_elixir_cost(self) -> int:
        return 240_000

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
        # High elixir training cost (~240k) requires a substantial loot pool to justify
        min_gold = _sanitize_val(thresholds.min_gold) if thresholds else 400_000
        min_elixir = _sanitize_val(thresholds.min_elixir) if thresholds else 400_000
        target_total = min_gold + min_elixir

        loot_margin = total_loot - self.estimated_elixir_cost
        if loot_margin >= target_total * 1.5:
            res_suitability = 95.0
            reasons.append(f"Substantial loot pool ({total_loot:,}) comfortably exceeds heavy air army cost.")
        elif loot_margin >= target_total:
            res_suitability = 78.0
            reasons.append(f"Profitable aerial target: {total_loot:,} total Gold/Elixir.")
        elif loot_margin >= 250_000:
            res_suitability = 52.0
            reasons.append("Moderate return; 240k elixir training investment represents significant overhead.")
        else:
            res_suitability = 25.0
            reasons.append("Insufficient loot to justify heavy Dragon army training cost.")

        # 2. Layout Suitability [0.0, 100.0] & Air Defense Analysis
        ad_locs = layout.air_defense_locations if layout else []
        layout_type = str(layout.metadata.get("layout_type", "")) if layout else ""

        # Normalize AD coords to [0.0, 1.0] if in pixels, skipping non-finite coords
        norm_ads: List[Tuple[float, float]] = []
        for ax, ay in ad_locs:
            try:
                fax, fay = float(ax), float(ay)
                if not (math.isfinite(fax) and math.isfinite(fay)):
                    continue
                nx = fax / 1920.0 if fax > 1.0 else fax
                ny = fay / 1080.0 if fay > 1.0 else fay
                norm_ads.append((max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))))
            except (TypeError, ValueError):
                continue

        if layout_type == "AIR_DEFENSE_CORRIDOR":
            layout_suitability = 94.0
            reasons.append("Air Defense Corridor layout detected: Air Defenses clustered on one quadrant, leaving an open aerial corridor.")
        elif len(norm_ads) <= 2:
            layout_suitability = 90.0
            reasons.append(f"Low Air Defense density ({len(norm_ads)} detected): Minimal aerial resistance.")
        elif self._is_ad_asymmetric(norm_ads):
            layout_suitability = 85.0
            reasons.append("Asymmetric Air Defense placement allows safe corridor approach from unfortified flank.")
        else:
            layout_suitability = 65.0
            reasons.append(f"Symmetric Air Defense coverage ({len(norm_ads)} ADs): Requires lightning strikes and tight corridor funneling.")

        # 3. Risk Assessment
        if layout_type == "AIR_DEFENSE_CORRIDOR" or len(norm_ads) <= 2:
            risk = RiskLevel.LOW
            reasons.append("Low air risk: Dragons can avoid head-on engagements with multiple Air Defenses.")
        elif self._is_ad_asymmetric(norm_ads):
            risk = RiskLevel.MEDIUM
            reasons.append("Medium air risk: Safe corridor available, but core freeze spells are required.")
        else:
            risk = RiskLevel.HIGH
            reasons.append("High air risk: Four-quadrant Air Defense web will engage dragons simultaneously.")

        # 4. Composite Score & Viability
        composite_score = round(0.50 * res_suitability + 0.50 * layout_suitability, 2)
        composite_score = max(0.0, min(100.0, composite_score if math.isfinite(composite_score) else 0.0))
        res_suitability = max(0.0, min(100.0, res_suitability if math.isfinite(res_suitability) else 0.0))
        layout_suitability = max(0.0, min(100.0, layout_suitability if math.isfinite(layout_suitability) else 0.0))
        viable = (total_loot >= 450_000) and (composite_score >= 45.0)

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

        # 1. Analyze Air Defenses to find the safest corridor
        ad_locs = layout.air_defense_locations if layout else []
        norm_ads: List[Tuple[float, float]] = []
        for ax, ay in ad_locs:
            try:
                fax, fay = float(ax), float(ay)
                if not (math.isfinite(fax) and math.isfinite(fay)):
                    continue
                nx = fax / float(cw) if fax > 1.0 else fax
                ny = fay / float(ch) if fay > 1.0 else fay
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
                norm_ads.append((nx, ny))
            except (TypeError, ValueError):
                continue

        # Safest approach angle (furthest from AD clusters)
        corridor_angle = self.find_safest_corridor_angle(norm_ads)
        if not math.isfinite(corridor_angle):
            corridor_angle = -math.pi / 2.0

        # Phase 1: Lightning Strikes on opposite / deep Air Defenses
        # Zap the AD furthest from the chosen corridor entry
        target_ad = self._find_most_dangerous_ad(norm_ads, corridor_angle)
        ad_x = target_ad[0] if (target_ad and math.isfinite(target_ad[0])) else 0.5
        ad_y = target_ad[1] if (target_ad and math.isfinite(target_ad[1])) else 0.5
        ad_x = max(0.0, min(1.0, ad_x))
        ad_y = max(0.0, min(1.0, ad_y))

        zap_px = max(0, min(cw, int(round(ad_x * cw))))
        zap_py = max(0, min(ch, int(round(ad_y * ch))))
        norm_zap_x = round(ad_x, 4)
        norm_zap_y = round(ad_y, 4)

        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Lightning Spell",
                target_x=zap_px,
                target_y=zap_py,
                normalized_x=norm_zap_x,
                normalized_y=norm_zap_y,
                delay_ms=current_delay,
                unit_count=3,
                note="Lightning Strike wave 1: Eliminate high-threat core Air Defense.",
            )
        )
        step_id += 1
        current_delay += 350

        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Lightning Spell",
                target_x=zap_px,
                target_y=zap_py,
                normalized_x=norm_zap_x,
                normalized_y=norm_zap_y,
                delay_ms=current_delay,
                unit_count=3,
                note="Lightning Strike wave 2: Destroy targeted Air Defense structure completely.",
            )
        )
        step_id += 1
        current_delay += 800

        # Phase 2: Dragon Funneling on Corridor Flanks
        flank_spread = math.pi / 5.0
        left_angle = corridor_angle - flank_spread
        right_angle = corridor_angle + flank_spread

        l_px, l_py, l_nx, l_ny = self.calculate_perimeter_point(
            left_angle, radius_fraction=0.42, canvas_width=cw, canvas_height=ch
        )
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Dragon",
                target_x=l_px,
                target_y=l_py,
                normalized_x=l_nx,
                normalized_y=l_ny,
                delay_ms=current_delay,
                unit_count=1,
                note="Left flank funnel: 1 Dragon to clear outer structures and anchor corridor.",
            )
        )
        step_id += 1
        current_delay += 300

        r_px, r_py, r_nx, r_ny = self.calculate_perimeter_point(
            right_angle, radius_fraction=0.42, canvas_width=cw, canvas_height=ch
        )
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Dragon",
                target_x=r_px,
                target_y=r_py,
                normalized_x=r_nx,
                normalized_y=r_ny,
                delay_ms=current_delay,
                unit_count=1,
                note="Right flank funnel: 1 Dragon to establish attack lane boundaries.",
            )
        )
        step_id += 1
        current_delay += 1000

        # Phase 3: Main Dragon Line along Safe Corridor
        main_px, main_py, main_nx, main_ny = self.calculate_perimeter_point(
            corridor_angle, radius_fraction=0.40, canvas_width=cw, canvas_height=ch
        )

        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Dragon",
                target_x=main_px,
                target_y=main_py,
                normalized_x=main_nx,
                normalized_y=main_ny,
                delay_ms=current_delay,
                unit_count=10,
                note="Main aerial wave: Deploy 10 Dragons in concentrated corridor spread.",
            )
        )
        step_id += 1
        current_delay += 800

        # Phase 4: Balloon Escort Squad
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Balloon",
                target_x=main_px,
                target_y=main_py,
                normalized_x=main_nx,
                normalized_y=main_ny,
                delay_ms=current_delay,
                unit_count=8,
                note="Balloon squad: Direct defense targeting behind dragon shield line.",
            )
        )
        step_id += 1
        current_delay += 1500

        # Phase 5: Rage Spell into base core
        entry_mid_x = int(round(main_px * 0.5 + center_x * 0.5))
        entry_mid_y = int(round(main_py * 0.5 + center_y * 0.5))

        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Rage Spell",
                target_x=entry_mid_x,
                target_y=entry_mid_y,
                normalized_x=round(entry_mid_x / cw, 4),
                normalized_y=round(entry_mid_y / ch, 4),
                delay_ms=current_delay,
                unit_count=1,
                note="Rage Spell 1: Accelerate dragons through initial defensive boundary.",
            )
        )
        step_id += 1
        current_delay += 1500

        # Freeze Spell on Core Defenses / Town Hall
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Freeze Spell",
                target_x=center_x,
                target_y=center_y,
                normalized_x=round(center_x / cw, 4),
                normalized_y=round(center_y / ch, 4),
                delay_ms=current_delay,
                unit_count=1,
                note="Freeze Spell: Lockdown core Town Hall Giga Inferno or Air Sweeper.",
            )
        )
        step_id += 1
        current_delay += 2000

        # Rage Spell 2: Core destruction
        plan.append(
            DeploymentStep(
                step_id=step_id,
                troop_type="Rage Spell",
                target_x=center_x,
                target_y=center_y,
                normalized_x=round(center_x / cw, 4),
                normalized_y=round(center_y / ch, 4),
                delay_ms=current_delay,
                unit_count=1,
                note="Rage Spell 2: Maximum damage multiplier over central storages and Town Hall.",
            )
        )

        # Strictly validate finiteness and clamp all deployment step coordinates to canvas bounds
        for step in plan:
            tx = float(step.target_x) if math.isfinite(step.target_x) else (cw / 2.0)
            ty = float(step.target_y) if math.isfinite(step.target_y) else (ch / 2.0)
            step.target_x = max(0, min(cw, int(round(tx))))
            step.target_y = max(0, min(ch, int(round(ty))))

            nx = float(step.normalized_x) if math.isfinite(step.normalized_x) else (step.target_x / cw)
            ny = float(step.normalized_y) if math.isfinite(step.normalized_y) else (step.target_y / ch)
            step.normalized_x = round(max(0.0, min(1.0, nx)), 4)
            step.normalized_y = round(max(0.0, min(1.0, ny)), 4)

        return plan

    @staticmethod
    def _is_ad_asymmetric(norm_ads: List[Tuple[float, float]]) -> bool:
        """Determines if Air Defenses are heavily biased toward one hemisphere."""
        if len(norm_ads) < 2:
            return True
        y_vals = [y for _, y in norm_ads]
        x_vals = [x for x, _ in norm_ads]
        # Check if all on north/south or east/west
        all_south = all(y > 0.55 for y in y_vals)
        all_north = all(y < 0.45 for y in y_vals)
        all_east = all(x > 0.55 for x in x_vals)
        all_west = all(x < 0.45 for x in x_vals)
        return all_south or all_north or all_east or all_west

    @staticmethod
    def _find_most_dangerous_ad(
        norm_ads: List[Tuple[float, float]],
        corridor_angle: float,
    ) -> Tuple[float, float]:
        """Finds the Air Defense that poses the greatest threat to the corridor."""
        if not norm_ads:
            return (0.5, 0.5)

        # The most dangerous AD is the one opposite to the corridor or closest to center
        cx, cy = 0.5, 0.5
        best_ad = norm_ads[0]
        max_danger = -1.0

        for ax, ay in norm_ads:
            dx = ax - cx
            dy = ay - cy
            angle = math.atan2(dy, dx)
            # Angular distance to corridor entry: opposite angles (diff ~ pi) are deep in base
            angle_diff = abs(angle - corridor_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            dist = math.sqrt(dx * dx + dy * dy)
            danger = angle_diff + (1.0 - dist)
            if danger > max_danger:
                max_danger = danger
                best_ad = (ax, ay)

        return best_ad
