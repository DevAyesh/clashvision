"""
ClashVision Farming Decision Subsystem — Strategy E: User-Defined Custom Strategy.
Configurable strategy architecture supporting user army compositions, custom thresholds,
target priorities, and procedural deployment sequences.
Derived from Requirement R3 and Feature 20.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
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


@dataclass
class CustomStrategyConfig:
    """
    User-configurable parameters for custom strategy execution.
    """
    name: str = "Custom Attack Strategy"
    description: str = "Configurable strategy tailored to user-defined army preferences."
    army_composition: Dict[str, int] = field(
        default_factory=lambda: {
            "Super Barbarian": 30,
            "Super Archer": 20,
            "Earthquake Spell": 4,
            "Rage Spell": 1,
        }
    )
    target_priorities: List[str] = field(
        default_factory=lambda: ["gold", "elixir", "dark_elixir", "town_hall"]
    )
    min_gold: int = 350_000
    min_elixir: int = 350_000
    min_dark_elixir: int = 2_500
    preferred_layout_types: List[str] = field(default_factory=list)
    requires_accessible_th: bool = False
    max_acceptable_risk: RiskLevel = RiskLevel.HIGH
    custom_steps: List[Dict[str, Any]] = field(default_factory=list)
    training_time_seconds: int = 1200
    elixir_cost: int = 120_000
    dark_elixir_cost: int = 400

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["max_acceptable_risk"] = self.max_acceptable_risk.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CustomStrategyConfig:
        risk = data.get("max_acceptable_risk", RiskLevel.HIGH.value)
        if isinstance(risk, str):
            risk = RiskLevel(risk)
        return cls(
            name=data.get("name", "Custom Attack Strategy"),
            description=data.get("description", "Configurable custom attack strategy."),
            army_composition=dict(data.get("army_composition", {})),
            target_priorities=list(data.get("target_priorities", ["gold", "elixir"])),
            min_gold=int(data.get("min_gold", 350_000)),
            min_elixir=int(data.get("min_elixir", 350_000)),
            min_dark_elixir=int(data.get("min_dark_elixir", 2_500)),
            preferred_layout_types=list(data.get("preferred_layout_types", [])),
            requires_accessible_th=bool(data.get("requires_accessible_th", False)),
            max_acceptable_risk=risk,
            custom_steps=list(data.get("custom_steps", [])),
            training_time_seconds=int(data.get("training_time_seconds", 1200)),
            elixir_cost=int(data.get("elixir_cost", 120_000)),
            dark_elixir_cost=int(data.get("dark_elixir_cost", 400)),
        )


class CustomStrategy(BaseStrategy):
    """
    Strategy E: User-Defined Custom Strategy.
    Accepts arbitrary user army configurations, customized resource targets,
    prerequisite layout constraints, and deployment templates.
    """

    def __init__(self, config: Optional[CustomStrategyConfig] = None) -> None:
        self.config = config or CustomStrategyConfig()

    def update_config(self, config: CustomStrategyConfig) -> None:
        """Dynamically reconfigures the strategy."""
        self.config = config

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.CUSTOM

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def description(self) -> str:
        return self.config.description

    @property
    def army_composition(self) -> Dict[str, int]:
        return dict(self.config.army_composition)

    @property
    def estimated_training_time_seconds(self) -> int:
        return self.config.training_time_seconds

    @property
    def estimated_elixir_cost(self) -> int:
        return self.config.elixir_cost

    @property
    def estimated_dark_elixir_cost(self) -> int:
        return self.config.dark_elixir_cost

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
                reasons=["No resource readings available for custom evaluation."],
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

        # 1. Custom Resource Suitability
        min_g = _sanitize_val(self.config.min_gold)
        min_e = _sanitize_val(self.config.min_elixir)
        min_de = _sanitize_val(self.config.min_dark_elixir)

        g_sat = min(2.0, gold / max(1, min_g))
        e_sat = min(2.0, elixir / max(1, min_e))
        de_sat = min(2.0, dark_elixir / max(1, min_de))

        res_suitability = ((g_sat + e_sat + de_sat * 1.5) / 3.5) * 50.0  # [0, 100]
        res_suitability = max(0.0, min(100.0, res_suitability))

        if gold >= min_g and elixir >= min_e:
            reasons.append("Base satisfies custom Gold and Elixir criteria.")
        else:
            reasons.append("Base resources fall short of user-configured minimums.")

        # 2. Custom Layout Suitability
        layout_suitability = 70.0
        layout_type = str(layout.metadata.get("layout_type", "")) if layout else ""
        th_accessible = layout.th_accessible if layout else False

        # Preferred layout bonus
        if self.config.preferred_layout_types:
            if layout_type in self.config.preferred_layout_types:
                layout_suitability = 95.0
                reasons.append(f"Base layout '{layout_type}' matches user preferred topology list.")
            else:
                layout_suitability = 55.0
                reasons.append(f"Layout '{layout_type}' is outside preferred topology set.")

        # TH accessibility prerequisite check
        prerequisites_met = True
        if self.config.requires_accessible_th:
            if not th_accessible:
                layout_suitability = max(10.0, layout_suitability - 40.0)
                prerequisites_met = False
                reasons.append("Constraint violation: Base requires accessible Town Hall, but TH is fortified in core.")
            else:
                reasons.append("Prerequisite met: Town Hall is accessible.")

        # Army non-empty validation
        has_army = any(c > 0 for c in self.config.army_composition.values()) or bool(self.config.custom_steps)
        if not has_army:
            prerequisites_met = False
            reasons.append("Empty custom army: No troops, spells, or custom steps configured.")
            layout_suitability = 0.0

        # 3. Risk Assessment
        risk = RiskLevel.MEDIUM
        exp_ratio = layout.exposed_collector_ratio if layout else 0.5
        if exp_ratio >= 0.7:
            risk = RiskLevel.LOW
        elif exp_ratio < 0.2:
            risk = RiskLevel.HIGH

        # Risk constraint check
        risk_rank = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3, RiskLevel.EXTREME: 4}
        if risk_rank[risk] > risk_rank[self.config.max_acceptable_risk]:
            prerequisites_met = False
            reasons.append(f"Risk level {risk.value} exceeds maximum user-tolerated threshold {self.config.max_acceptable_risk.value}.")

        # 4. Composite Score
        composite_score = round(0.50 * res_suitability + 0.50 * layout_suitability, 2)
        composite_score = max(0.0, min(100.0, composite_score if math.isfinite(composite_score) else 0.0))
        res_suitability = max(0.0, min(100.0, res_suitability if math.isfinite(res_suitability) else 0.0))
        layout_suitability = max(0.0, min(100.0, layout_suitability if math.isfinite(layout_suitability) else 0.0))
        viable = bool(has_army) and prerequisites_met and (composite_score >= 40.0)

        return StrategyEvaluation(
            strategy_type=self.strategy_type,
            score=composite_score,
            viable=viable,
            risk_level=risk,
            resource_suitability=round(res_suitability, 2),
            layout_suitability=round(layout_suitability, 2),
            prerequisites_met=prerequisites_met,
            reasons=reasons,
        )

    def generate_deployment_plan(
        self,
        layout: Optional[BaseLayoutFeatures],
        canvas_width: int = VIRTUAL_REFERENCE_WIDTH,
        canvas_height: int = VIRTUAL_REFERENCE_HEIGHT,
    ) -> List[DeploymentStep]:
        cw = max(1, canvas_width)
        ch = max(1, canvas_height)

        # If user provided explicit custom steps in configuration, parse and return them
        if self.config.custom_steps:
            plan: List[DeploymentStep] = []
            for idx, raw in enumerate(self.config.custom_steps):
                try:
                    unit_cnt = int(raw.get("unit_count", 1))
                except (TypeError, ValueError):
                    continue
                if unit_cnt <= 0:
                    continue

                # Safely parse and validate normalized coordinates
                try:
                    raw_nx = float(raw.get("normalized_x", 0.5))
                    nx = raw_nx if math.isfinite(raw_nx) else 0.5
                except (TypeError, ValueError):
                    nx = 0.5
                nx = max(0.0, min(1.0, nx))

                try:
                    raw_ny = float(raw.get("normalized_y", 0.5))
                    ny = raw_ny if math.isfinite(raw_ny) else 0.5
                except (TypeError, ValueError):
                    ny = 0.5
                ny = max(0.0, min(1.0, ny))

                # Safely parse and validate pixel coordinates
                raw_tx = raw.get("target_x")
                if raw_tx is not None:
                    try:
                        val_tx = float(raw_tx)
                        tx = int(round(val_tx)) if math.isfinite(val_tx) else int(round(nx * cw))
                    except (TypeError, ValueError):
                        tx = int(round(nx * cw))
                else:
                    tx = int(round(nx * cw))
                tx = max(0, min(cw, tx))

                raw_ty = raw.get("target_y")
                if raw_ty is not None:
                    try:
                        val_ty = float(raw_ty)
                        ty = int(round(val_ty)) if math.isfinite(val_ty) else int(round(ny * ch))
                    except (TypeError, ValueError):
                        ty = int(round(ny * ch))
                else:
                    ty = int(round(ny * ch))
                ty = max(0, min(ch, ty))

                try:
                    delay = int(raw.get("delay_ms", idx * 500))
                    delay = max(0, delay)
                except (TypeError, ValueError):
                    delay = idx * 500

                plan.append(
                    DeploymentStep(
                        step_id=len(plan) + 1,
                        troop_type=str(raw.get("troop_type", "Custom Unit")),
                        target_x=tx,
                        target_y=ty,
                        normalized_x=round(nx, 4),
                        normalized_y=round(ny, 4),
                        delay_ms=delay,
                        unit_count=unit_cnt,
                        note=str(raw.get("note", "Custom deployment action.")),
                    )
                )
            return plan

        # Procedurally generate a plan from custom army composition
        plan: List[DeploymentStep] = []
        step_id = 1
        current_delay = 0

        center_x = cw // 2
        center_y = ch // 2

        # 1. Earthquakes / Spells
        spell_troops = [k for k in self.config.army_composition if "Spell" in k]
        for spell in spell_troops:
            count = self.config.army_composition[spell]
            if count <= 0:
                continue
            plan.append(
                DeploymentStep(
                    step_id=step_id,
                    troop_type=spell,
                    target_x=center_x,
                    target_y=center_y,
                    normalized_x=round(center_x / cw, 4),
                    normalized_y=round(center_y / ch, 4),
                    delay_ms=current_delay,
                    unit_count=count,
                    note=f"Deploy {spell} to weaken core compartments.",
                )
            )
            step_id += 1
            current_delay += 800

        # 2. Main combat troops
        combat_troops = [k for k in self.config.army_composition if "Spell" not in k]
        valid_combat_troops = [k for k in combat_troops if self.config.army_composition[k] > 0]
        perimeter_points = self.get_cardinal_perimeter_points(
            count=max(4, len(valid_combat_troops) * 2),
            radius_fraction=0.42,
            canvas_width=cw,
            canvas_height=ch,
        )

        pt_idx = 0
        for troop in combat_troops:
            total_count = self.config.army_composition[troop]
            if total_count <= 0:
                continue
            # Split into 2 deployment points
            half_count = max(1, total_count // 2)
            rem = total_count - half_count

            px, py, nx, ny = perimeter_points[pt_idx % len(perimeter_points)]
            plan.append(
                DeploymentStep(
                    step_id=step_id,
                    troop_type=troop,
                    target_x=px,
                    target_y=py,
                    normalized_x=nx,
                    normalized_y=ny,
                    delay_ms=current_delay,
                    unit_count=half_count,
                    note=f"Deploy {half_count}x {troop} at perimeter sector {pt_idx + 1}.",
                )
            )
            step_id += 1
            pt_idx += 1
            current_delay += 400

            if rem > 0:
                px2, py2, nx2, ny2 = perimeter_points[pt_idx % len(perimeter_points)]
                plan.append(
                    DeploymentStep(
                        step_id=step_id,
                        troop_type=troop,
                        target_x=px2,
                        target_y=py2,
                        normalized_x=nx2,
                        normalized_y=ny2,
                        delay_ms=current_delay,
                        unit_count=rem,
                        note=f"Deploy remaining {rem}x {troop} at sector {pt_idx + 1}.",
                    )
                )
                step_id += 1
                pt_idx += 1
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
