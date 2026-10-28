"""
ClashVision Autonomous Farming Engine — Simulated Battle Outcome Resolver.
Determines battle result, destruction percentage, star acquisition (0-3),
and harvested resources based on strategy execution and base defenses.
Derived from Master Prompt Section 1, 5, 10.
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from clashvision.core.constants import StrategyType
from clashvision.core.models import (
    BaseLayoutFeatures,
    DeploymentStep,
    ResourceReading,
)

logger = logging.getLogger("clashvision.simulator.battle_resolver")


@dataclass
class SimulatedBattleResult:
    """
    Standardized result contract for an offline simulated attack.
    """
    strategy_type: StrategyType
    stars: int  # 0, 1, 2, or 3
    destruction_percent: int  # 0 - 100
    gold_looted: int
    elixir_looted: int
    dark_elixir_looted: int
    th_destroyed: bool
    duration_seconds: float
    total_troops_deployed: int
    steps_executed: int
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_type": self.strategy_type.value if isinstance(self.strategy_type, StrategyType) else str(self.strategy_type),
            "stars": self.stars,
            "destruction_percent": self.destruction_percent,
            "gold_looted": self.gold_looted,
            "elixir_looted": self.elixir_looted,
            "dark_elixir_looted": self.dark_elixir_looted,
            "th_destroyed": self.th_destroyed,
            "duration_seconds": round(self.duration_seconds, 2),
            "total_troops_deployed": self.total_troops_deployed,
            "steps_executed": self.steps_executed,
            "summary": self.summary,
        }


class BattleResolver:
    """
    Deterministic tactical battle resolution engine.
    Computes destruction percentage and loot yields based on:
    - Army type and deployment steps executed
    - Base defense hazard density and Town Hall accessibility
    - Collector locations (outside perimeter vs core)
    """

    @staticmethod
    def resolve(
        strategy_type: StrategyType,
        steps: List[DeploymentStep],
        available_resources: Optional[ResourceReading],
        layout: Optional[BaseLayoutFeatures],
        execution_fraction: float = 1.0,
    ) -> SimulatedBattleResult:
        """
        Resolves the outcome of a simulated battle.
        `execution_fraction` represents how much of the planned deployment was executed (0.0 to 1.0).
        """
        gold_avail = available_resources.gold if available_resources and available_resources.gold else 0
        elixir_avail = available_resources.elixir if available_resources and available_resources.elixir else 0
        de_avail = available_resources.dark_elixir if available_resources and available_resources.dark_elixir else 0

        total_troops = sum(
            getattr(s, "unit_count", 1)
            for s in steps
            if not getattr(s, "spell_name", None)
            and "Spell" not in getattr(s, "troop_type", "")
        )
        executed_steps = int(len(steps) * max(0.0, min(1.0, execution_fraction)))

        # Base vulnerability factors
        hazard_score = (
            layout.metadata.get("defense_hazard_score", 45.0)
            if layout and layout.metadata
            else 45.0
        )
        th_exposed = layout.th_accessible if layout else True
        collectors_outside = (
            (layout.exposed_collector_ratio > 0.4)
            if layout and hasattr(layout, "exposed_collector_ratio")
            else True
        )

        # Tactical resolution depending on Strategy Archetype
        th_destroyed = False
        destruction_pct = 0
        loot_harvest_ratio = 0.0

        if strategy_type == StrategyType.SNEAKY_GOBLIN:
            # Sneaky Goblins: High resource extraction, moderate destruction (40-60%), reliable 1-2 stars
            th_destroyed = th_exposed or (execution_fraction >= 0.85 and hazard_score < 65)
            # Collectors outside guarantee >= 80% loot
            if collectors_outside:
                loot_harvest_ratio = 0.85 + (0.15 * execution_fraction)
            else:
                loot_harvest_ratio = 0.65 + (0.25 * execution_fraction)

            destruction_pct = int(min(65, 35 + (25 * execution_fraction) + (10 if th_destroyed else 0)))

        elif strategy_type == StrategyType.BARCH:
            # BARCH: Perimeter clear, 50% 1-star focus, rarely destroys core Town Hall
            th_destroyed = th_exposed and (execution_fraction >= 0.90)
            destruction_pct = int(min(70, 42 + (20 * execution_fraction)))
            loot_harvest_ratio = 0.70 + (0.25 * execution_fraction) if collectors_outside else 0.50

        elif strategy_type == StrategyType.MASS_DRAGON:
            # Mass Dragon: Heavy destruction (70-100%), 2-3 stars, high core penetration
            air_def_penalty = max(0.0, (hazard_score - 40) * 0.4)
            destruction_pct = int(max(40, min(100, (85 * execution_fraction) - air_def_penalty + 15)))
            th_destroyed = destruction_pct >= 50
            loot_harvest_ratio = min(1.0, (destruction_pct / 100.0) * 1.05)

        elif strategy_type == StrategyType.MASS_MINER:
            # Mass Miner: Bypasses walls, vulnerable to heavy splash (Scattershots / Eagle)
            splash_penalty = max(0.0, (hazard_score - 45) * 0.35)
            destruction_pct = int(max(45, min(100, (82 * execution_fraction) - splash_penalty + 12)))
            th_destroyed = destruction_pct >= 50
            loot_harvest_ratio = min(1.0, (destruction_pct / 100.0) * 1.02)

        else:  # CUSTOM
            destruction_pct = int(min(90, 50 + (35 * execution_fraction)))
            th_destroyed = th_exposed or (destruction_pct >= 60)
            loot_harvest_ratio = min(1.0, (destruction_pct / 100.0))

        # Clamp metrics
        destruction_pct = max(0, min(100, destruction_pct))
        loot_harvest_ratio = max(0.0, min(1.0, loot_harvest_ratio))

        # Calculate Stars
        # Star 1: 50% Destruction
        # Star 2: Town Hall Destroyed
        # Star 3: 100% Destruction
        stars = 0
        if destruction_pct >= 50:
            stars += 1
        if th_destroyed:
            stars += 1
        if destruction_pct == 100:
            stars = 3

        # Actual Loot Acquired
        gold_looted = int(gold_avail * loot_harvest_ratio)
        elixir_looted = int(elixir_avail * loot_harvest_ratio)
        dark_elixir_looted = int(de_avail * loot_harvest_ratio)

        # Attack duration in seconds (typically 45s - 180s)
        duration_s = round(max(30.0, min(180.0, 45.0 + (executed_steps * 2.5))), 1)

        summary = (
            f"{strategy_type.value}: {stars} Stars ({destruction_pct}%) | "
            f"Loot: G {gold_looted:,} / E {elixir_looted:,} / DE {dark_elixir_looted:,} "
            f"in {duration_s}s"
        )

        return SimulatedBattleResult(
            strategy_type=strategy_type,
            stars=stars,
            destruction_percent=destruction_pct,
            gold_looted=gold_looted,
            elixir_looted=elixir_looted,
            dark_elixir_looted=dark_elixir_looted,
            th_destroyed=th_destroyed,
            duration_seconds=duration_s,
            total_troops_deployed=total_troops,
            steps_executed=executed_steps,
            summary=summary,
        )
