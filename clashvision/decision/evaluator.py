"""
ClashVision Farming Decision Subsystem — Base Resource Evaluator.
Evaluates enemy base resource totals against user thresholds and priority weights,
applying structural base layout modifiers (exposed collectors vs core storages).
Derived from Requirement R3 and Feature 14.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from clashvision.core.constants import (
    ResourcePriority,
    DEFAULT_MIN_GOLD,
    DEFAULT_MIN_ELIXIR,
    DEFAULT_MIN_DARK_ELIXIR,
)
from clashvision.core.models import (
    ResourceReading,
    BaseLayoutFeatures,
)
from clashvision.core.config import ResourceThresholds


@dataclass
class BaseEvaluationResult:
    """
    Detailed evaluation of an enemy base's loot value and layout accessibility.
    """
    normalized_score: float  # [0.0, 100.0]
    meets_thresholds: bool
    resource_score: float
    layout_modifier: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normalized_score": round(self.normalized_score, 2),
            "meets_thresholds": self.meets_thresholds,
            "resource_score": round(self.resource_score, 2),
            "layout_modifier": round(self.layout_modifier, 2),
            "breakdown": {k: round(v, 2) for k, v in self.breakdown.items()},
            "summary": self.summary,
        }


class BaseEvaluator:
    """
    Configurable base resource evaluator.
    Computes a normalized score in [0.0, 100.0] based on Gold, Elixir, Dark Elixir,
    resource priority weighting, user thresholds, and layout accessibility modifiers.
    """

    def __init__(self, default_thresholds: Optional[ResourceThresholds] = None) -> None:
        self.default_thresholds = default_thresholds or ResourceThresholds()

    def evaluate(
        self,
        resources: Optional[ResourceReading],
        layout: Optional[BaseLayoutFeatures] = None,
        thresholds: Optional[ResourceThresholds] = None,
    ) -> BaseEvaluationResult:
        """
        Evaluates a base given detected resources and layout features.
        """
        active_thresholds = thresholds or self.default_thresholds

        # Handle missing or empty resource reading
        if resources is None or not resources.has_any():
            return BaseEvaluationResult(
                normalized_score=0.0,
                meets_thresholds=False,
                resource_score=0.0,
                layout_modifier=0.0,
                breakdown={
                    "gold_score": 0.0,
                    "elixir_score": 0.0,
                    "dark_elixir_score": 0.0,
                    "raw_resource_score": 0.0,
                    "layout_modifier": 0.0,
                    "final_score": 0.0,
                },
                summary="No resources detected or OCR reading unavailable.",
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

        # Zero-loot protection: If all resources are 0 or negative, base is unviable
        if gold <= 0 and elixir <= 0 and dark_elixir <= 0:
            return BaseEvaluationResult(
                normalized_score=0.0,
                meets_thresholds=False,
                resource_score=0.0,
                layout_modifier=0.0,
                breakdown={
                    "gold_score": 0.0,
                    "elixir_score": 0.0,
                    "dark_elixir_score": 0.0,
                    "raw_resource_score": 0.0,
                    "layout_modifier": 0.0,
                    "final_score": 0.0,
                },
                summary="Zero harvestable loot detected on target base.",
            )

        # 1. Weight adjustments based on priority
        w_g = active_thresholds.gold_weight
        w_e = active_thresholds.elixir_weight
        w_de = active_thresholds.dark_elixir_weight

        priority = active_thresholds.priority
        if priority == ResourcePriority.GOLD_FIRST:
            w_g *= 2.2
            w_e *= 0.6
            w_de *= 0.8
        elif priority == ResourcePriority.ELIXIR_FIRST:
            w_e *= 2.2
            w_g *= 0.6
            w_de *= 0.8
        elif priority == ResourcePriority.DARK_ELIXIR_FIRST:
            w_de *= 2.5
            w_g *= 0.5
            w_e *= 0.5

        # 2. Threshold comparisons
        min_g = _sanitize_val(active_thresholds.min_gold)
        min_e = _sanitize_val(active_thresholds.min_elixir)
        min_de = _sanitize_val(active_thresholds.min_dark_elixir)

        g_ratio = (gold / min_g) if min_g > 0 else (1.0 if gold > 0 else 0.0)
        e_ratio = (elixir / min_e) if min_e > 0 else (1.0 if elixir > 0 else 0.0)
        de_ratio = (dark_elixir / min_de) if min_de > 0 else (1.0 if dark_elixir > 0 else 0.0)

        # Check threshold satisfaction according to priority
        meets_thresholds = self._check_meets_thresholds(
            gold, elixir, dark_elixir, min_g, min_e, min_de, priority
        )

        # 3. Individual resource scores [0.0, 100.0]
        # Benchmark: ratio = 1.0 -> score 60.0; ratio = 2.0 -> score 90.0; ratio >= 2.5 -> score 100.0
        g_score = self._ratio_to_subscore(g_ratio)
        e_score = self._ratio_to_subscore(e_ratio)
        de_score = self._ratio_to_subscore(de_ratio)

        # Weighted combination of resource subscores
        total_weight = w_g + w_e + (w_de / 100.0)  # scale DE weight proportionally
        if total_weight <= 0:
            total_weight = 1.0

        resource_score = (
            g_score * w_g + e_score * w_e + de_score * (w_de / 100.0)
        ) / total_weight
        resource_score = max(0.0, min(100.0, resource_score if math.isfinite(resource_score) else 0.0))

        # 4. Layout modifier calculations
        layout_mod = 0.0
        collector_bonus = 0.0
        th_bonus = 0.0
        dead_base_bonus = 0.0
        storage_penalty = 0.0

        if layout is not None:
            # Exposed collector bonus: up to +15 points if 100% collectors exposed
            exp_ratio = max(0.0, min(1.0, layout.exposed_collector_ratio))
            collector_bonus = exp_ratio * 15.0

            # Dead base detection
            is_dead = bool(layout.metadata.get("dead_base", False))
            if is_dead or (exp_ratio >= 0.85 and (gold + elixir) > 800_000):
                dead_base_bonus = 10.0

            # Town Hall accessibility bonus
            if layout.th_accessible:
                th_bonus = 5.0

            # Core storage penalty if collectors are not exposed and storages are in core
            storages_in_core = layout.metadata.get("storages_in_core", True)
            if storages_in_core and exp_ratio < 0.25 and not is_dead:
                storage_penalty = -8.0 * (1.0 - exp_ratio)

            layout_mod = collector_bonus + th_bonus + dead_base_bonus + storage_penalty

        # 5. Composite Normalized Score
        final_score = resource_score + layout_mod
        final_score = max(0.0, min(100.0, final_score if math.isfinite(final_score) else 0.0))

        # Build detailed breakdown
        breakdown = {
            "gold_score": g_score,
            "elixir_score": e_score,
            "dark_elixir_score": de_score,
            "weighted_resource_score": resource_score,
            "collector_exposure_bonus": collector_bonus,
            "th_accessibility_bonus": th_bonus,
            "dead_base_bonus": dead_base_bonus,
            "core_storage_penalty": storage_penalty,
            "layout_modifier": layout_mod,
            "final_score": final_score,
        }

        # Human-readable summary
        summary = self._generate_summary(
            final_score, gold, elixir, dark_elixir, meets_thresholds, layout
        )

        return BaseEvaluationResult(
            normalized_score=final_score,
            meets_thresholds=meets_thresholds,
            resource_score=resource_score,
            layout_modifier=layout_mod,
            breakdown=breakdown,
            summary=summary,
        )

    @staticmethod
    def _ratio_to_subscore(ratio: float) -> float:
        """
        Maps threshold achievement ratio to a 0-100 scale:
        - ratio == 0.0: 0.0
        - ratio == 1.0 (exact threshold): 60.0
        - ratio == 1.5: 80.0
        - ratio == 2.0: 92.0
        - ratio >= 2.5: 100.0
        """
        if ratio <= 0:
            return 0.0
        if ratio < 1.0:
            return ratio * 60.0
        if ratio <= 2.0:
            return 60.0 + (ratio - 1.0) * 32.0  # 60 to 92
        # Above 2.0x threshold
        return min(100.0, 92.0 + (ratio - 2.0) * 16.0)

    @staticmethod
    def _check_meets_thresholds(
        gold: int,
        elixir: int,
        dark_elixir: int,
        min_g: int,
        min_e: int,
        min_de: int,
        priority: ResourcePriority,
    ) -> bool:
        """Determines whether current resources satisfy target thresholds under priority rules."""
        if priority == ResourcePriority.GOLD_FIRST:
            return gold >= min_g
        elif priority == ResourcePriority.ELIXIR_FIRST:
            return elixir >= min_e
        elif priority == ResourcePriority.DARK_ELIXIR_FIRST:
            return dark_elixir >= min_de
        else:  # BALANCED
            # Either both regular resources meet, or dark elixir + at least one regular meets, or total regular resources >= min_g + min_e
            reg_total = gold + elixir
            min_reg_total = min_g + min_e
            return (
                (gold >= min_g and elixir >= min_e)
                or (reg_total >= min_reg_total and min_reg_total > 0)
                or (dark_elixir >= min_de and (gold >= min_g or elixir >= min_e))
            )

    @staticmethod
    def _generate_summary(
        score: float,
        gold: int,
        elixir: int,
        dark_elixir: int,
        meets: bool,
        layout: Optional[BaseLayoutFeatures],
    ) -> str:
        """Generates concise tactical summary text."""
        status = "Target accepted" if meets else "Below farming threshold"
        res_text = f"Gold: {gold:,}, Elixir: {elixir:,}, DE: {dark_elixir:,}"
        layout_text = ""
        if layout:
            if layout.metadata.get("dead_base", False):
                layout_text = " [DEAD BASE: Full collectors]"
            elif layout.exposed_collector_ratio >= 0.7:
                layout_text = f" [{int(layout.exposed_collector_ratio * 100)}% exposed collectors]"
            elif layout.th_accessible:
                layout_text = " [Town Hall perimeter exposed]"
        return f"{status} (Score {score:.1f}/100) — {res_text}{layout_text}"
