"""
ClashVision Autonomous Farming Engine — Automatic Target Selection & Loot Filtering.
Evaluates simulated enemy bases against configurable thresholds across 4 filtering modes:
ALL, ANY, WEIGHTED, and CUSTOM.
Derived from Master Prompt Section 4.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Callable, Dict, List, Optional, Tuple

from clashvision.core.constants import ResourcePriority
from clashvision.core.models import (
    BaseLayoutFeatures,
    ResourceReading,
)
from clashvision.core.config import ResourceThresholds

logger = logging.getLogger("clashvision.simulator.target_selector")


@unique
class FilterMode(str, Enum):
    """
    Loot filtering evaluation modes.
    """
    ALL = "ALL"              # All individual thresholds must be met
    ANY = "ANY"              # Any individual threshold is sufficient
    WEIGHTED = "WEIGHTED"    # Composite weighted resource total
    CUSTOM = "CUSTOM"        # User-defined predicate evaluation


@dataclass
class TargetFilterConfig:
    """
    Configuration specification for autonomous target selection.
    """
    min_gold: int = 600_000
    min_elixir: int = 600_000
    min_dark_elixir: int = 5_000
    filter_mode: FilterMode = FilterMode.ALL

    # Weighted mode parameters
    weight_gold: float = 1.0
    weight_elixir: float = 1.0
    weight_dark_elixir: float = 100.0  # DE valued ~100x standard resources
    min_weighted_total: float = 1_000_000.0

    # Minimum OCR confidence threshold to trust numbers
    min_ocr_confidence: float = 0.60

    # Maximum search iterations before auto-stopping
    max_search_attempts: int = 50

    # Custom callable predicate: (ResourceReading, Optional[BaseLayoutFeatures]) -> bool
    custom_predicate: Optional[Callable[[ResourceReading, Optional[BaseLayoutFeatures]], bool]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_gold": self.min_gold,
            "min_elixir": self.min_elixir,
            "min_dark_elixir": self.min_dark_elixir,
            "filter_mode": self.filter_mode.value,
            "weight_gold": self.weight_gold,
            "weight_elixir": self.weight_elixir,
            "weight_dark_elixir": self.weight_dark_elixir,
            "min_weighted_total": self.min_weighted_total,
            "min_ocr_confidence": self.min_ocr_confidence,
            "max_search_attempts": self.max_search_attempts,
        }


@dataclass
class TargetEvaluationResult:
    """
    Evaluation assessment for a single scouted base.
    """
    action: str  # "ATTACK" or "SKIP"
    is_suitable: bool
    farming_score: float  # [0.0, 100.0]
    weighted_value: float
    ocr_confidence: float
    reasons: List[str] = field(default_factory=list)
    resources: Optional[ResourceReading] = None
    layout: Optional[BaseLayoutFeatures] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "is_suitable": self.is_suitable,
            "farming_score": round(self.farming_score, 2),
            "weighted_value": round(self.weighted_value, 2),
            "ocr_confidence": round(self.ocr_confidence, 4),
            "reasons": self.reasons,
            "gold": self.resources.gold if self.resources else 0,
            "elixir": self.resources.elixir if self.resources else 0,
            "dark_elixir": self.resources.dark_elixir if self.resources else 0,
        }


class TargetSelector:
    """
    Automatic target selection engine.
    Evaluates OCR readings, enforces thresholds, calculates composite farming scores,
    and determines whether to ATTACK or SKIP candidate bases.
    """

    def __init__(self, config: Optional[TargetFilterConfig] = None) -> None:
        self.config = config or TargetFilterConfig()

    def update_config(self, config: TargetFilterConfig) -> None:
        self.config = config

    def calculate_weighted_value(self, gold: int, elixir: int, dark_elixir: int) -> float:
        """Calculates normalized economic value using configured resource weights."""
        return (
            (gold * self.config.weight_gold)
            + (elixir * self.config.weight_elixir)
            + (dark_elixir * self.config.weight_dark_elixir)
        )

    def evaluate(
        self,
        resources: Optional[ResourceReading],
        layout: Optional[BaseLayoutFeatures] = None,
        min_confidence: Optional[float] = None,
    ) -> TargetEvaluationResult:
        """
        Evaluates a candidate base against active loot thresholds.
        Returns TargetEvaluationResult with decision ATTACK or SKIP.
        """
        required_conf = min_confidence if min_confidence is not None else self.config.min_ocr_confidence

        if resources is None:
            return TargetEvaluationResult(
                action="SKIP",
                is_suitable=False,
                farming_score=0.0,
                weighted_value=0.0,
                ocr_confidence=0.0,
                reasons=["No resource readings available"],
            )

        # 1. OCR Confidence Validation
        confs = list(resources.confidence.values()) if resources.confidence else [1.0]
        avg_confidence = (sum(confs) / len(confs)) if confs else 0.8
        if avg_confidence < required_conf:
            return TargetEvaluationResult(
                action="SKIP",
                is_suitable=False,
                farming_score=0.0,
                weighted_value=0.0,
                ocr_confidence=avg_confidence,
                reasons=[f"OCR confidence {avg_confidence:.2f} below threshold {required_conf:.2f}"],
                resources=resources,
                layout=layout,
            )

        gold = max(0, resources.gold or 0)
        elixir = max(0, resources.elixir or 0)
        dark_elixir = max(0, resources.dark_elixir or 0)

        weighted_val = self.calculate_weighted_value(gold, elixir, dark_elixir)
        mode = self.config.filter_mode
        reasons: List[str] = []
        is_suitable = False

        # 2. Evaluate by FilterMode
        if mode == FilterMode.ALL:
            gold_ok = gold >= self.config.min_gold
            elixir_ok = elixir >= self.config.min_elixir
            de_ok = dark_elixir >= self.config.min_dark_elixir

            is_suitable = gold_ok and elixir_ok and de_ok
            if not gold_ok:
                reasons.append(f"Gold ({gold:,}) < {self.config.min_gold:,}")
            if not elixir_ok:
                reasons.append(f"Elixir ({elixir:,}) < {self.config.min_elixir:,}")
            if not de_ok:
                reasons.append(f"DE ({dark_elixir:,}) < {self.config.min_dark_elixir:,}")
            if is_suitable:
                reasons.append("All resource thresholds met")

        elif mode == FilterMode.ANY:
            gold_ok = gold >= self.config.min_gold
            elixir_ok = elixir >= self.config.min_elixir
            de_ok = dark_elixir >= self.config.min_dark_elixir

            is_suitable = gold_ok or elixir_ok or de_ok
            if is_suitable:
                met = []
                if gold_ok:
                    met.append(f"Gold ({gold:,})")
                if elixir_ok:
                    met.append(f"Elixir ({elixir:,})")
                if de_ok:
                    met.append(f"DE ({dark_elixir:,})")
                reasons.append(f"Met thresholds for: {', '.join(met)}")
            else:
                reasons.append("None of the individual thresholds were satisfied")

        elif mode == FilterMode.WEIGHTED:
            is_suitable = weighted_val >= self.config.min_weighted_total
            if is_suitable:
                reasons.append(f"Weighted value {weighted_val:,.0f} >= {self.config.min_weighted_total:,.0f}")
            else:
                reasons.append(f"Weighted value {weighted_val:,.0f} < {self.config.min_weighted_total:,.0f}")

        elif mode == FilterMode.CUSTOM:
            if callable(self.config.custom_predicate):
                try:
                    is_suitable = bool(self.config.custom_predicate(resources, layout))
                    reasons.append("Custom rule passed" if is_suitable else "Custom rule rejected target")
                except Exception as ex:
                    logger.error("Custom filter predicate error: %s", ex)
                    is_suitable = False
                    reasons.append(f"Custom rule evaluation exception: {ex}")
            else:
                is_suitable = gold >= self.config.min_gold
                reasons.append("Custom rule undefined; defaulted to Gold check")

        # 3. Calculate Normalized Farming Score (0.0 - 100.0)
        # Scaled against target benchmark (e.g. 1M gold, 1M elixir, 8k DE)
        norm_gold = min(1.0, gold / max(1, self.config.min_gold * 1.5))
        norm_elixir = min(1.0, elixir / max(1, self.config.min_elixir * 1.5))
        norm_de = min(1.0, dark_elixir / max(1, self.config.min_dark_elixir * 1.5))

        base_score = ((norm_gold + norm_elixir + norm_de) / 3.0) * 100.0

        # Optional layout modifiers (e.g. exposed collectors increase farming score)
        if layout is not None:
            if getattr(layout, "collectors_outside", False) or getattr(layout, "exposed_collector_ratio", 0.0) > 0.4:
                base_score = min(100.0, base_score + 10.0)
            if getattr(layout, "th_accessible", False):
                base_score = min(100.0, base_score + 5.0)

        farming_score = max(0.0, min(100.0, base_score))
        action = "ATTACK" if is_suitable else "SKIP"

        return TargetEvaluationResult(
            action=action,
            is_suitable=is_suitable,
            farming_score=round(farming_score, 2),
            weighted_value=weighted_val,
            ocr_confidence=avg_confidence,
            reasons=reasons,
            resources=resources,
            layout=layout,
        )
