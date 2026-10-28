"""
ClashVision Farming Decision Subsystem — Base Strategy Abstract Interface.
Provides common contract, army composition representations, spatial heuristics,
and deployment sequencing utilities for all attack strategies.
Derived from Requirement R3 and Feature 15-20.
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    Point,
    NormalizedPoint,
    ResourceReading,
    StrategyEvaluation,
)
from clashvision.core.config import ResourceThresholds
from clashvision.core.coordinates import (
    clamp_point_to_bounds,
    normalized_to_point,
    point_to_normalized,
)


class BaseStrategy(ABC):
    """
    Abstract Base Class for all tactical attack strategies.
    Every strategy encapsulates:
    - Army composition and resource economics
    - Heuristic evaluation of base viability, layout suitability, and risk
    - Procedural generation of step-by-step deployment sequences
    """

    @property
    @abstractmethod
    def strategy_type(self) -> StrategyType:
        """Enum strategy identifier."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Detailed description of the tactical approach."""
        raise NotImplementedError

    @property
    @abstractmethod
    def army_composition(self) -> Dict[str, int]:
        """Dictionary mapping troop/spell names to required unit counts."""
        raise NotImplementedError

    @property
    def estimated_training_time_seconds(self) -> int:
        """Estimated full army training time in seconds."""
        return 1200

    @property
    def estimated_elixir_cost(self) -> int:
        """Estimated resource cost to train the army."""
        return 150_000

    @property
    def estimated_dark_elixir_cost(self) -> int:
        """Estimated dark elixir cost to train the army."""
        return 0

    @abstractmethod
    def evaluate(
        self,
        resources: Optional[ResourceReading],
        layout: Optional[BaseLayoutFeatures],
        thresholds: Optional[ResourceThresholds] = None,
    ) -> StrategyEvaluation:
        """
        Evaluates the viability, resource suitability, layout suitability,
        and overall fitness score of this strategy for the given base.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_deployment_plan(
        self,
        layout: Optional[BaseLayoutFeatures],
        canvas_width: int = VIRTUAL_REFERENCE_WIDTH,
        canvas_height: int = VIRTUAL_REFERENCE_HEIGHT,
    ) -> List[DeploymentStep]:
        """
        Procedurally generates a timed, step-by-step deployment sequence
        targeting strategic vulnerabilities of the base layout.
        """
        raise NotImplementedError

    # =========================================================================
    # Shared Tactical Geometry & Coordinate Heuristics
    # =========================================================================

    @staticmethod
    def calculate_perimeter_point(
        angle_rad: float,
        radius_fraction: float = 0.42,
        center_norm: Tuple[float, float] = (0.5, 0.5),
        canvas_width: int = VIRTUAL_REFERENCE_WIDTH,
        canvas_height: int = VIRTUAL_REFERENCE_HEIGHT,
    ) -> Tuple[int, int, float, float]:
        """
        Calculates a deployment drop point along the outer perimeter at a specified angle.
        Returns (pixel_x, pixel_y, normalized_x, normalized_y).
        """
        cx, cy = center_norm
        cw = max(1, canvas_width)
        ch = max(1, canvas_height)
        # In a 16:9 canvas, adjust x radius to compensate aspect ratio
        aspect = cw / ch
        rx = radius_fraction / (aspect / 1.7778)
        ry = radius_fraction

        nx = cx + rx * math.cos(angle_rad)
        ny = cy + ry * math.sin(angle_rad)

        nx = max(0.02, min(0.98, nx))
        ny = max(0.02, min(0.98, ny))

        px = int(round(nx * cw))
        py = int(round(ny * ch))

        return (px, py, round(nx, 4), round(ny, 4))

    @staticmethod
    def get_cardinal_perimeter_points(
        count: int = 8,
        radius_fraction: float = 0.42,
        canvas_width: int = VIRTUAL_REFERENCE_WIDTH,
        canvas_height: int = VIRTUAL_REFERENCE_HEIGHT,
    ) -> List[Tuple[int, int, float, float]]:
        """
        Generates equidistant perimeter deployment coordinates circling the base.
        """
        points: List[Tuple[int, int, float, float]] = []
        for i in range(count):
            angle = (2.0 * math.pi * i) / count
            pt = BaseStrategy.calculate_perimeter_point(
                angle, radius_fraction, canvas_width=canvas_width, canvas_height=canvas_height
            )
            points.append(pt)
        return points

    @staticmethod
    def find_safest_corridor_angle(
        hazard_points: List[Tuple[float, float]],
        center_norm: Tuple[float, float] = (0.5, 0.5),
        num_sectors: int = 16,
    ) -> float:
        """
        Finds the angular corridor around the center with the lowest density of hazards
        (e.g., Air Defenses, splash clusters).
        Returns angle in radians [0.0, 2*pi).
        """
        if not hazard_points:
            # Default to North attack corridor
            return -math.pi / 2.0

        cx, cy = center_norm
        sector_width = (2.0 * math.pi) / num_sectors
        sector_threats = [0.0] * num_sectors

        # Precompute hazard angles and proximity weights
        hazards_processed: List[Tuple[float, float]] = []
        for hx, hy in hazard_points:
            dx = hx - cx
            dy = hy - cy
            dist = math.sqrt(dx * dx + dy * dy)
            h_angle = math.atan2(dy, dx)
            if h_angle < 0:
                h_angle += 2.0 * math.pi
            weight = 1.0 / max(0.05, dist)
            hazards_processed.append((h_angle, weight))

        # Evaluate threat at each sector center
        for s in range(num_sectors):
            s_angle = (s + 0.5) * sector_width
            total_threat = 0.0
            for h_angle, weight in hazards_processed:
                diff = abs(s_angle - h_angle)
                if diff > math.pi:
                    diff = 2.0 * math.pi - diff
                # 1 + cos(diff) is 2.0 at diff=0, 0.0 at diff=pi
                total_threat += weight * (1.0 + math.cos(diff))
            sector_threats[s] = total_threat

        best_sector = sector_threats.index(min(sector_threats))
        return (best_sector + 0.5) * sector_width
