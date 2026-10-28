"""
Synthetic base layout generator for ClashVision.
Constructs canonical tactical base layout topologies:
- EXPOSED_COLLECTORS (vulnerable to BARCH / Sneaky Goblins)
- CORE_STORAGES (well-defended storages deep inside compartments)
- AIR_DEFENSE_CORRIDOR (vulnerable to Mass Dragon)
- COMPARTMENT_RING (suitable for Mass Miner core pathing)
- DEAD_BASE (full collectors, inactive defenses, high loot)
- TOWN_HALL_EXPOSED (Town Hall snipable from outer perimeter)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


@unique
class BaseLayoutType(str, Enum):
    """Canonical base layout topologies."""
    EXPOSED_COLLECTORS = "EXPOSED_COLLECTORS"
    CORE_STORAGES = "CORE_STORAGES"
    AIR_DEFENSE_CORRIDOR = "AIR_DEFENSE_CORRIDOR"
    COMPARTMENT_RING = "COMPARTMENT_RING"
    DEAD_BASE = "DEAD_BASE"
    TOWN_HALL_EXPOSED = "TOWN_HALL_EXPOSED"


@dataclass
class Building:
    """Represents an individual structure on the base grid."""
    id: str
    name: str
    x: float  # Normalized [0.0, 1.0] or pixel coords
    y: float
    is_collector: bool = False
    is_storage: bool = False
    is_defense: bool = False
    is_air_defense: bool = False
    is_town_hall: bool = False
    is_exposed: bool = False
    loot_share: float = 0.0  # Fraction of loot stored here
    hit_points: int = 1000


@dataclass
class BaseLayoutModel:
    """Comprehensive topological and tactical representation of a base layout."""
    layout_type: BaseLayoutType
    buildings: List[Building] = field(default_factory=list)
    wall_coordinates: List[Tuple[float, float]] = field(default_factory=list)
    exposed_collector_ratio: float = 0.0
    th_accessible: bool = False
    storages_in_core: bool = True
    air_defense_locations: List[Tuple[float, float]] = field(default_factory=list)
    total_gold: int = 500_000
    total_elixir: int = 500_000
    total_dark_elixir: int = 3_500
    dead_base: bool = False

    @property
    def collector_count(self) -> int:
        return sum(1 for b in self.buildings if b.is_collector)

    @property
    def exposed_collector_count(self) -> int:
        return sum(1 for b in self.buildings if b.is_collector and b.is_exposed)

    @property
    def storage_count(self) -> int:
        return sum(1 for b in self.buildings if b.is_storage)

    def render_map(self, size: int = 800) -> np.ndarray:
        """Render a 2D top-down tactical representation as a numpy BGR image."""
        img = np.full((size, size, 3), (35, 125, 55), dtype=np.uint8)

        # Draw walls
        for wx, wy in self.wall_coordinates:
            px, py = int(wx * size), int(wy * size)
            cv2.rectangle(img, (px - 3, py - 3), (px + 3, py + 3), (180, 180, 180), -1)

        # Draw buildings
        for b in self.buildings:
            px, py = int(b.x * size), int(b.y * size)
            if b.is_town_hall:
                color = (0, 0, 220)  # Red TH
                rad = 20
            elif b.is_air_defense:
                color = (200, 200, 0)  # Cyan AD
                rad = 14
            elif b.is_defense:
                color = (40, 40, 200)  # Defense
                rad = 12
            elif b.is_collector:
                color = (0, 215, 255) if b.is_exposed else (0, 150, 180)  # Yellow Gold
                rad = 10
            elif b.is_storage:
                color = (220, 50, 220)  # Purple Elixir
                rad = 14
            else:
                color = (120, 120, 120)
                rad = 8

            cv2.circle(img, (px, py), rad, color, -1)
            cv2.circle(img, (px, py), rad, (255, 255, 255), 1)

        return img


class LayoutGenerator:
    """Generates canonical base layouts with known ground-truth tactical attributes."""

    @staticmethod
    def generate_layout(layout_type: BaseLayoutType) -> BaseLayoutModel:
        if layout_type == BaseLayoutType.EXPOSED_COLLECTORS:
            return LayoutGenerator._generate_exposed_collectors()
        elif layout_type == BaseLayoutType.CORE_STORAGES:
            return LayoutGenerator._generate_core_storages()
        elif layout_type == BaseLayoutType.AIR_DEFENSE_CORRIDOR:
            return LayoutGenerator._generate_air_defense_corridor()
        elif layout_type == BaseLayoutType.COMPARTMENT_RING:
            return LayoutGenerator._generate_compartment_ring()
        elif layout_type == BaseLayoutType.DEAD_BASE:
            return LayoutGenerator._generate_dead_base()
        elif layout_type == BaseLayoutType.TOWN_HALL_EXPOSED:
            return LayoutGenerator._generate_town_hall_exposed()
        raise ValueError(f"Unknown layout type: {layout_type}")

    @staticmethod
    def _generate_exposed_collectors() -> BaseLayoutModel:
        """Collectors placed on the outer ring outside walls; storages semi-protected."""
        buildings: List[Building] = []
        walls: List[Tuple[float, float]] = []

        # Outer collectors around perimeter (radius 0.40 from center 0.5, 0.5)
        num_collectors = 14
        for i in range(num_collectors):
            angle = 2 * math.pi * i / num_collectors
            x = 0.5 + 0.38 * math.cos(angle)
            y = 0.5 + 0.38 * math.sin(angle)
            buildings.append(Building(
                id=f"collector_{i}",
                name="Gold Mine / Elixir Collector",
                x=x, y=y,
                is_collector=True,
                is_exposed=True,
                loot_share=0.06,
            ))

        # Inner walls forming core ring (radius 0.25)
        for i in range(36):
            angle = 2 * math.pi * i / 36
            walls.append((0.5 + 0.25 * math.cos(angle), 0.5 + 0.25 * math.sin(angle)))

        # Core buildings: TH, storages, defenses
        buildings.append(Building(
            id="th", name="Town Hall 14", x=0.5, y=0.5, is_town_hall=True, is_exposed=False
        ))
        for i in range(4):
            angle = (math.pi / 2) * i + math.pi / 4
            buildings.append(Building(
                id=f"storage_{i}", name="Storage",
                x=0.5 + 0.15 * math.cos(angle), y=0.5 + 0.15 * math.sin(angle),
                is_storage=True, is_exposed=False,
            ))
            buildings.append(Building(
                id=f"ad_{i}", name="Air Defense",
                x=0.5 + 0.18 * math.cos(angle), y=0.5 + 0.18 * math.sin(angle),
                is_defense=True, is_air_defense=True, is_exposed=False,
            ))

        return BaseLayoutModel(
            layout_type=BaseLayoutType.EXPOSED_COLLECTORS,
            buildings=buildings,
            wall_coordinates=walls,
            exposed_collector_ratio=1.0,  # 100% exposed
            th_accessible=False,
            storages_in_core=True,
            air_defense_locations=[(b.x, b.y) for b in buildings if b.is_air_defense],
            total_gold=820_000,
            total_elixir=790_000,
            total_dark_elixir=6_200,
            dead_base=False,
        )

    @staticmethod
    def _generate_core_storages() -> BaseLayoutModel:
        """Storages completely surrounded by multi-layer walls in deep core."""
        buildings: List[Building] = []
        walls: List[Tuple[float, float]] = []

        # Inner and outer wall layers
        for r in [0.20, 0.32]:
            for i in range(30):
                angle = 2 * math.pi * i / 30
                walls.append((0.5 + r * math.cos(angle), 0.5 + r * math.sin(angle)))

        # Town Hall & Storages inside inner ring
        buildings.append(Building(
            id="th", name="Town Hall", x=0.5, y=0.5, is_town_hall=True, is_exposed=False
        ))
        for i in range(6):
            angle = 2 * math.pi * i / 6
            buildings.append(Building(
                id=f"core_storage_{i}", name="Resource Storage",
                x=0.5 + 0.12 * math.cos(angle), y=0.5 + 0.12 * math.sin(angle),
                is_storage=True, is_exposed=False, loot_share=0.15
            ))

        # Collectors protected between inner and outer ring
        for i in range(10):
            angle = 2 * math.pi * i / 10
            buildings.append(Building(
                id=f"inner_collector_{i}", name="Protected Collector",
                x=0.5 + 0.26 * math.cos(angle), y=0.5 + 0.26 * math.sin(angle),
                is_collector=True, is_exposed=False, loot_share=0.01
            ))

        return BaseLayoutModel(
            layout_type=BaseLayoutType.CORE_STORAGES,
            buildings=buildings,
            wall_coordinates=walls,
            exposed_collector_ratio=0.0,  # 0% exposed
            th_accessible=False,
            storages_in_core=True,
            air_defense_locations=[],
            total_gold=650_000,
            total_elixir=620_000,
            total_dark_elixir=5_000,
            dead_base=False,
        )

    @staticmethod
    def _generate_air_defense_corridor() -> BaseLayoutModel:
        """Air defenses clustered on south side, leaving northern air corridor open."""
        buildings: List[Building] = []
        walls: List[Tuple[float, float]] = []

        # Air Defenses all on south quadrant (y > 0.6)
        ad_coords = [(0.35, 0.65), (0.45, 0.70), (0.55, 0.70), (0.65, 0.65)]
        for idx, (ax, ay) in enumerate(ad_coords):
            buildings.append(Building(
                id=f"ad_{idx}", name="Air Defense",
                x=ax, y=ay, is_defense=True, is_air_defense=True, is_exposed=False
            ))

        # Town Hall in center
        buildings.append(Building(id="th", name="Town Hall", x=0.5, y=0.5, is_town_hall=True))

        # Collectors and storages distributed
        for i in range(8):
            angle = 2 * math.pi * i / 8
            buildings.append(Building(
                id=f"col_{i}", name="Collector",
                x=0.5 + 0.3 * math.cos(angle), y=0.5 + 0.3 * math.sin(angle),
                is_collector=True, is_exposed=(i % 2 == 0)
            ))

        return BaseLayoutModel(
            layout_type=BaseLayoutType.AIR_DEFENSE_CORRIDOR,
            buildings=buildings,
            wall_coordinates=walls,
            exposed_collector_ratio=0.5,
            th_accessible=False,
            storages_in_core=True,
            air_defense_locations=ad_coords,
            total_gold=700_000,
            total_elixir=750_000,
            total_dark_elixir=4_000,
            dead_base=False,
        )

    @staticmethod
    def _generate_compartment_ring() -> BaseLayoutModel:
        """Grid of multiple tight compartments, ideal for Mass Miner miner underground pathing."""
        buildings: List[Building] = []
        walls: List[Tuple[float, float]] = []

        # Grid of walls
        for x_line in [0.35, 0.50, 0.65]:
            for y_step in np.linspace(0.25, 0.75, 15):
                walls.append((x_line, float(y_step)))
        for y_line in [0.35, 0.50, 0.65]:
            for x_step in np.linspace(0.25, 0.75, 15):
                walls.append((float(x_step), y_line))

        buildings.append(Building(id="th", name="Town Hall", x=0.5, y=0.5, is_town_hall=True))
        for i, (bx, by) in enumerate([(0.42, 0.42), (0.58, 0.42), (0.42, 0.58), (0.58, 0.58)]):
            buildings.append(Building(id=f"storage_{i}", name="Storage", x=bx, y=by, is_storage=True))

        return BaseLayoutModel(
            layout_type=BaseLayoutType.COMPARTMENT_RING,
            buildings=buildings,
            wall_coordinates=walls,
            exposed_collector_ratio=0.2,
            th_accessible=False,
            storages_in_core=True,
            air_defense_locations=[(0.42, 0.42), (0.58, 0.58)],
            total_gold=800_000,
            total_elixir=800_000,
            total_dark_elixir=6_000,
            dead_base=False,
        )

    @staticmethod
    def _generate_dead_base() -> BaseLayoutModel:
        """Dead base: collectors 100% full on perimeter, empty core storages, inactive defenses."""
        base = LayoutGenerator._generate_exposed_collectors()
        base.layout_type = BaseLayoutType.DEAD_BASE
        base.dead_base = True
        base.total_gold = 1_100_000
        base.total_elixir = 1_250_000
        base.total_dark_elixir = 8_500
        # Overwrite loot shares: 90% of all loot is in the exposed collectors
        for b in base.buildings:
            if b.is_collector:
                b.loot_share = 0.90 / max(1, base.collector_count)
            elif b.is_storage:
                b.loot_share = 0.10 / max(1, base.storage_count)
        return base

    @staticmethod
    def _generate_town_hall_exposed() -> BaseLayoutModel:
        """Town Hall placed outside perimeter wall, 100% snipable."""
        buildings: List[Building] = []
        walls: List[Tuple[float, float]] = []

        # Central base compartment
        for i in range(24):
            angle = 2 * math.pi * i / 24
            walls.append((0.55 + 0.25 * math.cos(angle), 0.5 + 0.25 * math.sin(angle)))

        # Town Hall isolated on far left edge (x=0.15, outside walls)
        buildings.append(Building(
            id="th", name="Town Hall", x=0.15, y=0.5, is_town_hall=True, is_exposed=True
        ))

        # Storages protected inside
        for i in range(4):
            angle = (math.pi / 2) * i
            buildings.append(Building(
                id=f"storage_{i}", name="Storage",
                x=0.55 + 0.15 * math.cos(angle), y=0.5 + 0.15 * math.sin(angle),
                is_storage=True, is_exposed=False
            ))

        return BaseLayoutModel(
            layout_type=BaseLayoutType.TOWN_HALL_EXPOSED,
            buildings=buildings,
            wall_coordinates=walls,
            exposed_collector_ratio=0.5,
            th_accessible=True,  # 100% accessible snipe
            storages_in_core=True,
            air_defense_locations=[(0.55, 0.4), (0.55, 0.6)],
            total_gold=550_000,
            total_elixir=500_000,
            total_dark_elixir=3_500,
            dead_base=False,
        )


def generate_canonical_layout(layout_type: Union[str, BaseLayoutType]) -> BaseLayoutModel:
    """Convenience factory function."""
    if isinstance(layout_type, str):
        layout_type = BaseLayoutType(layout_type)
    return LayoutGenerator.generate_layout(layout_type)
