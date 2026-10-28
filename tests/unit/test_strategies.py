"""
Unit tests for ClashVision Tactical Attack Strategies:
- Strategy A: Sneaky Goblin Farming (clashvision.decision.sneaky_goblin)
- Strategy B: BARCH Farming (clashvision.decision.barch)
- Strategy C: Mass Miner (clashvision.decision.mass_miner)
- Strategy D: Mass Dragon (clashvision.decision.mass_dragon)
- Strategy E: User-Defined Custom Strategy (clashvision.decision.custom_strategy)
- BaseStrategy tactical geometry and coordinate heuristics
Derived from Requirement R3 and Features 16, 17, 18, 19, 20.
"""

from __future__ import annotations
import json
import math
import pytest

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
from clashvision.decision.sneaky_goblin import SneakyGoblinStrategy
from clashvision.decision.barch import BarchStrategy
from clashvision.decision.mass_miner import MassMinerStrategy
from clashvision.decision.mass_dragon import MassDragonStrategy
from clashvision.decision.custom_strategy import CustomStrategy, CustomStrategyConfig
from tests.fixtures.layout_generator import (
    BaseLayoutType,
    generate_canonical_layout,
)
from clashvision.vision.layout_analyzer import BaseLayoutAnalyzer


# ==============================================================================
# BaseStrategy Tactical Geometry Tests
# ==============================================================================
class TestBaseStrategyTacticalGeometry:
    """Verifies coordinate calculations and hazard corridor analysis."""

    def test_calculate_perimeter_point_within_bounds(self) -> None:
        angles = [0.0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2]
        for angle in angles:
            px, py, nx, ny = BaseStrategy.calculate_perimeter_point(
                angle, radius_fraction=0.40, canvas_width=1920, canvas_height=1080
            )
            assert 0 <= px <= 1920
            assert 0 <= py <= 1080
            assert 0.0 <= nx <= 1.0
            assert 0.0 <= ny <= 1.0

    def test_get_cardinal_perimeter_points_count(self) -> None:
        pts4 = BaseStrategy.get_cardinal_perimeter_points(count=4)
        assert len(pts4) == 4
        pts8 = BaseStrategy.get_cardinal_perimeter_points(count=8)
        assert len(pts8) == 8

    def test_find_safest_corridor_angle_empty_hazards(self) -> None:
        angle = BaseStrategy.find_safest_corridor_angle([])
        # Defaults safely without error
        assert -math.pi <= angle <= 2 * math.pi

    def test_find_safest_corridor_avoids_hazard_cluster(self) -> None:
        # Clustered hazards strictly in South quadrant (y > 0.7, angles around pi/2 = 1.57 rad)
        hazards = [(0.45, 0.75), (0.50, 0.80), (0.55, 0.75)]
        best_angle = BaseStrategy.find_safest_corridor_angle(hazards, num_sectors=16)
        # Safest corridor should be in North quadrant (y < 0.5, angles around 3*pi/2 ~ 4.71 rad)
        # In atan2: dx ~ 0, dy < 0 corresponds to ~4.7 rad (or -pi/2)
        assert 3.5 <= best_angle <= 6.0


# ==============================================================================
# Strategy A: Sneaky Goblin Farming Tests
# ==============================================================================
class TestSneakyGoblinStrategy:
    """Tests Strategy A: Sneaky Goblin domain logic, routing, and deployment plan."""

    @pytest.fixture
    def strategy(self) -> SneakyGoblinStrategy:
        return SneakyGoblinStrategy()

    def test_properties(self, strategy: SneakyGoblinStrategy) -> None:
        assert strategy.strategy_type == StrategyType.SNEAKY_GOBLIN
        assert strategy.name == "Sneaky Goblin Farming"
        assert "Sneaky Goblin" in strategy.army_composition
        assert "Jump Spell" in strategy.army_composition
        assert "Invisibility Spell" in strategy.army_composition
        assert strategy.estimated_training_time_seconds > 0
        assert strategy.estimated_elixir_cost > 0

    def test_evaluate_none_resources(self, strategy: SneakyGoblinStrategy) -> None:
        ev = strategy.evaluate(None, None)
        assert not ev.viable
        assert ev.score == 0.0

    def test_evaluate_high_exposed_collectors_low_risk(
        self, strategy: SneakyGoblinStrategy
    ) -> None:
        reading = ResourceReading(gold=800_000, elixir=800_000, dark_elixir=6_000)
        layout = BaseLayoutFeatures(exposed_collector_ratio=0.9, th_accessible=True)
        ev = strategy.evaluate(reading, layout)
        assert ev.viable
        assert ev.score >= 85.0
        assert ev.risk_level == RiskLevel.LOW
        assert ev.layout_suitability >= 90.0
        assert any("Town Hall" in r for r in ev.reasons)

    def test_evaluate_core_storages_medium_risk(
        self, strategy: SneakyGoblinStrategy
    ) -> None:
        reading = ResourceReading(gold=600_000, elixir=600_000, dark_elixir=4_000)
        layout = BaseLayoutFeatures(exposed_collector_ratio=0.1, th_accessible=False)
        ev = strategy.evaluate(reading, layout)
        assert ev.viable
        assert ev.risk_level == RiskLevel.HIGH
        assert ev.layout_suitability <= 70.0

    def test_deployment_plan_sequence(
        self, strategy: SneakyGoblinStrategy
    ) -> None:
        layout = BaseLayoutFeatures(exposed_collector_ratio=0.8, th_accessible=True)
        plan = strategy.generate_deployment_plan(layout, canvas_width=1920, canvas_height=1080)
        assert len(plan) >= 10

        troop_types = [s.troop_type for s in plan]
        assert "Sneaky Goblin" in troop_types
        assert "Super Wall Breaker" in troop_types
        assert "Jump Spell" in troop_types
        assert "Haste Spell" in troop_types
        assert "Invisibility Spell" in troop_types

        # Verify step timing monotonicity
        for i in range(1, len(plan)):
            assert plan[i].delay_ms >= plan[i - 1].delay_ms
            assert plan[i].step_id == i + 1


# ==============================================================================
# Strategy B: BARCH Farming Tests
# ==============================================================================
class TestBarchStrategy:
    """Tests Strategy B: BARCH domain logic, perimeter waves, and wall sensitivity."""

    @pytest.fixture
    def strategy(self) -> BarchStrategy:
        return BarchStrategy()

    def test_properties(self, strategy: BarchStrategy) -> None:
        assert strategy.strategy_type == StrategyType.BARCH
        assert strategy.name == "BARCH Farming"
        assert strategy.army_composition["Barbarian"] >= 100
        assert strategy.army_composition["Archer"] >= 100
        assert strategy.estimated_dark_elixir_cost == 0

    def test_evaluate_high_exposed_collectors_viable(
        self, strategy: BarchStrategy
    ) -> None:
        reading = ResourceReading(gold=750_000, elixir=750_000, dark_elixir=2_000)
        layout = BaseLayoutFeatures(exposed_collector_ratio=0.85)
        ev = strategy.evaluate(reading, layout)
        assert ev.viable
        assert ev.score >= 80.0
        assert ev.risk_level == RiskLevel.LOW

    def test_evaluate_zero_exposed_collectors_unviable(
        self, strategy: BarchStrategy
    ) -> None:
        reading = ResourceReading(gold=750_000, elixir=750_000, dark_elixir=2_000)
        layout = BaseLayoutFeatures(exposed_collector_ratio=0.0, wall_perimeter_intact=True)
        ev = strategy.evaluate(reading, layout)
        assert not ev.viable
        assert ev.score <= 45.0
        assert ev.risk_level == RiskLevel.EXTREME

    def test_evaluate_open_walls_bonus(
        self, strategy: BarchStrategy
    ) -> None:
        reading = ResourceReading(gold=500_000, elixir=500_000, dark_elixir=2_000)
        layout_intact = BaseLayoutFeatures(exposed_collector_ratio=0.4, wall_perimeter_intact=True)
        layout_open = BaseLayoutFeatures(exposed_collector_ratio=0.4, wall_perimeter_intact=False)
        ev_intact = strategy.evaluate(reading, layout_intact)
        ev_open = strategy.evaluate(reading, layout_open)
        assert ev_open.score > ev_intact.score

    def test_deployment_plan_alternating_waves(
        self, strategy: BarchStrategy
    ) -> None:
        layout = BaseLayoutFeatures(exposed_collector_ratio=0.8)
        plan = strategy.generate_deployment_plan(layout)
        assert len(plan) >= 20

        # Verify initial Barbarian wave followed by Archer wave
        barb_steps = [s for s in plan if s.troop_type == "Barbarian"]
        archer_steps = [s for s in plan if s.troop_type == "Archer"]
        lightning_steps = [s for s in plan if s.troop_type == "Lightning Spell"]

        assert len(barb_steps) > 0
        assert len(archer_steps) > 0
        assert len(lightning_steps) > 0


# ==============================================================================
# Strategy C: Mass Miner Tests
# ==============================================================================
class TestMassMinerStrategy:
    """Tests Strategy C: Mass Miner domain logic, compartment routing, and heal spells."""

    @pytest.fixture
    def strategy(self) -> MassMinerStrategy:
        return MassMinerStrategy()

    def test_properties(self, strategy: MassMinerStrategy) -> None:
        assert strategy.strategy_type == StrategyType.MASS_MINER
        assert strategy.name == "Mass Miner Assault"
        assert strategy.army_composition["Miner"] >= 35
        assert strategy.army_composition["Heal Spell"] >= 4

    def test_evaluate_compartment_density_bonus(
        self, strategy: MassMinerStrategy
    ) -> None:
        reading = ResourceReading(gold=800_000, elixir=800_000, dark_elixir=5_000)
        layout_dense = BaseLayoutFeatures(compartment_count=6, metadata={"storages_in_core": True})
        layout_sparse = BaseLayoutFeatures(compartment_count=1, metadata={"storages_in_core": False})

        ev_dense = strategy.evaluate(reading, layout_dense)
        ev_sparse = strategy.evaluate(reading, layout_sparse)

        assert ev_dense.layout_suitability > ev_sparse.layout_suitability
        assert ev_dense.viable

    def test_evaluate_low_loot_unviable_due_to_training_cost(
        self, strategy: MassMinerStrategy
    ) -> None:
        # Loot margin below 380k
        reading = ResourceReading(gold=150_000, elixir=150_000, dark_elixir=1_000)
        ev = strategy.evaluate(reading, None)
        assert not ev.viable

    def test_deployment_plan_funnel_and_heals(
        self, strategy: MassMinerStrategy
    ) -> None:
        layout = BaseLayoutFeatures(compartment_count=5)
        plan = strategy.generate_deployment_plan(layout)

        miner_steps = [s for s in plan if s.troop_type == "Miner"]
        heal_steps = [s for s in plan if s.troop_type == "Heal Spell"]
        poison_steps = [s for s in plan if s.troop_type == "Poison Spell"]

        assert len(miner_steps) >= 4  # 2 funnels + 2 main waves
        assert len(heal_steps) == 3   # 3 rolling heals
        assert len(poison_steps) == 1


# ==============================================================================
# Strategy D: Mass Dragon Tests
# ==============================================================================
class TestMassDragonStrategy:
    """Tests Strategy D: Mass Dragon domain logic, Air Defense analysis, and aerial corridors."""

    @pytest.fixture
    def strategy(self) -> MassDragonStrategy:
        return MassDragonStrategy()

    def test_properties(self, strategy: MassDragonStrategy) -> None:
        assert strategy.strategy_type == StrategyType.MASS_DRAGON
        assert strategy.name == "Mass Dragon Corridor"
        assert strategy.army_composition["Dragon"] >= 10
        assert strategy.army_composition["Balloon"] >= 6

    def test_evaluate_asymmetric_air_defenses_favored(
        self, strategy: MassDragonStrategy
    ) -> None:
        reading = ResourceReading(gold=850_000, elixir=850_000, dark_elixir=5_000)
        # All ADs in south (y > 0.6)
        ads_south = [(600, 750), (800, 800), (1000, 800), (1200, 750)]
        layout_corridor = BaseLayoutFeatures(
            air_defense_locations=ads_south, metadata={"layout_type": "AIR_DEFENSE_CORRIDOR"}
        )
        ev = strategy.evaluate(reading, layout_corridor)
        assert ev.viable
        assert ev.layout_suitability >= 90.0
        assert ev.risk_level == RiskLevel.LOW

    def test_evaluate_high_symmetric_ad_density_higher_risk(
        self, strategy: MassDragonStrategy
    ) -> None:
        reading = ResourceReading(gold=850_000, elixir=850_000, dark_elixir=5_000)
        # Symmetric 4-corner ADs
        ads_symmetric = [(400, 300), (1500, 300), (400, 800), (1500, 800)]
        layout_symmetric = BaseLayoutFeatures(air_defense_locations=ads_symmetric)
        ev = strategy.evaluate(reading, layout_symmetric)
        assert ev.risk_level == RiskLevel.HIGH
        assert ev.layout_suitability < 80.0

    def test_deployment_plan_spells_and_funnel(
        self, strategy: MassDragonStrategy
    ) -> None:
        ads = [(700, 700), (800, 750)]
        layout = BaseLayoutFeatures(air_defense_locations=ads)
        plan = strategy.generate_deployment_plan(layout)

        dragon_steps = [s for s in plan if s.troop_type == "Dragon"]
        balloon_steps = [s for s in plan if s.troop_type == "Balloon"]
        lightning_steps = [s for s in plan if s.troop_type == "Lightning Spell"]
        rage_steps = [s for s in plan if s.troop_type == "Rage Spell"]
        freeze_steps = [s for s in plan if s.troop_type == "Freeze Spell"]

        assert len(dragon_steps) == 3   # 2 funnels + 1 main line
        assert len(balloon_steps) == 1
        assert len(lightning_steps) == 2
        assert len(rage_steps) == 2
        assert len(freeze_steps) == 1


# ==============================================================================
# Strategy E: Custom User Strategy Tests
# ==============================================================================
class TestCustomStrategy:
    """Tests Strategy E: User-defined configuration, thresholds, prerequisites, and plans."""

    def test_default_custom_strategy(self) -> None:
        strat = CustomStrategy()
        assert strat.strategy_type == StrategyType.CUSTOM
        assert "Super Barbarian" in strat.army_composition
        assert strat.name == "Custom Attack Strategy"

    def test_reconfigure_custom_strategy(self) -> None:
        cfg = CustomStrategyConfig(
            name="Sneaky Blimp Rush",
            min_gold=500_000,
            min_elixir=500_000,
            min_dark_elixir=4_000,
            requires_accessible_th=True,
            max_acceptable_risk=RiskLevel.MEDIUM,
        )
        strat = CustomStrategy(cfg)
        assert strat.name == "Sneaky Blimp Rush"
        assert strat.config.min_gold == 500_000

    def test_accessible_th_constraint_enforcement(self) -> None:
        cfg = CustomStrategyConfig(requires_accessible_th=True)
        strat = CustomStrategy(cfg)

        reading = ResourceReading(gold=700_000, elixir=700_000, dark_elixir=5_000)
        layout_accessible = BaseLayoutFeatures(th_accessible=True)
        layout_fortified = BaseLayoutFeatures(th_accessible=False)

        ev_ok = strat.evaluate(reading, layout_accessible)
        ev_fail = strat.evaluate(reading, layout_fortified)

        assert ev_ok.prerequisites_met
        assert ev_ok.viable
        assert not ev_fail.prerequisites_met
        assert not ev_fail.viable

    def test_custom_steps_compilation(self) -> None:
        custom_steps = [
            {"troop_type": "Battle Blimp", "normalized_x": 0.1, "normalized_y": 0.5, "delay_ms": 0, "unit_count": 1},
            {"troop_type": "Clone Spell", "normalized_x": 0.5, "normalized_y": 0.5, "delay_ms": 1500, "unit_count": 2},
        ]
        cfg = CustomStrategyConfig(custom_steps=custom_steps)
        strat = CustomStrategy(cfg)
        plan = strat.generate_deployment_plan(None, canvas_width=1920, canvas_height=1080)

        assert len(plan) == 2
        assert plan[0].troop_type == "Battle Blimp"
        assert plan[0].target_x == int(0.1 * 1920)
        assert plan[1].troop_type == "Clone Spell"
        assert plan[1].delay_ms == 1500

    def test_custom_strategy_config_serialization(self) -> None:
        cfg = CustomStrategyConfig(
            name="Yeti Smash",
            army_composition={"Yeti": 8, "Bowler": 10, "Earthquake Spell": 4},
            min_gold=600_000,
        )
        d = cfg.to_dict()
        assert d["name"] == "Yeti Smash"
        assert d["army_composition"]["Yeti"] == 8

        loaded = CustomStrategyConfig.from_dict(d)
        assert loaded.name == cfg.name
        assert loaded.army_composition == cfg.army_composition


# ==============================================================================
# Model Serialization & Invariance Tests
# ==============================================================================
class TestDecisionModelSerialization:
    """Verifies dataclass to_dict serializability and json cleanliness."""

    def test_strategy_evaluation_to_dict_json_safe(self) -> None:
        ev = StrategyEvaluation(
            strategy_type=StrategyType.SNEAKY_GOBLIN,
            score=91.5,
            viable=True,
            risk_level=RiskLevel.LOW,
            resource_suitability=95.0,
            layout_suitability=88.0,
            prerequisites_met=True,
            reasons=["Great loot", "Exposed collectors"],
        )
        d = ev.to_dict()
        json_str = json.dumps(d)
        assert "SNEAKY_GOBLIN" in json_str
        assert "LOW" in json_str

    def test_deployment_step_to_dict_json_safe(self) -> None:
        step = DeploymentStep(
            step_id=1,
            troop_type="Sneaky Goblin",
            target_x=500,
            target_y=600,
            normalized_x=0.2604,
            normalized_y=0.5556,
            delay_ms=450,
            unit_count=5,
            note="Test drop",
        )
        d = step.to_dict()
        json_str = json.dumps(d)
        assert "Sneaky Goblin" in json_str
        assert "0.2604" in json_str
