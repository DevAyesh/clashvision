"""
ClashVision Farming Decision Subsystem.
Implements base resource evaluation, modular attack strategy frameworks,
and multi-strategy tactical recommendations.
"""

from clashvision.decision.evaluator import (
    BaseEvaluator,
    BaseEvaluationResult,
)
from clashvision.decision.base_strategy import (
    BaseStrategy,
)
from clashvision.decision.sneaky_goblin import (
    SneakyGoblinStrategy,
)
from clashvision.decision.barch import (
    BarchStrategy,
)
from clashvision.decision.mass_miner import (
    MassMinerStrategy,
)
from clashvision.decision.mass_dragon import (
    MassDragonStrategy,
)
from clashvision.decision.custom_strategy import (
    CustomStrategy,
    CustomStrategyConfig,
)
from clashvision.decision.recommendation_engine import (
    RecommendationEngine,
)

__all__ = [
    "BaseEvaluator",
    "BaseEvaluationResult",
    "BaseStrategy",
    "SneakyGoblinStrategy",
    "BarchStrategy",
    "MassMinerStrategy",
    "MassDragonStrategy",
    "CustomStrategy",
    "CustomStrategyConfig",
    "RecommendationEngine",
]
