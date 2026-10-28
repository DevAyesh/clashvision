"""
ClashVision Presentation Subsystem — View 4: Strategy Planner.
Displays multi-strategy comparative rankings:
- Side-by-side comparison table of all 5 strategies (Sneaky Goblin, BARCH, Miner, Dragon, Custom)
- Scoring, viability badges, risk assessments, and decision rationale
- Troop requirements breakdown
- Step-by-step deployment timeline
Derived from Requirement R5 and Feature 32.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from clashvision.core.constants import StrategyType, RiskLevel
from clashvision.core.models import (
    FarmingRecommendation,
    StrategyEvaluation,
    DeploymentStep,
)
from clashvision.ui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, Qt
)
from clashvision.ui.styles import (
    COLOR_NEON_GREEN, COLOR_NEON_GOLD, COLOR_NEON_RED,
    COLOR_BORDER
)


class StrategyView(QWidget):
    """
    Multi-strategy comparison and tactical deployment plan viewer.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("strategy_view")
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        # Header Title
        title_label = QLabel("Multi-Strategy Tactical Planner")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #f1f2f6;")
        main_layout.addWidget(title_label)

        # 1. Strategy Comparison Table
        table_title = QLabel("CANDIDATE STRATEGY EVALUATION & RANKING")
        table_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #a4b0be;")
        main_layout.addWidget(table_title)

        self.strategy_table = QTableWidget(5, 5)
        self.strategy_table.setHorizontalHeaderLabels(
            ["Strategy", "Suitability Score", "Viability", "Risk Level", "Rationale Summary"]
        )
        self.strategy_table.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(self.strategy_table, stretch=2)

        # Populate default empty strategy rows
        for row, strat in enumerate([
            StrategyType.SNEAKY_GOBLIN,
            StrategyType.BARCH,
            StrategyType.MASS_MINER,
            StrategyType.MASS_DRAGON,
            StrategyType.CUSTOM,
        ]):
            self.strategy_table.setItem(row, 0, QTableWidgetItem(strat.display_name))
            self.strategy_table.setItem(row, 1, QTableWidgetItem("--"))
            self.strategy_table.setItem(row, 2, QTableWidgetItem("PENDING"))
            self.strategy_table.setItem(row, 3, QTableWidgetItem("--"))
            self.strategy_table.setItem(row, 4, QTableWidgetItem("Awaiting target base analysis."))

        # 2. Lower Split: Troop Requirements & Deployment Steps
        lower_layout = QHBoxLayout()
        lower_layout.setSpacing(14)

        # Troop Composition Card
        troops_card = QFrame()
        troops_card.setStyleSheet(
            f"background-color: #1a1c23; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 12px;"
        )
        troops_layout = QVBoxLayout(troops_card)
        troops_title = QLabel("ESTIMATED ARMY REQUIREMENTS")
        troops_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #a4b0be;")
        troops_layout.addWidget(troops_title)

        self.troops_summary_label = QLabel("Primary Strategy: Not selected\nEstimated Training Cost: --")
        self.troops_summary_label.setStyleSheet("color: #f1f2f6; font-size: 12px; line-height: 1.5;")
        troops_layout.addWidget(self.troops_summary_label)
        troops_layout.addStretch()
        lower_layout.addWidget(troops_card, stretch=1)

        # Deployment Steps Table
        steps_card = QFrame()
        steps_card.setStyleSheet(
            f"background-color: #1a1c23; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 12px;"
        )
        steps_layout = QVBoxLayout(steps_card)
        steps_title = QLabel("STEP-BY-STEP DEPLOYMENT PLAN")
        steps_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #a4b0be;")
        steps_layout.addWidget(steps_title)

        self.steps_table = QTableWidget(0, 5)
        self.steps_table.setHorizontalHeaderLabels(["Step #", "Troop Type", "Target (X, Y)", "Delay (ms)", "Note"])
        self.steps_table.horizontalHeader().setStretchLastSection(True)
        steps_layout.addWidget(self.steps_table)
        lower_layout.addWidget(steps_card, stretch=2)

        main_layout.addLayout(lower_layout, stretch=2)

    def update_recommendation(self, rec: FarmingRecommendation) -> None:
        """Populate strategy comparisons and deployment steps from recommendation."""
        # Update strategy evaluation table
        rankings = rec.rankings or []
        self.strategy_table.setRowCount(len(rankings))

        for row, item in enumerate(rankings):
            strat_name = item.strategy_type.display_name if hasattr(item.strategy_type, "display_name") else str(item.strategy_type)
            self.strategy_table.setItem(row, 0, QTableWidgetItem(strat_name))
            self.strategy_table.setItem(row, 1, QTableWidgetItem(f"{item.score:.1f}"))

            viab_str = "VIABLE" if item.viable else "UNVIABLE"
            self.strategy_table.setItem(row, 2, QTableWidgetItem(viab_str))

            risk_str = item.risk_level.value if hasattr(item.risk_level, "value") else str(item.risk_level)
            self.strategy_table.setItem(row, 3, QTableWidgetItem(risk_str))

            reasons_text = "; ".join(item.reasons) if item.reasons else "Calculated heuristic baseline."
            self.strategy_table.setItem(row, 4, QTableWidgetItem(reasons_text))

        # Update primary strategy details
        primary_name = rec.primary_strategy.display_name if hasattr(rec.primary_strategy, "display_name") else str(rec.primary_strategy)
        self.troops_summary_label.setText(
            f"Selected Strategy: {primary_name}\n"
            f"Suitability Score: {rec.suitability_score:.1f} / 100\n"
            f"Tactical Rationale: {rec.explanation}"
        )

        # Update deployment steps table
        steps = rec.deployment_plan or []
        self.steps_table.setRowCount(len(steps))
        for row, step in enumerate(steps):
            self.steps_table.setItem(row, 0, QTableWidgetItem(str(step.step_id or (row + 1))))
            self.steps_table.setItem(row, 1, QTableWidgetItem(f"{step.troop_type} (x{step.unit_count})"))
            self.steps_table.setItem(row, 2, QTableWidgetItem(f"({step.target_x}, {step.target_y})"))
            self.steps_table.setItem(row, 3, QTableWidgetItem(f"{step.delay_ms} ms"))
            self.steps_table.setItem(row, 4, QTableWidgetItem(step.note or ""))
