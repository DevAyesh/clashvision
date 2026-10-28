"""
ClashVision Presentation Subsystem — View 1: Overview Dashboard.
Displays live system status, current detected GameState card with confidence,
resource counters (Gold, Elixir, DE), recommended strategy summary, and error/alert banner.
Derived from Requirement R5 and Feature 29.
"""

from __future__ import annotations
import time
from typing import Any, Dict, Optional

from clashvision.core.constants import GameState, StrategyType
from clashvision.core.models import (
    ResourceReading,
    FarmingRecommendation,
    VisionAnalysisResult,
)
from clashvision.ui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QProgressBar, Qt
)
from clashvision.ui.styles import (
    COLOR_NEON_GREEN, COLOR_NEON_GOLD, COLOR_NEON_RED,
    COLOR_NEON_PURPLE, COLOR_TEXT_MUTED, get_state_badge_style
)


class OverviewView(QWidget):
    """
    High-level operational overview dashboard view.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("overview_view")
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # 1. Title & System Status Banner
        header_row = QHBoxLayout()
        title_label = QLabel("ClashVision — System Overview")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #f1f2f6;")
        header_row.addWidget(title_label)

        header_row.addStretch()

        self.system_status_badge = QLabel("SYSTEM IDLE")
        self.system_status_badge.setStyleSheet(
            f"background-color: #242834; color: {COLOR_TEXT_MUTED}; "
            "border: 1px solid #323846; border-radius: 4px; padding: 4px 10px; font-weight: bold;"
        )
        header_row.addWidget(self.system_status_badge)
        main_layout.addLayout(header_row)

        # 2. Key Metrics Card Grid (State Card, Resources Card, Target Card)
        cards_row = QHBoxLayout()
        cards_row.setSpacing(16)

        # State Card
        self.state_card = QFrame()
        self.state_card.setProperty("class", "dashboard_card")
        self.state_card.setStyleSheet("background-color: #1a1c23; border: 1px solid #2c303b; border-radius: 8px; padding: 12px;")
        state_card_layout = QVBoxLayout(self.state_card)

        card_title = QLabel("CURRENT GAME STATE")
        card_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #a4b0be; letter-spacing: 1px;")
        state_card_layout.addWidget(card_title)

        self.state_badge = QLabel(GameState.UNKNOWN.value)
        self.state_badge.setStyleSheet(get_state_badge_style(GameState.UNKNOWN.value))
        state_card_layout.addWidget(self.state_badge)

        self.state_confidence_label = QLabel("Confidence: 0.0%")
        self.state_confidence_label.setStyleSheet("font-size: 12px; color: #747d8c;")
        state_card_layout.addWidget(self.state_confidence_label)
        cards_row.addWidget(self.state_card)

        # Resource Loot Card
        self.loot_card = QFrame()
        self.loot_card.setProperty("class", "dashboard_card")
        self.loot_card.setStyleSheet("background-color: #1a1c23; border: 1px solid #2c303b; border-radius: 8px; padding: 12px;")
        loot_card_layout = QVBoxLayout(self.loot_card)

        loot_title = QLabel("DETECTED BASE LOOT")
        loot_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #a4b0be; letter-spacing: 1px;")
        loot_card_layout.addWidget(loot_title)

        loot_values_layout = QHBoxLayout()
        self.gold_label = QLabel("Gold: --")
        self.gold_label.setStyleSheet(f"color: {COLOR_NEON_GOLD}; font-weight: bold; font-size: 13px;")
        loot_values_layout.addWidget(self.gold_label)

        self.elixir_label = QLabel("Elixir: --")
        self.elixir_label.setStyleSheet("color: #ff4757; font-weight: bold; font-size: 13px;")
        loot_values_layout.addWidget(self.elixir_label)

        self.dark_elixir_label = QLabel("DE: --")
        self.dark_elixir_label.setStyleSheet("color: #3742fa; font-weight: bold; font-size: 13px;")
        loot_values_layout.addWidget(self.dark_elixir_label)
        loot_card_layout.addLayout(loot_values_layout)

        self.th_status_label = QLabel("Town Hall: Unknown")
        self.th_status_label.setStyleSheet("font-size: 12px; color: #747d8c;")
        loot_card_layout.addWidget(self.th_status_label)
        cards_row.addWidget(self.loot_card)

        main_layout.addLayout(cards_row)

        # 3. Recommended Strategy Summary Card
        self.rec_card = QFrame()
        self.rec_card.setStyleSheet("background-color: #1a1c23; border: 1px solid #2c303b; border-radius: 8px; padding: 16px;")
        rec_card_layout = QVBoxLayout(self.rec_card)

        rec_header_layout = QHBoxLayout()
        rec_card_title = QLabel("TACTICAL RECOMMENDATION")
        rec_card_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #a4b0be;")
        rec_header_layout.addWidget(rec_card_title)
        rec_header_layout.addStretch()

        self.rec_viability_badge = QLabel("NO ACTIVE TARGET")
        self.rec_viability_badge.setStyleSheet(
            "background-color: #242834; color: #747d8c; border-radius: 4px; padding: 3px 8px; font-weight: bold;"
        )
        rec_header_layout.addWidget(self.rec_viability_badge)
        rec_card_layout.addLayout(rec_header_layout)

        self.rec_strategy_name = QLabel("Awaiting enemy base preview...")
        self.rec_strategy_name.setStyleSheet("font-size: 16px; font-weight: bold; color: #f1f2f6; margin-top: 4px;")
        rec_card_layout.addWidget(self.rec_strategy_name)

        suitability_row = QHBoxLayout()
        suitability_row.addWidget(QLabel("Suitability Score:"))
        self.rec_score_bar = QProgressBar()
        self.rec_score_bar.setRange(0, 100)
        self.rec_score_bar.setValue(0)
        suitability_row.addWidget(self.rec_score_bar)
        self.rec_score_label = QLabel("0.0 / 100")
        self.rec_score_label.setStyleSheet("font-weight: bold; min-width: 65px;")
        suitability_row.addWidget(self.rec_score_label)
        rec_card_layout.addLayout(suitability_row)

        self.rec_explanation_label = QLabel("Observation engine running in passive read-only mode.")
        self.rec_explanation_label.setStyleSheet("color: #a4b0be; font-size: 12px; margin-top: 4px;")
        rec_card_layout.addWidget(self.rec_explanation_label)
        main_layout.addWidget(self.rec_card)

        # 4. Error & Safety Banner Panel
        self.alert_banner = QFrame()
        self.alert_banner.setStyleSheet(
            "background-color: #201316; border: 1px solid #4d1d25; border-radius: 6px; padding: 10px;"
        )
        alert_layout = QHBoxLayout(self.alert_banner)
        self.alert_icon = QLabel("🛡️")
        self.alert_icon.setStyleSheet("font-size: 16px;")
        alert_layout.addWidget(self.alert_icon)

        self.alert_text = QLabel("Live Safety Barrier Active: Zero mouse/keyboard injection to emulator.")
        self.alert_text.setStyleSheet("color: #ff6b81; font-weight: 500; font-size: 12px;")
        alert_layout.addWidget(self.alert_text)
        alert_layout.addStretch()
        main_layout.addWidget(self.alert_banner)

        main_layout.addStretch()

    def update_state(self, state: GameState, confidence: float, evidence: Dict[str, Any]) -> None:
        """Update live game state card."""
        st_val = state.value if isinstance(state, GameState) else str(state)
        self.state_badge.setText(st_val)
        self.state_badge.setStyleSheet(get_state_badge_style(st_val))
        self.state_confidence_label.setText(f"Confidence: {confidence * 100:.1f}%")

        is_actionable = (
            state.is_actionable
            if isinstance(state, GameState)
            else (
                st_val
                in (
                    GameState.ENEMY_BASE_PREVIEW.value,
                    GameState.BATTLE_IN_PROGRESS.value,
                    GameState.HOME_VILLAGE.value,
                )
            )
        )
        if is_actionable:
            self.system_status_badge.setText("TARGET IDENTIFIED")
            self.system_status_badge.setStyleSheet(
                f"background-color: #142a1b; color: {COLOR_NEON_GREEN}; "
                f"border: 1px solid {COLOR_NEON_GREEN}; border-radius: 4px; padding: 4px 10px; font-weight: bold;"
            )
        else:
            self.system_status_badge.setText("OBSERVING")
            self.system_status_badge.setStyleSheet(
                f"background-color: #162844; color: #70a1ff; "
                "border: 1px solid #70a1ff; border-radius: 4px; padding: 4px 10px; font-weight: bold;"
            )

    def update_resources(self, reading: ResourceReading) -> None:
        """Update detected resource counts."""
        if reading.gold is not None:
            self.gold_label.setText(f"Gold: {reading.gold:,}")
        else:
            self.gold_label.setText("Gold: --")

        if reading.elixir is not None:
            self.elixir_label.setText(f"Elixir: {reading.elixir:,}")
        else:
            self.elixir_label.setText("Elixir: --")

        if reading.dark_elixir is not None:
            self.dark_elixir_label.setText(f"DE: {reading.dark_elixir:,}")
        else:
            self.dark_elixir_label.setText("DE: --")

    def update_recommendation(self, rec: FarmingRecommendation) -> None:
        """Update recommendation card with tactical evaluation."""
        self.rec_strategy_name.setText(rec.primary_strategy.display_name)
        score_int = int(round(rec.suitability_score))
        self.rec_score_bar.setValue(score_int)
        self.rec_score_label.setText(f"{rec.suitability_score:.1f} / 100")
        self.rec_explanation_label.setText(rec.explanation or "No explanation provided.")

        if rec.suitability_score >= 70.0:
            self.rec_viability_badge.setText("HIGH VALUE TARGET")
            self.rec_viability_badge.setStyleSheet(
                f"background-color: #142a1b; color: {COLOR_NEON_GREEN}; "
                f"border: 1px solid {COLOR_NEON_GREEN}; border-radius: 4px; padding: 3px 8px; font-weight: bold;"
            )
        elif rec.suitability_score >= 50.0:
            self.rec_viability_badge.setText("VIABLE TARGET")
            self.rec_viability_badge.setStyleSheet(
                "background-color: #3b2a12; color: #ffa502; border: 1px solid #ffa502; border-radius: 4px; padding: 3px 8px; font-weight: bold;"
            )
        else:
            self.rec_viability_badge.setText("SKIP / LOW LOOT")
            self.rec_viability_badge.setStyleSheet(
                "background-color: #331518; color: #ff4757; border: 1px solid #ff4757; border-radius: 4px; padding: 3px 8px; font-weight: bold;"
            )

    def show_alert(self, title: str, message: str) -> None:
        """Display an alert banner message."""
        self.alert_text.setText(f"{title}: {message}")
