"""
ClashVision Presentation Subsystem — View 3: Base Analyzer.
Provides spatial structural base analysis:
- Annotated base preview canvas
- Exposed collectors vs core storages resource split
- Town Hall accessibility assessment badge
- Quantitative suitability score breakdown (Resource score, Layout score, Risk discount)
Derived from Requirement R5 and Feature 31.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from clashvision.core.models import (
    VisionAnalysisResult,
    BaseLayoutFeatures,
    FarmingRecommendation,
)
from clashvision.ui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QProgressBar, Qt
)
from clashvision.ui.styles import (
    COLOR_NEON_GREEN, COLOR_NEON_GOLD, COLOR_NEON_RED,
    COLOR_NEON_CYAN, COLOR_BORDER
)


class BaseAnalyzerView(QWidget):
    """
    Spatial base layout and resource accessibility inspection view.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("base_analyzer_view")
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        # Header Title
        title_label = QLabel("Base Structural & Accessibility Analyzer")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #f1f2f6;")
        main_layout.addWidget(title_label)

        # Content Split Layout: Base Preview on Left, Breakdown Cards on Right
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        # Left: Annotated Base Preview
        preview_box = QFrame()
        preview_box.setStyleSheet(
            f"background-color: #0e0f14; border: 1px solid {COLOR_BORDER}; border-radius: 8px;"
        )
        preview_layout = QVBoxLayout(preview_box)
        preview_title = QLabel("ANNOTATED BASE SCHEMATIC")
        preview_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #a4b0be;")
        preview_layout.addWidget(preview_title)

        self.preview_canvas = QLabel("No active enemy base analyzed.")
        self.preview_canvas.setAlignment(Qt.AlignCenter)
        self.preview_canvas.setStyleSheet("color: #747d8c; min-height: 400px; min-width: 480px;")
        preview_layout.addWidget(self.preview_canvas, stretch=1)
        content_layout.addWidget(preview_box, stretch=3)

        # Right: Spatial Metrics & Accessibility Breakdown
        metrics_panel = QFrame()
        metrics_panel.setStyleSheet(
            f"background-color: #1a1c23; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 12px;"
        )
        metrics_layout = QVBoxLayout(metrics_panel)
        metrics_layout.setSpacing(12)

        panel_title = QLabel("SPATIAL FEATURE BREAKDOWN")
        panel_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #a4b0be;")
        metrics_layout.addWidget(panel_title)

        # Town Hall Accessibility Badge
        th_card = QFrame()
        th_card.setStyleSheet("background-color: #242834; border-radius: 6px; padding: 10px;")
        th_layout = QVBoxLayout(th_card)
        th_layout.addWidget(QLabel("Town Hall Accessibility:"))
        self.th_badge = QLabel("NOT DETECTED")
        self.th_badge.setStyleSheet(
            "background-color: #331518; color: #ff4757; border: 1px solid #ff4757; border-radius: 4px; padding: 4px 8px; font-weight: bold;"
        )
        th_layout.addWidget(self.th_badge)
        metrics_layout.addWidget(th_card)

        # Resource Split: Exposed vs Core
        split_card = QFrame()
        split_card.setStyleSheet("background-color: #242834; border-radius: 6px; padding: 10px;")
        split_layout = QVBoxLayout(split_card)
        split_layout.addWidget(QLabel("Exposed Collector vs Core Storage Split:"))

        self.collector_split_bar = QProgressBar()
        self.collector_split_bar.setRange(0, 100)
        self.collector_split_bar.setValue(50)
        split_layout.addWidget(self.collector_split_bar)

        self.split_label = QLabel("Exposed Collectors: 50.0% | Core: 50.0%")
        self.split_label.setStyleSheet("color: #a4b0be; font-size: 11px;")
        split_layout.addWidget(self.split_label)
        metrics_layout.addWidget(split_card)

        # Suitability Score Breakdown
        score_card = QFrame()
        score_card.setStyleSheet("background-color: #242834; border-radius: 6px; padding: 10px;")
        score_layout = QVBoxLayout(score_card)
        score_layout.addWidget(QLabel("Suitability Score Components:"))

        self.resource_score_label = QLabel("• Resource Loot Score: 0.0 / 100")
        score_layout.addWidget(self.resource_score_label)

        self.layout_score_label = QLabel("• Layout Accessibility Score: 0.0 / 100")
        score_layout.addWidget(self.layout_score_label)

        self.risk_discount_label = QLabel("• Risk Factor Penalty: 0.0")
        score_layout.addWidget(self.risk_discount_label)

        self.total_suitability_label = QLabel("Total Suitability: 0.0 / 100")
        self.total_suitability_label.setStyleSheet(
            f"color: {COLOR_NEON_GREEN}; font-weight: bold; font-size: 14px; margin-top: 6px;"
        )
        score_layout.addWidget(self.total_suitability_label)
        metrics_layout.addWidget(score_card)

        metrics_layout.addStretch()
        content_layout.addWidget(metrics_panel, stretch=2)

        main_layout.addLayout(content_layout)

    def update_analysis(
        self,
        analysis: VisionAnalysisResult,
        rec: Optional[FarmingRecommendation] = None,
    ) -> None:
        """Update base analyzer schematic and feature breakdown."""
        layout: Optional[BaseLayoutFeatures] = analysis.base_layout
        if layout is not None:
            ratio_pct = int(round(layout.exposed_collector_ratio * 100))
            self.collector_split_bar.setValue(ratio_pct)
            self.split_label.setText(
                f"Exposed Collectors: {ratio_pct}% | Core Storages: {100 - ratio_pct}%"
            )

            if layout.th_accessible:
                self.th_badge.setText("TH ACCESSIBLE / SNIPABLE")
                self.th_badge.setStyleSheet(
                    f"background-color: #142a1b; color: {COLOR_NEON_GREEN}; border: 1px solid {COLOR_NEON_GREEN}; border-radius: 4px; padding: 4px 8px; font-weight: bold;"
                )
            else:
                self.th_badge.setText("TH DEEPLY PROTECTED")
                self.th_badge.setStyleSheet(
                    f"background-color: #3b2a12; color: #ffa502; border: 1px solid #ffa502; border-radius: 4px; padding: 4px 8px; font-weight: bold;"
                )

        if rec is not None:
            bd = rec.score_breakdown or {}
            res_sc = bd.get("resource_score", rec.suitability_score)
            lay_sc = bd.get("layout_score", bd.get("accessibility_score", 50.0))
            risk_pen = bd.get("risk_discount", bd.get("risk_penalty", 0.0))

            self.resource_score_label.setText(f"• Resource Loot Score: {res_sc:.1f} / 100")
            self.layout_score_label.setText(f"• Layout Accessibility Score: {lay_sc:.1f} / 100")
            self.risk_discount_label.setText(f"• Risk Factor Penalty: -{risk_pen:.1f}")
            self.total_suitability_label.setText(f"Total Suitability: {rec.suitability_score:.1f} / 100")

        st_val = analysis.state.value if hasattr(analysis.state, "value") else str(analysis.state)
        self.preview_canvas.setText(
            f"Base Schematic Ready: State={st_val} | Conf={analysis.confidence * 100:.1f}%"
        )
