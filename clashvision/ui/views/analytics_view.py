"""
ClashVision Presentation Subsystem — View 6: Analytics Dashboard.
Displays local session analytics and historical loot metrics:
- Total bases scanned counter
- Target match rate (%)
- Average Gold, Elixir, and Dark Elixir per base
- Computer vision processing latency statistics
- Historical scans log table
Derived from Requirement R5 and Feature 34.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from clashvision.storage.repository import Repository
from clashvision.ui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, Qt
)
from clashvision.ui.styles import (
    COLOR_NEON_GREEN, COLOR_NEON_GOLD, COLOR_NEON_CYAN,
    COLOR_BORDER
)


class AnalyticsView(QWidget):
    """
    Session analytics and historical performance inspection view.
    """

    def __init__(self, repo: Optional[Repository] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("analytics_view")
        self.repo = repo
        self._latency_samples: List[float] = []
        self._init_ui()

    def record_cv_latency(self, latency_ms: float) -> None:
        """Record real computer vision processing latency sample."""
        if latency_ms > 0:
            self._latency_samples.append(latency_ms)
            if len(self._latency_samples) > 200:
                self._latency_samples.pop(0)

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        # Header with Refresh Button
        header_row = QHBoxLayout()
        title_label = QLabel("Session & Loot Analytics Dashboard")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #f1f2f6;")
        header_row.addWidget(title_label)

        header_row.addStretch()

        self.refresh_btn = QPushButton("🔄 Refresh Analytics")
        self.refresh_btn.clicked.connect(self.refresh_stats)
        header_row.addWidget(self.refresh_btn)

        main_layout.addLayout(header_row)

        # 1. Metric Cards Row
        cards_row = QHBoxLayout()
        cards_row.setSpacing(14)

        # Card 1: Total Scans
        self.card_total_scans = self._create_metric_card("TOTAL BASES SCANNED", "0", COLOR_NEON_CYAN)
        cards_row.addWidget(self.card_total_scans)

        # Card 2: Match Rate
        self.card_match_rate = self._create_metric_card("TARGET MATCH RATE", "0.0%", COLOR_NEON_GREEN)
        cards_row.addWidget(self.card_match_rate)

        # Card 3: Avg Gold Loot
        self.card_avg_gold = self._create_metric_card("AVERAGE GOLD", "0", COLOR_NEON_GOLD)
        cards_row.addWidget(self.card_avg_gold)

        # Card 4: Avg Elixir Loot
        self.card_avg_elixir = self._create_metric_card("AVERAGE ELIXIR", "0", "#ff4757")
        cards_row.addWidget(self.card_avg_elixir)

        # Card 5: Avg CV Latency
        self.card_cv_latency = self._create_metric_card("AVG CV LATENCY", "0.0 ms", "#a55eea")
        cards_row.addWidget(self.card_cv_latency)

        main_layout.addLayout(cards_row)

        # 2. Historical Scans Table
        table_title = QLabel("RECENT SCAN LOGS & RESOURCE READINGS")
        table_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #a4b0be; margin-top: 10px;")
        main_layout.addWidget(table_title)

        self.scans_table = QTableWidget(0, 7)
        self.scans_table.setHorizontalHeaderLabels([
            "Scan ID", "Timestamp", "State", "Confidence", "Gold", "Elixir", "Dark Elixir"
        ])
        self.scans_table.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(self.scans_table, stretch=1)

    def _create_metric_card(self, title: str, initial_value: str, value_color: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            f"background-color: #1a1c23; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 12px;"
        )
        layout = QVBoxLayout(card)
        layout.setSpacing(4)

        t_lbl = QLabel(title)
        t_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #a4b0be; letter-spacing: 0.5px;")
        layout.addWidget(t_lbl)

        v_lbl = QLabel(initial_value)
        v_lbl.setObjectName(f"val_{title.lower().replace(' ', '_')}")
        v_lbl.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {value_color};")
        layout.addWidget(v_lbl)
        card.value_label = v_lbl  # type: ignore[attr-defined]

        return card

    def set_repository(self, repo: Repository) -> None:
        self.repo = repo
        self.refresh_stats()

    def refresh_stats(self) -> None:
        """Fetch historical records from repository and update counters/table."""
        if self.repo is None:
            return

        scans = self.repo.get_base_scans(limit=100)
        total_count = self.repo.get_scan_count()
        sessions = self.repo.list_sessions(limit=50)

        total_matched = sum(s.matched_targets for s in sessions)
        total_session_scans = sum(s.total_scans for s in sessions)
        match_rate = (total_matched / total_session_scans * 100.0) if total_session_scans > 0 else 0.0

        gold_vals = [s["gold"] for s in scans if s["gold"] is not None]
        elixir_vals = [s["elixir"] for s in scans if s["elixir"] is not None]

        avg_gold = int(sum(gold_vals) / len(gold_vals)) if gold_vals else 0
        avg_elixir = int(sum(elixir_vals) / len(elixir_vals)) if elixir_vals else 0

        self.card_total_scans.value_label.setText(str(total_count))  # type: ignore[attr-defined]
        self.card_match_rate.value_label.setText(f"{match_rate:.1f}%")  # type: ignore[attr-defined]
        self.card_avg_gold.value_label.setText(f"{avg_gold:,}")  # type: ignore[attr-defined]
        # Dynamic CV latency calculation
        latencies = list(self._latency_samples)
        for s in scans:
            raw_json = s.get("raw_ocr_json")
            if raw_json and isinstance(raw_json, str) and "cv_latency_ms" in raw_json:
                try:
                    import json
                    data = json.loads(raw_json)
                    if isinstance(data, dict) and "cv_latency_ms" in data:
                        lat = float(data["cv_latency_ms"])
                        if lat > 0:
                            latencies.append(lat)
                except Exception:
                    pass

        if not scans and not self._latency_samples:
            self.card_cv_latency.value_label.setText("N/A")  # type: ignore[attr-defined]
        elif latencies:
            avg_lat = sum(latencies) / len(latencies)
            self.card_cv_latency.value_label.setText(f"{avg_lat:.1f} ms")  # type: ignore[attr-defined]
        else:
            self.card_cv_latency.value_label.setText("N/A")  # type: ignore[attr-defined]

        # Populate table
        self.scans_table.setRowCount(len(scans))
        for row, s in enumerate(scans):
            self.scans_table.setItem(row, 0, QTableWidgetItem(str(s["scan_id"])))
            self.scans_table.setItem(row, 1, QTableWidgetItem(f"{s['timestamp']:.2f}"))
            self.scans_table.setItem(row, 2, QTableWidgetItem(s["game_state"]))
            self.scans_table.setItem(row, 3, QTableWidgetItem(f"{s['confidence'] * 100:.1f}%"))
            self.scans_table.setItem(row, 4, QTableWidgetItem(f"{s['gold']:,}" if s['gold'] else "--"))
            self.scans_table.setItem(row, 5, QTableWidgetItem(f"{s['elixir']:,}" if s['elixir'] else "--"))
            self.scans_table.setItem(row, 6, QTableWidgetItem(f"{s['dark_elixir']:,}" if s['dark_elixir'] else "--"))
