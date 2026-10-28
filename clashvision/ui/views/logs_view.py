"""
ClashVision Presentation Subsystem — View 9: Structured Logs View.
Displays live virtualized application event logs:
- Virtualized log table (Timestamp, Level, Component, Message)
- Log level filter dropdown (ALL, INFO, WARNING, ERROR)
- Real-time text search filter
- Auto-scroll to latest log toggle
- Clear logs button
- Direct persistence and retrieval from SQLite repository
Derived from Requirement R5 and Feature 37.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional

from clashvision.storage.repository import Repository
from clashvision.ui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QComboBox, QLineEdit, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, Qt
)
from clashvision.ui.styles import (
    COLOR_NEON_GREEN, COLOR_NEON_RED, COLOR_NEON_GOLD,
    COLOR_BORDER
)


class LogsView(QWidget):
    """
    Live structured logging and system event inspection view.
    """

    def __init__(self, repo: Optional[Repository] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("logs_view")
        self.repo = repo
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Filter Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        toolbar.addWidget(QLabel("Filter Level:"))
        self.combo_level = QComboBox()
        self.combo_level.addItems(["ALL", "INFO", "WARNING", "ERROR"])
        self.combo_level.currentTextChanged.connect(self._on_filter_changed)
        toolbar.addWidget(self.combo_level)

        toolbar.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter messages by keyword...")
        self.search_input.textChanged.connect(self._on_filter_changed)
        toolbar.addWidget(self.search_input)

        self.chk_autoscroll = QCheckBox("Auto-scroll")
        self.chk_autoscroll.setChecked(True)
        toolbar.addWidget(self.chk_autoscroll)

        toolbar.addStretch()

        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.refresh_logs)
        toolbar.addWidget(self.refresh_btn)

        self.clear_btn = QPushButton("🗑️ Clear Logs")
        self.clear_btn.clicked.connect(self._on_clear_logs)
        toolbar.addWidget(self.clear_btn)

        main_layout.addLayout(toolbar)

        # 2. Virtualized Logs Table
        self.logs_table = QTableWidget(0, 4)
        self.logs_table.setHorizontalHeaderLabels(["Timestamp", "Level", "Component", "Message"])
        self.logs_table.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(self.logs_table, stretch=1)

        # Status Footer
        footer_row = QHBoxLayout()
        self.count_label = QLabel("Total Events: 0")
        self.count_label.setStyleSheet("color: #a4b0be; font-size: 11px;")
        footer_row.addWidget(self.count_label)
        footer_row.addStretch()
        main_layout.addLayout(footer_row)

    def set_repository(self, repo: Repository) -> None:
        self.repo = repo
        self.refresh_logs()

    def _on_filter_changed(self) -> None:
        self.refresh_logs()

    def _on_clear_logs(self) -> None:
        if self.repo:
            self.repo.clear_logs()
        self.logs_table.setRowCount(0)
        self.count_label.setText("Total Events: 0")

    def append_log_entry(
        self,
        level: str,
        message: str,
        logger_name: str = "clashvision",
        timestamp: Optional[float] = None,
    ) -> None:
        """Append an event to the UI table and repository."""
        ts = timestamp if timestamp is not None else time.time()
        if self.repo:
            self.repo.log_event(level=level, message=message, logger_name=logger_name, timestamp=ts)

        # Update table directly
        row = self.logs_table.rowCount()
        self.logs_table.setRowCount(row + 1)

        timestr = time.strftime("%H:%M:%S", time.localtime(ts))
        self.logs_table.setItem(row, 0, QTableWidgetItem(timestr))

        lvl_item = QTableWidgetItem(level.upper())
        self.logs_table.setItem(row, 1, lvl_item)

        self.logs_table.setItem(row, 2, QTableWidgetItem(logger_name))
        self.logs_table.setItem(row, 3, QTableWidgetItem(message))

        self.count_label.setText(f"Total Events: {self.logs_table.rowCount()}")

        if self.chk_autoscroll.isChecked():
            self.logs_table.scrollToBottom()

    def refresh_logs(self) -> None:
        """Query repository with current filters and populate table."""
        if not self.repo:
            return

        level_filter = self.combo_level.currentText()
        search_filter = self.search_input.text().strip()

        logs = self.repo.query_logs(limit=200, level=level_filter, search=search_filter or None)
        self.logs_table.setRowCount(len(logs))

        for row, entry in enumerate(logs):
            timestr = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
            self.logs_table.setItem(row, 0, QTableWidgetItem(timestr))
            self.logs_table.setItem(row, 1, QTableWidgetItem(entry["level"]))
            self.logs_table.setItem(row, 2, QTableWidgetItem(entry["logger_name"]))
            self.logs_table.setItem(row, 3, QTableWidgetItem(entry["message"]))

        self.count_label.setText(f"Total Events: {len(logs)}")
        if self.chk_autoscroll.isChecked():
            self.logs_table.scrollToBottom()
