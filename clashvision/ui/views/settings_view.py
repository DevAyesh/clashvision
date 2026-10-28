"""
ClashVision Presentation Subsystem — View 8: Settings View.
Provides runtime configuration controls:
- Resource thresholds (Min Gold, Min Elixir, Min Dark Elixir)
- Farming strategy priority weighting selector
- Screen grabber capture FPS (1-60)
- OCR Engine backend selector (RapidOCR, Pytesseract, Mock)
- Target emulator window / process selector
- Persistence to local SQLite settings table and AppSettings JSON
Derived from Requirement R5 and Feature 36.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from clashvision.core.config import AppSettings, ResourceThresholds, CaptureConfig, OCRConfig
from clashvision.core.constants import ResourcePriority, DEFAULT_CAPTURE_FPS
from clashvision.storage.repository import Repository
from clashvision.ui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QSpinBox, QComboBox, QLineEdit, QFormLayout, Qt
)
from clashvision.ui.styles import COLOR_NEON_GREEN, COLOR_BORDER


class SettingsView(QWidget):
    """
    Subsystem preferences and configuration management view.
    """

    def __init__(
        self,
        repo: Optional[Repository] = None,
        settings: Optional[AppSettings] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("settings_view")
        self.repo = repo
        self.settings = settings or AppSettings()
        self._init_ui()
        self.load_settings()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        # Title
        title_label = QLabel("ClashVision Subsystem Preferences")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #f1f2f6;")
        main_layout.addWidget(title_label)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        # Left Column: Farming Thresholds & Priorities
        left_card = QFrame()
        left_card.setStyleSheet(
            f"background-color: #1a1c23; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 14px;"
        )
        left_layout = QVBoxLayout(left_card)
        left_layout.setSpacing(10)

        left_title = QLabel("FARMING THRESHOLDS & STRATEGY PRIORITY")
        left_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #a4b0be;")
        left_layout.addWidget(left_title)

        form_left = QFormLayout()

        self.spin_gold = QSpinBox()
        self.spin_gold.setRange(0, 5000000)
        self.spin_gold.setSingleStep(50000)
        form_left.addRow("Min Gold Threshold:", self.spin_gold)

        self.spin_elixir = QSpinBox()
        self.spin_elixir.setRange(0, 5000000)
        self.spin_elixir.setSingleStep(50000)
        form_left.addRow("Min Elixir Threshold:", self.spin_elixir)

        self.spin_dark_elixir = QSpinBox()
        self.spin_dark_elixir.setRange(0, 50000)
        self.spin_dark_elixir.setSingleStep(500)
        form_left.addRow("Min Dark Elixir Threshold:", self.spin_dark_elixir)

        self.combo_priority = QComboBox()
        for prio in ResourcePriority:
            self.combo_priority.addItem(prio.value)
        form_left.addRow("Resource Priority Weighting:", self.combo_priority)

        left_layout.addLayout(form_left)
        left_layout.addStretch()
        content_layout.addWidget(left_card)

        # Right Column: Capture, OCR & Target Options
        right_card = QFrame()
        right_card.setStyleSheet(
            f"background-color: #1a1c23; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 14px;"
        )
        right_layout = QVBoxLayout(right_card)
        right_layout.setSpacing(10)

        right_title = QLabel("CAPTURE ENGINE & TARGET WINDOW")
        right_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #a4b0be;")
        right_layout.addWidget(right_title)

        form_right = QFormLayout()

        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 60)
        self.spin_fps.setValue(DEFAULT_CAPTURE_FPS)
        form_right.addRow("Capture Rate (FPS):", self.spin_fps)

        self.combo_ocr = QComboBox()
        self.combo_ocr.addItem("rapidocr")
        self.combo_ocr.addItem("pytesseract")
        self.combo_ocr.addItem("mock")
        form_right.addRow("Active OCR Engine:", self.combo_ocr)

        self.target_window_input = QLineEdit()
        self.target_window_input.setText("Clash of Clans / Google Play Games on PC")
        form_right.addRow("Target Window Title:", self.target_window_input)

        right_layout.addLayout(form_right)
        right_layout.addStretch()
        content_layout.addWidget(right_card)

        main_layout.addLayout(content_layout)

        # Bottom Action Bar
        action_bar = QHBoxLayout()
        self.save_btn = QPushButton("💾 Save & Apply Settings")
        self.save_btn.setProperty("class", "btn_primary")
        self.save_btn.setStyleSheet(f"background-color: {COLOR_NEON_GREEN}; color: #0b1a10; font-weight: bold; padding: 8px 18px;")
        self.save_btn.clicked.connect(self.save_settings)
        action_bar.addWidget(self.save_btn)

        self.reset_btn = QPushButton("🔄 Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_defaults)
        action_bar.addWidget(self.reset_btn)

        action_bar.addStretch()

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #a4b0be; font-size: 12px;")
        action_bar.addWidget(self.status_label)

        main_layout.addLayout(action_bar)
        main_layout.addStretch()

    def set_repository(self, repo: Repository) -> None:
        self.repo = repo
        self.load_settings()

    def load_settings(self) -> None:
        """Load settings from repository or AppSettings defaults."""
        res = self.settings.resources
        cap = self.settings.capture
        ocr = self.settings.ocr

        if self.repo:
            min_g = self.repo.get_setting("min_gold")
            min_e = self.repo.get_setting("min_elixir")
            min_de = self.repo.get_setting("min_dark_elixir")
            fps = self.repo.get_setting("target_fps")
            ocr_eng = self.repo.get_setting("ocr_engine")
            prio = self.repo.get_setting("resource_priority")

            if min_g: res.min_gold = int(min_g)
            if min_e: res.min_elixir = int(min_e)
            if min_de: res.min_dark_elixir = int(min_de)
            if fps: cap.target_fps = int(fps)
            if ocr_eng: ocr.engine_type = ocr_eng
            if prio: res.priority = ResourcePriority(prio)

        self.spin_gold.setValue(res.min_gold)
        self.spin_elixir.setValue(res.min_elixir)
        self.spin_dark_elixir.setValue(res.min_dark_elixir)
        self.spin_fps.setValue(cap.target_fps)
        self.combo_ocr.setCurrentText(ocr.engine_type)
        self.combo_priority.setCurrentText(res.priority.value)

    def save_settings(self) -> None:
        """Persist current UI settings to repository and AppSettings."""
        self.settings.resources.min_gold = self.spin_gold.value()
        self.settings.resources.min_elixir = self.spin_elixir.value()
        self.settings.resources.min_dark_elixir = self.spin_dark_elixir.value()
        self.settings.resources.priority = ResourcePriority(self.combo_priority.currentText())
        self.settings.capture.target_fps = self.spin_fps.value()
        self.settings.ocr.engine_type = self.combo_ocr.currentText()

        if self.repo:
            self.repo.set_setting("min_gold", self.spin_gold.value())
            self.repo.set_setting("min_elixir", self.spin_elixir.value())
            self.repo.set_setting("min_dark_elixir", self.spin_dark_elixir.value())
            self.repo.set_setting("resource_priority", self.combo_priority.currentText())
            self.repo.set_setting("target_fps", self.spin_fps.value())
            self.repo.set_setting("ocr_engine", self.combo_ocr.currentText())
            self.status_label.setText("Settings saved successfully to SQLite storage.")
        else:
            self.status_label.setText("Settings applied in-memory.")

    def reset_defaults(self) -> None:
        """Reset configuration controls to factory default settings."""
        self.settings = AppSettings()
        self.load_settings()
        self.status_label.setText("Reset to default settings.")
