"""
ClashVision Presentation Subsystem — Responsive Desktop MainWindow.
Hosts dark-themed shell with collapsible sidebar navigation, top safety status bar,
and a central QStackedWidget hosting all 9 functional views.
Seamlessly integrates SQLite persistence and asynchronous worker threads.
Derived from Requirement R5 and Feature 28.
"""

from __future__ import annotations
import logging
import sys
import time
from typing import Any, Dict, List, Optional

from clashvision.core.config import AppSettings
from clashvision.core.constants import GameState, StrategyType
from clashvision.core.models import (
    FramePacket,
    ResourceReading,
    VisionAnalysisResult,
    FarmingRecommendation,
    SimulationState,
)
from clashvision.storage.repository import Repository
from clashvision.ui.qt_compat import (
    QMainWindow, QWidget, QStackedWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QFrame, QPushButton, QEvent, Qt
)
from clashvision.ui.styles import (
    get_stylesheet, COLOR_NEON_GREEN, COLOR_NEON_PURPLE,
    COLOR_NEON_RED, COLOR_BG_SIDEBAR, COLOR_BORDER
)
from clashvision.ui.workers import (
    CaptureWorker,
    VisionWorker,
    DecisionWorker,
    SimulationWorker,
)
from clashvision.ui.views import (
    OverviewView,
    LiveCaptureView,
    BaseAnalyzerView,
    StrategyView,
    SimulatorView,
    AnalyticsView,
    DatasetView,
    SettingsView,
    LogsView,
)

logger = logging.getLogger("clashvision.ui.main_window")


VIEW_TITLES = [
    ("📊 Overview", "overview"),
    ("📹 Live Capture", "live_capture"),
    ("🔍 Base Analyzer", "base_analyzer"),
    ("♟️ Strategy Planner", "strategy"),
    ("⚔️ Battle Simulator", "simulator"),
    ("📈 Analytics", "analytics"),
    ("📁 Dataset Manager", "dataset"),
    ("⚙️ Settings", "settings"),
    ("📜 Logs", "logs"),
]


class MainWindow(QMainWindow):
    """
    Main application shell for ClashVision desktop application.
    """

    def __init__(
        self,
        repo: Optional[Repository] = None,
        settings: Optional[AppSettings] = None,
        start_workers: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("main_window")
        self.setWindowTitle("ClashVision — Read-Only CoC Vision & Farming Advisor")
        self.resize(1280, 800)

        self.repo = repo or Repository()
        self.settings = settings or AppSettings()
        self._current_session_id = f"sess_{int(time.time())}"

        # Initialize tracking session in database
        try:
            self.repo.create_session(self._current_session_id, notes="Desktop UI Session")
        except Exception as e:
            logger.warning("Could not initialize session in repository: %s", e)

        # Build UI layout
        self._init_ui()

        # Build and wire asynchronous worker pool
        self._init_workers()

        # Apply dark QSS theme
        self.setStyleSheet(get_stylesheet())

        if start_workers:
            self.start_all_workers()

    def _init_ui(self) -> None:
        root_container = QWidget()
        root_container.setObjectName("root_container")
        self.setCentralWidget(root_container)

        root_layout = QHBoxLayout(root_container)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Left Sidebar Navigation
        self.sidebar_frame = QFrame()
        self.sidebar_frame.setObjectName("sidebar")
        self.sidebar_frame.setStyleSheet(
            f"background-color: {COLOR_BG_SIDEBAR}; border-right: 1px solid {COLOR_BORDER}; min-width: 190px; max-width: 220px;"
        )
        sidebar_layout = QVBoxLayout(self.sidebar_frame)
        sidebar_layout.setContentsMargins(10, 16, 10, 16)
        sidebar_layout.setSpacing(6)

        # Brand / App Header
        brand_label = QLabel("CLASHVISION")
        brand_label.setStyleSheet("font-size: 16px; font-weight: 900; color: #00d26a; letter-spacing: 2px; padding: 6px;")
        sidebar_layout.addWidget(brand_label)

        ver_label = QLabel("v0.1.0 • Read-Only Observation")
        ver_label.setStyleSheet("font-size: 10px; color: #747d8c; padding-left: 6px; margin-bottom: 12px;")
        sidebar_layout.addWidget(ver_label)

        # 9 View Selection Buttons
        self.nav_buttons: List[QPushButton] = []
        for idx, (title, tag) in enumerate(VIEW_TITLES):
            btn = QPushButton(title)
            btn.setProperty("class", "sidebar_nav_btn")
            btn.setObjectName(f"nav_btn_{tag}")
            btn.setStyleSheet(
                "QPushButton { text-align: left; padding: 10px 14px; border: none; border-radius: 6px; font-size: 13px; color: #a4b0be; font-weight: 500; } "
                "QPushButton:hover { background-color: #242834; color: #f1f2f6; }"
            )
            btn.clicked.connect(lambda checked=False, i=idx: self.switch_view(i))
            sidebar_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        sidebar_layout.addStretch()

        # Sidebar Footer: Safety Badge
        safety_badge = QLabel("🛡️ ZERO INJECTION")
        safety_badge.setStyleSheet(
            "background-color: #16241b; color: #2ed573; border: 1px solid #2ed573; "
            "border-radius: 4px; padding: 4px 8px; font-size: 10px; font-weight: bold; text-align: center;"
        )
        sidebar_layout.addWidget(safety_badge)

        root_layout.addWidget(self.sidebar_frame)

        # 2. Main Content Area (Header Status Bar + QStackedWidget)
        content_area = QWidget()
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Top Header Status Bar (Requirement R4 & R5)
        self.top_status_bar = QFrame()
        self.top_status_bar.setObjectName("top_status_bar")
        self.top_status_bar.setStyleSheet(
            f"background-color: #1a1c23; border-bottom: 1px solid {COLOR_BORDER}; padding: 8px 16px;"
        )
        top_bar_layout = QHBoxLayout(self.top_status_bar)
        top_bar_layout.setContentsMargins(12, 6, 12, 6)
        top_bar_layout.setSpacing(12)

        # Live vs Simulation Mode Indicator
        self.mode_indicator = QLabel("LIVE OBSERVATION (READ-ONLY)")
        self.mode_indicator.setObjectName("mode_indicator")
        self.mode_indicator.setStyleSheet(
            f"background-color: #142a1b; color: {COLOR_NEON_GREEN}; border: 1px solid {COLOR_NEON_GREEN}; "
            "border-radius: 4px; padding: 4px 12px; font-weight: bold; font-size: 11px;"
        )
        top_bar_layout.addWidget(self.mode_indicator)

        # Window Target Indicator
        self.target_window_indicator = QLabel("Target: Google Play Games PC (Clash of Clans)")
        self.target_window_indicator.setStyleSheet("color: #a4b0be; font-size: 12px; font-weight: 500;")
        top_bar_layout.addWidget(self.target_window_indicator)

        top_bar_layout.addStretch()

        # Emergency Stop Button (Accessible from every view)
        self.top_emergency_stop_btn = QPushButton("🛑 EMERGENCY STOP")
        self.top_emergency_stop_btn.setObjectName("top_emergency_stop_btn")
        self.top_emergency_stop_btn.setStyleSheet(
            f"background-color: {COLOR_NEON_RED}; color: #ffffff; border: 1px solid #ff6b81; "
            "border-radius: 5px; font-weight: bold; font-size: 11px; padding: 5px 14px;"
        )
        self.top_emergency_stop_btn.clicked.connect(self.trigger_emergency_stop)
        top_bar_layout.addWidget(self.top_emergency_stop_btn)

        content_layout.addWidget(self.top_status_bar)

        # Central QStackedWidget hosting 9 Views
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setObjectName("central_stacked_widget")

        # Instantiate 9 Views
        self.view_overview = OverviewView(parent=self)
        self.view_live_capture = LiveCaptureView(parent=self)
        self.view_base_analyzer = BaseAnalyzerView(parent=self)
        self.view_strategy = StrategyView(parent=self)
        self.view_simulator = SimulatorView(parent=self)
        self.view_analytics = AnalyticsView(repo=self.repo, parent=self)
        self.view_dataset = DatasetView(repo=self.repo, parent=self)
        self.view_settings = SettingsView(repo=self.repo, settings=self.settings, parent=self)
        self.view_logs = LogsView(repo=self.repo, parent=self)

        self.views = [
            self.view_overview,
            self.view_live_capture,
            self.view_base_analyzer,
            self.view_strategy,
            self.view_simulator,
            self.view_analytics,
            self.view_dataset,
            self.view_settings,
            self.view_logs,
        ]

        for v in self.views:
            self.stacked_widget.addWidget(v)

        content_layout.addWidget(self.stacked_widget, stretch=1)
        root_layout.addWidget(content_area, stretch=1)

        # Set default view to Overview (Index 0)
        self.switch_view(0)

    def _init_workers(self) -> None:
        """Initialize decoupled background worker threads and wire Qt signals."""
        self.capture_worker = CaptureWorker(fps=float(self.settings.capture.target_fps), parent=self)
        self.vision_worker = VisionWorker(parent=self)
        self.decision_worker = DecisionWorker(parent=self)
        self.simulation_worker = SimulationWorker(parent=self)

        # CaptureWorker -> LiveCaptureView & VisionWorker
        self.capture_worker.sig_frame_captured.connect(self._on_frame_captured)
        self.capture_worker.sig_error.connect(self._on_worker_error)

        # VisionWorker -> OverviewView, BaseAnalyzerView, DecisionWorker
        self.vision_worker.sig_state_detected.connect(self.view_overview.update_state)
        self.vision_worker.sig_ocr_parsed.connect(self.view_overview.update_resources)
        self.vision_worker.sig_analysis_complete.connect(self._on_vision_complete)
        self.vision_worker.sig_error.connect(self._on_worker_error)

        # DecisionWorker -> OverviewView, StrategyView, SimulatorWorker, Database
        self.decision_worker.sig_decision_ready.connect(self._on_decision_ready)
        self.decision_worker.sig_error.connect(self._on_worker_error)

        # SimulationWorker -> SimulatorView
        self.simulation_worker.sig_sim_tick.connect(self.view_simulator.update_simulation_state)
        self.simulation_worker.sig_error.connect(self._on_worker_error)

        # SimulatorView controls -> SimulationWorker
        self.view_simulator.sig_start.connect(self.simulation_worker.start_playback)
        self.view_simulator.sig_pause.connect(self.simulation_worker.pause_playback)
        self.view_simulator.sig_stop.connect(self.simulation_worker.stop_playback)
        self.view_simulator.sig_step_forward.connect(self.simulation_worker.step_forward)
        self.view_simulator.sig_reset.connect(self.simulation_worker.reset_playback)
        self.view_simulator.sig_speed_changed.connect(self.simulation_worker.set_speed)
        self.view_simulator.sig_emergency_stop.connect(self.trigger_emergency_stop)

    def start_all_workers(self) -> None:
        """Start asynchronous worker threads."""
        self.capture_worker.start()
        self.vision_worker.start()
        self.decision_worker.start()

    def stop_all_workers(self) -> None:
        """Gracefully halt all worker threads."""
        self.capture_worker.stop()
        self.vision_worker.stop()
        self.decision_worker.stop()
        self.simulation_worker.stop_worker()

    def switch_view(self, index: int) -> None:
        """Switch central view to the given index and update active sidebar button."""
        if 0 <= index < len(self.views):
            self.stacked_widget.setCurrentIndex(index)

            # Update sidebar active styling
            for i, btn in enumerate(self.nav_buttons):
                if i == index:
                    btn.setStyleSheet(
                        f"QPushButton {{ text-align: left; padding: 10px 14px; border: none; border-radius: 6px; "
                        f"font-size: 13px; color: {COLOR_NEON_GREEN}; font-weight: bold; background-color: #2b3040; "
                        f"border-left: 3px solid {COLOR_NEON_GREEN}; }}"
                    )
                else:
                    btn.setStyleSheet(
                        "QPushButton { text-align: left; padding: 10px 14px; border: none; border-radius: 6px; "
                        "font-size: 13px; color: #a4b0be; font-weight: 500; background-color: transparent; } "
                        "QPushButton:hover { background-color: #242834; color: #f1f2f6; }"
                    )

            # Update top status indicator depending on whether in Simulator view
            if index == 4:  # Simulator View
                self.mode_indicator.setText("LOCAL SIMULATION (OFFLINE REPLAY)")
                self.mode_indicator.setStyleSheet(
                    f"background-color: #2b173d; color: {COLOR_NEON_PURPLE}; border: 1px solid {COLOR_NEON_PURPLE}; "
                    "border-radius: 4px; padding: 4px 12px; font-weight: bold; font-size: 11px;"
                )
            else:
                self.mode_indicator.setText("LIVE OBSERVATION (READ-ONLY)")
                self.mode_indicator.setStyleSheet(
                    f"background-color: #142a1b; color: {COLOR_NEON_GREEN}; border: 1px solid {COLOR_NEON_GREEN}; "
                    "border-radius: 4px; padding: 4px 12px; font-weight: bold; font-size: 11px;"
                )

    def trigger_emergency_stop(self) -> None:
        """Immediate Emergency Stop halting simulation and logging event (<= 50ms)."""
        logger.warning("Emergency Stop triggered on MainWindow.")
        self.simulation_worker.stop_playback()
        self.view_overview.show_alert("EMERGENCY STOP", "Playback halted immediately.")
        try:
            self.view_logs.append_log_entry("WARNING", "Emergency Stop button pressed.", logger_name="MainWindow")
        except Exception as ex:
            logger.warning("Failed to persist emergency stop log to repository: %s", ex)

    def _on_frame_captured(self, packet: FramePacket) -> None:
        self.view_live_capture.update_frame(packet)
        self.vision_worker.enqueue_frame(packet)

    def _on_vision_complete(self, result: VisionAnalysisResult) -> None:
        self.view_base_analyzer.update_analysis(result)
        self.decision_worker.enqueue_analysis(result)

        # Record latency on AnalyticsView
        if result.evidence and "cv_latency_ms" in result.evidence:
            try:
                self.view_analytics.record_cv_latency(float(result.evidence["cv_latency_ms"]))
            except Exception:
                pass

        # Persist base scan record in SQLite
        res = result.resources
        try:
            raw_ocr = res.raw_text if res else {}
            if result.evidence and "cv_latency_ms" in result.evidence:
                raw_ocr = dict(raw_ocr)
                raw_ocr["cv_latency_ms"] = result.evidence["cv_latency_ms"]
            self.repo.save_base_scan(
                session_id=self._current_session_id,
                timestamp=result.timestamp,
                game_state=result.state,
                confidence=result.confidence,
                gold=res.gold if res else None,
                elixir=res.elixir if res else None,
                dark_elixir=res.dark_elixir if res else None,
                th_accessible=result.base_layout.th_accessible if result.base_layout else False,
                raw_ocr_json=raw_ocr,
            )
        except Exception as e:
            logger.error("Failed to persist base scan: %s", e)

    def _on_decision_ready(self, rec: FarmingRecommendation) -> None:
        self.view_overview.update_recommendation(rec)
        self.view_strategy.update_recommendation(rec)
        self.simulation_worker.load_recommendation(rec)

        # Persist decision in SQLite
        try:
            self.repo.save_recommendation(rec, session_id=self._current_session_id)
        except Exception as e:
            logger.error("Failed to persist decision recommendation: %s", e)

    def _on_worker_error(self, source: str, error_msg: str) -> None:
        try:
            self.view_logs.append_log_entry("ERROR", error_msg, logger_name=source)
        except Exception as ex:
            logger.warning("Failed to persist worker error log to repository: %s", ex)
        self.view_overview.show_alert(f"{source} Error", error_msg)

    def closeEvent(self, event: Any) -> None:
        """Handle window close event and terminate workers gracefully."""
        self.stop_all_workers()
        if self.repo:
            try:
                self.repo.end_session(self._current_session_id, notes="Clean shutdown")
            except Exception:
                pass
        super().closeEvent(event)
