"""
Integration tests for ClashVision Presentation Subsystem (Headless Mode).
Tests Qt compatibility shim, MainWindow layout, view switching across all 9 views,
individual view model updates, worker signal propagation, and emergency stop latency.
"""

from __future__ import annotations
import os
import sys
import time
from typing import Generator
import pytest
import numpy as np

# Ensure offscreen platform before importing Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from clashvision.core.constants import GameState, StrategyType, RiskLevel
from clashvision.core.models import (
    FramePacket,
    WindowInfo,
    Rect,
    ResourceReading,
    BaseLayoutFeatures,
    VisionAnalysisResult,
    FarmingRecommendation,
    StrategyEvaluation,
    DeploymentStep,
    SimulationState,
)
from clashvision.storage.repository import Repository
from clashvision.storage.db import DatabaseManager
from clashvision.ui.qt_compat import get_qapplication, IS_PYSIDE6, IS_PYQT6, IS_HEADLESS_FALLBACK
from clashvision.ui.main_window import MainWindow, VIEW_TITLES
from clashvision.ui.workers import (
    CaptureWorker,
    VisionWorker,
    DecisionWorker,
    SimulationWorker,
)


@pytest.fixture(scope="session")
def qapp():
    """Initializes and provides the offscreen QApplication instance."""
    app = get_qapplication()
    yield app


@pytest.fixture
def memory_repo() -> Generator[Repository, None, None]:
    """In-memory Repository for clean test state."""
    db = DatabaseManager(db_path=":memory:", auto_init=True)
    repo = Repository(db=db)
    yield repo
    db.close()


@pytest.fixture
def main_window(qapp, memory_repo: Repository) -> Generator[MainWindow, None, None]:
    """MainWindow instance wired to in-memory database."""
    win = MainWindow(repo=memory_repo, start_workers=False)
    yield win
    win.stop_all_workers()
    win.close()


class TestQtCompatAbstraction:
    """Verifies seamless import abstraction across PySide6/PyQt6/Mock."""

    def test_qt_binding_active_or_mock(self, qapp) -> None:
        assert qapp is not None
        assert (IS_PYQT6 or IS_PYSIDE6 or IS_HEADLESS_FALLBACK) is True

    def test_qapplication_singleton(self) -> None:
        app1 = get_qapplication()
        app2 = get_qapplication()
        assert app1 is app2


class TestMainWindowLayoutAndNavigation:
    """Verifies MainWindow shell layout, sidebar buttons, and view switching."""

    def test_main_window_has_9_views(self, main_window: MainWindow) -> None:
        assert main_window.stacked_widget.count() == 9
        assert len(main_window.views) == 9
        assert len(main_window.nav_buttons) == 9

    def test_view_switching_across_all_9_views(self, main_window: MainWindow) -> None:
        for idx in range(9):
            main_window.switch_view(idx)
            assert main_window.stacked_widget.currentIndex() == idx

            # When on Simulator view (idx 4), mode banner should indicate Local Simulation
            if idx == 4:
                assert "LOCAL SIMULATION" in main_window.mode_indicator.text()
            else:
                assert "LIVE OBSERVATION" in main_window.mode_indicator.text()

    def test_sidebar_button_labels_match_feature_spec(self, main_window: MainWindow) -> None:
        for idx, (title, tag) in enumerate(VIEW_TITLES):
            btn = main_window.nav_buttons[idx]
            assert tag in btn.objectName()


class TestViewModelUpdates:
    """Verifies that view models correctly render and update their components."""

    def test_overview_view_updates(self, main_window: MainWindow) -> None:
        ov = main_window.view_overview
        # Update State
        ov.update_state(GameState.ENEMY_BASE_PREVIEW, 0.98, {"button": "next_button"})
        assert ov.state_badge.text() == "ENEMY_BASE_PREVIEW"
        assert "98.0%" in ov.state_confidence_label.text()
        assert "TARGET IDENTIFIED" in ov.system_status_badge.text()

        # Update Resources
        reading = ResourceReading(
            gold=750000,
            elixir=600000,
            dark_elixir=4200,
            timestamp=time.time(),
        )
        ov.update_resources(reading)
        assert "750,000" in ov.gold_label.text()
        assert "600,000" in ov.elixir_label.text()
        assert "4,200" in ov.dark_elixir_label.text()

        # Update Recommendation
        rec = FarmingRecommendation(
            suitability_score=82.0,
            primary_strategy=StrategyType.SNEAKY_GOBLIN,
            rankings=[],
            explanation="Superb collector loot on base perimeter.",
        )
        ov.update_recommendation(rec)
        assert ov.rec_score_bar.value() == 82
        assert "Sneaky Goblin" in ov.rec_strategy_name.text()
        assert "HIGH VALUE TARGET" in ov.rec_viability_badge.text()

    def test_live_capture_view_frame_updates_and_pause(self, main_window: MainWindow) -> None:
        lcv = main_window.view_live_capture
        packet = FramePacket(
            frame=np.zeros((1080, 1920, 3), dtype=np.uint8),
            timestamp=time.time(),
            window_info=WindowInfo(client_rect=Rect(0, 0, 1920, 1080)),
            frame_id=42,
        )
        lcv.update_frame(packet)
        assert "1920x1080" in lcv.stats_label.text()
        assert "Frame #42" in lcv.feed_canvas.text()

        # Toggle Pause
        lcv._on_toggle_pause()
        assert lcv.is_paused is True
        assert "FEED PAUSED" in lcv.feed_status_badge.text()

        # When paused, frames should not overwrite canvas
        packet2 = FramePacket(
            frame=np.zeros((1080, 1920, 3), dtype=np.uint8),
            timestamp=time.time(),
            window_info=WindowInfo(client_rect=Rect(0, 0, 1920, 1080)),
            frame_id=99,
        )
        lcv.update_frame(packet2)
        assert "Frame #42" in lcv.feed_canvas.text()

        # Resume
        lcv._on_toggle_pause()
        assert lcv.is_paused is False

    def test_base_analyzer_view_updates(self, main_window: MainWindow) -> None:
        bav = main_window.view_base_analyzer
        analysis = VisionAnalysisResult(
            state=GameState.ENEMY_BASE_PREVIEW,
            confidence=0.92,
            base_layout=BaseLayoutFeatures(
                exposed_collector_ratio=0.75,
                th_accessible=True,
            ),
        )
        rec = FarmingRecommendation(
            suitability_score=88.0,
            primary_strategy=StrategyType.SNEAKY_GOBLIN,
            score_breakdown={"resource_score": 90.0, "layout_score": 85.0, "risk_penalty": 2.0},
        )
        bav.update_analysis(analysis, rec)
        assert bav.collector_split_bar.value() == 75
        assert "TH ACCESSIBLE" in bav.th_badge.text()
        assert "88.0 / 100" in bav.total_suitability_label.text()

    def test_strategy_view_comparison_and_deployment_table(self, main_window: MainWindow) -> None:
        sv = main_window.view_strategy
        rec = FarmingRecommendation(
            suitability_score=75.0,
            primary_strategy=StrategyType.BARCH,
            rankings=[
                StrategyEvaluation(
                    strategy_type=StrategyType.BARCH,
                    score=75.0,
                    viable=True,
                    risk_level=RiskLevel.LOW,
                    reasons=["Exposed perimeter collectors"],
                ),
                StrategyEvaluation(
                    strategy_type=StrategyType.MASS_DRAGON,
                    score=45.0,
                    viable=False,
                    risk_level=RiskLevel.HIGH,
                    reasons=["Centralized high level air defense"],
                ),
            ],
            deployment_plan=[
                DeploymentStep(step_id=1, troop_type="Barbarian", target_x=200, target_y=300, delay_ms=100, unit_count=10),
                DeploymentStep(step_id=2, troop_type="Archer", target_x=210, target_y=310, delay_ms=200, unit_count=15),
            ],
            explanation="BARCH perimeter sweep.",
        )
        sv.update_recommendation(rec)

        assert sv.strategy_table.rowCount() == 2
        assert sv.steps_table.rowCount() == 2
        assert "BARCH" in sv.troops_summary_label.text()
        assert sv.strategy_table.item(0, 2).text() == "VIABLE"
        assert sv.strategy_table.item(1, 2).text() == "UNVIABLE"

    def test_simulator_view_playback_and_speed_control(self, main_window: MainWindow) -> None:
        sim_view = main_window.view_simulator
        state = SimulationState(
            step_index=3,
            total_steps=10,
            is_playing=True,
            current_time_ms=1500,
            total_duration_ms=5000,
            active_troops=[{"id": 1}, {"id": 2}],
        )
        sim_view.update_simulation_state(state)
        assert "1.50s / 5.00s" in sim_view.time_label.text()
        assert "Step: 3 / 10" in sim_view.step_label.text()

        # Test speed slider
        sim_view.speed_slider.setValue(20)  # 2.0x
        assert sim_view.speed_label.text() == "2.0x"

    def test_analytics_view_refresh_with_repository_data(self, main_window: MainWindow, memory_repo: Repository) -> None:
        # Save dummy scan in repo
        memory_repo.create_session("test_sess", notes="Analytics test session")
        memory_repo.save_base_scan(
            session_id="test_sess",
            timestamp=time.time(),
            game_state=GameState.ENEMY_BASE_PREVIEW,
            confidence=0.99,
            gold=850000,
            elixir=800000,
            dark_elixir=5000,
        )
        av = main_window.view_analytics
        av.refresh_stats()
        assert av.scans_table.rowCount() >= 1

    def test_dataset_view_add_and_export(self, main_window: MainWindow, memory_repo: Repository, tmp_path) -> None:
        dv = main_window.view_dataset
        dv.image_path_input.setText(str(tmp_path / "test_frame.png"))
        dv.state_combo.setCurrentText(GameState.HOME_VILLAGE.value)
        dv._on_add_bounding_box()
        dv._on_save_annotation()

        assert dv.annotations_table.rowCount() >= 1
        annotations = memory_repo.list_annotations()
        assert len(annotations) >= 1

    def test_settings_view_persistence(self, main_window: MainWindow, memory_repo: Repository) -> None:
        sv = main_window.view_settings
        sv.spin_gold.setValue(600000)
        sv.spin_elixir.setValue(550000)
        sv.spin_fps.setValue(25)
        sv.save_settings()

        assert memory_repo.get_setting("min_gold") == "600000"
        assert memory_repo.get_setting("min_elixir") == "550000"
        assert memory_repo.get_setting("target_fps") == "25"

    def test_logs_view_live_event_and_filtering(self, main_window: MainWindow) -> None:
        lv = main_window.view_logs
        lv.append_log_entry("INFO", "System initialized", logger_name="init")
        lv.append_log_entry("ERROR", "Target window disconnected", logger_name="capture")

        assert lv.logs_table.rowCount() >= 2

        # Filter by ERROR
        lv.combo_level.setCurrentText("ERROR")
        lv.refresh_logs()
        # All rows should have level ERROR
        for r in range(lv.logs_table.rowCount()):
            assert lv.logs_table.item(r, 1).text() == "ERROR"


class TestWorkerSignalPropagation:
    """Verifies asynchronous worker signal contracts and event propagation."""

    def test_capture_worker_emits_frame_packet(self, qapp) -> None:
        worker = CaptureWorker(fps=30.0)
        captured = []

        def on_frame(packet: FramePacket) -> None:
            captured.append(packet)

        worker.sig_frame_captured.connect(on_frame)
        worker.start()
        for _ in range(15):
            time.sleep(0.02)
            qapp.processEvents()
        worker.stop()
        qapp.processEvents()

        assert len(captured) > 0
        assert captured[0].frame is not None
        assert captured[0].width == 1920
        assert captured[0].height == 1080

    def test_emergency_stop_halts_simulation_within_50ms(self, main_window: MainWindow) -> None:
        sim_worker = main_window.simulation_worker
        sim_worker.load_steps([
            DeploymentStep(step_id=1, troop_type="Barbarian", target_x=100, target_y=100, delay_ms=1000)
        ])
        sim_worker.start_playback()

        t0 = time.perf_counter()
        main_window.trigger_emergency_stop()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        assert latency_ms < 50.0, f"Emergency stop latency {latency_ms:.2f}ms exceeded 50ms limit"
        assert sim_worker.engine.is_stopped is True or sim_worker.engine.is_paused is True
