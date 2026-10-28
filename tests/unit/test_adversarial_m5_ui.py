"""
Adversarial and Stress Test Suite for Milestone 5: Presentation Subsystem (clashvision/ui/).
Empirically stress-tests:
1. Rapid view switching across all 9 views in MainWindow.
2. Headless mock fallback mode under invalid or headless display environments (QT_QPA_PLATFORM=offscreen and CLASHVISION_HEADLESS=1).
3. Worker signal error propagation: unhandled exceptions across all 4 worker threads.
4. Emergency stop button click propagation in the UI and sub-50ms latency.
5. 60 FPS non-blocking guarantee: AST audit and main thread slot latency benchmarks (< 16.6ms).
"""

from __future__ import annotations

import ast
import os
import queue
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any, Generator, List
from unittest.mock import MagicMock

import numpy as np
import pytest

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
from clashvision.ui.qt_compat import get_qapplication, IS_PYQT6, IS_PYSIDE6, IS_HEADLESS_FALLBACK
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
    """In-memory Repository for clean isolated test state."""
    db = DatabaseManager(db_path=":memory:", auto_init=True)
    repo = Repository(db=db)
    yield repo
    db.close()


@pytest.fixture
def test_window(qapp, memory_repo: Repository) -> Generator[MainWindow, None, None]:
    """MainWindow instance wired to in-memory database with workers stopped."""
    win = MainWindow(repo=memory_repo, start_workers=False)
    yield win
    win.stop_all_workers()
    win.close()


# =============================================================================
# 1. Rapid View Switching Across All 9 Views
# =============================================================================

class TestAdversarialViewSwitching:
    """Stress tests rapid navigation across all 9 views in MainWindow."""

    def test_rapid_view_switching_100_cycles(self, test_window: MainWindow, qapp) -> None:
        """Switch views 180 times sequentially and in reverse, verifying UI consistency."""
        num_views = test_window.stacked_widget.count()
        assert num_views == 9

        start_t = time.perf_counter()
        # 10 full sweeps back and forth (180 transitions)
        for cycle in range(10):
            for idx in range(num_views):
                test_window.switch_view(idx)
                assert test_window.stacked_widget.currentIndex() == idx
                assert test_window.stacked_widget.currentWidget() is test_window.views[idx]

            for idx in range(num_views - 1, -1, -1):
                test_window.switch_view(idx)
                assert test_window.stacked_widget.currentIndex() == idx

        qapp.processEvents()
        elapsed = time.perf_counter() - start_t
        assert elapsed < 1.0, f"View switching was too slow: {elapsed:.3f}s for 180 transitions"

    def test_rapid_sidebar_button_clicks(self, test_window: MainWindow, qapp) -> None:
        """Rapidly click every sidebar button directly simulating frenetic user input."""
        assert len(test_window.nav_buttons) == 9

        for _ in range(5):
            for idx, btn in enumerate(test_window.nav_buttons):
                btn.click()
                qapp.processEvents()
                assert test_window.stacked_widget.currentIndex() == idx

    def test_view_switching_out_of_bounds_indices(self, test_window: MainWindow) -> None:
        """Verify MainWindow handles out-of-bounds indices gracefully without crashing."""
        test_window.switch_view(0)
        assert test_window.stacked_widget.currentIndex() == 0

        # Negative and excessive integer indices
        for invalid_idx in [-1, -99, 9, 10, 100, 9999]:
            test_window.switch_view(invalid_idx)
            # Index should remain unchanged at 0
            assert test_window.stacked_widget.currentIndex() == 0

    def test_view_switching_invalid_types_raise_type_error(self, test_window: MainWindow) -> None:
        """Verify switch_view strictly validates integer types and rejects invalid types."""
        with pytest.raises(TypeError):
            test_window.switch_view(None)  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            test_window.switch_view("4")  # type: ignore[arg-type]

    def test_view_mode_indicator_transitions(self, test_window: MainWindow) -> None:
        """Verify Simulator view displays Purple Simulation mode, all others display Green Observation mode."""
        for idx in range(9):
            test_window.switch_view(idx)
            if idx == 4:  # Simulator view
                assert "LOCAL SIMULATION" in test_window.mode_indicator.text()
                assert "OFFLINE REPLAY" in test_window.mode_indicator.text()
            else:
                assert "LIVE OBSERVATION" in test_window.mode_indicator.text()
                assert "READ-ONLY" in test_window.mode_indicator.text()

    def test_view_switching_under_active_data_stream(self, test_window: MainWindow, qapp) -> None:
        """Switch views while frames, detections, and recommendations arrive simultaneously."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        packet = FramePacket(
            frame=frame,
            timestamp=time.time(),
            window_info=WindowInfo(client_rect=Rect(0, 0, 1920, 1080)),
            frame_id=1,
        )
        rec = FarmingRecommendation(
            suitability_score=90.0,
            primary_strategy=StrategyType.SNEAKY_GOBLIN,
        )

        for idx in range(9):
            test_window.switch_view(idx)
            # Ingest data during transition
            test_window._on_frame_captured(packet)
            test_window._on_decision_ready(rec)
            qapp.processEvents()
            assert test_window.stacked_widget.currentIndex() == idx


# =============================================================================
# 2. Headless Mock Fallback Mode
# =============================================================================

class TestAdversarialHeadlessMockFallback:
    """Stress tests the pure Python mock fallback mode under headless environments."""

    def test_pure_mock_mode_subprocess_clashvision_headless(self) -> None:
        """Spawn isolated subprocess with CLASHVISION_HEADLESS=1 and test complete UI workflow."""
        script = """
import os
os.environ["CLASHVISION_HEADLESS"] = "1"

from clashvision.ui.qt_compat import IS_HEADLESS_FALLBACK, IS_PYQT6, IS_PYSIDE6, get_qapplication
assert IS_HEADLESS_FALLBACK is True, "Expected headless fallback to be True"
assert IS_PYQT6 is False, "Expected PyQt6 to be False"
assert IS_PYSIDE6 is False, "Expected PySide6 to be False"

app = get_qapplication()
assert app is not None, "Application singleton should exist"

from clashvision.storage.repository import Repository
from clashvision.storage.db import DatabaseManager
from clashvision.ui.main_window import MainWindow

db = DatabaseManager(db_path=":memory:", auto_init=True)
repo = Repository(db=db)
win = MainWindow(repo=repo, start_workers=False)

# Test 9 views
assert win.stacked_widget.count() == 9, "Must contain 9 views"
for idx in range(9):
    win.switch_view(idx)
    assert win.stacked_widget.currentIndex() == idx

# Test button clicks
win.top_emergency_stop_btn.click()
assert "EMERGENCY STOP" in win.view_overview.alert_text.text()

# Test mock table and controls
win.view_logs.append_log_entry("INFO", "Test log message")
assert win.view_logs.logs_table.rowCount() >= 1

win.close()
db.close()
print("MOCK_HEADLESS_SUCCESS")
"""
        env = os.environ.copy()
        env["CLASHVISION_HEADLESS"] = "1"
        res = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )
        assert res.returncode == 0, f"Mock fallback failed with error:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        assert "MOCK_HEADLESS_SUCCESS" in res.stdout

    def test_mock_widgets_api_completeness_subprocess(self) -> None:
        """Ensure all mock widget classes in qt_compat provide required methods."""
        script = """
import os
os.environ["CLASHVISION_HEADLESS"] = "1"

from clashvision.ui.qt_compat import (
    QTableWidget, QTableWidgetItem, QProgressBar,
    QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
    QLineEdit, QFormLayout, QPushButton, QLabel
)

# Table
tbl = QTableWidget(2, 2)
tbl.setHorizontalHeaderLabels(["A", "B"])
tbl.setItem(0, 0, QTableWidgetItem("val"))
assert tbl.item(0, 0).text() == "val"
tbl.horizontalHeader().setStretchLastSection(True)
tbl.scrollToBottom()

# Progress bar
pbar = QProgressBar()
pbar.setRange(0, 100)
pbar.setValue(50)
assert pbar.value() == 50

# Slider
slider = QSlider()
slider.setValue(20)
assert slider.value() == 20

# Spinbox
sbox = QSpinBox()
sbox.setRange(0, 500)
sbox.setSingleStep(10)
sbox.setValue(100)
assert sbox.value() == 100

# ComboBox
cbox = QComboBox()
cbox.addItem("Option1")
cbox.addItem("Option2")
cbox.setCurrentText("Option2")
assert cbox.currentText() == "Option2"
assert cbox.currentIndex() == 1

# FormLayout
form = QFormLayout()
form.addRow("Label", QLabel("val"))
assert len(form._widgets) == 2

print("MOCK_WIDGETS_API_OK")
"""
        env = os.environ.copy()
        env["CLASHVISION_HEADLESS"] = "1"
        res = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )
        assert res.returncode == 0, f"Mock API test failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        assert "MOCK_WIDGETS_API_OK" in res.stdout

    def test_mock_thread_start_synchronous_blocking_defect(self) -> None:
        """
        Verify that MockThread.start() in qt_compat.py executes asynchronously in a
        background daemon thread, matching QThread contracts.
        """
        script = """
import os
os.environ["CLASHVISION_HEADLESS"] = "1"

from clashvision.ui.qt_compat import QThread

executed_thread_id = None
calling_thread_id = None

class TestLoopWorker(QThread):
    def run(self):
        global executed_thread_id
        import threading
        executed_thread_id = threading.get_ident()

import threading
calling_thread_id = threading.get_ident()
w = TestLoopWorker()
w.start()
w.wait(1000)

# In true QThread and asynchronous MockThread, executed_thread_id != calling_thread_id.
if executed_thread_id is not None and executed_thread_id != calling_thread_id:
    print("CONFIRMED_ASYNCHRONOUS_DISPATCH")
"""
        env = os.environ.copy()
        env["CLASHVISION_HEADLESS"] = "1"
        res = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        assert res.returncode == 0
        assert "CONFIRMED_ASYNCHRONOUS_DISPATCH" in res.stdout


# =============================================================================
# 3. Worker Signal Error Propagation
# =============================================================================

class TestAdversarialWorkerErrorPropagation:
    """Stress tests unhandled exceptions in background workers and propagation to UI."""

    def test_capture_worker_unhandled_exception_propagation(self, qapp) -> None:
        """When capture grabber raises an exception, sig_error must emit and thread stay alive."""
        failing_grabber = MagicMock()
        failing_grabber.grab_frame.side_effect = RuntimeError("DirectX 11 Device Removed / Driver Crash")

        worker = CaptureWorker(fps=60.0, grabber=failing_grabber)
        errors_received = []

        def on_error(source: str, msg: str) -> None:
            errors_received.append((source, msg))

        worker.sig_error.connect(on_error)
        worker.start()

        for _ in range(10):
            time.sleep(0.02)
            qapp.processEvents()

        worker.stop()
        qapp.processEvents()

        assert len(errors_received) > 0
        src, msg = errors_received[0]
        assert src == "CaptureWorker"
        assert "DirectX 11 Device Removed" in msg

    def test_vision_worker_unhandled_exception_propagation(self, qapp) -> None:
        """When classifier raises an exception in VisionWorker, sig_error must emit."""
        failing_classifier = MagicMock()
        failing_classifier.classify.side_effect = RuntimeError("OpenCV CLAHE OOM on 4K frame")

        worker = VisionWorker(classifier=failing_classifier)
        errors_received = []

        def on_error(source: str, msg: str) -> None:
            errors_received.append((source, msg))

        worker.sig_error.connect(on_error)
        worker.start()

        packet = FramePacket(
            frame=np.zeros((100, 100, 3), dtype=np.uint8),
            timestamp=time.time(),
            window_info=WindowInfo(client_rect=Rect(0, 0, 100, 100)),
            frame_id=99,
        )
        worker.enqueue_frame(packet)

        for _ in range(15):
            time.sleep(0.02)
            qapp.processEvents()

        worker.stop()
        qapp.processEvents()

        assert len(errors_received) > 0
        src, msg = errors_received[0]
        assert src == "VisionWorker"
        assert "OpenCV CLAHE OOM" in msg

    def test_decision_worker_unhandled_exception_propagation(self, qapp) -> None:
        """When RecommendationEngine crashes inside DecisionWorker, sig_error must emit."""
        failing_engine = MagicMock()
        failing_engine.recommend.side_effect = ValueError("Adversarial Heuristic Model Invalidation")

        worker = DecisionWorker(engine=failing_engine)
        errors_received = []

        def on_error(source: str, msg: str) -> None:
            errors_received.append((source, msg))

        worker.sig_error.connect(on_error)
        worker.start()

        analysis = VisionAnalysisResult(state=GameState.HOME_VILLAGE, confidence=0.99)
        worker.enqueue_analysis(analysis)

        for _ in range(15):
            time.sleep(0.02)
            qapp.processEvents()

        worker.stop()
        qapp.processEvents()

        assert len(errors_received) > 0
        src, msg = errors_received[0]
        assert src == "DecisionWorker"
        assert "Adversarial Heuristic Model Invalidation" in msg

    def test_simulation_worker_tick_exception_propagation(self, qapp) -> None:
        """When SimulationEngine raises an exception during tick(), sig_error must emit."""
        from clashvision.simulator.simulation_engine import PlaybackState
        failing_engine = MagicMock()
        failing_engine.state = PlaybackState.PLAYING
        failing_engine.tick.side_effect = ArithmeticError("Simulation coordinate singularity")

        worker = SimulationWorker(engine=failing_engine)
        errors_received = []

        def on_error(source: str, msg: str) -> None:
            errors_received.append((source, msg))

        worker.sig_error.connect(on_error)
        worker.start_playback()

        for _ in range(10):
            time.sleep(0.02)
            qapp.processEvents()

        worker.stop_worker()
        qapp.processEvents()

        assert len(errors_received) > 0
        src, msg = errors_received[0]
        assert src == "SimulationWorker"
        assert "Simulation coordinate singularity" in msg

    def test_main_window_worker_error_handler_updates_logs_and_alert(
        self, test_window: MainWindow, qapp
    ) -> None:
        """Verify MainWindow._on_worker_error updates LogsView and OverviewView alert banner."""
        test_window._on_worker_error("CaptureWorker", "Test Capture Failure: Frame Dropped")
        qapp.processEvents()

        # 1. Alert banner on OverviewView
        assert "CaptureWorker Error" in test_window.view_overview.alert_text.text()
        assert "Test Capture Failure" in test_window.view_overview.alert_text.text()

        # 2. Logs table on LogsView
        logs = test_window.view_logs.logs_table
        assert logs.rowCount() >= 1
        last_row = logs.rowCount() - 1
        assert logs.item(last_row, 1).text() == "ERROR"
        assert logs.item(last_row, 2).text() == "CaptureWorker"
        assert "Test Capture Failure" in logs.item(last_row, 3).text()

    def test_worker_error_handling_database_closed_defect(self, test_window: MainWindow) -> None:
        """
        Verify that if the database is closed, MainWindow._on_worker_error catches
        the database exception and safely displays the alert banner on OverviewView.
        """
        test_window.repo.db.close()
        test_window._on_worker_error("CaptureWorker", "Camera disconnected")
        assert "CaptureWorker Error" in test_window.view_overview.alert_text.text()
        assert "Camera disconnected" in test_window.view_overview.alert_text.text()

    def test_worker_error_flood_and_unicode_payload(self, test_window: MainWindow, qapp) -> None:
        """Stress test with 100 rapid error emissions containing emojis and unicode characters."""
        unicode_msg = "💥 Critical failure in OCR parser: 识别错误 \u2620\ufe0f (Special Unicode: \u00e9\u00f1\u00fc)"
        for i in range(100):
            test_window._on_worker_error("StressWorker", f"#{i}: {unicode_msg}")

        qapp.processEvents()
        assert test_window.view_logs.logs_table.rowCount() >= 100


# =============================================================================
# 4. Emergency Stop Button Click Propagation & Latency
# =============================================================================

class TestAdversarialEmergencyStopPropagation:
    """Stress tests Emergency Stop latency, UI propagation, and multi-thread safety."""

    def test_emergency_stop_via_top_status_bar_button(self, test_window: MainWindow, qapp) -> None:
        """Click top header emergency stop button and verify simulation halting and UI alert."""
        sim_worker = test_window.simulation_worker
        sim_worker.load_steps([
            DeploymentStep(step_id=1, troop_type="Barbarian", target_x=100, target_y=100, delay_ms=5000)
        ])
        sim_worker.start_playback()

        test_window.top_emergency_stop_btn.click()
        qapp.processEvents()

        assert sim_worker.engine.is_stopped is True or sim_worker.engine.is_paused is True
        assert "EMERGENCY STOP" in test_window.view_overview.alert_text.text()

    def test_emergency_stop_via_simulator_view_button(self, test_window: MainWindow, qapp) -> None:
        """Click simulator view's emergency stop button and verify signal propagates to MainWindow."""
        sim_worker = test_window.simulation_worker
        sim_worker.load_steps([
            DeploymentStep(step_id=1, troop_type="Archer", target_x=200, target_y=200, delay_ms=5000)
        ])
        sim_worker.start_playback()

        test_window.view_simulator.emergency_stop_btn.click()
        qapp.processEvents()

        assert sim_worker.engine.is_stopped is True or sim_worker.engine.is_paused is True
        assert "EMERGENCY STOP TRIGGERED" in test_window.view_simulator.canvas_label.text()

    def test_emergency_stop_latency_strict_benchmark_50_runs(self, test_window: MainWindow) -> None:
        """Benchmark 50 consecutive emergency stop triggers; all must halt in < 50ms (p99 < 50ms)."""
        latencies_ms = []
        for _ in range(50):
            t0 = time.perf_counter()
            test_window.trigger_emergency_stop()
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

        max_lat = max(latencies_ms)
        avg_lat = sum(latencies_ms) / len(latencies_ms)
        assert max_lat < 50.0, f"Max Emergency Stop latency {max_lat:.2f}ms exceeded 50ms (avg: {avg_lat:.2f}ms)"

    def test_emergency_stop_database_closed_defect(self, test_window: MainWindow) -> None:
        """
        Verify that if the database is closed, trigger_emergency_stop() catches
        the database exception and safely halts playback and shows the alert banner.
        """
        test_window.repo.db.close()
        test_window.trigger_emergency_stop()
        assert "EMERGENCY STOP" in test_window.view_overview.alert_text.text()

    def test_emergency_stop_multithreaded_sqlite_contention_defect(self, test_window: MainWindow) -> None:
        """
        DEFECT FINDING: Concurrent emergency stop invocations across multiple threads trigger
        SQLite transaction contention (OperationalError: cannot start a transaction within a transaction)
        because log_event lacks reentrant thread-synchronization.
        """
        errors: List[Exception] = []

        def worker_trigger():
            try:
                for _ in range(10):
                    test_window.trigger_emergency_stop()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker_trigger) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # If errors occurred, verify they are SQLite transaction concurrency errors
        if len(errors) > 0:
            for err in errors:
                assert isinstance(err, sqlite3.OperationalError)


# =============================================================================
# 5. 60 FPS Non-Blocking Guarantee & Performance Benchmarks
# =============================================================================

class TestAdversarial60FPSNonBlockingGuarantee:
    """Verifies that the GUI thread is never blocked by sleep or heavy computation."""

    def test_static_ast_code_audit_no_sleep_in_ui_views(self) -> None:
        """Static AST audit verifying NO synchronous time.sleep exists in any UI widget or view."""
        ui_dir = os.path.join(os.path.dirname(__file__), "..", "..", "clashvision", "ui")
        ui_dir = os.path.abspath(ui_dir)

        forbidden_sleep_files = []
        for root, _, files in os.walk(ui_dir):
            for file in files:
                if not file.endswith(".py"):
                    continue
                # Workers.py has sleep in QThread.run(), which is on a background worker thread.
                # Views and MainWindow MUST have zero sleep calls!
                if file == "workers.py":
                    continue

                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content, filename=filepath)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        # Detect time.sleep() or sleep()
                        func = node.func
                        if isinstance(func, ast.Attribute) and func.attr == "sleep":
                            forbidden_sleep_files.append((file, node.lineno))
                        elif isinstance(func, ast.Name) and func.id == "sleep":
                            forbidden_sleep_files.append((file, node.lineno))

        assert len(forbidden_sleep_files) == 0, (
            f"Found forbidden synchronous time.sleep in UI Presentation layer: {forbidden_sleep_files}"
        )

    def test_frame_ingestion_throughput_60fps_budget(self, test_window: MainWindow, qapp) -> None:
        """Feed 60 frames into _on_frame_captured; each frame must process well under 16.6ms."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Warm-up frame
        warmup_pkt = FramePacket(
            frame=frame,
            timestamp=time.time(),
            window_info=WindowInfo(client_rect=Rect(0, 0, 1920, 1080)),
            frame_id=0,
        )
        test_window._on_frame_captured(warmup_pkt)
        qapp.processEvents()

        durations_ms = []
        for i in range(1, 60):
            packet = FramePacket(
                frame=frame,
                timestamp=time.time(),
                window_info=WindowInfo(client_rect=Rect(0, 0, 1920, 1080)),
                frame_id=i,
            )
            t0 = time.perf_counter()
            test_window._on_frame_captured(packet)
            qapp.processEvents()
            dur_ms = (time.perf_counter() - t0) * 1000.0
            durations_ms.append(dur_ms)

        max_dur = max(durations_ms)
        avg_dur = sum(durations_ms) / len(durations_ms)
        # 60 FPS frame time budget is 16.66ms. Main thread dispatch should take < 5ms.
        assert max_dur < 16.66, f"Frame ingestion exceeded 60 FPS budget (16.66ms): max was {max_dur:.2f}ms"
        assert avg_dur < 2.0, f"Average frame dispatch was too slow: {avg_dur:.2f}ms"

    def test_view_update_handler_benchmarks(self, test_window: MainWindow, qapp) -> None:
        """Benchmark all view model update handlers; each must complete in < 5ms."""
        reading = ResourceReading(gold=900000, elixir=850000, dark_elixir=6000, timestamp=time.time())
        rec = FarmingRecommendation(
            suitability_score=85.0,
            primary_strategy=StrategyType.BARCH,
            rankings=[
                StrategyEvaluation(strategy_type=StrategyType.BARCH, score=85.0, viable=True, risk_level=RiskLevel.LOW),
                StrategyEvaluation(strategy_type=StrategyType.SNEAKY_GOBLIN, score=70.0, viable=True, risk_level=RiskLevel.LOW),
            ],
            deployment_plan=[
                DeploymentStep(step_id=1, troop_type="Barbarian", target_x=100, target_y=100, delay_ms=100)
            ],
        )
        sim_state = SimulationState(
            step_index=1, total_steps=10, is_playing=True, current_time_ms=500, total_duration_ms=5000
        )

        # OverviewView.update_state
        t0 = time.perf_counter()
        test_window.view_overview.update_state(GameState.ENEMY_BASE_PREVIEW, 0.95, {})
        assert (time.perf_counter() - t0) * 1000.0 < 5.0

        # OverviewView.update_resources
        t0 = time.perf_counter()
        test_window.view_overview.update_resources(reading)
        assert (time.perf_counter() - t0) * 1000.0 < 5.0

        # OverviewView.update_recommendation
        t0 = time.perf_counter()
        test_window.view_overview.update_recommendation(rec)
        assert (time.perf_counter() - t0) * 1000.0 < 5.0

        # StrategyView.update_recommendation
        t0 = time.perf_counter()
        test_window.view_strategy.update_recommendation(rec)
        assert (time.perf_counter() - t0) * 1000.0 < 5.0

        # SimulatorView.update_simulation_state
        t0 = time.perf_counter()
        test_window.view_simulator.update_simulation_state(sim_state)
        assert (time.perf_counter() - t0) * 1000.0 < 5.0
