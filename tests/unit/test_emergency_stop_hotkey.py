"""
Tests for keyboard input and emergency stop hotkey integration across
AutonomousSimulatorView, MainWindow, and GlobalHotkeyListener.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from clashvision.capture.safety_guard import get_safety_guard
from clashvision.core.hotkey_listener import GlobalHotkeyListener, VK_ESCAPE, VK_F12
from clashvision.simulator.autonomous_engine import AutonomousEngine
from clashvision.simulator.autonomous_state_machine import FarmingState
from clashvision.simulator.safety_barrier import SafetyBarrier
from clashvision.ui.main_window import MainWindow
from clashvision.ui.qt_compat import (
    get_qapplication,
    Qt,
    QEvent,
)
from clashvision.ui.views.autonomous_simulator_view import AutonomousSimulatorView


class TestEmergencyStopHotkey(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_qapplication()

    def setUp(self) -> None:
        get_safety_guard().reset_emergency_stop()
        SafetyBarrier.get_instance().reset_emergency_stop()
        self.engine = AutonomousEngine()
        self.engine.barrier.reset_emergency_stop()

    def tearDown(self) -> None:
        get_safety_guard().reset_emergency_stop()
        SafetyBarrier.get_instance().reset_emergency_stop()

    def test_global_hotkey_listener_lifecycle(self) -> None:
        listener = GlobalHotkeyListener(poll_interval_s=0.01)
        callback_mock = MagicMock()
        listener.register_callback(callback_mock)

        # Starts and stops without error
        listener.start()
        self.assertTrue(listener._is_running)
        listener.stop()
        self.assertFalse(listener._is_running)

    def test_autonomous_simulator_view_shortcut_esc_triggers_emergency_stop(self) -> None:
        view = AutonomousSimulatorView(engine=self.engine)
        self.assertFalse(self.engine.barrier.is_emergency_stopped)

        # Simulate Escape shortcut activation
        view.shortcut_esc.activated.emit()

        self.assertTrue(self.engine.barrier.is_emergency_stopped)
        self.assertEqual(view.lbl_state.text(), "EMERGENCY STOPPED")

    def test_autonomous_simulator_view_shortcut_f12_triggers_emergency_stop(self) -> None:
        view = AutonomousSimulatorView(engine=self.engine)
        self.assertFalse(self.engine.barrier.is_emergency_stopped)

        # Simulate F12 shortcut activation
        view.shortcut_f12.activated.emit()

        self.assertTrue(self.engine.barrier.is_emergency_stopped)
        self.assertEqual(view.lbl_state.text(), "EMERGENCY STOPPED")

    def test_autonomous_simulator_view_keypress_event_triggers_emergency_stop(self) -> None:
        view = AutonomousSimulatorView(engine=self.engine)
        self.assertFalse(self.engine.barrier.is_emergency_stopped)

        # Create mock QKeyEvent with Key_Escape
        mock_event = MagicMock()
        mock_event.key.return_value = getattr(Qt, "Key_Escape", 0x01000000)

        view.keyPressEvent(mock_event)

        self.assertTrue(self.engine.barrier.is_emergency_stopped)
        self.assertEqual(view.lbl_state.text(), "EMERGENCY STOPPED")
        mock_event.accept.assert_called_once()

    def test_autonomous_simulator_view_start_shortcut_f5(self) -> None:
        view = AutonomousSimulatorView(engine=self.engine)
        self.engine.start = MagicMock()

        # Simulate F5 shortcut activation
        view.shortcut_start.activated.emit()
        self.engine.start.assert_called_once()

    def test_autonomous_simulator_view_pause_toggle_space(self) -> None:
        view = AutonomousSimulatorView(engine=self.engine)
        self.engine.start = MagicMock()
        self.engine.pause = MagicMock()
        self.engine.resume = MagicMock()

        # When in READY/STOPPED, Space starts
        view.shortcut_pause.activated.emit()
        self.engine.start.assert_called_once()

    def test_resolution_independent_button_click_mapping(self) -> None:
        from clashvision.core.constants import (
            SIMULATED_UI_BATTLE_BUTTON_NORM,
            SIMULATED_UI_FIND_MATCH_BUTTON_NORM,
            SIMULATED_UI_ATTACK_CONFIRM_BUTTON_NORM,
            SIMULATED_UI_NEXT_BUTTON_NORM,
            SIMULATED_UI_RETURN_HOME_BUTTON_NORM,
            SIMULATED_UI_BATTLE_BUTTON,
            SIMULATED_UI_FIND_MATCH_BUTTON,
            SIMULATED_UI_ATTACK_CONFIRM_BUTTON,
            SIMULATED_UI_NEXT_BUTTON,
            SIMULATED_UI_RETURN_HOME_BUTTON,
        )
        # Test custom canvas resolution (e.g. 1024x576)
        self.engine.executor.canvas_width = 1024
        self.engine.executor.canvas_height = 576
        self.engine.executor.controller.canvas_width = 1024
        self.engine.executor.controller.canvas_height = 576

        # Simulate clicks on each button
        self.engine._simulate_ui_click("Battle", SIMULATED_UI_BATTLE_BUTTON, SIMULATED_UI_BATTLE_BUTTON_NORM)
        self.engine._simulate_ui_click("Find a Match", SIMULATED_UI_FIND_MATCH_BUTTON, SIMULATED_UI_FIND_MATCH_BUTTON_NORM)
        self.engine._simulate_ui_click("Attack!", SIMULATED_UI_ATTACK_CONFIRM_BUTTON, SIMULATED_UI_ATTACK_CONFIRM_BUTTON_NORM)
        self.engine._simulate_ui_click("Next", SIMULATED_UI_NEXT_BUTTON, SIMULATED_UI_NEXT_BUTTON_NORM)
        self.engine._simulate_ui_click("Return Home", SIMULATED_UI_RETURN_HOME_BUTTON, SIMULATED_UI_RETURN_HOME_BUTTON_NORM)

        history = self.engine.executor.controller.action_history
        self.assertGreaterEqual(len(history), 5)
        # Verify all coordinates are within 1024x576 bounds
        for act in history:
            self.assertGreaterEqual(act.x, 0)
            self.assertLessEqual(act.x, 1024)
            self.assertGreaterEqual(act.y, 0)
            self.assertLessEqual(act.y, 576)

    def test_main_window_keyboard_shortcuts_and_auto_bot_emergency_stop(self) -> None:
        # Instantiate headless MainWindow with mock capture
        window = MainWindow(mock_capture=True)

        try:
            # Verify auto-simulator view is wired to MainWindow emergency stop
            auto_view = window.view_auto_simulator
            self.assertFalse(auto_view.engine.barrier.is_emergency_stopped)

            # 1. Trigger via Esc shortcut on MainWindow
            window.shortcut_esc.activated.emit()
            self.assertTrue(auto_view.engine.barrier.is_emergency_stopped)

            # Reset barrier
            auto_view.engine.barrier.reset_emergency_stop()
            self.assertFalse(auto_view.engine.barrier.is_emergency_stopped)

            # 2. Trigger via F12 shortcut on MainWindow
            window.shortcut_f12.activated.emit()
            self.assertTrue(auto_view.engine.barrier.is_emergency_stopped)

            # Reset barrier
            auto_view.engine.barrier.reset_emergency_stop()
            self.assertFalse(auto_view.engine.barrier.is_emergency_stopped)

            # 3. Trigger via keyPressEvent on MainWindow
            mock_event = MagicMock()
            mock_event.key.return_value = getattr(Qt, "Key_Escape", 0x01000000)
            window.keyPressEvent(mock_event)
            self.assertTrue(auto_view.engine.barrier.is_emergency_stopped)
            mock_event.accept.assert_called_once()
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
