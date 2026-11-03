"""
ClashVision Presentation Subsystem — View 10: Autonomous Simulator Dashboard.
Full-cycle autonomous farming dashboard providing real-time state visualization,
simulated execution controls, loot harvest metrics, action execution logs,
and configurable farming thresholds.
Derived from Master Prompt Section 9, 10.
"""

from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional

from clashvision.core.constants import StrategyType
from clashvision.simulator.autonomous_state_machine import FarmingState
from clashvision.simulator.target_selector import TargetFilterConfig, FilterMode
from clashvision.simulator.autonomous_engine import (
    AutonomousEngine,
    AutonomousEngineConfig,
    AutonomousSessionStats,
)
from clashvision.ui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QFrame, QPushButton, QProgressBar, QSpinBox, QComboBox,
    QSlider, QTextEdit, QScrollArea, Qt, Signal, Slot,
    QShortcut, QKeySequence
)
from clashvision.ui.styles import (
    COLOR_NEON_GREEN, COLOR_NEON_PURPLE, COLOR_NEON_RED,
    COLOR_NEON_GOLD, COLOR_NEON_CYAN, COLOR_BORDER, COLOR_BG_CARD,
    COLOR_BG_WINDOW
)

logger = logging.getLogger("clashvision.ui.views.autonomous_simulator")


class AutonomousSimulatorView(QWidget):
    """
    Dedicated Graphical Dashboard for the Autonomous Farming Engine.
    Controls simulation execution, monitors state transitions, and displays
    real-time farming analytics.
    """

    sig_emergency_stop = Signal()

    def __init__(
        self,
        engine: Optional[AutonomousEngine] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("autonomous_simulator_view")
        self.engine = engine or AutonomousEngine()
        self._init_ui()
        self._wire_engine_callbacks()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        # ----------------------------------------------------------------------
        # 1. Header & Emergency Stop Bar
        # ----------------------------------------------------------------------
        header_bar = QHBoxLayout()
        header_title = QLabel("🤖 Autonomous Farming Simulator (Local Sandbox)")
        header_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #f1f2f6;")
        header_bar.addWidget(header_title)

        header_bar.addStretch()

        self.badge_mode = QLabel("OFFLINE SIMULATOR (ZERO REAL INPUT)")
        self.badge_mode.setStyleSheet(
            f"background-color: #2b173d; color: {COLOR_NEON_PURPLE}; border: 1px solid {COLOR_NEON_PURPLE}; "
            "border-radius: 4px; padding: 4px 10px; font-weight: bold; font-size: 11px;"
        )
        header_bar.addWidget(self.badge_mode)

        self.btn_emergency_stop = QPushButton("🛑 EMERGENCY STOP (Esc / F12)")
        self.btn_emergency_stop.setToolTip("Immediate emergency stop for autonomous farming loop (Hotkey: Escape or F12)")
        self.btn_emergency_stop.setStyleSheet(
            f"background-color: #b71540; color: #ffffff; border: 1px solid {COLOR_NEON_RED}; "
            "border-radius: 4px; padding: 6px 16px; font-weight: bold; font-size: 12px;"
        )
        self.btn_emergency_stop.clicked.connect(self._on_emergency_stop)
        header_bar.addWidget(self.btn_emergency_stop)

        # Keyboard shortcuts for Autonomous Simulator
        self.shortcut_start = QShortcut(QKeySequence("F5"), self)
        self.shortcut_start.activated.connect(self._on_start)

        self.shortcut_stop = QShortcut(QKeySequence("Shift+F5"), self)
        self.shortcut_stop.activated.connect(self._on_stop)

        self.shortcut_pause = QShortcut(QKeySequence("Space"), self)
        self.shortcut_pause.activated.connect(self._toggle_pause_resume)

        self.shortcut_esc = QShortcut(QKeySequence("Escape"), self)
        self.shortcut_esc.activated.connect(self._on_emergency_stop)

        self.shortcut_f12 = QShortcut(QKeySequence("F12"), self)
        self.shortcut_f12.activated.connect(self._on_emergency_stop)

        main_layout.addLayout(header_bar)

        # ----------------------------------------------------------------------
        # 2. Control Toolbar
        # ----------------------------------------------------------------------
        controls_card = QFrame()
        controls_card.setStyleSheet(f"background-color: {COLOR_BG_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 10px;")
        controls_layout = QHBoxLayout(controls_card)

        self.btn_start = QPushButton("▶ Start Simulation (F5)")
        self.btn_start.setToolTip("Start autonomous farming engine (Hotkey: F5)")
        self.btn_start.setStyleSheet(f"background-color: #142a1b; color: {COLOR_NEON_GREEN}; border: 1px solid {COLOR_NEON_GREEN}; font-weight: bold; padding: 6px 14px;")
        self.btn_start.clicked.connect(self._on_start)
        controls_layout.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸ Pause (Space)")
        self.btn_pause.setToolTip("Pause farming cycle (Hotkey: Space)")
        self.btn_pause.setStyleSheet("padding: 6px 14px;")
        self.btn_pause.clicked.connect(self._on_pause)
        controls_layout.addWidget(self.btn_pause)

        self.btn_resume = QPushButton("▶ Resume (Space)")
        self.btn_resume.setToolTip("Resume farming cycle (Hotkey: Space)")
        self.btn_resume.setStyleSheet("padding: 6px 14px;")
        self.btn_resume.clicked.connect(self._on_resume)
        controls_layout.addWidget(self.btn_resume)

        self.btn_stop = QPushButton("⏹ Stop (Shift+F5)")
        self.btn_stop.setToolTip("Stop farming simulation cleanly (Hotkey: Shift+F5)")
        self.btn_stop.setStyleSheet("padding: 6px 14px;")
        self.btn_stop.clicked.connect(self._on_stop)
        controls_layout.addWidget(self.btn_stop)

        controls_layout.addSpacing(15)

        # Keyboard shortcuts visual badge
        self.lbl_shortcuts_info = QLabel("⌨ Hotkeys: Start [F5]  •  Pause/Resume [Space]  •  Stop [Shift+F5]  •  Emergency Stop [Esc / F12]")
        self.lbl_shortcuts_info.setToolTip("Keyboard shortcuts available to control simulation execution")
        self.lbl_shortcuts_info.setStyleSheet(
            "background-color: #1e222d; color: #00d2d3; border: 1px solid #2e3548; "
            "border-radius: 4px; padding: 5px 12px; font-size: 11px; font-weight: bold;"
        )
        controls_layout.addWidget(self.lbl_shortcuts_info)

        controls_layout.addSpacing(15)

        # Speed Multiplier
        controls_layout.addWidget(QLabel("Speed:"))
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.5x", "1.0x", "2.0x", "4.0x", "10.0x"])
        self.combo_speed.setCurrentText("2.0x")
        self.combo_speed.currentTextChanged.connect(self._on_speed_changed)
        controls_layout.addWidget(self.combo_speed)

        controls_layout.addStretch()
        main_layout.addWidget(controls_card)

        # ----------------------------------------------------------------------
        # 3. Real-Time Status & Metrics Grid
        # ----------------------------------------------------------------------
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(10)

        # State Card
        self.card_state = QFrame()
        self.card_state.setStyleSheet(f"background-color: {COLOR_BG_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 12px;")
        layout_state = QVBoxLayout(self.card_state)
        layout_state.addWidget(QLabel("CURRENT SIMULATION STATE"))
        self.lbl_state = QLabel("STOPPED")
        self.lbl_state.setStyleSheet("font-size: 16px; font-weight: 900; color: #a4b0be;")
        layout_state.addWidget(self.lbl_state)
        metrics_grid.addWidget(self.card_state, 0, 0)

        # Strategy Card
        self.card_strat = QFrame()
        self.card_strat.setStyleSheet(f"background-color: {COLOR_BG_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 12px;")
        layout_strat = QVBoxLayout(self.card_strat)
        layout_strat.addWidget(QLabel("SELECTED STRATEGY"))
        self.lbl_strat = QLabel("Awaiting target...")
        self.lbl_strat.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {COLOR_NEON_CYAN};")
        layout_strat.addWidget(self.lbl_strat)
        metrics_grid.addWidget(self.card_strat, 0, 1)

        # Cycles Card
        self.card_cycles = QFrame()
        self.card_cycles.setStyleSheet(f"background-color: {COLOR_BG_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 12px;")
        layout_cycles = QVBoxLayout(self.card_cycles)
        layout_cycles.addWidget(QLabel("COMPLETED CYCLES"))
        self.lbl_cycles = QLabel("0 / 10")
        self.lbl_cycles.setStyleSheet("font-size: 16px; font-weight: bold; color: #f1f2f6;")
        layout_cycles.addWidget(self.lbl_cycles)
        metrics_grid.addWidget(self.card_cycles, 0, 2)

        # Skipped Bases Card
        self.card_skipped = QFrame()
        self.card_skipped.setStyleSheet(f"background-color: {COLOR_BG_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 12px;")
        layout_skipped = QVBoxLayout(self.card_skipped)
        layout_skipped.addWidget(QLabel("SKIPPED BASES"))
        self.lbl_skipped = QLabel("0 targets")
        self.lbl_skipped.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffa502;")
        layout_skipped.addWidget(self.lbl_skipped)
        metrics_grid.addWidget(self.card_skipped, 0, 3)

        # Loot Collected Card (Spanning 4 columns)
        self.card_loot = QFrame()
        self.card_loot.setStyleSheet(f"background-color: {COLOR_BG_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 12px;")
        layout_loot = QHBoxLayout(self.card_loot)

        layout_loot.addWidget(QLabel("HARVESTED LOOT:"))
        self.lbl_gold = QLabel("Gold: 0")
        self.lbl_gold.setStyleSheet(f"font-weight: bold; color: {COLOR_NEON_GOLD}; font-size: 13px;")
        layout_loot.addWidget(self.lbl_gold)

        self.lbl_elixir = QLabel("Elixir: 0")
        self.lbl_elixir.setStyleSheet("font-weight: bold; color: #ff3838; font-size: 13px;")
        layout_loot.addWidget(self.lbl_elixir)

        self.lbl_de = QLabel("DE: 0")
        self.lbl_de.setStyleSheet("font-weight: bold; color: #3742fa; font-size: 13px;")
        layout_loot.addWidget(self.lbl_de)

        self.lbl_stars = QLabel("⭐ 0 Stars")
        self.lbl_stars.setStyleSheet("font-weight: bold; color: #ffd32a; font-size: 13px;")
        layout_loot.addWidget(self.lbl_stars)

        layout_loot.addStretch()
        self.lbl_rate = QLabel("Rate: 0 G/hr • 0 E/hr • 0 DE/hr")
        self.lbl_rate.setStyleSheet("color: #747d8c; font-size: 11px;")
        layout_loot.addWidget(self.lbl_rate)

        metrics_grid.addWidget(self.card_loot, 1, 0, 1, 4)
        main_layout.addLayout(metrics_grid)

        # ----------------------------------------------------------------------
        # 4. Two-Column Workspace (Settings Panel & Live Execution Log)
        # ----------------------------------------------------------------------
        body_split = QHBoxLayout()

        # Left Column: Configuration Settings
        config_frame = QFrame()
        config_frame.setStyleSheet(f"background-color: {COLOR_BG_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 14px;")
        config_layout = QVBoxLayout(config_frame)
        config_layout.addWidget(QLabel("⚙️ Autonomous Farming Preferences"))

        # Thresholds Form
        grid_conf = QGridLayout()

        grid_conf.addWidget(QLabel("Filter Mode:"), 0, 0)
        self.combo_filter_mode = QComboBox()
        self.combo_filter_mode.addItems(["ALL (Strict)", "ANY (Flexible)", "WEIGHTED (Score)", "CUSTOM (Rules)"])
        grid_conf.addWidget(self.combo_filter_mode, 0, 1)

        grid_conf.addWidget(QLabel("Min Gold:"), 1, 0)
        self.spin_gold = QSpinBox()
        self.spin_gold.setRange(0, 5_000_000)
        self.spin_gold.setSingleStep(50_000)
        self.spin_gold.setValue(600_000)
        grid_conf.addWidget(self.spin_gold, 1, 1)

        grid_conf.addWidget(QLabel("Min Elixir:"), 2, 0)
        self.spin_elixir = QSpinBox()
        self.spin_elixir.setRange(0, 5_000_000)
        self.spin_elixir.setSingleStep(50_000)
        self.spin_elixir.setValue(600_000)
        grid_conf.addWidget(self.spin_elixir, 2, 1)

        grid_conf.addWidget(QLabel("Min Dark Elixir:"), 3, 0)
        self.spin_de = QSpinBox()
        self.spin_de.setRange(0, 50_000)
        self.spin_de.setSingleStep(500)
        self.spin_de.setValue(5_000)
        grid_conf.addWidget(self.spin_de, 3, 1)

        grid_conf.addWidget(QLabel("Max Cycles:"), 4, 0)
        self.spin_max_cycles = QSpinBox()
        self.spin_max_cycles.setRange(1, 100)
        self.spin_max_cycles.setValue(10)
        grid_conf.addWidget(self.spin_max_cycles, 4, 1)

        grid_conf.addWidget(QLabel("Max Search Attempts:"), 5, 0)
        self.spin_max_search = QSpinBox()
        self.spin_max_search.setRange(5, 200)
        self.spin_max_search.setValue(40)
        grid_conf.addWidget(self.spin_max_search, 5, 1)

        config_layout.addLayout(grid_conf)

        self.btn_apply_conf = QPushButton("Apply Configuration")
        self.btn_apply_conf.clicked.connect(self._on_apply_config)
        config_layout.addWidget(self.btn_apply_conf)

        self.btn_export_analytics = QPushButton("📊 Export Session Analytics")
        self.btn_export_analytics.clicked.connect(self._on_export_session)
        config_layout.addWidget(self.btn_export_analytics)

        config_layout.addStretch()
        body_split.addWidget(config_frame, stretch=1)

        # Right Column: Live Action Execution Log
        log_frame = QFrame()
        log_frame.setStyleSheet(f"background-color: {COLOR_BG_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 14px;")
        log_layout = QVBoxLayout(log_frame)
        log_layout.addWidget(QLabel("📋 Real-Time Action & State Log"))

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet(
            f"background-color: {COLOR_BG_WINDOW}; color: #ced6e0; font-family: Consolas, monospace; "
            "font-size: 11px; border: 1px solid #242834; border-radius: 4px; padding: 8px;"
        )
        log_layout.addWidget(self.txt_log)

        body_split.addWidget(log_frame, stretch=2)
        main_layout.addLayout(body_split, stretch=1)

    def _wire_engine_callbacks(self) -> None:
        """Connects background engine listeners to Qt GUI slots."""
        self.engine.add_state_listener(self._handle_state_change)
        self.engine.add_cycle_listener(self._handle_cycle_complete)
        self.engine.add_log_listener(self._handle_log_entry)

    # --------------------------------------------------------------------------
    # Slot Handlers
    # --------------------------------------------------------------------------

    def _on_start(self) -> None:
        self._on_apply_config()
        self.engine.start()

    def _on_pause(self) -> None:
        self.engine.pause()

    def _on_resume(self) -> None:
        self.engine.resume()

    def _on_stop(self) -> None:
        self.engine.stop()

    def _on_emergency_stop(self) -> None:
        self.engine.emergency_stop()
        self.lbl_state.setText("EMERGENCY STOPPED")
        self.lbl_state.setStyleSheet(f"font-size: 16px; font-weight: 900; color: {COLOR_NEON_RED};")
        self.sig_emergency_stop.emit()

    def _on_speed_changed(self, speed_str: str) -> None:
        try:
            val = float(speed_str.replace("x", ""))
            self.engine.config.playback_speed = val
            self.engine.executor.controller.clear_history()
        except Exception:
            pass

    def _on_apply_config(self) -> None:
        mode_idx = self.combo_filter_mode.currentIndex()
        modes = [FilterMode.ALL, FilterMode.ANY, FilterMode.WEIGHTED, FilterMode.CUSTOM]
        mode = modes[mode_idx]

        filter_conf = TargetFilterConfig(
            min_gold=self.spin_gold.value(),
            min_elixir=self.spin_elixir.value(),
            min_dark_elixir=self.spin_de.value(),
            filter_mode=mode,
            max_search_attempts=self.spin_max_search.value(),
        )

        self.engine.config.max_cycles = self.spin_max_cycles.value()
        self.engine.config.max_search_attempts_per_cycle = self.spin_max_search.value()
        self.engine.config.target_filter = filter_conf
        self.engine.target_selector.update_config(filter_conf)
        self._append_log("INFO", "Autonomous farming configuration updated.")

    def _on_export_session(self) -> None:
        """Exports session stats summary to JSON/Log."""
        s = self.engine.stats.to_dict()
        self._append_log("INFO", f"Exported Session [{s.get('session_id')}]: {s.get('total_cycles_completed')} cycles, G: {s.get('total_gold_acquired'):,}, E: {s.get('total_elixir_acquired'):,}")

    def _handle_state_change(self, old: FarmingState, new: FarmingState, payload: Dict[str, Any]) -> None:
        st_name = new.value
        color = "#00d26a" if new.is_active else ("#ff4757" if new == FarmingState.ERROR else "#70a1ff")
        self.lbl_state.setText(st_name)
        self.lbl_state.setStyleSheet(f"font-size: 16px; font-weight: 900; color: {color};")

    def _handle_cycle_complete(self, data: Dict[str, Any]) -> None:
        stats: Dict[str, Any] = data.get("stats", {})
        cycle_num = data.get("cycle_number", 0)
        max_c = self.engine.config.max_cycles

        skips = stats.get("total_bases_skipped", 0)
        streak = stats.get("consecutive_skips", 0)
        max_streak = stats.get("max_consecutive_skips", 0)
        self.lbl_cycles.setText(f"{cycle_num} / {max_c if max_c > 0 else '∞'}")
        self.lbl_skipped.setText(f"{skips} skipped (Streak: {streak}, Max: {max_streak})")
        self.lbl_gold.setText(f"Gold: {stats.get('total_gold_acquired', 0):,}")
        self.lbl_elixir.setText(f"Elixir: {stats.get('total_elixir_acquired', 0):,}")
        self.lbl_de.setText(f"DE: {stats.get('total_dark_elixir_acquired', 0):,}")
        self.lbl_stars.setText(f"⭐ {stats.get('total_stars', 0)} Stars")

        r_rates = stats.get("resources_per_hour", {})
        lat = stats.get("average_cycle_latency_ms", 0.0)
        self.lbl_rate.setText(
            f"Rate: {r_rates.get('gold', 0):,.0f} G/hr • {r_rates.get('elixir', 0):,.0f} E/hr • Latency: {lat:.0f}ms"
        )

        strats = stats.get("strategy_counts", {})
        if strats:
            top_s = max(strats.items(), key=lambda x: x[1])[0]
            self.lbl_strat.setText(f"{top_s} ({strats[top_s]} attacks)")

    def _handle_log_entry(self, level: str, message: str) -> None:
        self._append_log(level, message)

    def _append_log(self, level: str, message: str) -> None:
        t_str = time.strftime("%H:%M:%S")
        color = "#ff4757" if level in ("ERROR", "CRITICAL") else ("#ffa502" if level == "WARNING" else "#ced6e0")
        line = f"<span style='color: #747d8c;'>[{t_str}]</span> <span style='color: {color};'><b>[{level}]</b> {message}</span>"
        self.txt_log.append(line)

    def _toggle_pause_resume(self) -> None:
        """Toggles between Pause and Resume, or Starts if currently stopped."""
        if self.engine.current_state == FarmingState.PAUSED:
            self._on_resume()
        elif self.engine.current_state in (FarmingState.STOPPED, FarmingState.READY):
            self._on_start()
        elif self.engine.current_state.is_active:
            self._on_pause()

    def keyPressEvent(self, event: Any) -> None:
        """Handle keyboard input for Start, Pause/Resume, Stop, and Emergency Stop."""
        key = getattr(event, "key", lambda: None)()
        esc_key = getattr(Qt, "Key_Escape", 0x01000000)
        f12_key = getattr(Qt, "Key_F12", 0x0100003B)
        f5_key = getattr(Qt, "Key_F5", 0x01000034)
        space_key = getattr(Qt, "Key_Space", 0x20)

        # 1. Emergency Stop (Esc, F12)
        if key in (esc_key, f12_key):
            self._on_emergency_stop()
            if hasattr(event, "accept"):
                event.accept()
            return

        # 2. Start / Stop (F5 / Shift+F5)
        if key == f5_key:
            modifiers = getattr(event, "modifiers", lambda: 0)()
            shift_mod = getattr(Qt, "ShiftModifier", 0x02000000)
            if modifiers & shift_mod:
                self._on_stop()
            else:
                self._on_start()
            if hasattr(event, "accept"):
                event.accept()
            return

        # 3. Pause / Resume (Space) - ignore if user is typing in a text field
        if key == space_key:
            self._toggle_pause_resume()
            if hasattr(event, "accept"):
                event.accept()
            return

        super().keyPressEvent(event)
