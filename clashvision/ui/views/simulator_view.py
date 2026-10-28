"""
ClashVision Presentation Subsystem — View 5: Battle Simulator.
Provides offline attack replay visualization and playback controls:
- Simulated canvas displaying troop deployment markers
- Playback controls: Start, Pause, Stop, Step Forward, Step Backward, Reset
- Playback speed slider (0.5x to 5.0x) with real-time multiplier label
- Timeline scrubber showing current playback position and duration
- Distinct Visual Mode Indicator banner (Local Simulation vs Live Observation)
- Emergency Stop safety button halting playback in <= 50ms
Derived from Requirement R4, R5 and Features 24, 26, 33.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from clashvision.core.models import SimulationState
from clashvision.ui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QSlider, QProgressBar, Qt, Signal
)
from clashvision.ui.styles import (
    COLOR_NEON_GREEN, COLOR_NEON_PURPLE, COLOR_NEON_RED,
    COLOR_BORDER
)


class SimulatorView(QWidget):
    """
    Offline battle attack simulation and replay workbench view.
    """

    sig_start = Signal()
    sig_pause = Signal()
    sig_stop = Signal()
    sig_step_forward = Signal()
    sig_step_backward = Signal()
    sig_reset = Signal()
    sig_speed_changed = Signal(float)
    sig_emergency_stop = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("simulator_view")
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Mode Header Banner (Requirement R4: Distinct Visual Status)
        header_bar = QHBoxLayout()
        self.mode_banner = QLabel("LOCAL SIMULATION (OFFLINE REPLAY)")
        self.mode_banner.setObjectName("mode_banner_sim")
        self.mode_banner.setStyleSheet(
            f"background-color: #2b173d; color: {COLOR_NEON_PURPLE}; "
            f"border: 1px solid {COLOR_NEON_PURPLE}; border-radius: 6px; padding: 6px 16px; font-weight: bold; font-size: 13px;"
        )
        header_bar.addWidget(self.mode_banner)

        header_bar.addStretch()

        # Prominent Emergency Stop Button (Requirement R4)
        self.emergency_stop_btn = QPushButton("EMERGENCY STOP (Esc)")
        self.emergency_stop_btn.setObjectName("emergency_stop_btn")
        self.emergency_stop_btn.setStyleSheet(
            f"background-color: {COLOR_NEON_RED}; color: #ffffff; border: 2px solid #ff6b81; "
            "border-radius: 6px; font-weight: bold; font-size: 13px; padding: 6px 18px;"
        )
        self.emergency_stop_btn.clicked.connect(self._on_emergency_stop)
        header_bar.addWidget(self.emergency_stop_btn)

        main_layout.addLayout(header_bar)

        # 2. Simulator Canvas Box
        canvas_box = QFrame()
        canvas_box.setStyleSheet(
            f"background-color: #0c0d12; border: 1px solid {COLOR_BORDER}; border-radius: 8px;"
        )
        canvas_layout = QVBoxLayout(canvas_box)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas_label = QLabel("Simulator Sandbox Ready. Load strategy deployment plan to begin.")
        self.canvas_label.setAlignment(Qt.AlignCenter)
        self.canvas_label.setStyleSheet("color: #747d8c; min-height: 420px; font-size: 14px;")
        canvas_layout.addWidget(self.canvas_label)

        main_layout.addWidget(canvas_box, stretch=1)

        # 3. Timeline Scrubber & Step Progress
        timeline_box = QFrame()
        timeline_box.setStyleSheet(
            f"background-color: #1a1c23; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 8px 12px;"
        )
        timeline_layout = QVBoxLayout(timeline_box)
        timeline_layout.setSpacing(6)

        timeline_labels = QHBoxLayout()
        self.time_label = QLabel("0.00s / 0.00s")
        self.time_label.setStyleSheet("font-weight: bold; color: #f1f2f6; font-size: 12px;")
        timeline_labels.addWidget(self.time_label)

        timeline_labels.addStretch()

        self.step_label = QLabel("Step: 0 / 0 | Active Troops: 0")
        self.step_label.setStyleSheet("color: #a4b0be; font-size: 12px;")
        timeline_labels.addWidget(self.step_label)
        timeline_layout.addLayout(timeline_labels)

        self.scrubber = QProgressBar()
        self.scrubber.setRange(0, 1000)
        self.scrubber.setValue(0)
        timeline_layout.addWidget(self.scrubber)

        main_layout.addWidget(timeline_box)

        # 4. Playback Controls Toolbar
        controls_panel = QFrame()
        controls_panel.setStyleSheet(
            f"background-color: #1a1c23; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 10px;"
        )
        controls_layout = QHBoxLayout(controls_panel)
        controls_layout.setSpacing(10)

        self.start_btn = QPushButton("▶ Play")
        self.start_btn.clicked.connect(self._on_start)
        controls_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self._on_pause)
        controls_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        controls_layout.addWidget(self.stop_btn)

        self.step_back_btn = QPushButton("⏮ Step Back")
        self.step_back_btn.clicked.connect(self._on_step_backward)
        controls_layout.addWidget(self.step_back_btn)

        self.step_fwd_btn = QPushButton("⏭ Step Fwd")
        self.step_fwd_btn.clicked.connect(self._on_step_forward)
        controls_layout.addWidget(self.step_fwd_btn)

        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.clicked.connect(self._on_reset)
        controls_layout.addWidget(self.reset_btn)

        controls_layout.addStretch()

        # Playback Speed Control
        controls_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(5, 50)  # 0.5x to 5.0x
        self.speed_slider.setValue(10)      # 1.0x default
        self.speed_slider.valueChanged.connect(self._on_speed_slider_changed)
        controls_layout.addWidget(self.speed_slider)

        self.speed_label = QLabel("1.0x")
        self.speed_label.setStyleSheet("font-weight: bold; min-width: 45px;")
        controls_layout.addWidget(self.speed_label)

        main_layout.addWidget(controls_panel)

    def _on_start(self) -> None:
        self.sig_start.emit()

    def _on_pause(self) -> None:
        self.sig_pause.emit()

    def _on_stop(self) -> None:
        self.sig_stop.emit()

    def _on_step_forward(self) -> None:
        self.sig_step_forward.emit()

    def _on_step_backward(self) -> None:
        self.sig_step_backward.emit()

    def _on_reset(self) -> None:
        self.sig_reset.emit()

    def _on_speed_slider_changed(self, value: int) -> None:
        speed_factor = value / 10.0
        self.speed_label.setText(f"{speed_factor:.1f}x")
        self.sig_speed_changed.emit(speed_factor)

    def _on_emergency_stop(self) -> None:
        self.sig_emergency_stop.emit()
        self.canvas_label.setText("EMERGENCY STOP TRIGGERED: Playback halted immediately.")

    def update_simulation_state(self, state: SimulationState) -> None:
        """Update timeline scrubber, troop markers, and status indicators."""
        curr_s = state.current_time_ms / 1000.0
        tot_s = state.total_duration_ms / 1000.0
        self.time_label.setText(f"{curr_s:.2f}s / {tot_s:.2f}s")
        self.step_label.setText(
            f"Step: {state.step_index} / {state.total_steps} | Active Troops: {len(state.active_troops)}"
        )
        self.scrubber.setValue(int(state.progress_fraction * 1000))

        if state.is_playing:
            self.canvas_label.setText(
                f"Replaying Attack Timeline: {curr_s:.2f}s ({len(state.active_troops)} active markers on battlefield)"
            )
        elif state.is_paused:
            self.canvas_label.setText(f"Playback Paused at {curr_s:.2f}s")
        elif state.is_stopped:
            self.canvas_label.setText("Playback Stopped.")
