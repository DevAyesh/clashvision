"""
ClashVision Presentation Subsystem — View 2: Live Capture Feed & Overlays.
Displays live screen capture video canvas with toggleable overlays:
- Bounding box overlays (Resource ROIs, layout boxes)
- OCR text detection zones
- State badges & capture FPS indicator
- Pause/Resume feed controls
Derived from Requirement R5 and Feature 30.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional

import numpy as np

from clashvision.core.models import FramePacket, Rect
from clashvision.ui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QCheckBox, Qt, Signal
)
from clashvision.ui.styles import COLOR_NEON_GREEN, COLOR_NEON_CYAN, COLOR_BORDER


class LiveCaptureView(QWidget):
    """
    Live video capture feed view with dynamic HUD overlays.
    """

    sig_toggle_pause = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("live_capture_view")
        self.is_paused = False
        self._last_frame_time = time.time()
        self._fps_counter = 0.0
        self._last_packet: Optional[FramePacket] = None
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Top Controls Bar
        controls_bar = QHBoxLayout()

        self.pause_resume_btn = QPushButton("Pause Feed")
        self.pause_resume_btn.setStyleSheet("min-width: 100px;")
        self.pause_resume_btn.clicked.connect(self._on_toggle_pause)
        controls_bar.addWidget(self.pause_resume_btn)

        # Toggleable Overlays
        self.chk_bounding_boxes = QCheckBox("Resource ROIs")
        self.chk_bounding_boxes.setChecked(True)
        self.chk_bounding_boxes.toggled.connect(self._on_overlay_changed)
        controls_bar.addWidget(self.chk_bounding_boxes)

        self.chk_ocr_zones = QCheckBox("OCR Detection Zones")
        self.chk_ocr_zones.setChecked(True)
        self.chk_ocr_zones.toggled.connect(self._on_overlay_changed)
        controls_bar.addWidget(self.chk_ocr_zones)

        self.chk_state_badge = QCheckBox("State Badge HUD")
        self.chk_state_badge.setChecked(True)
        self.chk_state_badge.toggled.connect(self._on_overlay_changed)
        controls_bar.addWidget(self.chk_state_badge)

        controls_bar.addStretch()

        self.stats_label = QLabel("Resolution: 1920x1080 | FPS: 0.0 | Latency: 0 ms")
        self.stats_label.setStyleSheet("color: #a4b0be; font-size: 12px; font-weight: 500;")
        controls_bar.addWidget(self.stats_label)

        main_layout.addLayout(controls_bar)

        # 2. Main Capture Canvas Area
        self.canvas_container = QFrame()
        self.canvas_container.setStyleSheet(
            f"background-color: #0d0e12; border: 1px solid {COLOR_BORDER}; border-radius: 8px;"
        )
        canvas_layout = QVBoxLayout(self.canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.feed_canvas = QLabel("Awaiting live window frames...")
        self.feed_canvas.setAlignment(Qt.AlignCenter)
        self.feed_canvas.setStyleSheet(
            "color: #747d8c; font-size: 14px; min-height: 480px; background-color: transparent;"
        )
        canvas_layout.addWidget(self.feed_canvas)

        main_layout.addWidget(self.canvas_container, stretch=1)

        # 3. Status Footer
        footer_layout = QHBoxLayout()
        self.status_footer = QLabel("Active Target: Google Play Games PC / Clash of Clans (Read-Only Capture)")
        self.status_footer.setStyleSheet("color: #747d8c; font-size: 11px;")
        footer_layout.addWidget(self.status_footer)
        footer_layout.addStretch()

        self.feed_status_badge = QLabel("LIVE FEED ACTIVE")
        self.feed_status_badge.setStyleSheet(
            f"background-color: #142a1b; color: {COLOR_NEON_GREEN}; border: 1px solid {COLOR_NEON_GREEN}; "
            "border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: bold;"
        )
        footer_layout.addWidget(self.feed_status_badge)
        main_layout.addLayout(footer_layout)

    def _on_toggle_pause(self) -> None:
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_resume_btn.setText("Resume Feed")
            self.feed_status_badge.setText("FEED PAUSED")
            self.feed_status_badge.setStyleSheet(
                "background-color: #382410; color: #ffa502; border: 1px solid #ffa502; "
                "border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: bold;"
            )
        else:
            self.pause_resume_btn.setText("Pause Feed")
            self.feed_status_badge.setText("LIVE FEED ACTIVE")
            self.feed_status_badge.setStyleSheet(
                f"background-color: #142a1b; color: {COLOR_NEON_GREEN}; border: 1px solid {COLOR_NEON_GREEN}; "
                "border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: bold;"
            )
        self.sig_toggle_pause.emit(self.is_paused)

    def _on_overlay_changed(self) -> None:
        """Callback when HUD overlay toggles are changed."""
        if self._last_packet is not None:
            self._render_canvas(self._last_packet)

    def update_frame(self, packet: FramePacket) -> None:
        """Update live feed canvas with captured frame."""
        if self.is_paused:
            return

        now = time.time()
        delta = now - self._last_frame_time
        if delta > 0:
            self._fps_counter = 0.9 * self._fps_counter + 0.1 * (1.0 / delta)
        self._last_frame_time = now

        w = packet.width
        h = packet.height
        latency_ms = int((now - packet.timestamp) * 1000)

        self.stats_label.setText(
            f"Resolution: {w}x{h} | FPS: {self._fps_counter:.1f} | Latency: {max(0, latency_ms)} ms"
        )
        self._last_packet = packet
        self._render_canvas(packet)

    def _render_canvas(self, packet: FramePacket) -> None:
        """Render frame and overlay annotations to feed canvas."""
        w = packet.width
        h = packet.height

        overlays = []
        if self.chk_bounding_boxes.isChecked():
            overlays.append("ROIs")
        if self.chk_ocr_zones.isChecked():
            overlays.append("OCR Zones")
        if self.chk_state_badge.isChecked():
            overlays.append("HUD")

        overlay_suffix = f" [HUD: {', '.join(overlays)}]" if overlays else ""
        self.feed_canvas.setText(f"Frame #{packet.frame_id} captured ({w}x{h}){overlay_suffix}")

    def get_annotated_frame(self) -> Optional[Any]:
        """Returns the current frame with visual overlay annotations drawn."""
        if self._last_packet is None or self._last_packet.frame is None:
            return None
        frame = self._last_packet.frame
        import numpy as np
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return frame
        import cv2
        fh, fw = frame.shape[:2]
        canvas_img = frame.copy()
        if self.chk_bounding_boxes.isChecked():
            cv2.rectangle(canvas_img, (int(0.04 * fw), int(0.03 * fh)), (int(0.22 * fw), int(0.21 * fh)), (0, 255, 255), 2)
        if self.chk_ocr_zones.isChecked():
            cv2.rectangle(canvas_img, (int(0.04 * fw), int(0.03 * fh)), (int(0.22 * fw), int(0.09 * fh)), (0, 215, 255), 1)
            cv2.rectangle(canvas_img, (int(0.04 * fw), int(0.09 * fh)), (int(0.22 * fw), int(0.15 * fh)), (255, 0, 255), 1)
            cv2.rectangle(canvas_img, (int(0.04 * fw), int(0.15 * fh)), (int(0.22 * fw), int(0.21 * fh)), (200, 50, 50), 1)
        return canvas_img
