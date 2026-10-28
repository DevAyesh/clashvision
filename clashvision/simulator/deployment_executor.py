"""
ClashVision Autonomous Farming Engine — Simulated Action & Troop Deployment Executor.
Executes phased, timed troop and spell drops strictly inside the quarantined local
simulation canvas, maintaining safety isolation from live game windows.
Derived from Master Prompt Section 6.
"""

from __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from clashvision.core.constants import (
    VIRTUAL_REFERENCE_WIDTH,
    VIRTUAL_REFERENCE_HEIGHT,
)
from clashvision.core.models import (
    DeploymentStep,
    Point,
    NormalizedPoint,
    Rect,
)
from clashvision.core.exceptions import (
    LiveInputSafetyViolation,
    EmergencyStopTriggered,
)
from clashvision.capture.safety_guard import SafetyGuard, get_safety_guard
from clashvision.simulator.safety_barrier import SafetyBarrier, get_safety_barrier
from clashvision.simulator.sandboxed_controller import SandboxedController, InputAction

logger = logging.getLogger("clashvision.simulator.deployment_executor")


@dataclass
class ExecutedStepRecord:
    """Audit verification record for an executed deployment step."""
    step_id: int
    troop_name: str
    unit_count: int
    target_pixel_x: int
    target_pixel_y: int
    normalized_x: float
    normalized_y: float
    spell_name: Optional[str]
    phase_name: str
    timestamp: float
    status: str = "CONFIRMED"  # "CONFIRMED", "FAILED", "SKIPPED"
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "troop_name": self.troop_name,
            "unit_count": self.unit_count,
            "target_pixel_x": self.target_pixel_x,
            "target_pixel_y": self.target_pixel_y,
            "normalized_x": round(self.normalized_x, 4),
            "normalized_y": round(self.normalized_y, 4),
            "spell_name": self.spell_name,
            "phase_name": self.phase_name,
            "timestamp": self.timestamp,
            "status": self.status,
            "error_message": self.error_message,
        }


class DeploymentExecutor:
    """
    Dedicated action executor for the local simulation environment.
    Orchestrates:
    - Step-by-step phased deployment
    - Coordinate scaling and bounds verification
    - Multi-point drop pacing
    - Sandboxed controller action dispatch
    - Complete isolation from live game capture
    """

    def __init__(
        self,
        controller: Optional[SandboxedController] = None,
        canvas_width: int = VIRTUAL_REFERENCE_WIDTH,
        canvas_height: int = VIRTUAL_REFERENCE_HEIGHT,
        guard: Optional[SafetyGuard] = None,
        barrier: Optional[SafetyBarrier] = None,
    ) -> None:
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.guard = guard or get_safety_guard()
        self.barrier = barrier or get_safety_barrier()

        # Initialize sandboxed controller if not provided
        self.controller = controller or SandboxedController(
            canvas_hwnd=99991,  # Default registered simulator mock canvas HWND
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            guard=self.guard,
            barrier=self.barrier,
            strict_sandbox=True,
        )

        # Explicitly register the simulator window with SafetyGuard
        if self.guard and self.controller.canvas_hwnd:
            self.guard.register_simulator_window(self.controller.canvas_hwnd)

        self._lock = threading.RLock()
        self._is_paused: bool = False
        self._is_cancelled: bool = False
        self._execution_history: List[ExecutedStepRecord] = []
        self._on_step_callbacks: List[Callable[[ExecutedStepRecord], None]] = []

    @property
    def history(self) -> List[ExecutedStepRecord]:
        with self._lock:
            return list(self._execution_history)

    def add_step_listener(self, callback: Callable[[ExecutedStepRecord], None]) -> None:
        with self._lock:
            if callback not in self._on_step_callbacks:
                self._on_step_callbacks.append(callback)

    def pause(self) -> None:
        with self._lock:
            self._is_paused = True

    def resume(self) -> None:
        with self._lock:
            self._is_paused = False

    def cancel(self) -> None:
        with self._lock:
            self._is_cancelled = True

    def reset(self) -> None:
        with self._lock:
            self._is_paused = False
            self._is_cancelled = False
            self._execution_history.clear()
            self.controller.clear_history()

    def map_coordinates(self, step: DeploymentStep) -> Tuple[int, int, float, float]:
        """
        Maps step target coordinates safely into simulator canvas boundaries.
        Returns (pixel_x, pixel_y, normalized_x, normalized_y).
        """
        cw = max(1, self.canvas_width)
        ch = max(1, self.canvas_height)

        nx_val = getattr(step, "normalized_x", 0.0)
        ny_val = getattr(step, "normalized_y", 0.0)
        tx_val = getattr(step, "target_x", 0)
        ty_val = getattr(step, "target_y", 0)

        if nx_val > 0.0 or ny_val > 0.0:
            nx = max(0.01, min(0.99, nx_val))
            ny = max(0.01, min(0.99, ny_val))
            px = int(round(nx * cw))
            py = int(round(ny * ch))
        elif tx_val > 0 or ty_val > 0:
            px = max(1, min(cw - 1, tx_val))
            py = max(1, min(ch - 1, ty_val))
            nx = px / float(cw)
            ny = py / float(ch)
        else:
            px = cw // 2
            py = ch // 2
            nx = 0.5
            ny = 0.5

        return px, py, nx, ny

    def execute_step(self, step: DeploymentStep) -> ExecutedStepRecord:
        """
        Executes an individual deployment action inside the simulator.
        Raises:
            EmergencyStopTriggered: If emergency stop is active.
            LiveInputSafetyViolation: If targeting any live emulator window.
        """
        if self.barrier.is_emergency_stopped:
            raise EmergencyStopTriggered("Cannot execute deployment step: Emergency Stop is active.")

        px, py, nx, ny = self.map_coordinates(step)

        troop_name = getattr(step, "troop_type", "") or getattr(step, "troop_name", "") or "Troop"
        spell_name = getattr(step, "spell_name", None) or (troop_name if "Spell" in troop_name else None)
        phase_name = getattr(step, "phase_name", "") or getattr(step, "note", "") or "Standard"
        unit_cnt = getattr(step, "unit_count", 1)

        # Dispatches strictly into sandboxed controller
        try:
            # 1. Simulate selecting troop/spell
            if spell_name:
                self.controller.click(px, py, button="left")
            else:
                # Multi-point or rapid tap simulation
                for _ in range(min(unit_cnt, 5)):
                    self.controller.click(px, py, button="left")

            record = ExecutedStepRecord(
                step_id=getattr(step, "step_id", 0),
                troop_name=troop_name,
                unit_count=unit_cnt,
                target_pixel_x=px,
                target_pixel_y=py,
                normalized_x=nx,
                normalized_y=ny,
                spell_name=spell_name,
                phase_name=phase_name,
                timestamp=time.time(),
                status="CONFIRMED",
            )
        except Exception as ex:
            logger.error("Deployment step execution failed: %s", ex)
            record = ExecutedStepRecord(
                step_id=getattr(step, "step_id", 0),
                troop_name=troop_name,
                unit_count=unit_cnt,
                target_pixel_x=px,
                target_pixel_y=py,
                normalized_x=nx,
                normalized_y=ny,
                spell_name=spell_name,
                phase_name=phase_name,
                timestamp=time.time(),
                status="FAILED",
                error_message=str(ex),
            )
            if isinstance(ex, (LiveInputSafetyViolation, EmergencyStopTriggered)):
                raise

        with self._lock:
            self._execution_history.append(record)

        # Notify listeners
        for cb in self._on_step_callbacks:
            try:
                cb(record)
            except Exception as e:
                logger.error("Step listener error: %s", e)

        return record

    def execute_plan(
        self,
        steps: List[DeploymentStep],
        speed: float = 1.0,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> List[ExecutedStepRecord]:
        """
        Executes a complete sequential deployment plan in the simulator.
        Honors step delays scaled by playback speed.
        """
        results: List[ExecutedStepRecord] = []
        speed_factor = max(0.1, min(10.0, speed))

        for step in steps:
            if self._is_cancelled:
                logger.info("Deployment plan cancelled.")
                break

            if stop_check and stop_check():
                logger.info("Deployment plan halted by stop condition.")
                break

            # Handle pause state
            while self._is_paused and not self._is_cancelled:
                if stop_check and stop_check():
                    break
                time.sleep(0.05)

            # Simulated pacing delay
            delay_s = (step.delay_ms / 1000.0) / speed_factor
            if delay_s > 0:
                time.sleep(min(delay_s, 0.5))  # Paced for simulation

            res = self.execute_step(step)
            results.append(res)

        return results
