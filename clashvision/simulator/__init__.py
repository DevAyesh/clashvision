"""
ClashVision Local Attack Simulator Subsystem.
Provides offline battle playback, troop deployment simulation, quarantined sandboxed controls,
emergency stop latch with <= 50ms latency, and replay recording.
"""

from clashvision.simulator.safety_barrier import (
    SafetyBarrier,
    VisualMode,
    get_safety_barrier,
)
from clashvision.simulator.sandboxed_controller import (
    SandboxedController,
    SandboxedInputController,
    InputAction,
)
from clashvision.simulator.simulation_engine import (
    SimulationEngine,
    PlaybackState,
    ActiveTroop,
    TroopMarker,
)
from clashvision.simulator.replay_recorder import (
    ReplayRecorder,
    ReplayEvent,
)

__all__ = [
    "SafetyBarrier",
    "VisualMode",
    "get_safety_barrier",
    "SandboxedController",
    "SandboxedInputController",
    "InputAction",
    "SimulationEngine",
    "PlaybackState",
    "ActiveTroop",
    "TroopMarker",
    "ReplayRecorder",
    "ReplayEvent",
]
