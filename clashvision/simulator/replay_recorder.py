"""
ClashVision Simulator Replay Recorder.
Captures simulation timeline events, serializes replays to JSON/dict,
and provides scrubbing support across timestamps and deployment steps.
Derived directly from Requirements R3, R4 and Features 22, 23, 24.
"""

from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from clashvision.core.models import (
    DeploymentStep,
    SimulationState,
)
from clashvision.simulator.simulation_engine import SimulationEngine, PlaybackState

logger = logging.getLogger("clashvision.simulator.recorder")


@dataclass
class ReplayEvent:
    """
    Individual event recorded during simulation playback.
    """
    timestamp_ms: int
    event_type: str  # "STEP", "TICK", "STATE_CHANGE", "ACTION", "EMERGENCY_STOP"
    step_index: int
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "event_type": self.event_type,
            "step_index": self.step_index,
            "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReplayEvent:
        if not isinstance(data, dict):
            data = {}
        ts_val = data.get("timestamp_ms")
        timestamp_ms = int(ts_val) if ts_val is not None else 0
        event_type = str(data.get("event_type") or "UNKNOWN")
        step_val = data.get("step_index")
        step_index = int(step_val) if step_val is not None else 0
        data_val = data.get("data")
        event_data = dict(data_val) if data_val is not None and isinstance(data_val, dict) else {}
        return cls(
            timestamp_ms=timestamp_ms,
            event_type=event_type,
            step_index=step_index,
            data=event_data,
        )


class ReplayRecorder:
    """
    Simulation timeline event recorder, scrubber, and serializer.
    Captures:
    - Step executions (troops, coordinates, delays).
    - Playback ticks and state changes.
    - Simulated input actions and emergency stop triggers.
    - Export / import of replay sessions via JSON or dictionary.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._events: List[ReplayEvent] = []
        self._metadata: Dict[str, Any] = metadata or {}
        self._recording: bool = True

    @property
    def events(self) -> List[ReplayEvent]:
        """Chronological list of all recorded replay events."""
        return list(self._events)

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def clear(self) -> None:
        """Clears all recorded events."""
        self._events.clear()

    # --------------------------------------------------------------------------
    # Event Recording Methods
    # --------------------------------------------------------------------------
    def record_step(
        self,
        step: DeploymentStep,
        timestamp_ms: int,
        troop_marker: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records the execution of a deployment step."""
        if not self._recording:
            return

        data = {
            "step_id": step.step_id,
            "troop_type": step.troop_type,
            "target_x": step.target_x,
            "target_y": step.target_y,
            "normalized_x": step.normalized_x,
            "normalized_y": step.normalized_y,
            "unit_count": step.unit_count,
            "note": step.note,
        }
        if troop_marker is not None:
            data["troop_marker"] = troop_marker

        event = ReplayEvent(
            timestamp_ms=timestamp_ms,
            event_type="STEP",
            step_index=step.step_id,
            data=data,
        )
        self._events.append(event)
        logger.debug("ReplayRecorder: Recorded STEP %d at %d ms", step.step_id, timestamp_ms)

    def record_tick(self, state: SimulationState) -> None:
        """Records a simulation tick snapshot."""
        if not self._recording:
            return

        event = ReplayEvent(
            timestamp_ms=state.current_time_ms,
            event_type="TICK",
            step_index=state.step_index,
            data={
                "progress_fraction": state.progress_fraction,
                "active_troops_count": len(state.active_troops),
                "is_playing": state.is_playing,
            },
        )
        self._events.append(event)

    def record_state_change(self, old_state: str, new_state: str, timestamp_ms: int) -> None:
        """Records a playback state change."""
        if not self._recording:
            return

        event = ReplayEvent(
            timestamp_ms=timestamp_ms,
            event_type="STATE_CHANGE",
            step_index=0,
            data={"old_state": str(old_state), "new_state": str(new_state)},
        )
        self._events.append(event)

    def record_action(
        self,
        action_type: str,
        x: int,
        y: int,
        timestamp_ms: int,
        **kwargs: Any,
    ) -> None:
        """Records an individual simulated input action."""
        if not self._recording:
            return

        data = {"action_type": action_type, "x": x, "y": y}
        data.update(kwargs)
        event = ReplayEvent(
            timestamp_ms=timestamp_ms,
            event_type="ACTION",
            step_index=0,
            data=data,
        )
        self._events.append(event)

    def record_emergency_stop(self, timestamp_ms: int, reason: str = "") -> None:
        """Records an emergency stop halt event."""
        if not self._recording:
            return

        event = ReplayEvent(
            timestamp_ms=timestamp_ms,
            event_type="EMERGENCY_STOP",
            step_index=0,
            data={"reason": reason},
        )
        self._events.append(event)

    # --------------------------------------------------------------------------
    # Scrubbing & Querying
    # --------------------------------------------------------------------------
    def get_events(self, event_type: Optional[str] = None) -> List[ReplayEvent]:
        """Returns recorded events, optionally filtered by event_type."""
        if event_type is None:
            return list(self._events)
        return [e for e in self._events if e.event_type == event_type]

    def seek_to_time(self, target_time_ms: int) -> Tuple[int, List[ReplayEvent]]:
        """
        Timeline scrubbing: returns highest step_index reached and all events
        up to target_time_ms.
        """
        matched = [e for e in self._events if e.timestamp_ms <= target_time_ms]
        highest_step = max([e.step_index for e in matched], default=0)
        return highest_step, matched

    def seek_to_step(self, target_step_index: int) -> Tuple[int, List[ReplayEvent]]:
        """
        Step scrubbing: returns the timestamp_ms and events up to target_step_index.
        When target_step_index exceeds recorded steps, target_timestamp resolves to
        the timestamp of the last available step/event instead of staying 0.
        """
        matched = [e for e in self._events if e.step_index <= target_step_index]
        step_events = [e for e in matched if e.event_type == "STEP"]
        if step_events:
            target_timestamp = max(e.timestamp_ms for e in step_events)
        elif matched:
            target_timestamp = max(e.timestamp_ms for e in matched)
        else:
            target_timestamp = 0
        return target_timestamp, matched

    # --------------------------------------------------------------------------
    # Serialization: JSON / Dict / File
    # --------------------------------------------------------------------------
    def export_replay(self) -> Dict[str, Any]:
        """Serializes the replay recorder session to a dictionary."""
        total_duration = max([e.timestamp_ms for e in self._events], default=0)
        step_events = [e for e in self._events if e.event_type == "STEP"]
        return {
            "version": "1.0",
            "exported_at": time.time(),
            "total_events": len(self._events),
            "total_duration_ms": total_duration,
            "step_count": len(step_events),
            "metadata": dict(self._metadata),
            "events": [e.to_dict() for e in self._events],
        }

    def import_replay(self, data: Dict[str, Any]) -> None:
        """Deserializes replay session data from a dictionary."""
        if not isinstance(data, dict):
            data = {}
        meta_val = data.get("metadata")
        self._metadata = dict(meta_val) if meta_val is not None and isinstance(meta_val, dict) else {}
        self._events.clear()
        raw_events = data.get("events") if data.get("events") is not None else []
        if isinstance(raw_events, list):
            for ev in raw_events:
                if isinstance(ev, dict):
                    self._events.append(ReplayEvent.from_dict(ev))
        logger.info("ReplayRecorder imported %d events.", len(self._events))

    def save_to_file(self, file_path: str) -> None:
        """Saves replay session to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.export_replay(), f, indent=2)
        logger.info("ReplayRecorder saved %d events to %s", len(self._events), file_path)

    def load_from_file(self, file_path: str) -> None:
        """Loads replay session from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.import_replay(data)

    # --------------------------------------------------------------------------
    # Engine Attachment
    # --------------------------------------------------------------------------
    def attach_to_engine(self, engine: SimulationEngine) -> None:
        """
        Attaches recorder directly to a SimulationEngine instance.
        Automatically registers hooks for on_step, on_tick, and on_state_change.
        """
        engine.register_on_step(
            lambda step, marker: self.record_step(
                step=step,
                timestamp_ms=engine.current_time_ms,
                troop_marker=marker,
            )
        )
        engine.register_on_tick(
            lambda state: self.record_tick(state)
        )
        engine.register_on_state_change(
            lambda old_s, new_s: self.record_state_change(
                old_state=old_s.value,
                new_state=new_s.value,
                timestamp_ms=engine.current_time_ms,
            )
        )
        logger.info("ReplayRecorder attached to SimulationEngine.")
