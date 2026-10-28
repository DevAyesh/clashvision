"""
ClashVision core typed data models and interface contracts.
Guarantees clean cross-subsystem communication between Capture, Vision,
Decision, Simulator, Storage, and UI layers.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

from clashvision.core.constants import (
    GameState,
    StrategyType,
    RiskLevel,
)


@dataclass(frozen=True)
class Point:
    """2D integer coordinate in pixel space."""
    x: int
    y: int

    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def distance_to(self, other: Point) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass(frozen=True)
class NormalizedPoint:
    """2D floating-point coordinate normalized to [0.0, 1.0]."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_pixel(self, width: int, height: int) -> Point:
        return Point(
            x=int(round(self.x * width)),
            y=int(round(self.y * height)),
        )


@dataclass
class Rect:
    """
    Axis-aligned 2D bounding rectangle in pixel space.
    """
    x: int
    y: int
    width: int
    height: int

    @property
    def left(self) -> int:
        return self.x

    @property
    def top(self) -> int:
        return self.y

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> int:
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        return self.y + self.height // 2

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0

    def contains_point(self, pt: Point | Tuple[int, int]) -> bool:
        px = pt.x if isinstance(pt, Point) else pt[0]
        py = pt.y if isinstance(pt, Point) else pt[1]
        return self.left <= px < self.right and self.top <= py < self.bottom

    def intersects(self, other: Rect) -> bool:
        return not (
            self.right <= other.left
            or self.left >= other.right
            or self.bottom <= other.top
            or self.top >= other.bottom
        )

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Rect:
        return cls(
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )


@dataclass
class NormalizedRect:
    """
    Axis-aligned rectangle with coordinates normalized to [0.0, 1.0].
    Resolution-independent representation for UI anchors and ROIs.
    """
    x: float
    y: float
    width: float
    height: float

    def to_pixel_rect(self, image_width: int, image_height: int) -> Rect:
        px = int(round(self.x * image_width))
        py = int(round(self.y * image_height))
        pw = int(round(self.width * image_width))
        ph = int(round(self.height * image_height))
        return Rect(x=px, y=py, width=pw, height=ph)

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizedRect:
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            width=float(data["width"]),
            height=float(data["height"]),
        )


@dataclass
class WindowInfo:
    """
    Snapshot of target emulator window properties and screen coordinates.
    """
    hwnd: int = 0
    title: str = ""
    process_name: str = ""
    client_rect: Rect = field(default_factory=lambda: Rect(0, 0, 0, 0))
    window_rect: Rect = field(default_factory=lambda: Rect(0, 0, 0, 0))
    is_visible: bool = False
    is_minimized: bool = False
    is_focused: bool = False
    scale_factor: float = 1.0
    dpi: int = 96

    @property
    def aspect_ratio(self) -> float:
        if self.client_rect.height <= 0:
            return 0.0
        return self.client_rect.width / self.client_rect.height

    @property
    def is_capturable(self) -> bool:
        """Determines if the window has a non-zero visible client area for capture."""
        return (
            self.is_visible
            and not self.is_minimized
            and self.client_rect.is_valid
            and self.client_rect.area > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hwnd": self.hwnd,
            "title": self.title,
            "process_name": self.process_name,
            "client_rect": self.client_rect.to_dict(),
            "window_rect": self.window_rect.to_dict(),
            "is_visible": self.is_visible,
            "is_minimized": self.is_minimized,
            "is_focused": self.is_focused,
            "scale_factor": self.scale_factor,
            "dpi": self.dpi,
            "is_capturable": self.is_capturable,
        }


@dataclass
class FramePacket:
    """
    Contract 1: CaptureSubsystem -> VisionSubsystem / UI Preview
    Encapsulates a single captured video frame and its associated metadata.
    """
    frame: np.ndarray
    timestamp: float
    window_info: WindowInfo
    frame_id: int

    @property
    def height(self) -> int:
        return int(self.frame.shape[0]) if self.frame is not None else 0

    @property
    def width(self) -> int:
        return int(self.frame.shape[1]) if self.frame is not None else 0

    @property
    def channels(self) -> int:
        return int(self.frame.shape[2]) if self.frame is not None and len(self.frame.shape) > 2 else 1


@dataclass
class ResourceReading:
    """
    Extracted in-game resource values and associated OCR confidences.
    """
    gold: Optional[int] = None
    elixir: Optional[int] = None
    dark_elixir: Optional[int] = None
    raw_text: Dict[str, str] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0

    @property
    def total_regular_resources(self) -> int:
        """Sum of Gold and Elixir."""
        return (self.gold or 0) + (self.elixir or 0)

    def has_any(self) -> bool:
        return (self.gold is not None) or (self.elixir is not None) or (self.dark_elixir is not None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gold": self.gold,
            "elixir": self.elixir,
            "dark_elixir": self.dark_elixir,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class BaseLayoutFeatures:
    """
    Structural spatial analysis of an enemy base for strategy evaluation.
    """
    exposed_collector_ratio: float = 0.0
    th_accessible: bool = False
    th_level: Optional[int] = None
    air_defense_locations: List[Tuple[int, int]] = field(default_factory=list)
    compartment_count: int = 0
    wall_perimeter_intact: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exposed_collector_ratio": self.exposed_collector_ratio,
            "th_accessible": self.th_accessible,
            "th_level": self.th_level,
            "air_defense_locations": self.air_defense_locations,
            "compartment_count": self.compartment_count,
            "wall_perimeter_intact": self.wall_perimeter_intact,
            "metadata": self.metadata,
        }


@dataclass
class StrategyEvaluation:
    """
    Scoring, viability, and risk assessment for a specific strategy candidate.
    """
    strategy_type: StrategyType
    score: float
    viable: bool
    risk_level: RiskLevel = RiskLevel.MEDIUM
    resource_suitability: float = 0.0
    layout_suitability: float = 0.0
    prerequisites_met: bool = True
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_type": self.strategy_type.value,
            "score": self.score,
            "viable": self.viable,
            "risk_level": self.risk_level.value,
            "resource_suitability": self.resource_suitability,
            "layout_suitability": self.layout_suitability,
            "prerequisites_met": self.prerequisites_met,
            "reasons": self.reasons,
        }


@dataclass
class DeploymentStep:
    """
    Individual troop or spell deployment instruction for attack planning and simulation.
    """
    step_id: int = 0
    troop_type: str = ""
    target_x: int = 0
    target_y: int = 0
    normalized_x: float = 0.0
    normalized_y: float = 0.0
    delay_ms: int = 0
    unit_count: int = 1
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "troop_type": self.troop_type,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "normalized_x": self.normalized_x,
            "normalized_y": self.normalized_y,
            "delay_ms": self.delay_ms,
            "unit_count": self.unit_count,
            "note": self.note,
        }


@dataclass
class FarmingRecommendation:
    """
    Contract 3: DecisionSubsystem -> Simulator / UI Presentation
    Complete tactical decision recommendation for an active enemy base.
    """
    suitability_score: float
    primary_strategy: StrategyType
    rankings: List[StrategyEvaluation] = field(default_factory=list)
    deployment_plan: List[DeploymentStep] = field(default_factory=list)
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suitability_score": self.suitability_score,
            "primary_strategy": self.primary_strategy.value,
            "rankings": [r.to_dict() for r in self.rankings],
            "deployment_plan": [s.to_dict() for s in self.deployment_plan],
            "score_breakdown": self.score_breakdown,
            "timestamp": self.timestamp,
            "explanation": self.explanation,
        }


@dataclass
class SimulationState:
    """
    Contract 5: SimulatorSubsystem playback status and timeline.
    """
    step_index: int = 0
    total_steps: int = 0
    is_playing: bool = False
    is_paused: bool = False
    is_stopped: bool = True
    current_time_ms: int = 0
    total_duration_ms: int = 0
    active_troops: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def progress_fraction(self) -> float:
        if self.total_duration_ms <= 0:
            return 0.0
        return min(1.0, max(0.0, self.current_time_ms / self.total_duration_ms))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "is_playing": self.is_playing,
            "is_paused": self.is_paused,
            "is_stopped": self.is_stopped,
            "current_time_ms": self.current_time_ms,
            "total_duration_ms": self.total_duration_ms,
            "progress_fraction": self.progress_fraction,
            "active_troops": self.active_troops,
        }


@dataclass
class VisionAnalysisResult:
    """
    Contract 2: VisionSubsystem -> DecisionSubsystem
    Unified result of state classification, OCR parsing, and layout extraction.
    """
    state: GameState
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    resources: Optional[ResourceReading] = None
    base_layout: Optional[BaseLayoutFeatures] = None
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "resources": self.resources.to_dict() if self.resources else None,
            "base_layout": self.base_layout.to_dict() if self.base_layout else None,
            "timestamp": self.timestamp,
        }


@dataclass
class SessionRecord:
    """Persistent session tracking for SQLite storage."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    total_scans: int = 0
    matched_targets: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionRecord:
    """Historical detection log entry for SQLite storage."""
    record_id: Optional[int] = None
    session_id: str = ""
    timestamp: float = 0.0
    state: GameState = GameState.UNKNOWN
    confidence: float = 0.0
    resources: Optional[ResourceReading] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "state": self.state.value,
            "confidence": self.confidence,
            "resources": self.resources.to_dict() if self.resources else None,
        }


@dataclass
class DatasetAnnotation:
    """Dataset item for screenshot labeling and export."""
    annotation_id: str
    image_path: str
    state: GameState
    labels: Dict[str, Any] = field(default_factory=dict)
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "image_path": self.image_path,
            "state": self.state.value,
            "labels": self.labels,
            "bounding_boxes": self.bounding_boxes,
        }
