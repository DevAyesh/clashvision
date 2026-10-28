"""
ClashVision application configuration dataclasses and persistence utilities.
Handles user preferences, capture options, OCR tuning, resource thresholds,
and simulator parameters.
"""

from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from clashvision.core.constants import (
    CaptureBackend,
    ResourcePriority,
    DEFAULT_CAPTURE_FPS,
    DEFAULT_TARGET_PROCESS_NAMES,
    DEFAULT_TARGET_WINDOW_TITLES,
    DEFAULT_MIN_GOLD,
    DEFAULT_MIN_ELIXIR,
    DEFAULT_MIN_DARK_ELIXIR,
    VIRTUAL_REFERENCE_WIDTH,
    VIRTUAL_REFERENCE_HEIGHT,
)
from clashvision.core.exceptions import ConfigurationError


@dataclass
class CaptureConfig:
    """Settings governing the screen grabber and window detector."""
    target_fps: int = DEFAULT_CAPTURE_FPS
    backend: CaptureBackend = CaptureBackend.MSS
    auto_pause_on_minimized: bool = True
    target_process_names: List[str] = field(default_factory=lambda: list(DEFAULT_TARGET_PROCESS_NAMES))
    target_window_titles: List[str] = field(default_factory=lambda: list(DEFAULT_TARGET_WINDOW_TITLES))
    mock_mode: bool = False
    mock_window_width: int = VIRTUAL_REFERENCE_WIDTH
    mock_window_height: int = VIRTUAL_REFERENCE_HEIGHT
    dpi_aware: bool = True

    def validate(self) -> None:
        if self.target_fps < 1 or self.target_fps > 60:
            raise ConfigurationError(f"target_fps must be between 1 and 60, got {self.target_fps}")
        if self.mock_window_width <= 0 or self.mock_window_height <= 0:
            raise ConfigurationError("Mock window dimensions must be positive.")


@dataclass
class ResourceThresholds:
    """Farming resource targets and priority weighting."""
    min_gold: int = DEFAULT_MIN_GOLD
    min_elixir: int = DEFAULT_MIN_ELIXIR
    min_dark_elixir: int = DEFAULT_MIN_DARK_ELIXIR
    priority: ResourcePriority = ResourcePriority.BALANCED
    gold_weight: float = 1.0
    elixir_weight: float = 1.0
    dark_elixir_weight: float = 100.0  # DE is generally 100x more valuable than gold/elixir

    def validate(self) -> None:
        if self.min_gold < 0 or self.min_elixir < 0 or self.min_dark_elixir < 0:
            raise ConfigurationError("Resource thresholds cannot be negative.")
        if self.gold_weight < 0 or self.elixir_weight < 0 or self.dark_elixir_weight < 0:
            raise ConfigurationError("Resource weights cannot be negative.")


@dataclass
class OCRConfig:
    """OCR engine selection and preprocessing parameters."""
    engine_type: str = "rapidocr"  # "rapidocr", "pytesseract", or "mock"
    tesseract_cmd: str = "tesseract"
    confidence_threshold: float = 0.60
    debug_save_crops: bool = False

    def validate(self) -> None:
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ConfigurationError("confidence_threshold must be between 0.0 and 1.0")


@dataclass
class SimulatorConfig:
    """Attack simulator playback parameters and safety limits."""
    emergency_stop_hotkey: str = "Escape"
    playback_speed: float = 1.0
    canvas_width: int = VIRTUAL_REFERENCE_WIDTH
    canvas_height: int = VIRTUAL_REFERENCE_HEIGHT
    enforce_strict_sandbox: bool = True
    max_step_delay_ms: int = 5000

    def validate(self) -> None:
        if (
            isinstance(self.playback_speed, bool)
            or not isinstance(self.playback_speed, (int, float))
            or not math.isfinite(self.playback_speed)
            or self.playback_speed < 0.1
            or self.playback_speed > 10.0
        ):
            raise ConfigurationError("playback_speed must be between 0.1 and 10.0")
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            raise ConfigurationError("Simulator canvas dimensions must be positive.")


@dataclass
class AppSettings:
    """
    Root application configuration containing all subsystem preferences.
    Can be serialized to and deserialized from JSON.
    """
    app_name: str = "ClashVision"
    version: str = "0.1.0"
    dark_mode: bool = True
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    resources: ResourceThresholds = field(default_factory=ResourceThresholds)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    db_path: str = "clashvision.db"
    log_level: str = "INFO"

    def validate(self) -> None:
        self.capture.validate()
        self.resources.validate()
        self.ocr.validate()
        self.simulator.validate()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert enums to string values
        data["capture"]["backend"] = self.capture.backend.value
        data["resources"]["priority"] = self.resources.priority.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AppSettings:
        cap_data = dict(data.get("capture", {}))
        if "backend" in cap_data and isinstance(cap_data["backend"], str):
            cap_data["backend"] = CaptureBackend(cap_data["backend"])
        capture = CaptureConfig(**cap_data)

        res_data = dict(data.get("resources", {}))
        if "priority" in res_data and isinstance(res_data["priority"], str):
            res_data["priority"] = ResourcePriority(res_data["priority"])
        resources = ResourceThresholds(**res_data)

        ocr = OCRConfig(**data.get("ocr", {}))
        sim = SimulatorConfig(**data.get("simulator", {}))

        return cls(
            app_name=data.get("app_name", "ClashVision"),
            version=data.get("version", "0.1.0"),
            dark_mode=data.get("dark_mode", True),
            capture=capture,
            resources=resources,
            ocr=ocr,
            simulator=sim,
            db_path=data.get("db_path", "clashvision.db"),
            log_level=data.get("log_level", "INFO"),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> AppSettings:
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save_to_file(self, file_path: str) -> None:
        self.validate()
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load_from_file(cls, file_path: str) -> AppSettings:
        if not os.path.exists(file_path):
            # Return defaults if file does not exist
            return cls()
        with open(file_path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())
