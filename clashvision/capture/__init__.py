"""
ClashVision Screen Capture Subsystem.
Provides window discovery, DPI-aware coordinate mapping, high-performance MSS grabbing,
and architectural read-only safety guardrails.
"""

from clashvision.capture.safety_guard import (
    SafetyGuard,
    get_safety_guard,
)
from clashvision.capture.window_detector import (
    WindowDetector,
)
from clashvision.capture.mss_grabber import (
    MSSGrabber,
)

__all__ = [
    "SafetyGuard",
    "get_safety_guard",
    "WindowDetector",
    "MSSGrabber",
]
