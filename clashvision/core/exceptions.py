"""
ClashVision core exception hierarchy.
Includes safety violation exceptions enforcing read-only capture.
"""


class ClashVisionError(Exception):
    """Base exception for all ClashVision runtime and configuration errors."""
    pass


class CaptureError(ClashVisionError):
    """Raised when screen capture operations encounter an error."""
    pass


class WindowNotFoundError(CaptureError):
    """Raised when the target process or window cannot be located."""
    pass


class WindowMinimizedError(CaptureError):
    """Raised when the target window is minimized, occluded, or has zero client area."""
    pass


class CaptureBackendError(CaptureError):
    """Raised when the underlying capture backend (e.g. MSS) fails."""
    pass


class OCRParseError(ClashVisionError):
    """Raised when OCR parsing fails on an image or text stream."""
    def __init__(self, message: str, raw_text: str = "", field: str = ""):
        super().__init__(message)
        self.raw_text = raw_text
        self.field = field


class LiveInputSafetyViolation(ClashVisionError):
    """
    CRITICAL SAFETY EXCEPTION:
    Raised whenever an automated input, click, or keystroke is attempted
    against a live emulator window or outside the quarantined local simulator canvas.
    Enforces Requirement R4 and the Integrity Mandate.
    """
    def __init__(self, message: str = "Automated input injection attempted against live target window."):
        super().__init__(message)


class EmergencyStopTriggered(ClashVisionError):
    """Raised when an emergency stop interrupt halts simulation playback."""
    def __init__(self, message: str = "Emergency stop triggered; simulation halted immediately."):
        super().__init__(message)


class ConfigurationError(ClashVisionError):
    """Raised when application configuration is invalid or missing required keys."""
    pass


class StateClassifierError(ClashVisionError):
    """Raised when state classification encounters an irrecoverable model or input error."""
    pass
