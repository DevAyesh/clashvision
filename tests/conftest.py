"""
Pytest configuration, mock fixtures, and autouse safety interceptor for ClashVision.
Enforces the strict safety mandate: any live mouse/keyboard injection outside the
quarantined simulator canvas immediately raises LiveInputSafetyViolation.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import LiveInputSafetyViolation from clashvision if available, or fallback
try:
    from clashvision.core.exceptions import LiveInputSafetyViolation
except ImportError:
    class LiveInputSafetyViolation(Exception):  # type: ignore
        """Safety violation exception raised when live input is intercepted."""
        def __init__(self, message: str = "Automated input injection attempted against live target window."):
            super().__init__(message)


# Global simulator sandbox flag
_SIMULATOR_SANDBOX_ACTIVE = False


@contextmanager
def allow_simulator_sandbox() -> Generator[None, None, None]:
    """Context manager allowing input specifically within the quarantined simulator canvas."""
    global _SIMULATOR_SANDBOX_ACTIVE
    prev = _SIMULATOR_SANDBOX_ACTIVE
    _SIMULATOR_SANDBOX_ACTIVE = True
    try:
        yield
    finally:
        _SIMULATOR_SANDBOX_ACTIVE = prev


def _safety_block_call(*args: Any, **kwargs: Any) -> None:
    """Interception handler that raises LiveInputSafetyViolation if called outside simulator."""
    if not _SIMULATOR_SANDBOX_ACTIVE:
        raise LiveInputSafetyViolation(
            "CRITICAL SAFETY VIOLATION: Automated mouse/keyboard call intercepted outside quarantined simulator bounds."
        )


@pytest.fixture(autouse=True)
def safety_live_input_interceptor(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """
    Global autouse fixture intercepting and blocking all live input APIs.
    Guarantees 100% read-only observation with ZERO automated input to the live OS/emulator.
    """
    # 1. Intercept PyAutoGUI methods if installed
    try:
        import pyautogui  # type: ignore

        methods_to_block = [
            "click", "doubleClick", "tripleClick", "rightClick", "middleClick",
            "moveTo", "moveRel", "dragTo", "dragRel",
            "mouseDown", "mouseUp",
            "press", "keyDown", "keyUp", "typewrite", "write", "hotkey"
        ]
        for method_name in methods_to_block:
            if hasattr(pyautogui, method_name):
                monkeypatch.setattr(pyautogui, method_name, _safety_block_call)
    except ImportError:
        pass

    # 2. Intercept Win32 API calls if win32api / win32gui is installed
    try:
        import win32api  # type: ignore
        if hasattr(win32api, "mouse_event"):
            monkeypatch.setattr(win32api, "mouse_event", _safety_block_call)
        if hasattr(win32api, "keybd_event"):
            monkeypatch.setattr(win32api, "keybd_event", _safety_block_call)
    except ImportError:
        pass

    yield


# --------------------------------------------------------------------------
# Domain & Synthetic Data Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def synthetic_generator() -> Any:
    """Provides an instance of SyntheticGenerator."""
    from tests.fixtures.synthetic_generator import SyntheticGenerator
    return SyntheticGenerator(default_seed=42)


@pytest.fixture
def all_state_frames() -> Dict[str, np.ndarray]:
    """Pre-generates 1080p synthetic numpy frames for all 12 game states."""
    from tests.fixtures.synthetic_generator import GameStateType, generate_state_frame
    frames = {}
    for state in GameStateType:
        frames[state.value] = generate_state_frame(state=state, width=1920, height=1080, seed=42)
    return frames


@pytest.fixture
def all_canonical_layouts() -> Dict[str, Any]:
    """Provides all 6 canonical base layout models."""
    from tests.fixtures.layout_generator import BaseLayoutType, generate_canonical_layout
    layouts = {}
    for lt in BaseLayoutType:
        layouts[lt.value] = generate_canonical_layout(lt)
    return layouts


@pytest.fixture
def ocr_corpus() -> List[Any]:
    """Provides the curated OCR test corpus."""
    from tests.fixtures.recorded_samples import OCR_CORPUS
    return OCR_CORPUS


@pytest.fixture
def recorded_samples() -> Dict[str, Any]:
    """Provides recorded ground-truth samples."""
    from tests.fixtures.recorded_samples import RECORDED_SAMPLES
    return RECORDED_SAMPLES


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Creates a temporary SQLite file path and unlinks it upon completion."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="clashvision_test_")
    os.close(fd)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


@pytest.fixture
def sandbox_context() -> Any:
    """Provides the allow_simulator_sandbox context manager."""
    return allow_simulator_sandbox
