"""
ClashVision Presentation Subsystem.
PySide6/PyQt6 seamless compatibility, dark QSS styles, asynchronous workers, and MainWindow.
"""

from clashvision.ui.qt_compat import (
    IS_PYSIDE6,
    IS_PYQT6,
    IS_HEADLESS_FALLBACK,
    get_qapplication,
)
from clashvision.ui.styles import get_stylesheet, DARK_THEME_QSS
from clashvision.ui.workers import (
    CaptureWorker,
    VisionWorker,
    DecisionWorker,
    SimulationWorker,
)
from clashvision.ui.main_window import MainWindow

__all__ = [
    "IS_PYSIDE6",
    "IS_PYQT6",
    "IS_HEADLESS_FALLBACK",
    "get_qapplication",
    "get_stylesheet",
    "DARK_THEME_QSS",
    "CaptureWorker",
    "VisionWorker",
    "DecisionWorker",
    "SimulationWorker",
    "MainWindow",
]
