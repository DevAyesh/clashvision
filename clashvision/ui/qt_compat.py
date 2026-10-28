"""
ClashVision Presentation Subsystem — Seamless Qt Compatibility Abstraction.
Unifies PySide6, PyQt6, and provides a pure Python headless mock fallback.
Guarantees zero import errors in any CI, headless server, or test environment.
Derived from Requirement R5 and Feature 28.
"""

from __future__ import annotations
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

IS_PYSIDE6 = False
IS_PYQT6 = False
IS_HEADLESS_FALLBACK = False

# Check if headless mock is explicitly forced via environment variable
_FORCE_MOCK = os.environ.get("CLASHVISION_HEADLESS", "").lower() in ("1", "true", "mock") or \
              os.environ.get("FORCE_QT_MOCK", "").lower() in ("1", "true", "mock")

if not _FORCE_MOCK:
    # Attempt 1: PySide6
    try:
        import PySide6
        from PySide6 import QtCore, QtGui, QtWidgets
        from PySide6.QtCore import Signal, Slot, Property, Qt, QObject, QThread, QTimer, QSize, QRect, QPoint, QEvent
        from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QPixmap, QImage, QIcon
        from PySide6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QStackedWidget,
            QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
            QPushButton, QLabel, QTableWidget, QTableWidgetItem,
            QProgressBar, QSlider, QSpinBox, QDoubleSpinBox,
            QComboBox, QLineEdit, QTextEdit, QScrollArea,
            QFrame, QSplitter, QGroupBox, QCheckBox, QRadioButton,
            QHeaderView, QAbstractItemView, QDialog, QMessageBox,
        )
        IS_PYSIDE6 = True
    except ImportError:
        # Attempt 2: PyQt6
        try:
            import PyQt6
            from PyQt6 import QtCore, QtGui, QtWidgets
            from PyQt6.QtCore import (
                pyqtSignal as Signal,
                pyqtSlot as Slot,
                pyqtProperty as Property,
                Qt, QObject, QThread, QTimer, QSize, QRect, QPoint, QEvent
            )
            from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QPixmap, QImage, QIcon
            from PyQt6.QtWidgets import (
                QApplication, QMainWindow, QWidget, QStackedWidget,
                QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
                QPushButton, QLabel, QTableWidget, QTableWidgetItem,
                QProgressBar, QSlider, QSpinBox, QDoubleSpinBox,
                QComboBox, QLineEdit, QTextEdit, QScrollArea,
                QFrame, QSplitter, QGroupBox, QCheckBox, QRadioButton,
                QHeaderView, QAbstractItemView, QDialog, QMessageBox,
            )
            IS_PYQT6 = True

            # Patch Qt unscoped enums for backwards/cross-compatibility
            for enum_cls in (
                getattr(Qt, "AlignmentFlag", None),
                getattr(Qt, "Orientation", None),
                getattr(Qt, "ItemDataRole", None),
                getattr(Qt, "FocusPolicy", None),
                getattr(Qt, "ScrollBarPolicy", None),
            ):
                if enum_cls is not None:
                    for name, val in enum_cls.__members__.items():
                        if not hasattr(Qt, name):
                            try:
                                setattr(Qt, name, val)
                            except Exception:
                                pass
        except ImportError:
            IS_HEADLESS_FALLBACK = True
else:
    IS_HEADLESS_FALLBACK = True


# =============================================================================
# Headless Mock Implementation for Environments Without Qt
# =============================================================================

if IS_HEADLESS_FALLBACK:
    class MockSignalInstance:
        """Signal instance bound to a specific object."""
        def __init__(self) -> None:
            self._slots: List[Callable[..., Any]] = []

        def connect(self, slot: Callable[..., Any]) -> None:
            if slot not in self._slots:
                self._slots.append(slot)

        def disconnect(self, slot: Optional[Callable[..., Any]] = None) -> None:
            if slot is None:
                self._slots.clear()
            elif slot in self._slots:
                self._slots.remove(slot)

        def emit(self, *args: Any, **kwargs: Any) -> None:
            for slot in list(self._slots):
                try:
                    slot(*args, **kwargs)
                except Exception:
                    pass

    class MockSignalDescriptor:
        """Signal class descriptor emulating PySide6.QtCore.Signal."""
        def __init__(self, *types: Any) -> None:
            self.types = types
            self._name = ""

        def __set_name__(self, owner: type, name: str) -> None:
            self._name = name

        def __get__(self, instance: Any, owner: type) -> Any:
            if instance is None:
                return self
            attr_name = f"_mock_sig_{self._name}"
            if not hasattr(instance, attr_name):
                setattr(instance, attr_name, MockSignalInstance())
            return getattr(instance, attr_name)

    Signal = MockSignalDescriptor

    def Slot(*args: Any, **kwargs: Any) -> Callable[[Any], Any]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn
        return decorator

    def Property(*args: Any, **kwargs: Any) -> Any:
        return property

    class MockQtEnum:
        def __init__(self, val: int) -> None:
            self.val = val
        def __eq__(self, other: Any) -> bool:
            return isinstance(other, MockQtEnum) and self.val == other.val

    class MockAlignmentFlag:
        AlignCenter = MockQtEnum(1)
        AlignLeft = MockQtEnum(2)
        AlignRight = MockQtEnum(3)
        AlignTop = MockQtEnum(4)
        AlignBottom = MockQtEnum(5)

    class MockOrientation:
        Horizontal = MockQtEnum(6)
        Vertical = MockQtEnum(7)

    class MockQt:
        AlignCenter = MockAlignmentFlag.AlignCenter
        AlignLeft = MockAlignmentFlag.AlignLeft
        AlignRight = MockAlignmentFlag.AlignRight
        AlignTop = MockAlignmentFlag.AlignTop
        AlignBottom = MockAlignmentFlag.AlignBottom
        Horizontal = MockOrientation.Horizontal
        Vertical = MockOrientation.Vertical
        DisplayRole = MockQtEnum(8)
        UserRole = MockQtEnum(9)
        ScrollBarAlwaysOff = MockQtEnum(10)
        ScrollBarAsNeeded = MockQtEnum(11)
        NoEditTriggers = MockQtEnum(12)
        SelectRows = MockQtEnum(13)

        AlignmentFlag = MockAlignmentFlag
        Orientation = MockOrientation

    Qt = MockQt()

    class MockSize:
        def __init__(self, w: int = 0, h: int = 0) -> None:
            self.w = w
            self.h = h
        def width(self) -> int: return self.w
        def height(self) -> int: return self.h

    QSize = MockSize

    class MockRect:
        def __init__(self, x: int = 0, y: int = 0, w: int = 0, h: int = 0) -> None:
            self.x = x
            self.y = y
            self.w = w
            self.h = h
        def width(self) -> int: return self.w
        def height(self) -> int: return self.h

    QRect = MockRect

    class MockPoint:
        def __init__(self, x: int = 0, y: int = 0) -> None:
            self.x = x
            self.y = y

    QPoint = MockPoint
    QEvent = object

    class MockObject:
        def __init__(self, parent: Optional[Any] = None) -> None:
            self._parent = parent
            self._object_name = ""
            self._properties: Dict[str, Any] = {}

        def setObjectName(self, name: str) -> None:
            self._object_name = name

        def objectName(self) -> str:
            return self._object_name

        def setProperty(self, name: str, value: Any) -> bool:
            self._properties[name] = value
            return True

        def property(self, name: str) -> Any:
            return self._properties.get(name)

        def parent(self) -> Any:
            return self._parent

    QObject = MockObject

    class MockThread(MockObject):
        sig_started = MockSignalDescriptor()
        sig_finished = MockSignalDescriptor()

        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._is_running = False
            self._worker_thread: Optional[Any] = None

        def start(self) -> None:
            self._is_running = True
            self.sig_started.emit()
            import threading
            self._worker_thread = threading.Thread(target=self._run_wrapper, daemon=True)
            self._worker_thread.start()

        def _run_wrapper(self) -> None:
            try:
                self.run()
            finally:
                self._is_running = False
                self.sig_finished.emit()

        def run(self) -> None:
            pass

        def quit(self) -> None:
            self._is_running = False

        def wait(self, msecs: int = 0) -> bool:
            self._is_running = False
            if self._worker_thread is not None and self._worker_thread.is_alive():
                timeout = (msecs / 1000.0) if msecs > 0 else None
                self._worker_thread.join(timeout=timeout)
            return True

        def isRunning(self) -> bool:
            if self._worker_thread is not None:
                return self._worker_thread.is_alive()
            return self._is_running

    QThread = MockThread

    class MockTimer(MockObject):
        timeout = MockSignalDescriptor()

        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._interval = 0
            self._active = False

        def start(self, interval: Optional[int] = None) -> None:
            if interval is not None:
                self._interval = interval
            self._active = True

        def stop(self) -> None:
            self._active = False

        def isActive(self) -> bool:
            return self._active

        def setInterval(self, ms: int) -> None:
            self._interval = ms

    QTimer = MockTimer

    class MockWidget(MockObject):
        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._layout: Optional[Any] = None
            self._visible = True
            self._enabled = True
            self._style_sheet = ""
            self._width = 640
            self._height = 480

        def setLayout(self, layout: Any) -> None:
            self._layout = layout

        def layout(self) -> Any:
            return self._layout

        def setStyleSheet(self, qss: str) -> None:
            self._style_sheet = qss

        def styleSheet(self) -> str:
            return self._style_sheet

        def show(self) -> None:
            self._visible = True

        def hide(self) -> None:
            self._visible = False

        def setVisible(self, v: bool) -> None:
            self._visible = v

        def isVisible(self) -> bool:
            return self._visible

        def setEnabled(self, e: bool) -> None:
            self._enabled = e

        def isEnabled(self) -> bool:
            return self._enabled

        def update(self) -> None:
            pass

        def setFixedSize(self, w: int, h: int) -> None:
            self._width = w
            self._height = h

        def setMinimumSize(self, w: int, h: int) -> None:
            pass

        def resize(self, w: int, h: int) -> None:
            self._width = w
            self._height = h

        def setWindowTitle(self, title: str) -> None:
            pass

        def close(self) -> bool:
            return True

        def closeEvent(self, event: Any) -> None:
            pass

    QWidget = MockWidget

    class MockMainWindow(MockWidget):
        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._central_widget: Optional[MockWidget] = None

        def setCentralWidget(self, widget: MockWidget) -> None:
            self._central_widget = widget

        def centralWidget(self) -> Optional[MockWidget]:
            return self._central_widget

        def closeEvent(self, event: Any) -> None:
            pass

    QMainWindow = MockMainWindow

    class MockLayout(MockObject):
        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._widgets: List[Any] = []

        def addWidget(self, widget: Any, *args: Any, **kwargs: Any) -> None:
            self._widgets.append(widget)

        def addLayout(self, layout: Any, *args: Any, **kwargs: Any) -> None:
            self._widgets.append(layout)

        def addStretch(self, stretch: int = 0) -> None:
            pass

        def setContentsMargins(self, left: int, top: int, right: int, bottom: int) -> None:
            pass

        def setSpacing(self, spacing: int) -> None:
            pass

    QVBoxLayout = MockLayout
    QHBoxLayout = MockLayout
    QGridLayout = MockLayout

    class MockFormLayout(MockLayout):
        def addRow(self, label_or_widget: Any, widget: Optional[Any] = None) -> None:
            self._widgets.append(label_or_widget)
            if widget is not None:
                self._widgets.append(widget)

    QFormLayout = MockFormLayout

    class MockStackedWidget(MockWidget):
        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._widgets: List[MockWidget] = []
            self._current_index: int = 0

        def addWidget(self, widget: MockWidget) -> int:
            self._widgets.append(widget)
            return len(self._widgets) - 1

        def setCurrentIndex(self, index: int) -> None:
            if 0 <= index < len(self._widgets):
                self._current_index = index

        def currentIndex(self) -> int:
            return self._current_index

        def currentWidget(self) -> Optional[MockWidget]:
            if 0 <= self._current_index < len(self._widgets):
                return self._widgets[self._current_index]
            return None

        def count(self) -> int:
            return len(self._widgets)

    QStackedWidget = MockStackedWidget

    class MockButton(MockWidget):
        clicked = MockSignalDescriptor()

        def __init__(self, text: str = "", parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._text = text

        def setText(self, text: str) -> None:
            self._text = text

        def text(self) -> str:
            return self._text

        def click(self) -> None:
            self.clicked.emit()

    QPushButton = MockButton

    class MockLabel(MockWidget):
        def __init__(self, text: str = "", parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._text = text

        def setText(self, text: str) -> None:
            self._text = text

        def text(self) -> str:
            return self._text

        def setAlignment(self, align: Any) -> None:
            pass

        def setPixmap(self, pixmap: Any) -> None:
            pass

    QLabel = MockLabel

    class MockTableWidgetItem:
        def __init__(self, text: str = "") -> None:
            self._text = text
        def setText(self, text: str) -> None: self._text = text
        def text(self) -> str: return self._text

    QTableWidgetItem = MockTableWidgetItem

    class MockTableWidget(MockWidget):
        def __init__(self, rows: int = 0, cols: int = 0, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._rows = rows
            self._cols = cols
            self._items: Dict[Tuple[int, int], MockTableWidgetItem] = {}

        def setRowCount(self, count: int) -> None: self._rows = count
        def rowCount(self) -> int: return self._rows
        def setColumnCount(self, count: int) -> None: self._cols = count
        def columnCount(self) -> int: return self._cols
        def setHorizontalHeaderLabels(self, labels: List[str]) -> None: pass
        def setItem(self, row: int, col: int, item: MockTableWidgetItem) -> None:
            self._items[(row, col)] = item
        def item(self, row: int, col: int) -> Optional[MockTableWidgetItem]:
            return self._items.get((row, col))
        def clearContents(self) -> None: self._items.clear()
        def scrollToBottom(self) -> None: pass
        def horizontalHeader(self) -> Any:
            class MockHeader:
                def setSectionResizeMode(self, *a: Any) -> None: pass
                def setStretchLastSection(self, s: bool) -> None: pass
            return MockHeader()

    QTableWidget = MockTableWidget

    class MockProgressBar(MockWidget):
        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._val = 0
            self._max = 100
        def setValue(self, v: int) -> None: self._val = v
        def value(self) -> int: return self._val
        def setRange(self, mn: int, mx: int) -> None: self._max = mx

    QProgressBar = MockProgressBar

    class MockSlider(MockWidget):
        valueChanged = MockSignalDescriptor()
        def __init__(self, orientation: Any = None, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._val = 0
        def setValue(self, v: int) -> None:
            self._val = v
            self.valueChanged.emit(v)
        def value(self) -> int: return self._val
        def setRange(self, mn: int, mx: int) -> None: pass

    QSlider = MockSlider

    class MockSpinBox(MockWidget):
        valueChanged = MockSignalDescriptor()
        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._val = 0
        def setValue(self, v: int) -> None:
            self._val = v
            self.valueChanged.emit(v)
        def value(self) -> int: return self._val
        def setRange(self, mn: int, mx: int) -> None: pass
        def setSingleStep(self, s: int) -> None: pass

    QSpinBox = MockSpinBox

    class MockDoubleSpinBox(MockWidget):
        valueChanged = MockSignalDescriptor()
        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._val = 0.0
        def setValue(self, v: float) -> None:
            self._val = v
            self.valueChanged.emit(v)
        def value(self) -> float: return self._val
        def setRange(self, mn: float, mx: float) -> None: pass
        def setSingleStep(self, s: float) -> None: pass

    QDoubleSpinBox = MockDoubleSpinBox

    class MockComboBox(MockWidget):
        currentTextChanged = MockSignalDescriptor()
        currentIndexChanged = MockSignalDescriptor()
        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._items: List[str] = []
            self._current_idx = 0
        def addItem(self, item: str) -> None:
            self._items.append(item)
        def addItems(self, items: List[str]) -> None:
            self._items.extend(items)
        def currentText(self) -> str:
            return self._items[self._current_idx] if self._items else ""
        def currentIndex(self) -> int: return self._current_idx
        def setCurrentIndex(self, idx: int) -> None:
            self._current_idx = idx
            self.currentIndexChanged.emit(idx)
            if self._items:
                self.currentTextChanged.emit(self._items[idx])
        def setCurrentText(self, text: str) -> None:
            if text in self._items:
                self.setCurrentIndex(self._items.index(text))
        def count(self) -> int: return len(self._items)
        def clear(self) -> None:
            self._items.clear()
            self._current_idx = 0

    QComboBox = MockComboBox

    class MockLineEdit(MockWidget):
        textChanged = MockSignalDescriptor()
        def __init__(self, text: str = "", parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._text = text
        def setText(self, text: str) -> None:
            self._text = text
            self.textChanged.emit(text)
        def text(self) -> str: return self._text
        def setPlaceholderText(self, text: str) -> None: pass

    QLineEdit = MockLineEdit

    class MockTextEdit(MockWidget):
        def __init__(self, text: str = "", parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._text = text
        def setPlainText(self, text: str) -> None: self._text = text
        def toPlainText(self) -> str: return self._text
        def append(self, text: str) -> None: self._text += f"\n{text}"
        def clear(self) -> None: self._text = ""

    QTextEdit = MockTextEdit

    class MockScrollArea(MockWidget):
        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._w: Optional[MockWidget] = None
        def setWidget(self, w: MockWidget) -> None: self._w = w
        def setWidgetResizable(self, r: bool) -> None: pass

    QScrollArea = MockScrollArea

    class MockFrame(MockWidget):
        pass
    QFrame = MockFrame
    QSplitter = MockWidget
    QGroupBox = MockWidget

    class MockCheckBox(MockWidget):
        toggled = MockSignalDescriptor()
        def __init__(self, text: str = "", parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._text = text
            self._checked = False
        def setChecked(self, c: bool) -> None:
            self._checked = c
            self.toggled.emit(c)
        def isChecked(self) -> bool: return self._checked
        def text(self) -> str: return self._text

    QCheckBox = MockCheckBox
    QRadioButton = MockCheckBox

    class MockHeaderView(MockWidget):
        pass
    QHeaderView = MockHeaderView

    class MockAbstractItemView:
        SelectRows = MockQtEnum(1)
        NoEditTriggers = MockQtEnum(2)
    QAbstractItemView = MockAbstractItemView

    QDialog = MockWidget
    QMessageBox = MockWidget

    # Drawing and pixmaps
    class MockColor:
        def __init__(self, *args: Any) -> None: pass
    QColor = MockColor

    class MockPixmap:
        def __init__(self, *args: Any) -> None: pass
        def isNull(self) -> bool: return True
        def scaled(self, *args: Any) -> Any: return self
    QPixmap = MockPixmap
    QImage = MockPixmap
    QIcon = MockPixmap
    QPainter = object
    QPen = object
    QBrush = object
    QFont = object

    class MockApplication:
        _instance: Optional[MockApplication] = None
        def __init__(self, argv: Optional[List[str]] = None) -> None:
            MockApplication._instance = self
        @classmethod
        def instance(cls) -> Optional[MockApplication]:
            return cls._instance
        def exec(self) -> int: return 0
        def exec_(self) -> int: return 0
        def quit(self) -> None: pass
        def processEvents(self) -> None: pass

    QApplication = MockApplication


def get_qapplication() -> Any:
    """
    Get or create a QApplication singleton instance.
    In testing or headless environments, ensures QT_QPA_PLATFORM is offscreen.
    """
    if IS_HEADLESS_FALLBACK:
        return QApplication.instance() or QApplication([])

    if "QT_QPA_PLATFORM" not in os.environ and sys.platform.startswith("linux"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    app = QApplication.instance()
    if app is None:
        # Offscreen platform fallback if running headless or CI
        if os.environ.get("HEADLESS", "").lower() in ("1", "true") or os.environ.get("CI"):
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QApplication(sys.argv[:1])
    return app
