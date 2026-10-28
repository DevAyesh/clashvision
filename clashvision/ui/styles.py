"""
ClashVision Presentation Subsystem — Dark Modern QSS Stylesheet & Color Palettes.
Matches Requirement R5 and Feature 28 specifications:
- Dark charcoal / slate backgrounds (#121318, #1a1c23, #15161c)
- Neon accents (#00d26a neon emerald, #1e90ff neon blue, #a55eea violet, #ffa502 gold)
- Badge styling for all 12 GameState variants
- Responsive card panels, custom scrollbars, tables, sliders, and buttons
"""

from typing import Dict
from clashvision.core.constants import GameState


# Primary Dark Palette
COLOR_BG_WINDOW = "#121318"
COLOR_BG_SIDEBAR = "#15161c"
COLOR_BG_CARD = "#1a1c23"
COLOR_BG_HOVER = "#232630"
COLOR_BG_SELECTED = "#2b3040"
COLOR_BORDER = "#2c303b"
COLOR_BORDER_FOCUS = "#00d26a"

# Text Colors
COLOR_TEXT_PRIMARY = "#f1f2f6"
COLOR_TEXT_SECONDARY = "#a4b0be"
COLOR_TEXT_MUTED = "#747d8c"

# Neon Accent Colors
COLOR_NEON_GREEN = "#00d26a"
COLOR_NEON_CYAN = "#00b4d8"
COLOR_NEON_BLUE = "#1e90ff"
COLOR_NEON_PURPLE = "#a55eea"
COLOR_NEON_GOLD = "#ffa502"
COLOR_NEON_RED = "#ff4757"
COLOR_NEON_ORANGE = "#ff793f"

# GameState Badge Color Map (Background, Text)
GAME_STATE_COLORS: Dict[str, Dict[str, str]] = {
    GameState.HOME_VILLAGE.value: {"bg": "#1b3a24", "text": "#2ed573", "border": "#2ed573"},
    GameState.MULTIPLAYER_INTERFACE.value: {"bg": "#142c44", "text": "#1e90ff", "border": "#1e90ff"},
    GameState.MATCHMAKING_INTERFACE.value: {"bg": "#331844", "text": "#e056fd", "border": "#e056fd"},
    GameState.ENEMY_BASE_PREVIEW.value: {"bg": "#441a22", "text": "#ff4757", "border": "#ff4757"},
    GameState.BATTLE_IN_PROGRESS.value: {"bg": "#47182a", "text": "#ff6b81", "border": "#ff6b81"},
    GameState.BATTLE_RESULT.value: {"bg": "#422e11", "text": "#ffa502", "border": "#ffa502"},
    GameState.TRAINING_INTERFACE.value: {"bg": "#162844", "text": "#70a1ff", "border": "#70a1ff"},
    GameState.ARMY_CONFIGURATION.value: {"bg": "#1a1a44", "text": "#5352ed", "border": "#5352ed"},
    GameState.LOADING_SCREEN.value: {"bg": "#22262e", "text": "#a4b0be", "border": "#747d8c"},
    GameState.CONNECTION_ERROR.value: {"bg": "#4a1212", "text": "#eb4d4b", "border": "#eb4d4b"},
    GameState.UNEXPECTED_POPUP.value: {"bg": "#442211", "text": "#ff793f", "border": "#ff793f"},
    GameState.UNKNOWN.value: {"bg": "#1e2129", "text": "#747d8c", "border": "#57606f"},
}


def get_state_badge_style(state: str) -> str:
    """Returns CSS style snippet for a GameState badge label."""
    info = GAME_STATE_COLORS.get(state, GAME_STATE_COLORS[GameState.UNKNOWN.value])
    return (
        f"background-color: {info['bg']}; "
        f"color: {info['text']}; "
        f"border: 1px solid {info['border']}; "
        f"border-radius: 4px; "
        f"padding: 4px 10px; "
        f"font-weight: bold; "
        f"font-size: 12px;"
    )


DARK_THEME_QSS = f"""
/* Root Window Styling */
QMainWindow, QDialog, QWidget#root_container {{
    background-color: {COLOR_BG_WINDOW};
    color: {COLOR_TEXT_PRIMARY};
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
}}

QWidget {{
    background-color: transparent;
    color: {COLOR_TEXT_PRIMARY};
}}

/* Sidebar Frame */
QFrame#sidebar {{
    background-color: {COLOR_BG_SIDEBAR};
    border-right: 1px solid {COLOR_BORDER};
}}

/* Navigation Sidebar Buttons */
QPushButton.sidebar_nav_btn {{
    background-color: transparent;
    color: {COLOR_TEXT_SECONDARY};
    border: none;
    border-radius: 6px;
    padding: 10px 14px;
    text-align: left;
    font-size: 13px;
    font-weight: 500;
}}

QPushButton.sidebar_nav_btn:hover {{
    background-color: {COLOR_BG_HOVER};
    color: {COLOR_TEXT_PRIMARY};
}}

QPushButton.sidebar_nav_btn:checked, QPushButton.sidebar_nav_btn.active {{
    background-color: {COLOR_BG_SELECTED};
    color: {COLOR_NEON_GREEN};
    font-weight: bold;
    border-left: 3px solid {COLOR_NEON_GREEN};
}}

/* Header Status Bar Frame */
QFrame#top_status_bar {{
    background-color: {COLOR_BG_CARD};
    border-bottom: 1px solid {COLOR_BORDER};
    padding: 6px 12px;
}}

/* Cards & Panel Containers */
QFrame.dashboard_card {{
    background-color: {COLOR_BG_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 12px;
}}

QFrame.dashboard_card:hover {{
    border-color: #3e4452;
}}

/* Standard Push Buttons */
QPushButton {{
    background-color: #242834;
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 5px;
    padding: 6px 14px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: #2e3342;
    border-color: #43495b;
}}

QPushButton:pressed {{
    background-color: #1a1d26;
}}

QPushButton:disabled {{
    background-color: #181a20;
    color: {COLOR_TEXT_MUTED};
    border-color: #20232b;
}}

/* Primary Action Buttons */
QPushButton.btn_primary {{
    background-color: {COLOR_NEON_GREEN};
    color: #0b1a10;
    border: 1px solid {COLOR_NEON_GREEN};
    font-weight: bold;
}}

QPushButton.btn_primary:hover {{
    background-color: #00e876;
}}

/* Emergency Stop Button */
QPushButton#emergency_stop_btn {{
    background-color: {COLOR_NEON_RED};
    color: #ffffff;
    border: 2px solid #ff6b81;
    border-radius: 6px;
    font-weight: bold;
    font-size: 13px;
    padding: 6px 16px;
}}

QPushButton#emergency_stop_btn:hover {{
    background-color: #ff3344;
}}

/* Mode Banners */
QLabel#mode_banner_live {{
    background-color: #142a1b;
    color: {COLOR_NEON_GREEN};
    border: 1px solid {COLOR_NEON_GREEN};
    border-radius: 5px;
    padding: 4px 12px;
    font-weight: bold;
}}

QLabel#mode_banner_sim {{
    background-color: #2e1c3b;
    color: {COLOR_NEON_PURPLE};
    border: 1px solid {COLOR_NEON_PURPLE};
    border-radius: 5px;
    padding: 4px 12px;
    font-weight: bold;
}}

/* Line Edits & Text Inputs */
QLineEdit, QTextEdit {{
    background-color: #161820;
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 5px;
    padding: 6px 10px;
    selection-background-color: {COLOR_NEON_BLUE};
}}

QLineEdit:focus, QTextEdit:focus {{
    border-color: {COLOR_NEON_GREEN};
}}

/* Combo Boxes & Spin Boxes */
QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: #161820;
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 5px;
    padding: 5px 8px;
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLOR_BG_CARD};
    color: {COLOR_TEXT_PRIMARY};
    selection-background-color: {COLOR_BG_SELECTED};
    border: 1px solid {COLOR_BORDER};
}}

/* Table Widgets */
QTableWidget {{
    background-color: {COLOR_BG_CARD};
    color: {COLOR_TEXT_PRIMARY};
    gridline-color: {COLOR_BORDER};
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
}}

QTableWidget::item {{
    padding: 6px;
}}

QTableWidget::item:selected {{
    background-color: {COLOR_BG_SELECTED};
    color: {COLOR_NEON_CYAN};
}}

QHeaderView::section {{
    background-color: #14161d;
    color: {COLOR_TEXT_SECONDARY};
    padding: 6px 8px;
    border: none;
    border-bottom: 1px solid {COLOR_BORDER};
    font-weight: 600;
}}

/* Progress Bars */
QProgressBar {{
    background-color: #14161d;
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    text-align: center;
    color: {COLOR_TEXT_PRIMARY};
    height: 16px;
}}

QProgressBar::chunk {{
    background-color: {COLOR_NEON_GREEN};
    border-radius: 3px;
}}

/* Sliders */
QSlider::groove:horizontal {{
    border: none;
    height: 6px;
    background: #242834;
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background: {COLOR_NEON_GREEN};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: #f1f2f6;
    border: 2px solid {COLOR_NEON_GREEN};
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}}

/* Scrollbars */
QScrollBar:vertical {{
    background: #121318;
    width: 10px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background: #2b303d;
    min-height: 20px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical:hover {{
    background: #3e4456;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""


def get_stylesheet() -> str:
    """Returns the full dark theme application stylesheet."""
    return DARK_THEME_QSS
