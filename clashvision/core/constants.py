"""
ClashVision core constants and enumerations.
Defines game states, strategy types, resource priorities, coordinate reference dimensions,
and default application settings.
"""

from enum import Enum, unique


@unique
class GameState(str, Enum):
    """
    12 Game states identifiable by the vision classification engine.
    Derived directly from Requirement R2.
    """
    HOME_VILLAGE = "HOME_VILLAGE"
    MULTIPLAYER_INTERFACE = "MULTIPLAYER_INTERFACE"
    MATCHMAKING_INTERFACE = "MATCHMAKING_INTERFACE"
    ENEMY_BASE_PREVIEW = "ENEMY_BASE_PREVIEW"
    BATTLE_IN_PROGRESS = "BATTLE_IN_PROGRESS"
    BATTLE_RESULT = "BATTLE_RESULT"
    TRAINING_INTERFACE = "TRAINING_INTERFACE"
    ARMY_CONFIGURATION = "ARMY_CONFIGURATION"
    LOADING_SCREEN = "LOADING_SCREEN"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    UNEXPECTED_POPUP = "UNEXPECTED_POPUP"
    UNKNOWN = "UNKNOWN"

    @property
    def display_name(self) -> str:
        """User-friendly display name for GUI views."""
        return self.value.replace("_", " ").title()

    @property
    def is_actionable(self) -> bool:
        """Whether this game state can undergo tactical decision-making."""
        return self in (
            GameState.ENEMY_BASE_PREVIEW,
            GameState.BATTLE_IN_PROGRESS,
            GameState.HOME_VILLAGE,
        )


@unique
class StrategyType(str, Enum):
    """
    Modular attack strategies supported by the recommendation engine.
    Derived directly from Requirement R3.
    """
    SNEAKY_GOBLIN = "SNEAKY_GOBLIN"
    BARCH = "BARCH"
    MASS_MINER = "MASS_MINER"
    MASS_DRAGON = "MASS_DRAGON"
    CUSTOM = "CUSTOM"

    @property
    def display_name(self) -> str:
        names = {
            StrategyType.SNEAKY_GOBLIN: "Sneaky Goblin Farming",
            StrategyType.BARCH: "BARCH Farming",
            StrategyType.MASS_MINER: "Mass Miner Assault",
            StrategyType.MASS_DRAGON: "Mass Dragon Corridor",
            StrategyType.CUSTOM: "User-Defined Custom Strategy",
        }
        return names.get(self, self.value)


@unique
class RiskLevel(str, Enum):
    """Risk category for strategy recommendations."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@unique
class ResourcePriority(str, Enum):
    """Resource weighting preferences for base scoring."""
    BALANCED = "BALANCED"
    GOLD_FIRST = "GOLD_FIRST"
    ELIXIR_FIRST = "ELIXIR_FIRST"
    DARK_ELIXIR_FIRST = "DARK_ELIXIR_FIRST"


@unique
class CaptureBackend(str, Enum):
    """Screen grabber capture backend selection."""
    MSS = "MSS"
    MOCK = "MOCK"
    GDI = "GDI"


# Normalized virtual reference resolution (16:9 standard full HD)
VIRTUAL_REFERENCE_WIDTH: int = 1920
VIRTUAL_REFERENCE_HEIGHT: int = 1080
REFERENCE_ASPECT_RATIO: float = VIRTUAL_REFERENCE_WIDTH / VIRTUAL_REFERENCE_HEIGHT  # 16:9 ~ 1.7777777778

# Process matching signatures for Google Play Games PC and compatible emulators
DEFAULT_TARGET_PROCESS_NAMES = [
    "PlayGamesPC.exe",
    "PlayGamesPC",
    "crosvm.exe",
    "crosvm",
    "Service.exe",
    "HD-Player.exe",
    "HD-Player",
    "BlueStacks.exe",
    "BlueStacks",
    "dnplayer.exe",
    "dnplayer",
    "Nox.exe",
    "Nox",
    "MEmu.exe",
    "MEmu",
]

DEFAULT_TARGET_WINDOW_TITLES = [
    r"Clash of Clans",
    r"Google Play Games",
    r"Google Play Games on PC",
    r"BlueStacks",
    r"LDPlayer",
    r"NoxPlayer",
    r"MEmu",
]

# Standard default operational bounds
DEFAULT_CAPTURE_FPS: int = 15
MIN_CAPTURE_FPS: int = 1
MAX_CAPTURE_FPS: int = 60

# Default resource thresholds for farming recommendations
DEFAULT_MIN_GOLD: int = 400_000
DEFAULT_MIN_ELIXIR: int = 400_000
DEFAULT_MIN_DARK_ELIXIR: int = 3_000

# Emergency Stop safety latency limit in milliseconds (R4 Requirement)
EMERGENCY_STOP_LATENCY_MAX_MS: int = 50

# Normalized [0.0..1.0] UI Navigation Button Locations (Resolution Invariant)
# Derived from exact ground-truth bounding box centroids in Clash of Clans
SIMULATED_UI_BATTLE_BUTTON_NORM: tuple[float, float] = (0.0547, 0.9097)            # "Battle" button at Home Village
SIMULATED_UI_FIND_MATCH_BUTTON_NORM: tuple[float, float] = (0.1426, 0.7031)         # "Find a Match" button (1,500 Gold)
SIMULATED_UI_ATTACK_CONFIRM_BUTTON_NORM: tuple[float, float] = (0.8438, 0.8524)    # Green "Attack!" confirmation button
SIMULATED_UI_NEXT_BUTTON_NORM: tuple[float, float] = (0.9375, 0.9375)              # "Next" / Skip candidate button
SIMULATED_UI_RETURN_HOME_BUTTON_NORM: tuple[float, float] = (0.0547, 0.9097)        # "Return Home" / "End Battle" button

# Simulated UI Navigation Reference Coordinates (Standard 1920x1080 Full HD)
SIMULATED_UI_BATTLE_BUTTON: tuple[int, int] = (105, 982)            # "Battle" button at Home Village (bottom-left)
SIMULATED_UI_FIND_MATCH_BUTTON: tuple[int, int] = (274, 759)       # "Find a Match" button (1,500 Gold)
SIMULATED_UI_ATTACK_CONFIRM_BUTTON: tuple[int, int] = (1620, 921)  # Green "Attack!" confirmation button
SIMULATED_UI_NEXT_BUTTON: tuple[int, int] = (1800, 1012)           # "Next" / Skip candidate button (bottom-right)
SIMULATED_UI_RETURN_HOME_BUTTON: tuple[int, int] = (105, 982)      # "Return Home" / "End Battle" button (bottom-left)

# Standard Keyboard Shortcuts for Autonomous Simulation Control
HOTKEY_SIMULATION_START: str = "F5"
HOTKEY_SIMULATION_PAUSE: str = "Space"
HOTKEY_SIMULATION_STOP: str = "Shift+F5"
HOTKEY_SIMULATION_EMERGENCY_STOP: str = "Escape"
HOTKEY_SIMULATION_EMERGENCY_STOP_ALT: str = "F12"
