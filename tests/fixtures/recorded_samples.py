"""
Recorded samples and ground-truth validation dataset for ClashVision offline testing.
Contains curated ground truths for game states, OCR strings, layout metrics, and strategy recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RecordedSample:
    """Ground truth metadata for an offline test scenario."""
    sample_id: str
    state: str
    confidence_expected: float
    key_landmarks: List[str]
    gold: Optional[int] = None
    elixir: Optional[int] = None
    dark_elixir: Optional[int] = None
    timer: Optional[str] = None
    troop_counts: Dict[str, int] = field(default_factory=dict)
    recommended_strategy: Optional[str] = None
    resolution: Tuple[int, int] = (1920, 1080)
    notes: str = ""


# Comprehensive Ground-Truth Dataset for all 12 Game States
RECORDED_SAMPLES: Dict[str, RecordedSample] = {
    "SAMPLE_HOME_01": RecordedSample(
        sample_id="SAMPLE_HOME_01",
        state="HOME_VILLAGE",
        confidence_expected=0.95,
        key_landmarks=["town_hall", "top_hud_gold", "top_hud_elixir", "attack_button_bottom_left"],
        gold=850_000,
        elixir=720_000,
        dark_elixir=4_500,
        troop_counts={"sneaky_goblin": 80, "super_wall_breaker": 4, "jump_spell": 2},
        notes="Standard maxed home village with attack button ready.",
    ),
    "SAMPLE_MULTI_01": RecordedSample(
        sample_id="SAMPLE_MULTI_01",
        state="MULTIPLAYER_INTERFACE",
        confidence_expected=0.92,
        key_landmarks=["multiplayer_modal", "find_a_match_button", "shield_timer_banner"],
        gold=850_000,
        elixir=720_000,
        dark_elixir=4_500,
        notes="Multiplayer matchmaking confirmation dialogue active.",
    ),
    "SAMPLE_MATCH_01": RecordedSample(
        sample_id="SAMPLE_MATCH_01",
        state="MATCHMAKING_INTERFACE",
        confidence_expected=0.90,
        key_landmarks=["cloud_screen", "magnifying_glass_spinner", "searching_text"],
        gold=None,
        elixir=None,
        dark_elixir=None,
        notes="Active matchmaking clouds transition screen.",
    ),
    "SAMPLE_ENEMY_01": RecordedSample(
        sample_id="SAMPLE_ENEMY_01",
        state="ENEMY_BASE_PREVIEW",
        confidence_expected=0.96,
        key_landmarks=["loot_cart_box_top_left", "next_button_bottom_right", "troop_tray_bottom"],
        gold=920_000,
        elixir=880_000,
        dark_elixir=7_500,
        timer="0:30",
        recommended_strategy="SNEAKY_GOBLIN",
        notes="High-resource enemy base scout with 30s scout timer.",
    ),
    "SAMPLE_BATTLE_01": RecordedSample(
        sample_id="SAMPLE_BATTLE_01",
        state="BATTLE_IN_PROGRESS",
        confidence_expected=0.94,
        key_landmarks=["battle_timer_top_center", "damage_percentage", "end_battle_button_bottom_left"],
        gold=650_000,
        elixir=500_000,
        dark_elixir=3_200,
        timer="2:15",
        notes="Live combat engagement active at 2 minutes 15 seconds remaining.",
    ),
    "SAMPLE_RESULT_01": RecordedSample(
        sample_id="SAMPLE_RESULT_01",
        state="BATTLE_RESULT",
        confidence_expected=0.93,
        key_landmarks=["victory_banner", "stars_container", "loot_stolen_summary", "return_home_button"],
        gold=890_000,
        elixir=840_000,
        dark_elixir=7_100,
        notes="Post-battle victory screen with 3 stars and stolen loot totals.",
    ),
    "SAMPLE_TRAIN_01": RecordedSample(
        sample_id="SAMPLE_TRAIN_01",
        state="TRAINING_INTERFACE",
        confidence_expected=0.91,
        key_landmarks=["training_tab_bar", "barracks_queue_slots", "quick_train_tab"],
        notes="Army camp / Barracks troop training queue screen.",
    ),
    "SAMPLE_ARMY_01": RecordedSample(
        sample_id="SAMPLE_ARMY_01",
        state="ARMY_CONFIGURATION",
        confidence_expected=0.88,
        key_landmarks=["army_preset_rows", "edit_army_button", "troop_capacity_meter"],
        notes="Army configuration and preset customization view.",
    ),
    "SAMPLE_LOAD_01": RecordedSample(
        sample_id="SAMPLE_LOAD_01",
        state="LOADING_SCREEN",
        confidence_expected=0.97,
        key_landmarks=["splash_logo", "progress_bar_bottom", "loading_text"],
        notes="App boot / stage reload splash screen.",
    ),
    "SAMPLE_CONN_01": RecordedSample(
        sample_id="SAMPLE_CONN_01",
        state="CONNECTION_ERROR",
        confidence_expected=0.95,
        key_landmarks=["connection_lost_modal", "reload_button", "sync_error_text"],
        notes="Network drop or out-of-sync alert modal.",
    ),
    "SAMPLE_POPUP_01": RecordedSample(
        sample_id="SAMPLE_POPUP_01",
        state="UNEXPECTED_POPUP",
        confidence_expected=0.89,
        key_landmarks=["modal_backdrop", "close_x_button", "promo_offer_banner"],
        notes="Promotional in-game announcement with dismiss button.",
    ),
    "SAMPLE_UNKNOWN_01": RecordedSample(
        sample_id="SAMPLE_UNKNOWN_01",
        state="UNKNOWN",
        confidence_expected=0.10,
        key_landmarks=[],
        notes="Unrecognized screen or desktop window backdrop.",
    ),
}


# Curated OCR String Testing Corpus with Expected Parse Results
# Format: (raw_string, expected_value, is_valid, description)
OCR_CORPUS: List[Tuple[str, Optional[int], bool, str]] = [
    # Standard comma-separated numbers
    ("800,000", 800_000, True, "Standard 6-digit with comma"),
    ("1,250,000", 1_250_000, True, "Standard 7-digit million with comma"),
    ("50,000", 50_000, True, "Standard 5-digit with comma"),
    ("3,500", 3_500, True, "Standard 4-digit with comma"),
    ("0", 0, True, "Zero"),
    ("10,000,000", 10_000_000, True, "Ten million max capacity"),
    ("12", 12, True, "Small integer"),

    # European dot-separated numbers
    ("800.000", 800_000, True, "European dot separator 800k"),
    ("1.250.000", 1_250_000, True, "European dot separator 1.25M"),

    # Compact 'K' notation
    ("800K", 800_000, True, "Compact uppercase K"),
    ("800k", 800_000, True, "Compact lowercase k"),
    ("350K", 350_000, True, "Compact 350K"),
    ("5.5K", 5_500, True, "Decimal K notation"),
    ("0K", 0, True, "Zero K"),

    # Compact 'M' notation
    ("1.2M", 1_200_000, True, "Compact decimal 1.2M"),
    ("1.25M", 1_250_000, True, "Compact two-decimal 1.25M"),
    ("2M", 2_000_000, True, "Integer million notation 2M"),
    ("0.8M", 800_000, True, "Fractional million 0.8M"),
    ("1.2m", 1_200_000, True, "Lowercase m"),

    # Text noise / prefixes
    ("Gold: 850,000", 850_000, True, "Prefixed with label 'Gold:'"),
    ("  720,000  ", 720_000, True, "Surrounded with whitespace"),
    ("Elixir: 1.5M", 1_500_000, True, "Prefixed with label 'Elixir:' and compact M"),
    ("DE: 4,500", 4_500, True, "Dark Elixir abbreviation"),

    # Malformed / Unrecognized strings (Must return None, ZERO hallucination)
    ("???", None, False, "Question marks"),
    ("---", None, False, "Dashes"),
    ("N/A", None, False, "Not applicable string"),
    ("Not Found", None, False, "Alpha text with no digits"),
    ("", None, False, "Empty string"),
    ("   ", None, False, "Whitespace only"),
    ("abcxyz", None, False, "Random letters"),
    ("$$$", None, False, "Currency symbols only"),
]


# Timer Parsing Corpus
# Format: (timer_string, expected_seconds, is_valid)
TIMER_CORPUS: List[Tuple[str, Optional[int], bool]] = [
    ("3:00", 180, True),
    ("2:45", 165, True),
    ("1:15", 75, True),
    ("0:59", 59, True),
    ("0:01", 1, True),
    ("0:00", 0, True),
    ("invalid", None, False),
    ("", None, False),
    ("99:99", 6039, True),
]



def get_sample_by_state(state: str) -> Optional[RecordedSample]:
    """Retrieve recorded sample for a game state name."""
    for s in RECORDED_SAMPLES.values():
        if s.state == state:
            return s
    return None
