"""
ClashVision Vision Subsystem — 12 Game State Classifier & Multi-Frame Debouncer.
Implements a multi-feature heuristic ensemble classifying all 12 game states:
1. HOME_VILLAGE
2. MULTIPLAYER_INTERFACE
3. MATCHMAKING_INTERFACE
4. ENEMY_BASE_PREVIEW
5. BATTLE_IN_PROGRESS
6. BATTLE_RESULT
7. TRAINING_INTERFACE
8. ARMY_CONFIGURATION
9. LOADING_SCREEN
10. CONNECTION_ERROR
11. UNEXPECTED_POPUP
12. UNKNOWN
Emits structured VisionAnalysisResult with confidence strictly in [0.0, 1.0],
explanatory evidence dictionary, and a 2-frame debouncing filter.
Derived directly from Requirement R2 and Features 6, 7, 8.
"""

from __future__ import annotations
import time
from typing import Dict, Any, Tuple, Optional, List

import cv2
import numpy as np

from clashvision.core.constants import GameState
from clashvision.core.models import VisionAnalysisResult


class StateDebouncer:
    """
    2-frame temporal consistency filter eliminating transient classification noise and flicker (Feature 8).
    A state transition occurs only after observing the new state for `required_consecutive` frames.
    """

    def __init__(self, required_consecutive: int = 2) -> None:
        self.required = max(1, required_consecutive)
        self.current_state: GameState = GameState.UNKNOWN
        self.candidate_state: GameState = GameState.UNKNOWN
        self.consecutive_count: int = 0

    def update(self, raw_state: GameState) -> GameState:
        """
        Updates the debouncing filter with the latest raw classified state.
        Returns the debounced state.
        """
        if raw_state == self.candidate_state:
            self.consecutive_count += 1
        else:
            self.candidate_state = raw_state
            self.consecutive_count = 1

        if self.consecutive_count >= self.required:
            self.current_state = self.candidate_state

        return self.current_state

    def reset(self, initial_state: GameState = GameState.UNKNOWN) -> None:
        """Resets the debouncer state."""
        self.current_state = initial_state
        self.candidate_state = initial_state
        self.consecutive_count = 0


class StateClassifier:
    """
    Ensemble classifier for all 12 Clash of Clans game states.
    Combines HSV color distribution analysis, UI landmark anchors, button shapes,
    and structural contours to produce high-confidence classifications and supporting evidence.
    """

    def __init__(self, enable_debouncing: bool = False, debounce_frames: int = 2) -> None:
        self.enable_debouncing = enable_debouncing
        self.debouncer = StateDebouncer(required_consecutive=debounce_frames)

    def classify(self, frame: np.ndarray, timestamp: Optional[float] = None) -> VisionAnalysisResult:
        """
        Classifies an input BGR frame into one of the 12 GameStates.
        Returns a VisionAnalysisResult containing state, confidence, and evidence.
        """
        now = timestamp if timestamp is not None else time.time()

        if frame is None or frame.size == 0 or len(frame.shape) < 2:
            return VisionAnalysisResult(
                state=GameState.UNKNOWN,
                confidence=0.0,
                evidence={"error": "empty_frame"},
                timestamp=now,
            )

        # Sanitize and normalize non-uint8 frames safely to uint8 for OpenCV compatibility
        if frame.dtype != np.uint8:
            if frame.dtype == bool:
                frame = (frame.astype(np.uint8) * 255)
            elif np.issubdtype(frame.dtype, np.floating):
                clean_f = np.nan_to_num(frame, copy=True, nan=0.0, posinf=255.0, neginf=0.0)
                if clean_f.size > 0 and clean_f.max() <= 1.0 and clean_f.min() >= 0.0 and clean_f.max() > 0.0:
                    frame = np.clip(clean_f * 255.0, 0, 255).astype(np.uint8)
                else:
                    frame = np.clip(clean_f, 0, 255).astype(np.uint8)
            elif frame.dtype == np.uint16:
                if frame.size > 0 and frame.max() > 255:
                    frame = (frame / 256).astype(np.uint8)
                else:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Standardize 3-channel BGR
        if len(frame.shape) == 2:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] == 1:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] > 4:
            bgr = frame[:, :, :3]
        else:
            bgr = frame

        h, w = bgr.shape[:2]
        if h < 20 or w < 20:
            return VisionAnalysisResult(
                state=GameState.UNKNOWN,
                confidence=0.0,
                evidence={"error": "frame_too_small"},
                timestamp=now,
            )

        # Convert to HSV for robust color analysis
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Extract features and landmark evidence
        evidence: Dict[str, Any] = {}
        scores: Dict[GameState, float] = {state: 0.0 for state in GameState}

        # 1. Global Color Analysis
        self._analyze_global_colors(bgr, hsv, evidence)

        # 2. Matchmaking Clouds Detection
        self._detect_matchmaking(bgr, hsv, evidence, scores)

        # 3. Loading Screen Detection (requires dark background)
        if evidence.get("mean_brightness", 0.0) < 120:
            self._detect_loading_screen(bgr, evidence, scores)

        # 4. Connection Error Modal Detection
        self._detect_connection_error(bgr, evidence, scores)

        # 5. Battle Result Screen Detection
        self._detect_battle_result(bgr, evidence, scores)

        # 6. In-Game Battle Detection (Timer + End Battle)
        self._detect_battle_in_progress(bgr, evidence, scores)

        # 7. Enemy Base Preview Detection (Available Loot + Next)
        self._detect_enemy_preview(bgr, evidence, scores)

        # 8. Modals: Multiplayer, Training, Army Config, Popup
        has_strong = any(
            scores[k] >= 2.5
            for k in (
                GameState.MATCHMAKING_INTERFACE,
                GameState.LOADING_SCREEN,
                GameState.CONNECTION_ERROR,
                GameState.BATTLE_RESULT,
                GameState.BATTLE_IN_PROGRESS,
                GameState.ENEMY_BASE_PREVIEW,
            )
        )
        if not has_strong and evidence.get("grass_ratio", 0.0) > 0.15:
            self._detect_modal_interfaces(bgr, evidence, scores)

        # 9. Home Village Detection (Lush Grass + Top HUD + Attack Button)
        if not any(v >= 2.5 for k, v in scores.items() if k != GameState.UNKNOWN):
            self._detect_home_village(bgr, evidence, scores)

        # Determine winner
        # Exclude UNKNOWN from initial selection
        candidate_scores = {k: v for k, v in scores.items() if k != GameState.UNKNOWN}
        best_state, best_score = max(candidate_scores.items(), key=lambda item: item[1])

        # Confidence normalization
        if best_score < 1.0:
            raw_state = GameState.UNKNOWN
            confidence = max(0.05, min(0.35, best_score * 0.35))
        else:
            raw_state = best_state
            # Sigmoid-like scaling for confidence: score >= 2.0 -> 0.85+, score >= 3.0 -> 0.95+
            confidence = min(0.99, max(0.65, 0.50 + 0.15 * min(3.5, best_score)))

        evidence["best_score"] = round(best_score, 2)
        evidence["raw_state"] = raw_state.value
        evidence["scores"] = {k.value: round(v, 2) for k, v in scores.items() if v > 0}

        # Apply debouncing if active
        if self.enable_debouncing:
            final_state = self.debouncer.update(raw_state)
            evidence["debounced_state"] = final_state.value
            evidence["debouncer_consecutive"] = self.debouncer.consecutive_count
        else:
            final_state = raw_state

        return VisionAnalysisResult(
            state=final_state,
            confidence=round(confidence, 4),
            evidence=evidence,
            timestamp=now,
        )

    # --------------------------------------------------------------------------
    # Feature Extraction Helpers
    # --------------------------------------------------------------------------

    def _analyze_global_colors(self, bgr: np.ndarray, hsv: np.ndarray, evidence: Dict[str, Any]) -> None:
        h, w = bgr.shape[:2]
        total_pixels = h * w

        # Grass terrain (Isometric green)
        # H: 30-85, S: 35-255, V: 35-255
        lower_green = np.array([30, 35, 35], dtype=np.uint8)
        upper_green = np.array([85, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        grass_ratio = float(np.count_nonzero(green_mask)) / total_pixels
        evidence["grass_ratio"] = round(grass_ratio, 3)

        # White/Cloud swirl (High V, Low S)
        lower_white = np.array([0, 0, 195], dtype=np.uint8)
        upper_white = np.array([180, 50, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        white_ratio = float(np.count_nonzero(white_mask)) / total_pixels
        evidence["white_ratio"] = round(white_ratio, 3)

        # Dark overlay ratio
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0], dtype=np.uint8), np.array([180, 255, 60], dtype=np.uint8))
        dark_ratio = float(np.count_nonzero(dark_mask)) / total_pixels
        evidence["dark_ratio"] = round(dark_ratio, 3)

        # Mean brightness
        evidence["mean_brightness"] = round(float(np.mean(hsv[:, :, 2])), 1)

    def _detect_matchmaking(
        self, bgr: np.ndarray, hsv: np.ndarray, evidence: Dict[str, Any], scores: Dict[GameState, float]
    ) -> None:
        h, w = bgr.shape[:2]
        # Matchmaking clouds: high white ratio, very low grass
        white_ratio = evidence.get("white_ratio", 0.0)
        grass_ratio = evidence.get("grass_ratio", 0.0)
        mean_v = evidence.get("mean_brightness", 0.0)

        if white_ratio > 0.40 and grass_ratio < 0.10 and mean_v > 190:
            # Check center magnifying glass circle
            cx, cy = w // 2, h // 2
            cr = max(20, int(min(w, h) * 0.06))
            center_crop = bgr[max(0, cy - cr):min(h, cy + cr), max(0, cx - cr):min(w, cx + cr)]
            if center_crop.size > 0:
                gray_c = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_c, 50, 150)
                edge_count = np.count_nonzero(edges)
                has_spinner = edge_count > 50
                evidence["magnifying_glass_detected"] = has_spinner
                if has_spinner:
                    scores[GameState.MATCHMAKING_INTERFACE] += 3.5
                else:
                    scores[GameState.MATCHMAKING_INTERFACE] += 2.5

    def _detect_loading_screen(
        self, bgr: np.ndarray, evidence: Dict[str, Any], scores: Dict[GameState, float]
    ) -> None:
        h, w = bgr.shape[:2]
        grass_ratio = evidence.get("grass_ratio", 0.0)
        if grass_ratio > 0.10:
            return

        # Loading screen has blue/purple gradient (B > G and R in bottom/top)
        # Sample bottom loading bar area: y ~ 0.85*h to 0.95*h, centered
        bar_y1, bar_y2 = int(h * 0.85), int(h * 0.95)
        bar_x1, bar_x2 = int(w * 0.25), int(w * 0.75)
        bar_crop = bgr[bar_y1:bar_y2, bar_x1:bar_x2]

        if bar_crop.size > 0:
            # Cyan/yellow progress bar fill: BGR ~ (25, 180, 240)
            hsv_bar = cv2.cvtColor(bar_crop, cv2.COLOR_BGR2HSV)
            bar_fill_mask = cv2.inRange(hsv_bar, np.array([15, 100, 100]), np.array([45, 255, 255]))
            has_progress_fill = np.count_nonzero(bar_fill_mask) > (bar_crop.shape[0] * bar_crop.shape[1] * 0.05)

            # Check center "CLASH OF CLANS" logo text
            center_y1, center_y2 = int(h * 0.40), int(h * 0.55)
            center_crop = bgr[center_y1:center_y2, int(w * 0.20):int(w * 0.80)]
            gray_logo = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY) if center_crop.size > 0 else None
            has_center_white_text = gray_logo is not None and np.count_nonzero(gray_logo > 220) > 100

            # Blue gradient background test: B > G and B > 20
            b_mean = np.mean(bgr[:, :, 0])
            g_mean = np.mean(bgr[:, :, 1])
            is_blue_gradient = b_mean > g_mean and b_mean > 30

            if (has_progress_fill or has_center_white_text) and is_blue_gradient:
                evidence["loading_progress_bar"] = has_progress_fill
                evidence["loading_splash_logo"] = has_center_white_text
                scores[GameState.LOADING_SCREEN] += 3.5

    def _detect_connection_error(
        self, bgr: np.ndarray, evidence: Dict[str, Any], scores: Dict[GameState, float]
    ) -> None:
        h, w = bgr.shape[:2]
        grass_ratio = evidence.get("grass_ratio", 0.0)
        mean_v = evidence.get("mean_brightness", 0.0)

        # Connection error has dark background (mean_v < 60) and red modal border (BGR 180, 70, 70)
        if grass_ratio < 0.05 and mean_v < 65:
            # Check modal bounding region (w ~ 0.35 to 0.65, h ~ 0.30 to 0.70)
            mx1, mx2 = int(w * 0.30), int(w * 0.70)
            my1, my2 = int(h * 0.30), int(h * 0.70)
            modal_crop = bgr[my1:my2, mx1:mx2]
            if modal_crop.size > 0:
                # Red border: R > 130, R > G + 40, R > B + 40
                r_chan = modal_crop[:, :, 2].astype(np.int16)
                g_chan = modal_crop[:, :, 1].astype(np.int16)
                b_chan = modal_crop[:, :, 0].astype(np.int16)
                red_border_pixels = np.count_nonzero((r_chan > 120) & (r_chan - g_chan > 35) & (r_chan - b_chan > 35))

                # Reload button (Blue button B > R + 50)
                blue_btn_pixels = np.count_nonzero((b_chan > 120) & (b_chan - r_chan > 40))

                if red_border_pixels > 50 and blue_btn_pixels > 40:
                    evidence["connection_lost_modal"] = True
                    evidence["reload_game_button"] = True
                    scores[GameState.CONNECTION_ERROR] += 3.5

    def _detect_battle_result(
        self, bgr: np.ndarray, evidence: Dict[str, Any], scores: Dict[GameState, float]
    ) -> None:
        h, w = bgr.shape[:2]
        # Victory / Defeat banner across top: y ~ 0.08*h to 0.22*h, centered
        banner_crop = bgr[int(h * 0.07):int(h * 0.22), int(w * 0.25):int(w * 0.75)]
        # Return Home button in bottom: y ~ 0.60*h to 0.75*h
        btn_crop = bgr[int(h * 0.60):int(h * 0.75), int(w * 0.35):int(w * 0.65)]

        if banner_crop.size > 0 and btn_crop.size > 0:
            # Banner: orange/blue banner with white text "VICTORY!"
            hsv_banner = cv2.cvtColor(banner_crop, cv2.COLOR_BGR2HSV)
            banner_orange = cv2.inRange(hsv_banner, np.array([10, 100, 100]), np.array([25, 255, 255]))
            has_banner = np.count_nonzero(banner_orange) > (banner_crop.shape[0] * banner_crop.shape[1] * 0.15)

            # Return home orange button
            hsv_btn = cv2.cvtColor(btn_crop, cv2.COLOR_BGR2HSV)
            btn_orange = cv2.inRange(hsv_btn, np.array([10, 100, 100]), np.array([25, 255, 255]))
            has_return_home = np.count_nonzero(btn_orange) > (btn_crop.shape[0] * btn_crop.shape[1] * 0.08)

            # Check for stars around y ~ 0.25*h to 0.35*h
            star_crop = bgr[int(h * 0.25):int(h * 0.35), int(w * 0.35):int(w * 0.65)]
            hsv_stars = cv2.cvtColor(star_crop, cv2.COLOR_BGR2HSV) if star_crop.size > 0 else None
            yellow_stars = (
                np.count_nonzero(cv2.inRange(hsv_stars, np.array([20, 120, 120]), np.array([35, 255, 255])))
                if hsv_stars is not None else 0
            )

            if has_banner and has_return_home:
                evidence["victory_banner"] = True
                evidence["return_home_button"] = True
                evidence["stars_detected"] = yellow_stars > 20
                scores[GameState.BATTLE_RESULT] += 3.5

    def _detect_battle_in_progress(
        self, bgr: np.ndarray, evidence: Dict[str, Any], scores: Dict[GameState, float]
    ) -> None:
        h, w = bgr.shape[:2]
        grass_ratio = evidence.get("grass_ratio", 0.0)
        if grass_ratio < 0.10:
            return

        # 1. Top-center Battle Timer Box: y ~ 0.015*h to 0.08*h, x ~ 0.42*w to 0.58*w
        timer_crop = bgr[int(h * 0.015):int(h * 0.08), int(w * 0.42):int(w * 0.58)]
        has_timer_box = False
        if timer_crop.size > 0:
            gray_t = cv2.cvtColor(timer_crop, cv2.COLOR_BGR2GRAY)
            white_text = np.count_nonzero(gray_t > 200)
            dark_bg = np.count_nonzero(gray_t < 50)
            has_timer_box = (white_text > 20) and (dark_bg > 50)

        # 2. Bottom-left "End Battle" Red Button: y ~ 0.88*h to 0.98*h, x ~ 0.01*w to 0.14*w
        eb_crop = bgr[int(h * 0.88):int(h * 0.98), int(w * 0.01):int(w * 0.14)]
        has_end_battle = False
        if eb_crop.size > 0:
            r = eb_crop[:, :, 2].astype(np.int16)
            g = eb_crop[:, :, 1].astype(np.int16)
            b = eb_crop[:, :, 0].astype(np.int16)
            red_pixels = np.count_nonzero((r > 130) & (r - g > 40) & (r - b > 40))
            has_end_battle = red_pixels > (eb_crop.shape[0] * eb_crop.shape[1] * 0.15)

        # 3. Top-left "Damage:" box: y ~ 0.015*h to 0.08*h, x ~ 0.01*w to 0.15*w
        dmg_crop = bgr[int(h * 0.015):int(h * 0.08), int(w * 0.01):int(w * 0.15)]
        has_damage_box = False
        if dmg_crop.size > 0:
            gray_d = cv2.cvtColor(dmg_crop, cv2.COLOR_BGR2GRAY)
            has_damage_box = np.count_nonzero(gray_d < 50) > 100

        if has_timer_box and has_end_battle:
            evidence["battle_timer_detected"] = True
            evidence["end_battle_button"] = True
            scores[GameState.BATTLE_IN_PROGRESS] += 3.5
        elif has_end_battle and has_damage_box:
            evidence["end_battle_button"] = True
            evidence["damage_box"] = True
            scores[GameState.BATTLE_IN_PROGRESS] += 2.8

    def _detect_enemy_preview(
        self, bgr: np.ndarray, evidence: Dict[str, Any], scores: Dict[GameState, float]
    ) -> None:
        h, w = bgr.shape[:2]
        grass_ratio = evidence.get("grass_ratio", 0.0)
        if grass_ratio < 0.10:
            return

        # 1. Top-left "Available Loot" Box: x ~ 0.01*w to 0.22*w, y ~ 0.02*h to 0.22*h
        loot_crop = bgr[int(h * 0.02):int(h * 0.22), int(w * 0.01):int(w * 0.22)]
        has_loot_box = False
        if loot_crop.size > 0:
            gray_l = cv2.cvtColor(loot_crop, cv2.COLOR_BGR2GRAY)
            dark_bg = np.count_nonzero(gray_l < 50)
            hsv_l = cv2.cvtColor(loot_crop, cv2.COLOR_BGR2HSV)
            # Gold color text (H: 20-35, S: 150-255, V: 150-255)
            gold_pixels = np.count_nonzero(cv2.inRange(hsv_l, np.array([20, 120, 120]), np.array([35, 255, 255])))
            # Elixir magenta text (H: 140-165)
            magenta_pixels = np.count_nonzero(cv2.inRange(hsv_l, np.array([140, 100, 100]), np.array([165, 255, 255])))
            has_loot_box = dark_bg > 200 and (gold_pixels > 20 or magenta_pixels > 20)

        # 2. Bottom-right "Next" Button: x ~ 0.85*w to 0.99*w, y ~ 0.88*h to 0.98*h
        next_crop = bgr[int(h * 0.88):int(h * 0.98), int(w * 0.85):int(w * 0.99)]
        has_next_button = False
        if next_crop.size > 0:
            hsv_n = cv2.cvtColor(next_crop, cv2.COLOR_BGR2HSV)
            orange_mask = cv2.inRange(hsv_n, np.array([10, 100, 100]), np.array([25, 255, 255]))
            has_next_button = np.count_nonzero(orange_mask) > (next_crop.shape[0] * next_crop.shape[1] * 0.15)

        if has_loot_box and has_next_button:
            evidence["loot_box_top_left"] = True
            evidence["next_button_bottom_right"] = True
            scores[GameState.ENEMY_BASE_PREVIEW] += 3.5
        elif has_next_button:
            evidence["next_button_bottom_right"] = True
            scores[GameState.ENEMY_BASE_PREVIEW] += 2.5
        elif has_loot_box:
            evidence["loot_box_top_left"] = True
            # Weak indicator without Next button (never outscores Home Village)
            scores[GameState.ENEMY_BASE_PREVIEW] += 0.6

    def _detect_modal_interfaces(
        self, bgr: np.ndarray, evidence: Dict[str, Any], scores: Dict[GameState, float]
    ) -> None:
        h, w = bgr.shape[:2]
        scale = w / 1920.0
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Central modal box bounding region
        mx1, mx2 = int(w * 0.15), int(w * 0.85)
        my1, my2 = int(h * 0.10), int(h * 0.90)
        modal_crop = bgr[my1:my2, mx1:mx2]

        if modal_crop.size == 0:
            return

        # Check if the central area is actually covered by a modal box rather than open village grass
        modal_sample = hsv[int(h * 0.35):int(h * 0.45), int(w * 0.40):int(w * 0.60)]
        if modal_sample.size > 0:
            modal_green = np.count_nonzero(cv2.inRange(modal_sample, np.array([30, 40, 40]), np.array([85, 255, 255])))
            modal_green_ratio = modal_green / (modal_sample.shape[0] * modal_sample.shape[1])
            if modal_green_ratio > 0.30:
                # Center is open grass terrain, not covered by a modal dialog
                return

        # 1. UNEXPECTED_POPUP: Close 'X' red circle in upper-right quadrant
        red1 = cv2.inRange(hsv, np.array([0, 130, 120]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 130, 120]), np.array([180, 255, 255]))
        popup_red = cv2.bitwise_or(red1, red2)[int(h * 0.18):int(h * 0.35), int(w * 0.60):int(w * 0.78)]
        if np.count_nonzero(popup_red) > 30:
            evidence["popup_close_x_button"] = True
            evidence["modal_detected"] = "UNEXPECTED_POPUP"
            scores[GameState.UNEXPECTED_POPUP] += 4.5
            scores[GameState.HOME_VILLAGE] = 0.0
            return

        # 2. MULTIPLAYER_INTERFACE: Big orange "Find a Match" button inside modal
        # BGR (30, 160, 230) -> H ~ 15, S > 150, V > 150
        orange = cv2.inRange(hsv, np.array([10, 150, 150]), np.array([25, 255, 255]))
        btn_area = orange[int(h * 0.50):int(h * 0.75), int(w * 0.35):int(w * 0.65)]
        if np.count_nonzero(btn_area) > 400:
            evidence["find_a_match_button"] = True
            evidence["modal_detected"] = "MULTIPLAYER_INTERFACE"
            scores[GameState.MULTIPLAYER_INTERFACE] += 4.5
            scores[GameState.HOME_VILLAGE] = 0.0
            return

        # 3. ARMY_CONFIGURATION vs TRAINING_INTERFACE
        # Army Config has center title "Army Overview & Presets"
        # Training Interface has tab on left "Train Troops"
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        center_title_crop = gray[int(h * 0.16):int(h * 0.25), int(w * 0.40):int(w * 0.75)]
        center_white = np.count_nonzero(center_title_crop > 200)

        left_tab_crop = gray[int(h * 0.14):int(h * 0.22), int(w * 0.18):int(w * 0.35)]
        left_white = np.count_nonzero(left_tab_crop > 200)

        if center_white > 300:
            evidence["army_preset_rows"] = True
            evidence["modal_detected"] = "ARMY_CONFIGURATION"
            scores[GameState.ARMY_CONFIGURATION] += 4.5
            scores[GameState.HOME_VILLAGE] = 0.0
            return

        if left_white > 150:
            evidence["training_tab_detected"] = True
            evidence["modal_detected"] = "TRAINING_INTERFACE"
            scores[GameState.TRAINING_INTERFACE] += 4.5
            scores[GameState.HOME_VILLAGE] = 0.0
            return

    def _detect_home_village(
        self, bgr: np.ndarray, evidence: Dict[str, Any], scores: Dict[GameState, float]
    ) -> None:
        if scores.get(GameState.HOME_VILLAGE, 0.0) == 0.0 and "modal_detected" in evidence:
            return

        h, w = bgr.shape[:2]
        grass_ratio = evidence.get("grass_ratio", 0.0)
        if grass_ratio < 0.20:
            return

        # 1. Top-right resource meters: Gold (yellow icon), Elixir (magenta icon)
        top_right_crop = bgr[int(h * 0.01):int(h * 0.22), int(w * 0.70):int(w * 0.99)]
        has_top_hud = False
        if top_right_crop.size > 0:
            hsv_tr = cv2.cvtColor(top_right_crop, cv2.COLOR_BGR2HSV)
            # Gold meter (H: 15-38, S: 70-255, V: 80-255)
            gold_m = np.count_nonzero(cv2.inRange(hsv_tr, np.array([15, 70, 80]), np.array([38, 255, 255])))
            # Magenta elixir meter (H: 135-170, S: 60-255, V: 70-255)
            elixir_m = np.count_nonzero(cv2.inRange(hsv_tr, np.array([135, 60, 70]), np.array([170, 255, 255])))
            has_top_hud = (gold_m > 25 and elixir_m > 25) or (gold_m > 50)

        # 2. Bottom-left "Attack!" Button: x ~ 0.01*w to 0.18*w, y ~ 0.80*h to 0.99*h
        atk_crop = bgr[int(h * 0.80):int(h * 0.99), int(w * 0.01):int(w * 0.18)]
        has_attack_button = False
        if atk_crop.size > 0:
            hsv_atk = cv2.cvtColor(atk_crop, cv2.COLOR_BGR2HSV)
            orange_atk = cv2.inRange(hsv_atk, np.array([10, 100, 100]), np.array([28, 255, 255]))
            has_attack_button = np.count_nonzero(orange_atk) > 30

        # 3. Top-left Player Level Badge (Blue circle/star)
        top_left_crop = bgr[int(h * 0.02):int(h * 0.14), int(w * 0.01):int(w * 0.10)]
        has_player_badge = False
        if top_left_crop.size > 0:
            b_val = top_left_crop[:, :, 0].astype(np.int16)
            r_val = top_left_crop[:, :, 2].astype(np.int16)
            has_player_badge = np.count_nonzero((b_val > 110) & (b_val - r_val > 30)) > 15

        evidence["top_hud_detected"] = has_top_hud
        evidence["attack_button_detected"] = has_attack_button
        evidence["player_badge_detected"] = has_player_badge

        if has_top_hud and has_attack_button:
            scores[GameState.HOME_VILLAGE] += 3.8
        elif has_attack_button or has_top_hud:
            scores[GameState.HOME_VILLAGE] += 3.0
        elif has_player_badge and grass_ratio > 0.25:
            scores[GameState.HOME_VILLAGE] += 2.5
        elif grass_ratio > 0.40:
            scores[GameState.HOME_VILLAGE] += 1.8

