"""
Synthetic image and crop fixture generator for ClashVision.
Produces pixel-accurate, deterministic 1080p and 720p synthetic frames for all 12 game states,
resource ROI crops with standard and formatted strings ("800,000", "800K", "1.2M", "2:45"),
and noise-corrupted variants for robustness testing.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class GameStateType(str, Enum):
    """Mirror of GameState enumeration for decoupled offline fixture generation."""
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


@dataclass
class SyntheticFrameConfig:
    """Configuration parameters for synthetic full-frame generation."""
    state: GameStateType = GameStateType.HOME_VILLAGE
    width: int = 1920
    height: int = 1080
    seed: int = 42
    gold_amount: int = 850_000
    elixir_amount: int = 720_000
    dark_elixir_amount: int = 4_500
    battle_timer: str = "2:45"
    destruction_percent: int = 0
    stars: int = 0
    modal_title: str = ""
    add_noise: bool = False
    noise_sigma: float = 0.0


@dataclass
class SyntheticCropConfig:
    """Configuration for OCR text crop generation."""
    text: str = "800,000"
    width: int = 160
    height: int = 40
    bg_color: Tuple[int, int, int] = (25, 25, 30)  # BGR
    text_color: Tuple[int, int, int] = (255, 255, 255)  # BGR
    font_scale: float = 0.8
    thickness: int = 2
    noise_level: float = 0.0
    blur_kernel: int = 0
    low_contrast: bool = False
    invert: bool = False


class SyntheticGenerator:
    """
    Deterministic programmatic generator of synthetic Clash of Clans frames and OCR crops.
    Emits genuine BGR numpy arrays and PIL Images with visual anchors for computer vision testing.
    """

    def __init__(self, default_seed: int = 42) -> None:
        self.default_seed = default_seed

    def generate_frame(
        self,
        config: Optional[SyntheticFrameConfig] = None,
        as_pil: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, Image.Image]:
        """
        Generate a synthetic frame for a specified game state.
        Returns a numpy array (BGR, uint8) or PIL Image.
        """
        if config is None:
            config = SyntheticFrameConfig(**kwargs)
        else:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        # Set random seed for determinism
        rng = np.random.RandomState(config.seed)

        # Generate base frame
        img = np.zeros((config.height, config.width, 3), dtype=np.uint8)

        # State-specific drawing
        if config.state == GameStateType.HOME_VILLAGE:
            img = self._draw_home_village(img, config, rng)
        elif config.state == GameStateType.MULTIPLAYER_INTERFACE:
            img = self._draw_multiplayer_interface(img, config, rng)
        elif config.state == GameStateType.MATCHMAKING_INTERFACE:
            img = self._draw_matchmaking_interface(img, config, rng)
        elif config.state == GameStateType.ENEMY_BASE_PREVIEW:
            img = self._draw_enemy_base_preview(img, config, rng)
        elif config.state == GameStateType.BATTLE_IN_PROGRESS:
            img = self._draw_battle_in_progress(img, config, rng)
        elif config.state == GameStateType.BATTLE_RESULT:
            img = self._draw_battle_result(img, config, rng)
        elif config.state == GameStateType.TRAINING_INTERFACE:
            img = self._draw_training_interface(img, config, rng)
        elif config.state == GameStateType.ARMY_CONFIGURATION:
            img = self._draw_army_configuration(img, config, rng)
        elif config.state == GameStateType.LOADING_SCREEN:
            img = self._draw_loading_screen(img, config, rng)
        elif config.state == GameStateType.CONNECTION_ERROR:
            img = self._draw_connection_error(img, config, rng)
        elif config.state == GameStateType.UNEXPECTED_POPUP:
            img = self._draw_unexpected_popup(img, config, rng)
        else:  # UNKNOWN
            img = self._draw_unknown_state(img, config, rng)

        # Add optional Gaussian noise
        if config.add_noise and config.noise_sigma > 0:
            noise = rng.normal(0, config.noise_sigma, img.shape).astype(np.float32)
            noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            img = noisy

        if as_pil:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        return img

    def generate_crop(
        self,
        config: Optional[SyntheticCropConfig] = None,
        as_pil: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, Image.Image]:
        """
        Generate a synthetic OCR text crop (e.g. resource counter or timer).
        """
        if config is None:
            config = SyntheticCropConfig(**kwargs)
        else:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        # Create background canvas
        crop = np.full((config.height, config.width, 3), config.bg_color, dtype=np.uint8)

        # Determine font positioning
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, baseline = cv2.getTextSize(
            config.text, font, config.font_scale, config.thickness
        )
        text_x = max(4, (config.width - text_size[0]) // 2)
        text_y = max(config.height - 6, (config.height + text_size[1]) // 2)

        # Text color
        tc = config.text_color
        if config.low_contrast:
            # Shift text color towards background color
            tc = tuple(int(0.6 * c + 0.4 * b) for c, b in zip(tc, config.bg_color))

        cv2.putText(
            crop,
            config.text,
            (text_x, text_y),
            font,
            config.font_scale,
            tc,
            config.thickness,
            cv2.LINE_AA,
        )

        # Apply noise if configured
        if config.noise_level > 0:
            noise = np.random.normal(0, config.noise_level * 50, crop.shape).astype(np.float32)
            crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Apply blur if requested
        if config.blur_kernel > 1:
            k = config.blur_kernel if config.blur_kernel % 2 == 1 else config.blur_kernel + 1
            crop = cv2.GaussianBlur(crop, (k, k), 0)

        # Invert if requested
        if config.invert:
            crop = cv2.bitwise_not(crop)

        if as_pil:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        return crop

    # --------------------------------------------------------------------------
    # Visual Generators for each of the 12 Game States
    # --------------------------------------------------------------------------

    def _draw_grass_terrain(self, img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Draw isometric green grass terrain typical of Clash of Clans base screens."""
        h, w = img.shape[:2]
        # Base lush green: B=35, G=125, R=55
        img[:] = (35, 125, 55)
        # Add subtle isometric grid / texture
        grid_step = max(30, int(w * 0.03))
        overlay = img.copy()
        for x in range(0, w, grid_step):
            cv2.line(overlay, (x, 0), (x + h // 2, h), (45, 140, 65), 1)
            cv2.line(overlay, (x, 0), (x - h // 2, h), (30, 115, 50), 1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        return img

    def _draw_top_hud(self, img: np.ndarray, config: SyntheticFrameConfig) -> np.ndarray:
        """Draw top-right resource meters (Gold, Elixir, Gems) and top-left profile."""
        h, w = img.shape[:2]
        scale = w / 1920.0

        # Top-left player level badge (Blue star)
        badge_center = (int(60 * scale), int(60 * scale))
        cv2.circle(img, badge_center, int(25 * scale), (200, 120, 20), -1)  # Blue circle
        cv2.putText(
            img, "145", (int(45 * scale), int(68 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255), int(2 * scale), cv2.LINE_AA
        )

        # Top-right Resource Boxes
        # 1. Gold Box (Top right)
        self._draw_resource_meter(
            img,
            x=int(w - 380 * scale),
            y=int(25 * scale),
            width=int(350 * scale),
            height=int(40 * scale),
            label="Gold",
            value=f"{config.gold_amount:,}",
            color=(0, 215, 255),  # Gold BGR
            scale=scale,
        )

        # 2. Elixir Box
        self._draw_resource_meter(
            img,
            x=int(w - 380 * scale),
            y=int(75 * scale),
            width=int(350 * scale),
            height=int(40 * scale),
            label="Elixir",
            value=f"{config.elixir_amount:,}",
            color=(210, 50, 210),  # Magenta Elixir BGR
            scale=scale,
        )

        # 3. Dark Elixir Box
        self._draw_resource_meter(
            img,
            x=int(w - 380 * scale),
            y=int(125 * scale),
            width=int(350 * scale),
            height=int(40 * scale),
            label="Dark Elixir",
            value=f"{config.dark_elixir_amount:,}",
            color=(30, 30, 30),  # Black/Dark Elixir
            scale=scale,
        )
        return img

    def _draw_resource_meter(
        self,
        img: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        label: str,
        value: str,
        color: Tuple[int, int, int],
        scale: float,
    ) -> None:
        """Draws an individual resource HUD meter."""
        # Background bar
        cv2.rectangle(img, (x, y), (x + width, y + height), (40, 40, 40), -1)
        cv2.rectangle(img, (x, y), (x + width, y + height), (90, 90, 90), 2)
        # Resource icon box
        icon_w = int(height * 1.1)
        cv2.rectangle(img, (x + width - icon_w, y), (x + width, y + height), color, -1)
        # Resource text value
        cv2.putText(
            img,
            value,
            (x + int(15 * scale), y + int(27 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65 * scale,
            (255, 255, 255),
            int(2 * scale),
            cv2.LINE_AA,
        )

    def _draw_home_village(
        self, img: np.ndarray, config: SyntheticFrameConfig, rng: np.random.RandomState
    ) -> np.ndarray:
        """1. HOME_VILLAGE: Green terrain, Town Hall, top HUD, bottom Attack button."""
        h, w = img.shape[:2]
        scale = w / 1920.0
        self._draw_grass_terrain(img, rng)
        self._draw_top_hud(img, config)

        # Town Hall building in center
        cx, cy = w // 2, h // 2
        th_w, th_h = int(140 * scale), int(140 * scale)
        cv2.rectangle(img, (cx - th_w // 2, cy - th_h // 2), (cx + th_w // 2, cy + th_h // 2), (50, 70, 180), -1)
        cv2.putText(
            img, "TH 15", (cx - int(35 * scale), cy + int(10 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (255, 255, 255), int(2 * scale), cv2.LINE_AA
        )

        # Bottom-left "Attack" button (iconic yellow/orange rounded button)
        atk_x, atk_y = int(30 * scale), int(h - 110 * scale)
        atk_w, atk_h = int(170 * scale), int(80 * scale)
        cv2.rectangle(img, (atk_x, atk_y), (atk_x + atk_w, atk_y + atk_h), (20, 140, 240), -1)
        cv2.rectangle(img, (atk_x, atk_y), (atk_x + atk_w, atk_y + atk_h), (255, 255, 255), 3)
        cv2.putText(
            img, "Attack!", (atk_x + int(25 * scale), atk_y + int(50 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (255, 255, 255), int(2 * scale), cv2.LINE_AA
        )
        return img

    def _draw_multiplayer_interface(
        self, img: np.ndarray, config: SyntheticFrameConfig, rng: np.random.RandomState
    ) -> np.ndarray:
        """2. MULTIPLAYER_INTERFACE: Village dimmed in background, 'Find a Match' modal dialogue."""
        self._draw_home_village(img, config, rng)
        h, w = img.shape[:2]
        scale = w / 1920.0

        # Dark overlay
        overlay = np.zeros_like(img)
        cv2.addWeighted(img, 0.4, overlay, 0.6, 0, img)

        # Modal Dialogue Box
        mw, mh = int(900 * scale), int(600 * scale)
        mx, my = (w - mw) // 2, (h - mh) // 2
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (45, 55, 65), -1)
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (190, 175, 150), int(4 * scale))

        # Title
        cv2.putText(
            img, "Multiplayer Battle", (mx + int(250 * scale), my + int(80 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (255, 255, 255), int(3 * scale), cv2.LINE_AA
        )

        # "Find a Match" big attack button
        bx, by = mx + int(250 * scale), my + int(360 * scale)
        bw, bh = int(400 * scale), int(100 * scale)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (30, 160, 230), -1)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 255, 255), int(3 * scale))
        cv2.putText(
            img, "Find a Match", (bx + int(60 * scale), by + int(60 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1 * scale, (255, 255, 255), int(3 * scale), cv2.LINE_AA
        )
        return img

    def _draw_matchmaking_interface(
        self, img: np.ndarray, config: SyntheticFrameConfig, rng: np.random.RandomState
    ) -> np.ndarray:
        """3. MATCHMAKING_INTERFACE: Iconic animated cloud screen with magnifying glass."""
        h, w = img.shape[:2]
        scale = w / 1920.0

        # Cloudy backdrop: white/light-cyan swirls
        img[:] = (235, 230, 225)
        for _ in range(40):
            cx = rng.randint(0, w)
            cy = rng.randint(0, h)
            rad = rng.randint(int(100 * scale), int(300 * scale))
            cv2.circle(img, (cx, cy), rad, (255, 250, 245), -1)

        # Magnifying glass circle in center
        center = (w // 2, h // 2)
        cv2.circle(img, center, int(70 * scale), (80, 80, 90), int(10 * scale))
        cv2.line(
            img,
            (center[0] + int(50 * scale), center[1] + int(50 * scale)),
            (center[0] + int(110 * scale), center[1] + int(110 * scale)),
            (70, 70, 80),
            int(18 * scale),
        )

        # "Searching for opponents..." text
        cv2.putText(
            img,
            "Searching for opponents...",
            (w // 2 - int(240 * scale), h // 2 + int(180 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1 * scale,
            (60, 60, 60),
            int(3 * scale),
            cv2.LINE_AA,
        )
        return img

    def _draw_enemy_base_preview(
        self, img: np.ndarray, config: SyntheticFrameConfig, rng: np.random.RandomState
    ) -> np.ndarray:
        """4. ENEMY_BASE_PREVIEW: Enemy village, available loot in top-left, Next button on bottom-right."""
        h, w = img.shape[:2]
        scale = w / 1920.0
        self._draw_grass_terrain(img, rng)

        # Enemy Town Hall in center
        cx, cy = w // 2, h // 2
        cv2.rectangle(
            img,
            (cx - int(70 * scale), cy - int(70 * scale)),
            (cx + int(70 * scale), cy + int(70 * scale)),
            (140, 60, 60),
            -1,
        )

        # Top-left Available Loot Box
        loot_box_x, loot_box_y = int(30 * scale), int(30 * scale)
        loot_w, loot_h = int(360 * scale), int(190 * scale)
        cv2.rectangle(img, (loot_box_x, loot_box_y), (loot_box_x + loot_w, loot_box_y + loot_h), (25, 25, 25), -1)
        cv2.rectangle(img, (loot_box_x, loot_box_y), (loot_box_x + loot_w, loot_box_y + loot_h), (80, 80, 80), 2)

        cv2.putText(
            img, "Available Loot:", (loot_box_x + int(15 * scale), loot_box_y + int(30 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65 * scale, (200, 200, 200), int(2 * scale), cv2.LINE_AA
        )

        # Gold text
        cv2.putText(
            img, f"{config.gold_amount:,}",
            (loot_box_x + int(60 * scale), loot_box_y + int(75 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 215, 255), int(2 * scale), cv2.LINE_AA
        )
        # Elixir text
        cv2.putText(
            img, f"{config.elixir_amount:,}",
            (loot_box_x + int(60 * scale), loot_box_y + int(120 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (220, 50, 220), int(2 * scale), cv2.LINE_AA
        )
        # Dark Elixir text
        cv2.putText(
            img, f"{config.dark_elixir_amount:,}",
            (loot_box_x + int(60 * scale), loot_box_y + int(165 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (180, 180, 180), int(2 * scale), cv2.LINE_AA
        )

        # Bottom-right "Next" button (Orange rounded button)
        nx_w, nx_h = int(180 * scale), int(75 * scale)
        nx_x, nx_y = w - nx_w - int(30 * scale), h - nx_h - int(30 * scale)
        cv2.rectangle(img, (nx_x, nx_y), (nx_x + nx_w, nx_y + nx_h), (25, 140, 235), -1)
        cv2.rectangle(img, (nx_x, nx_y), (nx_x + nx_w, nx_y + nx_h), (255, 255, 255), 2)
        cv2.putText(
            img, "Next", (nx_x + int(50 * scale), nx_y + int(48 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (255, 255, 255), int(2 * scale), cv2.LINE_AA
        )

        # Bottom troop deployment tray
        tray_h = int(110 * scale)
        tray_w = int(1000 * scale)
        tx = (w - tray_w) // 2
        ty = h - tray_h - int(10 * scale)
        cv2.rectangle(img, (tx, ty), (tx + tray_w, ty + tray_h), (30, 35, 40), -1)
        return img

    def _draw_battle_in_progress(
        self, img: np.ndarray, config: SyntheticFrameConfig, rng: np.random.RandomState
    ) -> np.ndarray:
        """5. BATTLE_IN_PROGRESS: Active battle timer (e.g. 2:45), % destruction, Surrender button."""
        h, w = img.shape[:2]
        scale = w / 1920.0
        self._draw_grass_terrain(img, rng)

        # Top-center Battle Timer Box
        timer_w, timer_h = int(220 * scale), int(60 * scale)
        timer_x = (w - timer_w) // 2
        timer_y = int(20 * scale)
        cv2.rectangle(img, (timer_x, timer_y), (timer_x + timer_w, timer_y + timer_h), (20, 20, 20), -1)
        cv2.rectangle(img, (timer_x, timer_y), (timer_x + timer_w, timer_y + timer_h), (200, 200, 200), 2)
        cv2.putText(
            img, config.battle_timer, (timer_x + int(40 * scale), timer_y + int(42 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1 * scale, (255, 255, 255), int(2 * scale), cv2.LINE_AA
        )

        # Top-left Damage / Destruction %
        dmg_w, dmg_h = int(200 * scale), int(55 * scale)
        dmg_x, dmg_y = int(30 * scale), int(20 * scale)
        cv2.rectangle(img, (dmg_x, dmg_y), (dmg_x + dmg_w, dmg_y + dmg_h), (20, 20, 20), -1)
        cv2.putText(
            img, f"Damage: {config.destruction_percent}%",
            (dmg_x + int(15 * scale), dmg_y + int(38 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75 * scale, (255, 215, 0), int(2 * scale), cv2.LINE_AA
        )

        # Bottom-left "End Battle" button (Red button)
        eb_w, eb_h = int(180 * scale), int(65 * scale)
        eb_x, eb_y = int(30 * scale), h - eb_h - int(25 * scale)
        cv2.rectangle(img, (eb_x, eb_y), (eb_x + eb_w, eb_y + eb_h), (40, 40, 200), -1)
        cv2.putText(
            img, "End Battle", (eb_x + int(20 * scale), eb_y + int(42 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (255, 255, 255), int(2 * scale), cv2.LINE_AA
        )
        return img

    def _draw_battle_result(
        self, img: np.ndarray, config: SyntheticFrameConfig, rng: np.random.RandomState
    ) -> np.ndarray:
        """6. BATTLE_RESULT: Victory / Defeat screen with stars, loot stolen, 'Return Home' button."""
        h, w = img.shape[:2]
        scale = w / 1920.0
        # Darkened battlefield background
        self._draw_grass_terrain(img, rng)
        overlay = np.zeros_like(img)
        cv2.addWeighted(img, 0.3, overlay, 0.7, 0, img)

        # Victory Banner
        banner_w, banner_h = int(800 * scale), int(120 * scale)
        bx, by = (w - banner_w) // 2, int(100 * scale)
        cv2.rectangle(img, (bx, by), (bx + banner_w, by + banner_h), (20, 160, 230), -1)
        cv2.putText(
            img, "VICTORY!", (bx + int(240 * scale), by + int(80 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.8 * scale, (255, 255, 255), int(4 * scale), cv2.LINE_AA
        )

        # 3 Stars in center
        for i in range(3):
            sx = (w // 2) - int(120 * scale) + (i * int(120 * scale))
            sy = int(300 * scale)
            star_color = (0, 215, 255) if i < config.stars else (100, 100, 100)
            cv2.circle(img, (sx, sy), int(40 * scale), star_color, -1)

        # Stolen Loot summary box
        sb_w, sb_h = int(600 * scale), int(200 * scale)
        sx, sy = (w - sb_w) // 2, int(420 * scale)
        cv2.rectangle(img, (sx, sy), (sx + sb_w, sy + sb_h), (35, 40, 45), -1)
        cv2.putText(
            img, f"Gold Stolen: {config.gold_amount:,}", (sx + int(50 * scale), sy + int(60 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 215, 255), int(2 * scale), cv2.LINE_AA
        )
        cv2.putText(
            img, f"Elixir Stolen: {config.elixir_amount:,}", (sx + int(50 * scale), sy + int(110 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (220, 50, 220), int(2 * scale), cv2.LINE_AA
        )

        # "Return Home" button
        rh_w, rh_h = int(260 * scale), int(75 * scale)
        rx, ry = (w - rh_w) // 2, int(680 * scale)
        cv2.rectangle(img, (rx, ry), (rx + rh_w, ry + rh_h), (25, 140, 235), -1)
        cv2.putText(
            img, "Return Home", (rx + int(35 * scale), ry + int(48 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (255, 255, 255), int(2 * scale), cv2.LINE_AA
        )
        return img

    def _draw_training_interface(
        self, img: np.ndarray, config: SyntheticFrameConfig, rng: np.random.RandomState
    ) -> np.ndarray:
        """7. TRAINING_INTERFACE: Barracks troop training queue modal with tabs."""
        self._draw_home_village(img, config, rng)
        h, w = img.shape[:2]
        scale = w / 1920.0

        # Modal Dialogue
        mw, mh = int(1200 * scale), int(800 * scale)
        mx, my = (w - mw) // 2, (h - mh) // 2
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (40, 45, 55), -1)
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (170, 160, 140), int(3 * scale))

        # Tabs: "Train Troops", "Brew Spells", "Siege Machines"
        tab_w = int(240 * scale)
        tab_h = int(60 * scale)
        cv2.rectangle(img, (mx + int(20 * scale), my + int(20 * scale)), (mx + int(20 * scale) + tab_w, my + int(20 * scale) + tab_h), (60, 70, 85), -1)
        cv2.putText(
            img, "Train Troops", (mx + int(45 * scale), my + int(60 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (255, 255, 255), int(2 * scale), cv2.LINE_AA
        )

        # Troop Slots grid
        for i in range(8):
            slot_x = mx + int(40 * scale) + (i % 4) * int(270 * scale)
            slot_y = my + int(120 * scale) + (i // 4) * int(280 * scale)
            cv2.rectangle(img, (slot_x, slot_y), (slot_x + int(240 * scale), slot_y + int(240 * scale)), (30, 35, 42), -1)
            cv2.rectangle(img, (slot_x, slot_y), (slot_x + int(240 * scale), slot_y + int(240 * scale)), (70, 80, 95), 2)
        return img

    def _draw_army_configuration(
        self, img: np.ndarray, config: SyntheticFrameConfig, rng: np.random.RandomState
    ) -> np.ndarray:
        """8. ARMY_CONFIGURATION: Quick train presets, troop count summary, edit army."""
        self._draw_home_village(img, config, rng)
        h, w = img.shape[:2]
        scale = w / 1920.0

        mw, mh = int(1200 * scale), int(750 * scale)
        mx, my = (w - mw) // 2, (h - mh) // 2
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (38, 43, 52), -1)
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (150, 140, 120), int(3 * scale))

        cv2.putText(
            img, "Army Overview & Presets", (mx + int(350 * scale), my + int(60 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1 * scale, (255, 255, 255), int(2 * scale), cv2.LINE_AA
        )

        # Preset rows
        for i in range(3):
            ry = my + int(110 * scale) + (i * int(180 * scale))
            cv2.rectangle(img, (mx + int(40 * scale), ry), (mx + mw - int(40 * scale), ry + int(150 * scale)), (48, 55, 66), -1)
            cv2.putText(
                img, f"Preset {i+1}: Sneaky Farming", (mx + int(70 * scale), ry + int(50 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (255, 255, 255), int(2 * scale), cv2.LINE_AA
            )
        return img

    def _draw_loading_screen(
        self, img: np.ndarray, config: SyntheticFrameConfig, rng: np.random.RandomState
    ) -> np.ndarray:
        """9. LOADING_SCREEN: Dark Clash art backdrop, logo, bottom loading progress bar."""
        h, w = img.shape[:2]
        scale = w / 1920.0
        # Deep blue/purple gradient
        for y in range(h):
            val = int(20 + 40 * (y / h))
            img[y, :] = (val, val // 2, val * 2)

        # Supercell / Clash of Clans logo text in center
        cv2.putText(
            img, "CLASH OF CLANS", (w // 2 - int(340 * scale), h // 2 - int(40 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0 * scale, (255, 255, 255), int(5 * scale), cv2.LINE_AA
        )

        # Bottom loading bar
        bar_w, bar_h = int(800 * scale), int(26 * scale)
        bx, by = (w - bar_w) // 2, h - int(120 * scale)
        cv2.rectangle(img, (bx, by), (bx + bar_w, by + bar_h), (40, 40, 40), -1)
        cv2.rectangle(img, (bx, by), (bx + int(bar_w * 0.65), by + bar_h), (25, 180, 240), -1)
        cv2.rectangle(img, (bx, by), (bx + bar_w, by + bar_h), (200, 200, 200), 2)
        cv2.putText(
            img, "Loading...", (w // 2 - int(60 * scale), by - int(15 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (220, 220, 220), int(2 * scale), cv2.LINE_AA
        )
        return img

    def _draw_connection_error(
        self, img: np.ndarray, config: SyntheticFrameConfig, rng: np.random.RandomState
    ) -> np.ndarray:
        """10. CONNECTION_ERROR: Modal alert: 'Connection Lost', 'Try Again' button."""
        h, w = img.shape[:2]
        scale = w / 1920.0
        img[:] = (20, 20, 25)

        # Error modal
        mw, mh = int(700 * scale), int(400 * scale)
        mx, my = (w - mw) // 2, (h - mh) // 2
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (45, 50, 60), -1)
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (180, 70, 70), int(3 * scale))

        cv2.putText(
            img, "Connection Lost", (mx + int(180 * scale), my + int(100 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3 * scale, (255, 255, 255), int(3 * scale), cv2.LINE_AA
        )
        cv2.putText(
            img, "Client and server are out of sync.", (mx + int(90 * scale), my + int(180 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (200, 200, 200), int(2 * scale), cv2.LINE_AA
        )

        # "Reload Game" button
        bw, bh = int(260 * scale), int(70 * scale)
        bx, by = (w - bw) // 2, my + mh - int(100 * scale)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (30, 150, 230), -1)
        cv2.putText(
            img, "Reload Game", (bx + int(35 * scale), by + int(45 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85 * scale, (255, 255, 255), int(2 * scale), cv2.LINE_AA
        )
        return img

    def _draw_unexpected_popup(
        self, img: np.ndarray, config: SyntheticFrameConfig, rng: np.random.RandomState
    ) -> np.ndarray:
        """11. UNEXPECTED_POPUP: Promotional / Event / News popup with Close 'X' in upper right."""
        self._draw_home_village(img, config, rng)
        h, w = img.shape[:2]
        scale = w / 1920.0

        # Modal
        mw, mh = int(800 * scale), int(550 * scale)
        mx, my = (w - mw) // 2, (h - mh) // 2
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (50, 45, 40), -1)
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (0, 215, 255), int(3 * scale))

        title = config.modal_title or "Special Event Offer!"
        cv2.putText(
            img, title, (mx + int(180 * scale), my + int(80 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1 * scale, (255, 255, 255), int(2 * scale), cv2.LINE_AA
        )

        # Close 'X' button
        xx, xy = mx + mw - int(50 * scale), my + int(20 * scale)
        cv2.circle(img, (xx, xy), int(20 * scale), (40, 40, 210), -1)
        cv2.putText(
            img, "X", (xx - int(9 * scale), xy + int(9 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (255, 255, 255), int(2 * scale), cv2.LINE_AA
        )
        return img

    def _draw_unknown_state(
        self, img: np.ndarray, config: SyntheticFrameConfig, rng: np.random.RandomState
    ) -> np.ndarray:
        """12. UNKNOWN: Desktop window, browser, black screen, or unstructured pattern."""
        h, w = img.shape[:2]
        # Draw random geometric shapes or windows desktop pattern
        img[:] = (60, 40, 30)
        for _ in range(15):
            x1, y1 = rng.randint(0, w), rng.randint(0, h)
            x2, y2 = rng.randint(0, w), rng.randint(0, h)
            color = tuple(int(c) for c in rng.randint(0, 255, 3))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        return img


# Module-level convenience functions
def generate_state_frame(
    state: Union[str, GameStateType] = GameStateType.HOME_VILLAGE,
    width: int = 1920,
    height: int = 1080,
    seed: int = 42,
    as_pil: bool = False,
    **kwargs: Any,
) -> Union[np.ndarray, Image.Image]:
    """Generate a synthetic full frame for testing."""
    if isinstance(state, str):
        state = GameStateType(state)
    gen = SyntheticGenerator(default_seed=seed)
    cfg = SyntheticFrameConfig(state=state, width=width, height=height, seed=seed, **kwargs)
    return gen.generate_frame(cfg, as_pil=as_pil)


def generate_resource_crop(
    text: str = "800,000",
    width: int = 160,
    height: int = 40,
    noise_level: float = 0.0,
    blur_kernel: int = 0,
    low_contrast: bool = False,
    invert: bool = False,
    as_pil: bool = False,
    **kwargs: Any,
) -> Union[np.ndarray, Image.Image]:
    """Generate an individual synthetic resource text crop."""
    gen = SyntheticGenerator()
    cfg = SyntheticCropConfig(
        text=text,
        width=width,
        height=height,
        noise_level=noise_level,
        blur_kernel=blur_kernel,
        low_contrast=low_contrast,
        invert=invert,
        **kwargs,
    )
    return gen.generate_crop(cfg, as_pil=as_pil)


def generate_timer_crop(
    time_str: str = "2:45",
    as_pil: bool = False,
) -> Union[np.ndarray, Image.Image]:
    """Generate a synthetic battle timer crop."""
    return generate_resource_crop(text=time_str, width=120, height=45, as_pil=as_pil)

