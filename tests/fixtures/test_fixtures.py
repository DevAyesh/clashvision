"""
Verification tests for synthetic generators, layout models, and recorded datasets.
Ensures fixtures generate cleanly, deterministically, and conform to the data contracts.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tests.conftest import LiveInputSafetyViolation, allow_simulator_sandbox
from tests.fixtures.layout_generator import (
    BaseLayoutType,
    LayoutGenerator,
    generate_canonical_layout,
)
from tests.fixtures.recorded_samples import (
    OCR_CORPUS,
    RECORDED_SAMPLES,
    TIMER_CORPUS,
    get_sample_by_state,
)
from tests.fixtures.synthetic_generator import (
    GameStateType,
    SyntheticCropConfig,
    SyntheticFrameConfig,
    SyntheticGenerator,
    generate_resource_crop,
    generate_state_frame,
    generate_timer_crop,
)


class TestSyntheticGenerator:
    """Tests for synthetic frame and crop generation across all states and resolutions."""

    @pytest.mark.parametrize("state", list(GameStateType))
    def test_all_12_states_1080p_generation(self, state: GameStateType) -> None:
        """Every game state must generate a valid 1920x1080x3 uint8 BGR image."""
        img = generate_state_frame(state=state, width=1920, height=1080, seed=42)
        assert isinstance(img, np.ndarray)
        assert img.shape == (1080, 1920, 3)
        assert img.dtype == np.uint8
        # Ensure image is not pure blank (has pixel variation)
        assert np.std(img) > 1.0

    @pytest.mark.parametrize("state", list(GameStateType))
    def test_all_12_states_720p_generation(self, state: GameStateType) -> None:
        """Every game state must support dynamic downscaled 1280x720 resolution."""
        img = generate_state_frame(state=state, width=1280, height=720, seed=42)
        assert isinstance(img, np.ndarray)
        assert img.shape == (720, 1280, 3)
        assert img.dtype == np.uint8
        assert np.std(img) > 1.0

    def test_deterministic_reproducibility(self) -> None:
        """Same seed and parameters must produce byte-identical images."""
        frame1 = generate_state_frame(state=GameStateType.HOME_VILLAGE, seed=12345)
        frame2 = generate_state_frame(state=GameStateType.HOME_VILLAGE, seed=12345)
        assert np.array_equal(frame1, frame2)

    def test_different_seeds_produce_different_images(self) -> None:
        """Different seeds or noise configs must produce different image arrays."""
        frame1 = generate_state_frame(state=GameStateType.HOME_VILLAGE, seed=1)
        frame2 = generate_state_frame(state=GameStateType.HOME_VILLAGE, seed=2, add_noise=True, noise_sigma=10.0)
        assert not np.array_equal(frame1, frame2)

    def test_pil_image_generation(self) -> None:
        """Fixture generator must successfully emit PIL Images when requested."""
        pil_img = generate_state_frame(state=GameStateType.ENEMY_BASE_PREVIEW, as_pil=True)
        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (1920, 1080)
        assert pil_img.mode == "RGB"

    def test_resource_crop_variations(self) -> None:
        """OCR crop generator handles text formatting, blur, noise, and inversion."""
        crop_clean = generate_resource_crop(text="800,000")
        assert crop_clean.shape == (40, 160, 3)
        assert np.std(crop_clean) > 0

        crop_noisy = generate_resource_crop(text="1.2M", noise_level=0.5)
        assert crop_noisy.shape == (40, 160, 3)

        crop_blurred = generate_resource_crop(text="50K", blur_kernel=5)
        assert crop_blurred.shape == (40, 160, 3)

        crop_inverted = generate_resource_crop(text="0", invert=True)
        assert crop_inverted.shape == (40, 160, 3)

    def test_battle_timer_crop_generation(self) -> None:
        """Battle timer crops generate with appropriate dimensions."""
        timer_crop = generate_timer_crop(time_str="2:45")
        assert timer_crop.shape == (45, 120, 3)
        assert timer_crop.dtype == np.uint8


class TestLayoutGenerator:
    """Tests for canonical base layout generation and topological features."""

    @pytest.mark.parametrize("layout_type", list(BaseLayoutType))
    def test_canonical_layouts_creation(self, layout_type: BaseLayoutType) -> None:
        """All 6 canonical layout types construct valid models."""
        model = generate_canonical_layout(layout_type)
        assert model.layout_type == layout_type
        assert len(model.buildings) > 0
        assert 0.0 <= model.exposed_collector_ratio <= 1.0
        assert isinstance(model.th_accessible, bool)
        assert isinstance(model.storages_in_core, bool)

    def test_exposed_collectors_layout_properties(self) -> None:
        """Exposed collectors layout must have 100% exposed collector ratio."""
        model = generate_canonical_layout(BaseLayoutType.EXPOSED_COLLECTORS)
        assert model.exposed_collector_ratio == 1.0
        assert model.exposed_collector_count == model.collector_count
        assert model.collector_count >= 10
        assert not model.th_accessible

    def test_core_storages_layout_properties(self) -> None:
        """Core storages layout must have 0% exposed collector ratio and storages in core."""
        model = generate_canonical_layout(BaseLayoutType.CORE_STORAGES)
        assert model.exposed_collector_ratio == 0.0
        assert model.storages_in_core is True
        assert model.storage_count >= 4

    def test_dead_base_layout_properties(self) -> None:
        """Dead base must have dead_base flag set, high loot, and high collector ratio."""
        model = generate_canonical_layout(BaseLayoutType.DEAD_BASE)
        assert model.dead_base is True
        assert model.total_gold >= 1_000_000
        assert model.total_elixir >= 1_000_000
        assert model.exposed_collector_ratio >= 0.8

    def test_town_hall_exposed_properties(self) -> None:
        """Town hall exposed layout must indicate th_accessible is True."""
        model = generate_canonical_layout(BaseLayoutType.TOWN_HALL_EXPOSED)
        assert model.th_accessible is True

    def test_air_defense_corridor_properties(self) -> None:
        """Air defense corridor layout must contain localized air defenses."""
        model = generate_canonical_layout(BaseLayoutType.AIR_DEFENSE_CORRIDOR)
        assert len(model.air_defense_locations) == 4

    def test_layout_map_rendering(self) -> None:
        """Base layout model renders a 2D numpy BGR visualization."""
        model = generate_canonical_layout(BaseLayoutType.EXPOSED_COLLECTORS)
        img = model.render_map(size=600)
        assert isinstance(img, np.ndarray)
        assert img.shape == (600, 600, 3)
        assert img.dtype == np.uint8


class TestRecordedSamplesDataset:
    """Tests verifying the recorded samples and OCR ground-truth corpora."""

    def test_all_12_states_have_recorded_samples(self) -> None:
        """Recorded sample dictionary contains ground truth for all 12 game states."""
        for state in GameStateType:
            sample = get_sample_by_state(state.value)
            assert sample is not None, f"Missing recorded sample for state {state.value}"
            assert sample.state == state.value
            assert 0.0 <= sample.confidence_expected <= 1.0

    def test_ocr_corpus_completeness(self) -> None:
        """OCR corpus contains both valid numbers and invalid edge cases."""
        assert len(OCR_CORPUS) >= 20
        valid_items = [item for item in OCR_CORPUS if item[2] is True]
        invalid_items = [item for item in OCR_CORPUS if item[2] is False]
        assert len(valid_items) >= 12
        assert len(invalid_items) >= 6

    def test_timer_corpus_completeness(self) -> None:
        """Timer corpus covers standard and edge-case strings."""
        assert len(TIMER_CORPUS) >= 6


class TestConftestSafetyInterceptor:
    """Verifies that the autouse safety fixture actively intercepts live input calls."""

    def test_live_pyautogui_click_blocked_outside_sandbox(self) -> None:
        """Attempting live pyautogui click outside simulator canvas raises LiveInputSafetyViolation."""
        try:
            import pyautogui  # type: ignore
            with pytest.raises(LiveInputSafetyViolation):
                pyautogui.click(100, 100)
        except ImportError:
            pytest.skip("pyautogui not installed in environment")

    def test_live_pyautogui_allowed_inside_sandbox(self) -> None:
        """Simulated inputs succeed when wrapped within allow_simulator_sandbox."""
        try:
            import pyautogui  # type: ignore
            with allow_simulator_sandbox():
                # Should not raise LiveInputSafetyViolation
                pyautogui.click(100, 100)
        except ImportError:
            pytest.skip("pyautogui not installed in environment")
